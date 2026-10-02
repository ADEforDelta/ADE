'''
***********************************************************************
ADE: Accurate, Inference-efficient, and Tunable Delta Compression for
     Task-specific Fine-tuned Foundation Models

Authors: Anonymous

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: serving.py
- Model used for evaluation: the base weights stay dense and the
  compressed delta is applied at every forward pass with ADE's kernels,
  as in the serving path of Section 3.3.1 / Appendix K.
- ADEKernelLinear : y = W_b x + Delta(x). For T <= decode_max_tokens
                    tokens in the forward, factored application with the
                    packed INT4 decode kernels; otherwise reconstructed
                    application W_b + U^ V^^T with the INT4 Tensor Core
                    kernel (eq. 7). Both read the same packed codes.
- build_kernel_model : aligned model (FP16) whose compressed linear
                    modules are replaced by ADEKernelLinear.

Version: 1.0
***********************************************************************
'''

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptation import LINEAR_MODULE_NAMES
from kernels import decode_forward_packed, decode_kernel_status, pack_int4, recon_int4, recon_kernel_status
from quantization import PACKED_FIELDS, reconstruct_delta
from utils import load_packed


class ADEKernelLinear(nn.Module):
    """
    Linear module W_b + U^ V^^T applied with ADE's kernels (FP16): factored application for
    T <= decode_max_tokens tokens, reconstructed application otherwise.
    """

    def __init__(self, W_b: torch.Tensor, packed: dict, group_size: int = 128,
                 decode_max_tokens: int = 128, use_decode: bool = True, use_recon: bool = True,
                 split_k: bool = False):
        """
        Parameters:
            W_b: Base weight (d_o, d_i).
            packed: Packed factors of the module (gamma / rho folded into the scales).
            group_size: Group size G (128).
            decode_max_tokens: Largest T handled by factored application.
            use_decode: Use the decode kernels (otherwise float reconstruction for every T).
            use_recon: Use the INT4 Tensor Core reconstruction (otherwise float reconstruction).
            split_k: Allow split-K launch plans for the decode kernels.
        """
        super().__init__()
        self.group_size = group_size
        self.decode_max_tokens = int(decode_max_tokens)
        self.use_decode, self.use_recon, self.split_k = use_decode, use_recon, split_k
        self.out_features, self.in_features = W_b.shape
        self.register_buffer("weight", W_b.detach().to(torch.float16).contiguous())
        p4 = pack_int4(packed, group_size)
        for k in ("uq", "vq", "R", "C"):
            self.register_buffer(f"p4_{k}", p4[k])
        if not (use_decode and use_recon):          # float reconstruction fallback
            for k in PACKED_FIELDS:
                self.register_buffer(f"packed_{k}", packed[k].clone())
        self._p4 = None

    def _stored_form(self):
        """
        Stored form on the module's device.

        Returns:
            Dict uq, vq, R, C.
        """
        if self._p4 is None or self._p4["uq"].device != self.p4_uq.device:
            self._p4 = {"uq": self.p4_uq, "vq": self.p4_vq, "R": self.p4_R, "C": self.p4_C}
        return self._p4

    def _float_reconstructed(self):
        """
        W_b + U^ V^^T with the PyTorch reconstruction of eq. (7) (fallback path).

        Returns:
            Merged FP16 weight.
        """
        packed = {k: getattr(self, f"packed_{k}") for k in PACKED_FIELDS}
        return (self.weight.float() + reconstruct_delta(packed, self.group_size)).to(torch.float16)

    def forward(self, x):
        """
        Parameters:
            x: Input (..., d_i).

        Returns:
            Output (..., d_o) in FP16.
        """
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.float16).contiguous()
        T = x2.shape[0]
        if T <= self.decode_max_tokens and self.use_decode:
            y = decode_forward_packed(x2, F.linear(x2, self.weight), self._stored_form(),
                                      self.group_size, split_k=self.split_k)
        elif self.use_recon:
            y = F.linear(x2, recon_int4(self._stored_form(), self.weight, self.group_size))
        else:
            y = F.linear(x2, self._float_reconstructed())
        return y.reshape(*shape[:-1], self.out_features)


def _decoder_layers(model):
    """
    Decoder layers of a LLaMA-family causal LM or of a llava-hf LLaVA model.

    Parameters:
        model: LlamaForCausalLM or LlavaForConditionalGeneration.

    Returns:
        The ModuleList of decoder layers.
    """
    lm = getattr(getattr(model, "model", None), "language_model", None)   # LLaVA (model.language_model)
    if lm is None:
        lm = getattr(model, "language_model", None)                      # LLaVA (older layout)
    if lm is not None:
        return getattr(lm, "model", lm).layers
    return model.model.layers


def build_kernel_model(task: str, aligned_path: str, base_path: str, packed_path: str,
                       llava_hf_path: str | None = None, device: str = "cuda",
                       decode_max_tokens: int = 128, split_k: bool = False):
    """
    FP16 evaluation model: the aligned model (llava-hf for LLaVA) with every compressed
    linear module replaced by an ADEKernelLinear.

    Parameters:
        task: "math", "code" or "multimodal".
        aligned_path: Aligned language model directory (math / code).
        base_path: Base model directory.
        packed_path: Packed compressed delta (compressed.pt or tuned.pt).
        llava_hf_path: llava-hf checkpoint (multimodal).
        device: CUDA device.
        decode_max_tokens: Largest T for factored application.
        split_k: Allow split-K launch plans for the decode kernels.

    Returns:
        (model, tokenizer or processor).
    """
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    dec_ok, dec_reason = decode_kernel_status(device)
    rec_ok, rec_reason = recon_kernel_status(device)
    if not rec_ok:
        print(f"[kernels] WARNING: INT4 Tensor Core reconstruction unavailable ({rec_reason}); "
              f"inputs with T > {decode_max_tokens} tokens use float reconstruction")
    if not dec_ok:
        print(f"[kernels] WARNING: packed decode kernels unavailable ({dec_reason}); "
              f"every forward uses float reconstruction")
    if dec_ok:
        print(f"[kernels] factored application (packed decode kernels) for T <= {decode_max_tokens} "
              f"tokens, reconstructed application ({'INT4 Tensor Core kernel' if rec_ok else 'float'}) above")

    if task == "multimodal":
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(llava_hf_path, torch_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(llava_hf_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(aligned_path, torch_dtype=torch.float16)
        processor = AutoTokenizer.from_pretrained(aligned_path)
    base_sd = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16).state_dict()
    per_module, meta = load_packed(packed_path)
    G = int(meta["group_size"])

    layers = _decoder_layers(model)
    n = 0
    for li in range(len(layers)):
        for sn in LINEAR_MODULE_NAMES:
            prefix = f"model.layers.{li}.{sn}"
            if prefix not in per_module:
                continue
            parts = sn.split(".")
            parent = layers[li].get_submodule(".".join(parts[:-1]))
            setattr(parent, parts[-1], ADEKernelLinear(base_sd[prefix + ".weight"], per_module[prefix], G,
                                                       decode_max_tokens, use_decode=dec_ok,
                                                       use_recon=rec_ok and dec_ok, split_k=split_k))
            n += 1
    del base_sd, per_module
    model = model.to(device).eval()
    print(f"[kernels] {n} linear modules apply the compressed delta at every forward")
    return model, processor
