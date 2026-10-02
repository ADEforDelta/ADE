'''
***********************************************************************
ADE: Accurate, Inference-efficient, and Tunable Delta Compression for
     Task-specific Fine-tuned Foundation Models

Authors: Anonymous

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: models.py
- resolve_model                : local directory of a model ($ADE_MODEL_ROOT
                                 or a Hugging Face Hub snapshot).
- extract_llava_language_model : language-model part of LLaVA-1.5 as a
                                 LlamaForCausalLM checkpoint (the aligned
                                 model that is compressed).

Version: 1.0
***********************************************************************
'''

from __future__ import annotations

import glob
import os
import shutil

import torch


def resolve_model(name: str) -> str:
    """
    Local directory of a model: `name` itself, $ADE_MODEL_ROOT/<basename of name>, or a
    Hugging Face Hub snapshot.

    Parameters:
        name: Hugging Face repository id or directory.

    Returns:
        Path of a local model directory.
    """
    if os.path.isdir(name):
        return name
    root = os.environ.get("ADE_MODEL_ROOT", "./models")
    local = os.path.join(root, os.path.basename(name.rstrip("/")))
    if os.path.isdir(local):
        return local
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=name)


def extract_llava_language_model(llava_model: str, base_model: str, out_dir: str):
    """
    Build the aligned language model of LLaVA-1.5 as a LlamaForCausalLM checkpoint, with the
    configuration and tokenizer of Vicuna-v1.5.

    Parameters:
        llava_model: LLaVA-1.5 checkpoint directory (original format).
        base_model: Vicuna-v1.5 directory.
        out_dir: Output directory.
    """
    from safetensors.torch import load_file, save_file
    tmp = out_dir + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    sd = {}
    shards = sorted(glob.glob(os.path.join(llava_model, 'pytorch_model-*.bin')))
    for shard in shards:
        sd.update(torch.load(shard, map_location="cpu", weights_only=False))
    for shard in sorted(glob.glob(os.path.join(llava_model, '*.safetensors'))):
        sd.update(load_file(shard))
    keep = {k: v.contiguous() for k, v in sd.items()
            if k.startswith(("model.layers.", "model.embed_tokens.", "model.norm.", "lm_head."))}
    if not keep:
        raise RuntimeError(f"no language-model weights found under {llava_model}")
    save_file(keep, os.path.join(tmp, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(base_model):
        if f.endswith(('.bin', '.safetensors', '.index.json')) or os.path.isdir(os.path.join(base_model, f)):
            continue
        shutil.copy2(os.path.join(base_model, f), os.path.join(tmp, f))
    os.replace(tmp, out_dir)
    print(f"[llava] language model of {llava_model} -> {out_dir} ({len(keep)} tensors)")
