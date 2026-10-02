'''
***********************************************************************
ADE: Accurate, Inference-efficient, and Tunable Delta Compression for
     Task-specific Fine-tuned Foundation Models

Authors: Anonymous

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: utils.py
- Command-line arguments and YAML configurations.
- Storage of the compressed delta (packed INT4 codes, scales and
  zero-points per module).

Version: 1.0
***********************************************************************
'''

from __future__ import annotations

import argparse
import os

import torch
import yaml

from quantization import PACKED_FIELDS


PACKED_FORMAT = "ade_packed_v1"

TASKS = ("math", "code", "multimodal")

# configuration keys and defaults
CONFIG_DEFAULTS = {
    "task": None,                 # math | code | multimodal
    "base_model": None,           # backbone W_b
    "aligned_model": None,        # aligned model W_a
    "llava_hf_model": None,       # llava-hf checkpoint (multimodal)
    "alpha": 0.0625,              # target compression ratio
    "calib_dataset": None,        # metamath | magicoder | alpaca
    "r_min": 128,                 # R_min per module
    "alloc_kappa": 1.0,           # sensitivity exponent kappa
    "epochs": 0,                  # fold-in epochs (0 = ADE_-L)
    "train_data": None,           # metamath | metamath_gsm | magicoder | alpaca
    "num_train_samples": 0,
    "learning_rate": 1.0e-3,
    "weight_decay": 0.0,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "batch_size": 1,
    "grad_accum": 6,
    "lambda_lm": 1.0,
    "lambda_kd": 1.0,
    "lambda_re": 3000.0,
    "kd_reduction": "batchmean",  # batchmean | tokenmean
}


def get_args(argv=None):
    """
    Parse the command-line arguments of the ADE pipeline.

    Parameters:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description="ADE: compress (I1, I2), adapt (I3) and evaluate")
    parser.add_argument("--config", required=True, help="YAML configuration (configs/*.yaml)")
    parser.add_argument("--output_dir", required=True, help="output directory of this run")
    parser.add_argument("--alpha", type=float, default=None,
                        help="target compression ratio; overrides the configuration")
    parser.add_argument("--alloc_kappa", type=float, default=None,
                        help="exponent kappa on the sensitivities in the rank allocation (rank share "
                             "beyond R_min proportional to S^kappa; 1 = the paper's rule); overrides "
                             "the configuration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher_device", default=None,
                        help="device of the aligned (teacher) model during adaptation, e.g. cuda:1 "
                             "(default: same device as the adapted model)")
    parser.add_argument("--bits", type=int, default=4, help="bit-width b_quant of the factor codes")
    parser.add_argument("--group_size", type=int, default=128, help="group size G along the rank axis")
    parser.add_argument("--scale_bits", type=int, default=16, help="bit-width b_s of each group scale")
    parser.add_argument("--zero_bits", type=int, default=None,
                        help="bit-width b_z of each group zero-point (default: --bits)")
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_seqlen", type=int, default=2048)
    parser.add_argument("--calib_chunk", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=2048,
                        help="sequence length of the adaptation data")
    parser.add_argument("--grad_ckpt", action="store_true",
                        help="gradient checkpointing during adaptation (lower memory)")
    parser.add_argument("--decode_max_tokens", type=int, default=128,
                        help="forward passes with at most this many tokens use factored application "
                             "(packed decode kernels), longer ones reconstructed application (INT4 "
                             "Tensor Core kernel)")
    parser.add_argument("--split_k", action="store_true",
                        help="allow split-K launch plans for the decode kernels (faster; FP32 atomics "
                             "make the result not bitwise reproducible)")
    parser.add_argument("--delta", default=None,
                        help="evaluate an existing compressed delta (compressed.pt or tuned.pt of an "
                             "earlier run) instead of running stages 1-3")
    parser.add_argument("--eval_limit", type=int, default=0,
                        help="evaluate only the first N items (smoke tests; 0 = full benchmark)")
    parser.add_argument("--skip_eval", action="store_true", help="stop after writing the compressed delta")
    args = parser.parse_args(argv)
    if args.zero_bits is None:
        args.zero_bits = args.bits
    return args


def load_config(path: str) -> dict:
    """
    Load a YAML configuration and fill in the defaults.

    Parameters:
        path: YAML file.

    Returns:
        Dict with every key of CONFIG_DEFAULTS.
    """
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    unknown = sorted(set(cfg) - set(CONFIG_DEFAULTS))
    if unknown:
        raise ValueError(f"{path}: unknown configuration keys {unknown}")
    out = dict(CONFIG_DEFAULTS)
    out.update(cfg)
    for key in ("task", "base_model", "aligned_model", "calib_dataset"):
        if out[key] is None:
            raise ValueError(f"{path}: '{key}' is required")
    if out["task"] not in TASKS:
        raise ValueError(f"{path}: task must be one of {TASKS}")
    if out["task"] == "multimodal" and out["llava_hf_model"] is None:
        raise ValueError(f"{path}: multimodal configurations need 'llava_hf_model'")
    if int(out["epochs"]) > 0 and out["train_data"] is None:
        raise ValueError(f"{path}: 'train_data' is required when epochs > 0")
    return out


def save_packed(path: str, per_module: dict, meta: dict) -> None:
    """
    Save a compressed delta: per module the codes, scales and zero-points of both factors.

    Parameters:
        path: Output .pt file.
        per_module: Mapping module prefix -> packed dict (PACKED_FIELDS).
        meta: Metadata (bits, group_size, ...), stored under "__meta__".
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    flat = {"__meta__": dict(meta, format=PACKED_FORMAT)}
    for prefix, packed in per_module.items():
        for field in PACKED_FIELDS:
            flat[f"{prefix}.{field}"] = packed[field]
    torch.save(flat, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_packed(path: str):
    """
    Load a compressed delta written by save_packed.

    Parameters:
        path: .pt file.

    Returns:
        (per_module, meta).
    """
    flat = torch.load(path, map_location="cpu")
    meta = flat.pop("__meta__", None)
    if not isinstance(meta, dict) or meta.get("format") != PACKED_FORMAT:
        raise ValueError(f"{path} is not an ADE compressed delta (meta={meta!r})")
    per_module = {}
    for k, v in flat.items():
        prefix, field = k.rsplit(".", 1)
        per_module.setdefault(prefix, {})[field] = v
    for prefix, packed in per_module.items():
        missing = [f for f in PACKED_FIELDS if f not in packed]
        if missing:
            raise ValueError(f"{prefix} misses {missing} in {path}")
    return per_module, meta


def print_table(title, d, indent=2, max_val=100):
    """
    Print a dict as a titled two-column table.

    Parameters:
        title: Table title.
        d: Dict to print.
        indent: Left indentation.
        max_val: Maximum printed value width.
    """
    print(title)
    print("-" * max(len(title), 40))
    width = max((len(str(k)) for k in d), default=0)
    for k, v in d.items():
        s = str(v)
        if len(s) > max_val:
            s = s[:max_val - 3] + "..."
        print(f"{' ' * indent}{str(k):<{width}}  {s}")
    print()
