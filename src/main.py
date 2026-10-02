'''
***********************************************************************
ADE: Accurate, Inference-efficient, and Tunable Delta Compression for
     Task-specific Fine-tuned Foundation Models

Authors: Anonymous

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: main.py
- End-to-end pipeline of ADE (Algorithm 1, Appendix B) for one aligned
  model:
  1. Output-aware capacity allocation (I1): calibration activations,
     SVD of every delta, sensitivity at the probe rank, rank allocation
     under the bit budget (codes + scales + zero-points).
  2. Inference-aligned quantization (I2): Sigma^{1/2} absorption, shared
     rotation H, Q_r / Q_c along the rank axis -> compressed.pt (ADE_-L).
  3. Light fold-in adaptation (I3, epochs > 0): train gamma / rho, fold
     them into the scales -> tuned.pt (ADE).
  4. Evaluation model: the base weights stay dense and the compressed
     delta is applied at every forward pass with ADE's kernels
     (serving.py): factored application with the packed INT4 decode
     kernels for short inputs, reconstructed application with the INT4
     Tensor Core kernel (eq. 7) for long ones.
  5. Evaluation on the task's benchmark (GSM8K / MBPP+ / TextVQA).

Version: 1.0
***********************************************************************
'''

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import copy
import gc
import json
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptation import (LINEAR_MODULE_NAMES, attach_fold_in, export_folded, free_memory,
                        same_device, train_fold_in)
from allocation import (allocate_ranks, bit_budget, collect_input_covariance, group_cost_bits,
                        probe_rank, sensitivity)
from data import get_calibration, get_train_data
from evaluate import evaluate
from models import extract_llava_language_model, resolve_model
from quantization import compress_module, reconstruct_delta
from serving import build_kernel_model
from utils import get_args, load_config, load_packed, print_table, save_packed


def stage1_allocation_inputs(args, cfg, base_path, aligned_path):
    """
    Stage 1 of I1: SVD factors of every delta and their sensitivities at the probe rank.

    Parameters:
        args: Command-line arguments.
        cfg: Configuration.
        base_path: Base model directory.
        aligned_path: Aligned model directory.

    Returns:
        (cache of (U, Sigma, V, W_b) per weight, layer_specs, scores S(W_a), b_orig).
    """
    cache_root = os.environ.get("ADE_CACHE_DIR", "./cache/stage1")
    key = (f"{cfg['task']}__{os.path.basename(base_path.rstrip('/'))}"
           f"__{os.path.basename(aligned_path.rstrip('/'))}"
           f"__{cfg['calib_dataset']}_{args.calib_samples}x{args.calib_seqlen}_s{args.seed}")
    cache_dir = os.path.join(cache_root, key)
    factors_path = os.path.join(cache_dir, "factors.pt")
    scores_path = os.path.join(cache_dir, f"scores_a{args.alpha:g}_b{args.bits}_s{args.scale_bits}"
                                          f"_z{args.zero_bits}_g{args.group_size}.json")
    factors_cached = os.path.isfile(factors_path)
    hit = factors_cached and os.path.isfile(scores_path)

    base_mod = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    b_orig = next(iter(base_mod.parameters())).element_size() * 8
    cache, layer_specs, scores = {}, [], {}

    if hit:
        print(f"[stage 1] cache hit: {cache_dir}")
        f = torch.load(factors_path, map_location="cpu")
        with open(scores_path) as fh:
            scores = json.load(fh)
        base_sd = base_mod.state_dict()
        for spec in f["layer_specs"]:
            U, S, V = f["factors"][spec["name"]]
            cache[spec["name"]] = (U, S, V, base_sd[spec["name"]].detach().cpu())
            layer_specs.append(dict(spec))
        del f, base_sd, base_mod
        gc.collect()
        return cache, layer_specs, scores, b_orig

    # base weights stay on the CPU
    aligned = AutoModelForCausalLM.from_pretrained(aligned_path, torch_dtype=torch.bfloat16).to(args.device)
    tok = AutoTokenizer.from_pretrained(aligned_path)
    calib_ids = get_calibration(cfg["calib_dataset"], tok, n_samples=args.calib_samples,
                                seqlen=args.calib_seqlen, seed=args.seed)
    print(f"[stage 1] calibration data {cfg['calib_dataset']}: {tuple(calib_ids.shape)}")
    covariances = collect_input_covariance(aligned, calib_ids, device=args.device,
                                           chunk_size=args.calib_chunk)

    aligned_sd = aligned.state_dict()
    for k, W_b in tqdm(base_mod.state_dict().items(), desc="SVD + sensitivity"):
        if not (("self_attn" in k or "mlp" in k) and k.endswith(".weight")):
            continue
        W_b_dev = W_b.to(args.device)
        delta = aligned_sd[k] - W_b_dev
        d_o, d_i = W_b.shape
        cov = covariances[k.replace(".weight", "")]
        U, S, V = torch.svd(delta.to(device=args.device, dtype=torch.float32))
        S_cpu = S.detach().cpu().to(torch.float32)
        r_probe = probe_rank(d_o, d_i, args.alpha, b_orig, args.bits, args.scale_bits,
                             args.zero_bits, args.group_size)
        scores[k] = sensitivity(S, V, cov, W_b=W_b_dev, delta=delta, r_probe=r_probe, device=args.device)
        cache[k] = (U.detach().cpu().to(torch.float16), S_cpu,
                    V.detach().cpu().to(torch.float16), W_b.detach().cpu())
        layer_specs.append({"name": k, "d_i": d_i, "d_o": d_o})
        del U, S, V, delta, W_b_dev
        torch.cuda.empty_cache()
    del aligned_sd, base_mod, aligned, covariances, calib_ids
    free_memory()

    os.makedirs(cache_dir, exist_ok=True)
    kept = False
    if factors_cached:
        # keep the cached factors
        try:
            f = torch.load(factors_path, map_location="cpu", mmap=True)
            if sorted(sp["name"] for sp in f["layer_specs"]) == sorted(cache):
                for k, (U_c, S_c, V_c) in f["factors"].items():
                    cache[k] = (U_c, S_c, V_c, cache[k][3])
                kept = True
            del f
        except Exception as e:
            print(f"[stage 1] cached factors unusable ({type(e).__name__}: {e}); rewriting them")
    if not kept:
        torch.save({"layer_specs": layer_specs,
                    "factors": {k: (U, S, V) for k, (U, S, V, _) in cache.items()}},
                   factors_path + ".tmp")
        os.replace(factors_path + ".tmp", factors_path)
    with open(scores_path + ".tmp", "w") as fh:
        json.dump(scores, fh)
    os.replace(scores_path + ".tmp", scores_path)
    print(f"[stage 1] cached {cache_dir}")
    return cache, layer_specs, scores, b_orig


def compress_and_adapt(args, cfg, base_path, aligned_path):
    """
    Stages 1-3: output-aware capacity allocation (I1), inference-aligned quantization (I2) and,
    for epochs > 0, light fold-in adaptation (I3).

    Parameters:
        args: Command-line arguments.
        cfg: Configuration.
        base_path: Base model directory.
        aligned_path: Aligned model directory.

    Returns:
        Path of the packed compressed delta to evaluate (compressed.pt or tuned.pt).
    """
    # ---- 1. output-aware capacity allocation (I1) ----------------------------
    t0 = time.time()
    cache, layer_specs, scores, b_orig = stage1_allocation_inputs(args, cfg, base_path, aligned_path)
    probes = {}
    for s in layer_specs:
        shape = (s["d_o"], s["d_i"])
        if shape not in probes:
            probes[shape] = probe_rank(s["d_o"], s["d_i"], args.alpha, b_orig, args.bits,
                                       args.scale_bits, args.zero_bits, args.group_size)
    print("[stage 1] probe ranks r~ (d_o x d_i -> r~): "
          + ", ".join(f"{o}x{i}->{r}" for (o, i), r in probes.items()))

    budget = bit_budget(layer_specs, args.alpha, b_orig)
    ranks = allocate_ranks(scores, layer_specs, budget, r_min=int(cfg["r_min"]),
                           group_size=args.group_size, b_quant=args.bits,
                           kappa=float(cfg["alloc_kappa"]), b_s=args.scale_bits, b_z=args.zero_bits)
    r_vals = list(ranks.values())
    code_bits = sum(args.bits * ranks[s["name"]] * (s["d_o"] + s["d_i"]) for s in layer_specs)
    stored_bits = sum(group_cost_bits(s["d_o"], s["d_i"], args.bits, args.scale_bits, args.zero_bits,
                                      args.group_size) * (ranks[s["name"]] // args.group_size)
                      for s in layer_specs)
    dense_bits = b_orig * sum(s["d_o"] * s["d_i"] for s in layer_specs)
    assert all(r % args.group_size == 0 for r in r_vals)
    assert stored_bits <= budget, "codes + scales + zero-points exceed the budget"
    print(f"[stage 1] ranks min/mean/max = {min(r_vals)}/{sum(r_vals) / len(r_vals):.1f}/{max(r_vals)}; "
          f"stored (codes + scales + zero-points) = 1/{dense_bits / stored_bits:.2f} of the original "
          f"weights (alpha = 1/{1 / args.alpha:g}, {stored_bits}/{budget} bits), codes only = "
          f"1/{dense_bits / max(code_bits, 1):.2f}   [{time.time() - t0:.0f} s]")

    # ---- 2. inference-aligned quantization (I2) ------------------------------
    t0 = time.time()
    generator = torch.Generator().manual_seed(args.seed)
    layer_ids = sorted({int(s["name"].split(".")[2]) for s in layer_specs})
    compressed, err_sum, n_mod = {}, 0.0, 0
    for li in tqdm(layer_ids, desc="Quantization"):
        for name in LINEAR_MODULE_NAMES:
            key = f"model.layers.{li}.{name}.weight"
            r = ranks.get(key, 0)
            if r == 0:                  # base weight only
                continue
            U, S, V, _ = cache[key]
            U = U[:, :r].contiguous().to(torch.bfloat16).to(args.device)
            S = S[:r].contiguous().to(torch.bfloat16).to(args.device)
            V = V[:, :r].contiguous().to(torch.bfloat16).to(args.device)
            packed = compress_module(U, S, V, bits=args.bits, group_size=args.group_size,
                                     generator=generator)
            with torch.no_grad():
                delta_r = U.float() @ torch.diag(S.float()) @ V.float().T
                delta_q = reconstruct_delta({k: v.to(delta_r.device) for k, v in packed.items()},
                                            args.group_size)
                err_sum += ((delta_q - delta_r).norm() / delta_r.norm().clamp_min(1e-8)).item()
                n_mod += 1
            compressed[f"model.layers.{li}.{name}"] = packed
            del U, S, V, delta_r, delta_q
            if args.device != "cpu":
                torch.cuda.empty_cache()
    del cache
    gc.collect()
    meta = {"bits": args.bits, "group_size": args.group_size, "scale_bits": args.scale_bits,
            "zero_bits": args.zero_bits, "alpha": args.alpha}
    compressed_path = os.path.join(args.output_dir, "compressed.pt")
    save_packed(compressed_path, compressed, meta)
    print(f"[stage 2] {n_mod} modules, mean relative error of U^V^^T vs the rank-r SVD "
          f"{err_sum / max(n_mod, 1):.4f}; wrote {compressed_path}   [{time.time() - t0:.0f} s]")

    # ---- 3. light fold-in adaptation (I3) ------------------------------------
    delta_path = compressed_path
    if int(cfg["epochs"]) > 0:
        t0 = time.time()
        torch.manual_seed(args.seed)
        teacher_device = args.teacher_device or args.device
        split_teacher = not same_device(teacher_device, args.device)
        per_module, _ = load_packed(compressed_path)
        student = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
        teacher = AutoModelForCausalLM.from_pretrained(aligned_path, torch_dtype=torch.bfloat16)
        n = attach_fold_in(student, per_module, args.group_size)
        del per_module
        student = student.to(args.device)
        teacher = teacher.to(teacher_device)
        # embeddings and LM head of the aligned model
        if split_teacher:
            student.set_input_embeddings(copy.deepcopy(teacher.get_input_embeddings()).to(args.device))
            student.set_output_embeddings(copy.deepcopy(teacher.get_output_embeddings()).to(args.device))
        else:
            student.set_input_embeddings(teacher.get_input_embeddings())
            student.set_output_embeddings(teacher.get_output_embeddings())
        student.config.vocab_size = teacher.config.vocab_size
        student.config.pad_token_id = teacher.config.pad_token_id
        print(f"[stage 3] {n} fold-in modules (gamma, rho)")

        tokenizer = AutoTokenizer.from_pretrained(aligned_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ids, labels = get_train_data(cfg["train_data"], tokenizer, int(cfg["num_train_samples"]),
                                     context_length=args.context_length, seed=args.seed)
        print(f"[stage 3] training data {cfg['train_data']}: {ids.shape[0]} sequences of {args.context_length}")
        loader = DataLoader(TensorDataset(ids, labels), batch_size=int(cfg["batch_size"]),
                            shuffle=False, num_workers=4, pin_memory=True)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        hp = {k: cfg[k] for k in ("epochs", "learning_rate", "weight_decay", "warmup_ratio",
                                  "max_grad_norm", "grad_accum", "lambda_lm", "lambda_kd",
                                  "lambda_re", "kd_reduction")}
        train_fold_in(student, teacher, loader, hp, args.device,
                      teacher_device=teacher_device if split_teacher else None,
                      grad_ckpt=args.grad_ckpt)
        del teacher
        free_memory()
        tuned = export_folded(student)
        del student
        free_memory()
        delta_path = os.path.join(args.output_dir, "tuned.pt")
        save_packed(delta_path, tuned, meta)
        del tuned
        print(f"[stage 3] wrote {delta_path}   [{time.time() - t0:.0f} s]")

    return delta_path


def main(argv=None):
    """
    Run the ADE pipeline for one configuration.

    Parameters:
        argv: Optional argument list.
    """
    args = get_args(argv)
    cfg = load_config(args.config)
    if args.alpha is None:
        args.alpha = float(cfg["alpha"])
    if args.alloc_kappa is not None:
        cfg["alloc_kappa"] = args.alloc_kappa
    if not torch.cuda.is_available():
        args.device = "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump(dict(cfg, alpha=args.alpha), fh, sort_keys=False)
    print_table("ARGUMENTS", vars(args))
    print_table(f"CONFIGURATION ({args.config})", cfg)

    base_path = resolve_model(cfg["base_model"])
    if cfg["task"] == "multimodal":
        # LLaVA-1.5: its language model
        llava_path = resolve_model(cfg["aligned_model"])
        llava_hf_path = resolve_model(cfg["llava_hf_model"])
        aligned_path = os.path.join(os.environ.get("ADE_MODEL_ROOT", "./models"),
                                    os.path.basename(llava_path.rstrip("/")) + "-lm")
        if not os.path.isdir(aligned_path):
            extract_llava_language_model(llava_path, base_path, aligned_path)
    else:
        aligned_path = resolve_model(cfg["aligned_model"])
    print(f"base model   : {base_path}\naligned model: {aligned_path}\n")

    delta_path = args.delta or compress_and_adapt(args, cfg, base_path, aligned_path)

    # ---- 4-5. evaluation: delta applied by ADE's kernels -----------------------
    if args.skip_eval:
        return
    t0 = time.time()
    model, processor = build_kernel_model(cfg["task"], aligned_path, base_path, delta_path,
                                          llava_hf_path=llava_hf_path if cfg["task"] == "multimodal" else None,
                                          device=args.device, decode_max_tokens=args.decode_max_tokens,
                                          split_k=args.split_k)
    print(f"[stage 4] evaluation model ready   [{time.time() - t0:.0f} s]")
    t0 = time.time()
    res = evaluate(cfg["task"], model, processor, os.path.join(args.output_dir, "eval"), limit=args.eval_limit)
    print(f"[stage 5] {json.dumps(res)}   [{time.time() - t0:.0f} s]")

if __name__ == "__main__":
    main(sys.argv[1:])
