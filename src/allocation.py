'''
***********************************************************************
ADE: Accurate, Inference-efficient, and Tunable Delta Compression for
     Task-specific Fine-tuned Foundation Models

Authors: Anonymous

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: allocation.py
- I1. Output-aware capacity allocation (Section 3.2, Appendices C and D).
- collect_input_covariance : calibration activations X of every linear
                             module, kept as the covariance X X^T / T.
- probe_rank               : uniform-allocation probe rank r~ (eq. 13).
- sensitivity              : sensitivity S(W_a) at the probe rank (eq. 4).
- bit_budget / allocate_ranks : sensitivity-proportional ranks under the
                             bit budget of eqs. (11)-(12), with the storage
                             cost c_l = (b_quant + (b_s + b_z)/G)(d_o + d_i).

Version: 1.0
***********************************************************************
'''

from __future__ import annotations

import gc
import math
from fractions import Fraction

import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def collect_input_covariance(model, calib_ids, device, chunk_size=4):
    """
    Run the calibration data through the aligned model layer by layer and accumulate,
    for every linear module, the covariance of its input activations X.

    Parameters:
        model: HF causal LM (the aligned model) whose linear modules are probed.
        calib_ids: Tokenized calibration inputs of shape (n_samples, seq_len).
        device: Device on which the forward passes run.
        chunk_size: Mini-batch size for the chunked forward passes.

    Returns:
        Mapping from module name to its (d_i, d_i) input covariance X X^T / T (CPU, fp32).
    """
    model.eval()
    base = model.model

    all_hidden = []
    for i in range(0, calib_ids.shape[0], chunk_size):
        ids = calib_ids[i:i + chunk_size].to(device)
        all_hidden.append(base.embed_tokens(ids).cpu())
    all_hidden = torch.cat(all_hidden, dim=0)

    seq_len = calib_ids.shape[1]
    dtype = next(base.parameters()).dtype
    mask_val = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((seq_len, seq_len), mask_val, device=device, dtype=dtype),
        diagonal=1,
    )[None, None, :, :]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    rotary_emb = getattr(base, 'rotary_emb', None)

    covariances = {}
    for layer_idx, layer in enumerate(tqdm(base.layers, desc="Calibration")):
        layer_covs, layer_counts = {}, {}
        hooks = []

        def make_hook(full_name):
            """
            Create a forward hook that accumulates the input covariance of one linear module.

            Parameters:
                full_name: Fully qualified module name used as the storage key.

            Returns:
                Forward hook to register on a torch.nn.Linear module.
            """
            def fn(module, inp, out):
                """
                Forward hook: accumulate x^T x and the number of tokens seen.

                Parameters:
                    module: The torch.nn.Linear the hook is attached to.
                    inp: Tuple of positional inputs to the module.
                    out: Module output (unused).
                """
                x = inp[0].detach().to(torch.float32).reshape(-1, inp[0].shape[-1])
                xtx = x.T @ x
                if full_name not in layer_covs:
                    layer_covs[full_name] = torch.zeros_like(xtx)
                    layer_counts[full_name] = 0
                layer_covs[full_name] += xtx
                layer_counts[full_name] += x.shape[0]
            return fn

        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full = f"model.layers.{layer_idx}.{name}"
                hooks.append(module.register_forward_hook(make_hook(full)))

        next_hidden = []
        for i in range(0, all_hidden.shape[0], chunk_size):
            chunk = all_hidden[i:i + chunk_size].to(device)
            bs = chunk.shape[0]
            pos = position_ids.expand(bs, -1)
            kwargs = dict(attention_mask=causal.expand(bs, -1, -1, -1),
                          position_ids=pos)
            if rotary_emb is not None:
                kwargs['position_embeddings'] = rotary_emb(chunk, pos)
            out = layer(chunk, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            next_hidden.append(out.cpu())

        for h in hooks:
            h.remove()
        for full in layer_covs:
            covariances[full] = (layer_covs[full] / layer_counts[full]).cpu()
        del layer_covs, layer_counts

        all_hidden = torch.cat(next_hidden, dim=0)
        del next_hidden
        gc.collect()
        torch.cuda.empty_cache()

    return covariances


def _exact_ratio(alpha) -> Fraction:
    """
    Turn a compression ratio into an exact fraction (0.0625 -> 1/16).

    Parameters:
        alpha: Compression ratio as float, int or Fraction.

    Returns:
        Fraction equal to alpha.
    """
    if isinstance(alpha, Fraction):
        return alpha
    if isinstance(alpha, int):
        return Fraction(alpha)
    return Fraction(repr(float(alpha)))


def default_zero_bits(b_z, b_quant: int) -> int:
    """
    Bit-width b_z of a stored zero-point: by default the code bit-width (INT-b zero-points).

    Parameters:
        b_z: Explicit zero-point bit-width, or None for the default.
        b_quant: Bit-width b_quant of the quantized factors.

    Returns:
        Zero-point bit-width in bits.
    """
    return int(b_quant) if b_z is None else int(b_z)


def group_cost_bits(d_o: int, d_i: int, b_quant: int = 4, b_s: int = 16,
                    b_z=None, group_size: int = 128) -> int:
    """
    Storage cost, in bits, of G ranks of a module (Appendix C):
    G c_l = (G b_quant + b_s + b_z)(d_o + d_i).

    Parameters:
        d_o: Output dimension.
        d_i: Input dimension.
        b_quant: Bit-width of the quantized factors.
        b_s: Bit-width of each per-group scale.
        b_z: Bit-width of each per-group zero-point (None -> b_quant).
        group_size: Group size G along the rank axis.

    Returns:
        Cost in bits of G ranks of this module.
    """
    G = max(1, int(group_size))
    bz = default_zero_bits(b_z, b_quant)
    return (G * int(b_quant) + int(b_s) + bz) * (int(d_o) + int(d_i))


def probe_rank(d_o: int, d_i: int, alpha: float, b_orig: int = 16, b_quant: int = 4,
               b_s: int = 16, b_z=None, group_size: int = 128) -> int:
    """
    Probe rank r~ of uniform allocation at ratio alpha (Appendix D, eq. 13):
    r~ = clip(floor(alpha b_orig d_o d_i / c_l), 1, min(d_o, d_i)).

    Parameters:
        d_o: Output dimension.
        d_i: Input dimension.
        alpha: Target compression ratio.
        b_orig: Bit-width of the original weights.
        b_quant: Bit-width of the quantized factors.
        b_s: Bit-width of each per-group scale.
        b_z: Bit-width of each per-group zero-point (None -> b_quant).
        group_size: Group size G along the rank axis.

    Returns:
        Probe rank clipped to [1, min(d_o, d_i)].
    """
    G = max(1, int(group_size))
    num = _exact_ratio(alpha) * int(b_orig) * int(d_i) * int(d_o) * G
    den = group_cost_bits(d_o, d_i, b_quant, b_s, b_z, G)
    r = math.floor(num / den)
    return max(1, min(r, min(d_i, d_o)))


def _diag_v_cov_v(V: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """
    Compute the diagonal of V^T cov V column by column.

    Parameters:
        V: Right singular vectors (as columns), shape (d_i, rank).
        cov: Input covariance, shape (d_i, d_i).

    Returns:
        Non-negative tensor of shape (rank,) holding (V^T cov V)_{kk}.
    """
    cov_v = cov @ V
    return (V * cov_v).sum(dim=0).clamp(min=0)


@torch.no_grad()
def sensitivity(sigma: torch.Tensor,
                V: torch.Tensor,
                cov: torch.Tensor,
                W_b: torch.Tensor,
                delta: torch.Tensor,
                r_probe: int,
                device: str = "cuda") -> float:
    """
    Sensitivity S(W_a) at the probe rank r~ (eq. 4):
    ||W_a X - (W_b + Delta W_r~) X||_F^2 / ||W_a X||_F^2.

    Parameters:
        sigma: Singular values of Delta W, shape (rank,).
        V: Right singular vectors (as columns), shape (d_i, rank).
        cov: Input covariance C of the module, shape (d_i, d_i).
        W_b: Base weight W_b, shape (d_o, d_i).
        delta: Delta weight Delta W = W_a - W_b, shape (d_o, d_i).
        r_probe: Probe rank r~.
        device: Device for the computation.

    Returns:
        Scalar sensitivity score.
    """
    full_rank = sigma.shape[0]
    r = max(0, min(int(r_probe), full_rank))
    if r >= full_rank:
        return 0.0

    sigma = sigma.to(device=device, dtype=torch.float32)
    sigma2 = sigma.pow(2)
    V_d = V.to(device=device, dtype=torch.float32)
    cov_d = cov.to(device=device, dtype=torch.float32)
    diag = _diag_v_cov_v(V_d, cov_d)
    weighted_tail = (sigma2 * diag)[r:].sum()

    W_a = (W_b.to(device=device, dtype=torch.float32)
           + delta.to(device=device, dtype=torch.float32))
    cov_W_a_T = cov_d @ W_a.T
    denom = (W_a * cov_W_a_T.T).sum().clamp(min=1e-30)
    del W_a, cov_W_a_T
    return float((weighted_tail / denom).item())


def bit_budget(layer_specs: list[dict], alpha: float, b_orig: int = 16) -> int:
    """
    Total bit budget alpha * b_orig * sum_l d_o d_i of the compressed delta (eq. 12).

    Parameters:
        layer_specs: List of dicts with 'd_o' / 'd_i' per module.
        alpha: Target compression ratio.
        b_orig: Bit-width of the original weights.

    Returns:
        Budget in bits (floor of the exact rational value).
    """
    return math.floor(_exact_ratio(alpha) * int(b_orig)
                      * sum(int(s['d_o']) * int(s['d_i']) for s in layer_specs))


@torch.no_grad()
def allocate_ranks(scores: dict[str, float],
                   layer_specs: list[dict],
                   budget_bits: int,
                   r_min: int = 128,
                   group_size: int = 128,
                   b_quant: int = 4,
                   kappa: float = 1.0,
                   b_s: int = 16,
                   b_z=None,
                   ) -> dict[str, int]:
    """
    Sensitivity-proportional rank allocation under the bit budget (Appendix C, eqs. 11-12),
    r_l = R_min + R_sen^(l), in multiples of G.

    Parameters:
        scores: Mapping module name -> sensitivity S (non-negative).
        layer_specs: List of dicts with 'name', 'd_o', 'd_i' per module.
        budget_bits: Total bit budget (see bit_budget).
        r_min: Minimum rank R_min per module (multiple of group_size).
        group_size: Quantization group size G along the rank axis.
        b_quant: Bit-width of the quantized factors.
        kappa: Exponent on the sensitivities (1 = rule of the paper).
        b_s: Bit-width of each per-group scale.
        b_z: Bit-width of each per-group zero-point (None -> b_quant).

    Returns:
        Dict module name -> allocated rank.
    """
    if group_size > 1 and r_min % group_size != 0:
        raise ValueError(f"r_min={r_min} must be a multiple of group_size={group_size}")
    n = len(layer_specs)
    names = [s['name'] for s in layer_specs]
    unit = group_size
    # cost of G ranks and of one rank (c_l)
    unit_cost = [group_cost_bits(s['d_o'], s['d_i'], b_quant, b_s, b_z, unit)
                 for s in layer_specs]
    cost = [unit_cost[i] / unit for i in range(n)]
    u_max = [min(s['d_i'], s['d_o']) // unit for s in layer_specs]
    u_min = [min(r_min // unit, u_max[i]) for i in range(n)]
    floor_bits = (r_min // unit) * sum(unit_cost)      # = R_min * sum_l c_l
    B_bits = budget_bits - floor_bits
    if B_bits < 0:
        raise ValueError(f"budget {budget_bits} bits < R_min floor {floor_bits} bits "
                         f"(r_min={r_min}, n_modules={n})")
    if budget_bits > sum(unit_cost[i] * u_max[i] for i in range(n)):
        raise ValueError("the bit budget exceeds the rank cap of the modules")
    s_list = [max(0.0, float(scores[name])) ** kappa for name in names]
    s_sum = sum(s_list)
    if s_sum <= 0:
        s_list = [1.0] * n
        s_sum = float(n)
    share = [x / s_sum for x in s_list]
    c_S = sum(share[i] * cost[i] for i in range(n))           # average cost of one rank
    B_rank = B_bits / c_S
    # round to the nearest group
    units = [min(max(int(round((r_min + share[i] * B_rank) / unit)), u_min[i]), u_max[i]) for i in range(n)]
    used = sum(unit_cost[i] * units[i] for i in range(n))
    by_score = sorted(range(n), key=lambda i: s_list[i], reverse=True)
    for i in reversed(by_score):                 # over budget: trim least sensitive
        while used > budget_bits and units[i] > u_min[i]:
            units[i] -= 1
            used -= unit_cost[i]
    for i in by_score:                           # under budget: top up most sensitive
        step = unit_cost[i]
        while units[i] < u_max[i] and budget_bits - used >= step:
            units[i] += 1
            used += step
    assert used <= budget_bits
    assert budget_bits - used < max(unit_cost)
    return {names[i]: units[i] * unit for i in range(n)}
