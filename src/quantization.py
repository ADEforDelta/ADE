'''
***********************************************************************
ADE: Accurate, Inference-efficient, and Tunable Delta Compression for
     Task-specific Fine-tuned Foundation Models

Authors: Anonymous

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: quantization.py
- I2. Inference-aligned quantization (Section 3.3, Appendix F).
- rotation_matrix   : shared orthogonal rotation H (normalized Sylvester
                      Hadamard for power-of-two ranks, seeded random
                      orthogonal matrix from the QR decomposition of a
                      Gaussian matrix otherwise).
- quantize_rowwise  : Q_r, row-wise groups of G along the rank axis
                      (applied to U Sigma^{1/2} H).
- quantize_colwise  : Q_c, column-wise groups of G along the rank axis
                      (applied to H^T Sigma^{1/2} V^T).
- compress_module   : eq. (5) for one module -> packed INT4 codes, 16-bit
                      scales and INT4 zero-points.
- reconstruct_delta : low-bit reconstruction of eq. (7) in PyTorch
                      (integer contraction per group, one rescale per
                      group); reference of the CUDA kernels and their
                      fallback on GPUs without INT4 Tensor Cores.

Version: 1.0
***********************************************************************
'''

from __future__ import annotations

import math

import torch


PACKED_FIELDS = ("U_codes", "U_scale", "U_zero", "Vt_codes", "Vt_scale", "Vt_zero")


def _is_pow2(n):
    """
    Check whether an integer is a positive power of two.

    Parameters:
        n: Integer to test.

    Returns:
        True if n is a positive power of two, else False.
    """
    return n > 0 and (n & (n - 1)) == 0


def _sylvester_hadamard(k, dtype=torch.float32):
    """
    Build the normalized k x k Sylvester Hadamard matrix.

    Parameters:
        k: Matrix order (a power of two).
        dtype: Torch dtype of the returned matrix.

    Returns:
        The k x k orthonormal Hadamard matrix (entries +-1/sqrt(k)).
    """
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < k:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(k)


def _random_orthogonal(k, dtype=torch.float32, generator=None):
    """
    Sample a k x k random orthogonal matrix from the QR decomposition of a Gaussian matrix.

    Parameters:
        k: Matrix order.
        dtype: Torch dtype for sampling.
        generator: torch.Generator for reproducibility.

    Returns:
        A k x k orthogonal matrix (signs fixed by the diagonal of R).
    """
    A = torch.randn(k, k, dtype=dtype, generator=generator)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def rotation_matrix(rank, generator=None):
    """
    Shared rotation H of eq. (8) (Appendix F): normalized Sylvester Hadamard for a
    power-of-two rank, seeded random orthogonal matrix otherwise.

    Parameters:
        rank: Rank r of the module (a multiple of G).
        generator: torch.Generator consumed for non-power-of-two ranks.

    Returns:
        An (r, r) orthogonal matrix.
    """
    if _is_pow2(rank):
        H = _sylvester_hadamard(rank)
    else:
        H = _random_orthogonal(rank, generator=generator)
    return torch.block_diag(H)


def _pin_zero_in_range(xmin, xmax, scale, zero, maxq):
    """
    Keep every integer zero-point inside the code range [0, 2^b - 1]; groups of one sign
    are extended to include 0.

    Parameters:
        xmin: Per-group minimum (keepdim).
        xmax: Per-group maximum (keepdim).
        scale: Per-group scale.
        zero: Per-group zero-point.
        maxq: Largest code, 2^b - 1.

    Returns:
        (scale, zero) with every zero-point in [0, maxq].
    """
    lo = zero < 0                     # all values > 0: zero-point 0
    hi = zero > maxq                  # all values < 0: zero-point maxq
    if not bool((lo | hi).any()):
        return scale, zero
    new_scale = torch.where(lo, xmax.clamp(min=1e-5) / maxq, (-xmin).clamp(min=1e-5) / maxq)
    scale = torch.where(lo | hi, new_scale, scale)
    zero = torch.where(lo, torch.zeros_like(zero), torch.where(hi, torch.full_like(zero, float(maxq)), zero))
    return scale, zero


def quantize_rowwise(X, bits, group_size):
    """
    Q_r of eq. (5): asymmetric uniform quantization of each row of X in groups of G
    consecutive entries along the rank axis (eq. 2).

    Parameters:
        X: Float tensor of shape (d_o, r), here U Sigma^{1/2} H.
        bits: Bit-width b of the codes.
        group_size: Group size G along the rank axis.

    Returns:
        codes: uint8 codes in [0, 2^b - 1], shape (d_o, r).
        scale: bfloat16 per-group scales u_{o,g}, shape (d_o, r / G).
        zero: bfloat16 per-group integer zero-points mu_{o,g} in [0, 2^b - 1], shape (d_o, r / G).
    """
    maxq = 2 ** bits - 1
    d_o, rank = X.shape
    assert rank % group_size == 0
    n_groups = rank // group_size
    X_r = X.reshape(d_o, n_groups, group_size)
    xmin = X_r.amin(dim=-1, keepdim=True)
    xmax = X_r.amax(dim=-1, keepdim=True)
    both_zero = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both_zero, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(both_zero, torch.full_like(xmax, 1.0), xmax)
    scale = (xmax - xmin).clamp(min=1e-5) / maxq
    zero = torch.round(-xmin / scale)
    scale, zero = _pin_zero_in_range(xmin, xmax, scale, zero, maxq)
    q = torch.clamp(torch.round(X_r / scale) + zero, 0, maxq).to(torch.uint8)
    return (q.reshape(d_o, rank),
            scale.squeeze(-1).to(torch.bfloat16),
            zero.squeeze(-1).to(torch.bfloat16))


def quantize_colwise(X, bits, group_size):
    """
    Q_c of eq. (5): asymmetric uniform quantization of each column of X in groups of G
    consecutive entries along the rank axis (eq. 2).

    Parameters:
        X: Float tensor of shape (r, d_i), here H^T Sigma^{1/2} V^T.
        bits: Bit-width b of the codes.
        group_size: Group size G along the rank axis.

    Returns:
        codes: uint8 codes in [0, 2^b - 1], shape (r, d_i).
        scale: bfloat16 per-group scales v_{g,j}, shape (r / G, d_i).
        zero: bfloat16 per-group integer zero-points nu_{g,j} in [0, 2^b - 1], shape (r / G, d_i).
    """
    maxq = 2 ** bits - 1
    rank, d_i = X.shape
    assert rank % group_size == 0
    n_groups = rank // group_size
    X_r = X.reshape(n_groups, group_size, d_i)
    xmin = X_r.amin(dim=1, keepdim=True)
    xmax = X_r.amax(dim=1, keepdim=True)
    both_zero = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both_zero, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(both_zero, torch.full_like(xmax, 1.0), xmax)
    scale = (xmax - xmin).clamp(min=1e-5) / maxq
    zero = torch.round(-xmin / scale)
    scale, zero = _pin_zero_in_range(xmin, xmax, scale, zero, maxq)
    q = torch.clamp(torch.round(X_r / scale) + zero, 0, maxq).to(torch.uint8)
    return (q.reshape(rank, d_i),
            scale.squeeze(1).to(torch.bfloat16),
            zero.squeeze(1).to(torch.bfloat16))


def dequantize_rowwise(codes, scale, zero, group_size):
    """
    Dequantize Q_r codes (eq. 6): U^_{o,k} = u_{o,g} (U_{o,k} - mu_{o,g}).

    Parameters:
        codes: uint8 codes of shape (d_o, r).
        scale: Per-group scales of shape (d_o, r / G).
        zero: Per-group zero-points of shape (d_o, r / G).
        group_size: Group size G.

    Returns:
        Float32 tensor of shape (d_o, r).
    """
    d_o, rank = codes.shape
    n_groups = rank // group_size
    q = codes.reshape(d_o, n_groups, group_size).to(torch.float32)
    s = scale.to(torch.float32).unsqueeze(-1)
    z = zero.to(torch.float32).round().unsqueeze(-1)
    return ((q - z) * s).reshape(d_o, rank)


def dequantize_colwise(codes, scale, zero, group_size):
    """
    Dequantize Q_c codes (eq. 6): V^T^_{k,j} = v_{g,j} (V^T_{k,j} - nu_{g,j}).

    Parameters:
        codes: uint8 codes of shape (r, d_i).
        scale: Per-group scales of shape (r / G, d_i).
        zero: Per-group zero-points of shape (r / G, d_i).
        group_size: Group size G.

    Returns:
        Float32 tensor of shape (r, d_i).
    """
    rank, d_i = codes.shape
    n_groups = rank // group_size
    q = codes.reshape(n_groups, group_size, d_i).to(torch.float32)
    s = scale.to(torch.float32).unsqueeze(1)
    z = zero.to(torch.float32).round().unsqueeze(1)
    return ((q - z) * s).reshape(rank, d_i)


def compress_module(U, S, V, bits, group_size, generator):
    """
    Inference-aligned quantization of one module's truncated SVD (eqs. 5 and 8):
    Q_r(U Sigma^{1/2} H) and Q_c(H^T Sigma^{1/2} V^T).

    Parameters:
        U: Left singular vectors, shape (d_o, r).
        S: Singular values, shape (r,).
        V: Right singular vectors (as columns), shape (d_i, r).
        bits: Bit-width of the codes.
        group_size: Group size G along the rank axis (r must be a multiple of G).
        generator: torch.Generator for non-power-of-two ranks.

    Returns:
        Dict of CPU tensors U_codes, U_scale, U_zero, Vt_codes, Vt_scale, Vt_zero.
    """
    U = U.float()
    S = S.float().clamp(min=1e-8)
    Vt = V.transpose(-1, -2).contiguous().float()

    rank = S.shape[0]
    if rank % group_size != 0:
        raise ValueError(f"compress_module: rank={rank} is not a multiple of group_size={group_size}")

    sqrt_S = torch.sqrt(S)
    U = U * sqrt_S.unsqueeze(0)
    Vt = sqrt_S.unsqueeze(1) * Vt

    H = rotation_matrix(rank, generator=generator).to(U.device)
    U = U @ H
    Vt = H.T @ Vt

    U_codes, U_scale, U_zero = quantize_rowwise(U, bits, group_size)
    Vt_codes, Vt_scale, Vt_zero = quantize_colwise(Vt, bits, group_size)
    return {
        "U_codes": U_codes.cpu(),
        "U_scale": U_scale.cpu(),
        "U_zero": U_zero.cpu(),
        "Vt_codes": Vt_codes.cpu(),
        "Vt_scale": Vt_scale.cpu(),
        "Vt_zero": Vt_zero.cpu(),
    }


def reconstruct_delta(packed, group_size):
    """
    Low-bit reconstruction of eq. (7) in PyTorch (integer contraction, one rescale per
    group); fallback of the CUDA kernels.

    Parameters:
        packed: Dict with U_codes, U_scale, U_zero, Vt_codes, Vt_scale, Vt_zero.
        group_size: Group size G.

    Returns:
        Float32 delta of shape (d_o, d_i), on the device of the packed tensors.
    """
    U_q, V_q = packed["U_codes"], packed["Vt_codes"]
    n_groups = U_q.shape[1] // group_size
    acc = torch.zeros(U_q.shape[0], V_q.shape[1], device=U_q.device, dtype=torch.float32)
    for g in range(n_groups):
        cs = slice(g * group_size, (g + 1) * group_size)
        u_i = (U_q[:, cs].to(torch.int16)
               - packed["U_zero"][:, g].to(torch.int16).unsqueeze(1)).to(torch.int8)
        v_i = (V_q[cs, :].to(torch.int16)
               - packed["Vt_zero"][g, :].to(torch.int16).unsqueeze(0)).to(torch.int8)
        if u_i.is_cuda and u_i.shape[0] % 8 == 0 and v_i.shape[1] % 8 == 0:
            prod = torch._int_mm(u_i.contiguous(), v_i.contiguous())
        else:
            prod = torch.matmul(u_i.to(torch.int32), v_i.to(torch.int32))
        row = packed["U_scale"][:, g].to(torch.float32)
        col = packed["Vt_scale"][g, :].to(torch.float32)
        acc += prod.to(torch.float32) * (row.unsqueeze(1) * col.unsqueeze(0))
        del u_i, v_i, prod
    return acc
