'''
***********************************************************************
ADE: Accurate, Inference-efficient, and Tunable Delta Compression for
     Task-specific Fine-tuned Foundation Models

Authors: Anonymous

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: kernels.py
- ADE's two CUDA kernels (Section 3.3.1, Appendices E and K). Both read
  the same stored form of a compressed module (pack_int4): the 4-bit
  codes of U and V^T packed eight per 32-bit word along the rank axis,
  plus per-group FP32 tables derived from the scales and zero-points.
- Reconstructed application (long inputs): W_b + U^ V^^T of eq. (7).
  Each group of G = 128 ranks is contracted on the raw codes with INT4
  Tensor Core instructions (mma.sync.m16n8k64.u4.u4.s32, INT32
  accumulation); the zero-points enter through precomputed code sums,
      <U - mu, V^T - nu> = <U, V^T> - mu colsum(V^T) - nu rowsum(U) + G mu nu,
  and each group is rescaled once in FP32.
- Factored application (short inputs): y = y_dense + U^ (V^^T x).
  Kernel A computes h = V^^T x with the scale v applied to the input once
  per group (Appendix E.3); kernel B computes y_dense + sum_g u_g (h_g
  (U - mu)_g) with one rescale per group. Both use FP16 mma.m16n8k16 with
  FP32 accumulation; (code - zero) is formed exactly in FP16.
- Requirements: a CUDA toolkit (nvcc 12.x). The reconstruction kernel
  needs INT4 Tensor Cores, i.e. compute capability 8.x (sm_80 / sm_86 /
  sm_89); the decode kernels run on sm_80 and newer. Both are compiled on
  first use for the architecture of the current GPU.

Version: 1.0
***********************************************************************
'''

from __future__ import annotations

import os
import shutil

import torch
from torch.utils.cpp_extension import load_inline


def _cdiv(a, b):
    """
    Ceiling division.

    Parameters:
        a: Dividend.
        b: Divisor.

    Returns:
        ceil(a / b).
    """
    return -(-a // b)


def _capability(device):
    """
    Compute capability of a CUDA device.

    Parameters:
        device: Device.

    Returns:
        (major, minor), or None when the device is not a CUDA GPU.
    """
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return None
    return torch.cuda.get_device_capability(torch.device(device))


def _have_nvcc():
    """
    Whether a CUDA compiler is available for building the extensions.

    Returns:
        True if CUDA_HOME is set up or nvcc is on the PATH.
    """
    from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME is not None or shutil.which("nvcc") is not None


def recon_kernel_status(device="cuda"):
    """
    Whether the INT4 Tensor Core reconstruction kernel can run on `device`.

    Parameters:
        device: Target device.

    Returns:
        (ok, reason).
    """
    cap = _capability(device)
    if cap is None:
        return False, "no CUDA device"
    if cap[0] != 8:
        return False, (f"compute capability {cap[0]}.{cap[1]} has no INT4 Tensor Core MMA "
                       f"(mma.m16n8k64.u4 needs sm_80 / sm_86 / sm_89)")
    if not _have_nvcc():
        return False, "nvcc not found (set CUDA_HOME to a CUDA toolkit)"
    return True, f"sm_{cap[0]}{cap[1]}"


def decode_kernel_status(device="cuda"):
    """
    Whether the packed INT4 decode kernels can run on `device`.

    Parameters:
        device: Target device.

    Returns:
        (ok, reason).
    """
    cap = _capability(device)
    if cap is None:
        return False, "no CUDA device"
    if cap[0] < 8:
        return False, f"compute capability {cap[0]}.{cap[1]} < 8.0 (cp.async / mma.m16n8k16 need sm_80+)"
    if not _have_nvcc():
        return False, "nvcc not found (set CUDA_HOME to a CUDA toolkit)"
    return True, f"sm_{cap[0]}{cap[1]}"


def _arch_flags(device):
    """
    nvcc flags for the architecture of the current GPU.

    Parameters:
        device: CUDA device.

    Returns:
        (arch string, list of flags).
    """
    major, minor = torch.cuda.get_device_capability(torch.device(device))
    arch = f"{major}{minor}"
    return arch, [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


# ======================================================================
# Stored form shared by both kernels
# ======================================================================
def _pack_nibbles(codes_u8):
    """
    Pack 4-bit codes eight per 32-bit word along the last axis.

    Parameters:
        codes_u8: uint8 codes in [0, 15] of shape (..., K), K a multiple of 8.

    Returns:
        int32 tensor of shape (..., K / 8); code k occupies bits 4k..4k+3 of its word.
    """
    K = codes_u8.shape[-1]
    assert K % 8 == 0
    c = codes_u8.to(torch.int64).reshape(*codes_u8.shape[:-1], K // 8, 8)
    shifts = torch.arange(8, device=codes_u8.device, dtype=torch.int64) * 4
    words = (c << shifts).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words)   # same bit pattern as int32
    return words.to(torch.int32).contiguous()


def pack_int4(packed, group_size=128):
    """
    Stored form of one compressed module read by both kernels.

    Parameters:
        packed: Dict with U_codes, U_scale, U_zero, Vt_codes, Vt_scale, Vt_zero.
        group_size: Group size G (128).

    Returns:
        Dict with uq, vq (packed codes of U, V) and R, C (FP32 per-group tables).
    """
    uq8 = packed["U_codes"].to(torch.uint8)
    vq8 = packed["Vt_codes"].to(torch.uint8)
    M, rank = uq8.shape
    N = vq8.shape[1]
    ng = rank // group_size
    us = packed["U_scale"].float()                     # [M, ng]
    mu = packed["U_zero"].float().round()              # [M, ng]
    vs = packed["Vt_scale"].float()                    # [ng, N]
    nu = packed["Vt_zero"].float().round()             # [ng, N]
    rowsum = uq8.view(M, ng, group_size).float().sum(-1)      # [M, ng]
    colsum = vq8.view(ng, group_size, N).float().sum(1)       # [ng, N]
    R = torch.stack([us, us * mu, us * rowsum, us * mu * group_size], -1).contiguous()
    C = torch.stack([vs, vs * colsum, vs * nu, torch.zeros_like(vs)], -1).transpose(0, 1).contiguous()
    return {"uq": _pack_nibbles(uq8), "vq": _pack_nibbles(vq8.t().contiguous()), "R": R, "C": C}


# ======================================================================
# INT4 Tensor Core reconstruction (reconstructed application, eq. 7)
# ======================================================================
_INT4_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BM 64
#define BN 128
#define WARPS_M 1
#define WARPS_N 4
#define WM (BM / WARPS_M)   // 64
#define WN (BN / WARPS_N)   // 32
#define MT (WM / 16)        // 4 m16 sub-tiles per warp
#define NT (WN / 8)         // 4 n8 sub-tiles per warp
#define NTHREADS (32 * WARPS_M * WARPS_N)
#define GW 16               // words per row per group (128 codes)
#define SROW 20             // padded smem row stride (words)
#define A_WORDS (BM * SROW)
#define B_WORDS (BN * SROW)
#define R_WORDS (BM * 4)
#define C_WORDS (BN * 4)
#define BUF_WORDS (A_WORDS + B_WORDS + R_WORDS + C_WORDS)

// D = A(16x64, u4, row) * B(64x8, u4, col) + C, s32 accumulate.
__device__ __forceinline__ void mma_u4(int32_t* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// stage one group's code tiles and R / C rows into smem
__device__ __forceinline__ void stage_group(uint32_t* buf,
                                            const uint32_t* __restrict__ uq, const uint32_t* __restrict__ vq,
                                            const float* __restrict__ R, const float* __restrict__ C,
                                            int m_base, int n_base, int kw, int NG, int g, int tid) {
    uint32_t* As = buf;
    uint32_t* Bs = buf + A_WORDS;
    uint32_t* Rs = buf + A_WORDS + B_WORDS;
    uint32_t* Cs = Rs + R_WORDS;
    #pragma unroll
    for (int c = tid; c < (BM + BN) * 4; c += NTHREADS) {          // 16 B chunks of codes
        const int row = c >> 2, chunk = c & 3;
        if (row < BM)
            cp_async16(As + row * SROW + chunk * 4, uq + (size_t)(m_base + row) * kw + g * GW + chunk * 4);
        else
            cp_async16(Bs + (row - BM) * SROW + chunk * 4, vq + (size_t)(n_base + row - BM) * kw + g * GW + chunk * 4);
    }
    for (int c = tid; c < BM + BN; c += NTHREADS) {                // one float4 per row / column
        if (c < BM)
            cp_async16(Rs + c * 4, R + ((size_t)(m_base + c) * NG + g) * 4);
        else
            cp_async16(Cs + (c - BM) * 4, C + ((size_t)(n_base + c - BM) * NG + g) * 4);
    }
    cp_async_commit();
}

// uq: [M, RANK/8], vq: [N, RANK/8] packed codes; R: [M, NG, 4], C: [N, NG, 4] fp32
// out = base + sum_g (R0 C0 P - R1 C1 - R2 C2 + R3 C2)
__global__ void __launch_bounds__(NTHREADS)
ade_recon_int4_kernel(const uint32_t* __restrict__ uq, const uint32_t* __restrict__ vq,
                      const float* __restrict__ R, const float* __restrict__ C,
                      const __half* __restrict__ base, __half* __restrict__ out,
                      int M, int N, int RANK, int NG) {
    __shared__ __align__(16) uint32_t smem[2][BUF_WORDS];   // [buf][A | B | R | C]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int mb = blockIdx.y * BM, nb = blockIdx.x * BN;
    const int m0 = mb + wm * WM, n0 = nb + wn * WN;
    const int kw = RANK / 8;
    const int gid = lane >> 2, tig = lane & 3;

    float acc[MT][NT][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

    stage_group(smem[0], uq, vq, R, C, mb, nb, kw, NG, 0, tid);

    for (int g = 0; g < NG; ++g) {
        const int cur = g & 1;
        if (g + 1 < NG) {
            stage_group(smem[cur ^ 1], uq, vq, R, C, mb, nb, kw, NG, g + 1, tid);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        const uint32_t* As = smem[cur];
        const uint32_t* Bs = smem[cur] + A_WORDS;
        const float* Rs = reinterpret_cast<const float*>(smem[cur] + A_WORDS + B_WORDS);
        const float* Cs = Rs + R_WORDS;

        int32_t p[MT][NT][4];
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int j = 0; j < NT; ++j)
                #pragma unroll
                for (int r = 0; r < 4; ++r) p[i][j][r] = 0;

        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {           // two k64 steps per 128-group
            uint32_t a[MT][4], b[NT][2];
            #pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r0 = wm * WM + i * 16 + gid;
                a[i][0] = As[r0 * SROW + ks * 8 + tig];
                a[i][1] = As[(r0 + 8) * SROW + ks * 8 + tig];
                a[i][2] = As[r0 * SROW + ks * 8 + 4 + tig];
                a[i][3] = As[(r0 + 8) * SROW + ks * 8 + 4 + tig];
            }
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int c0 = wn * WN + j * 8 + gid;
                b[j][0] = Bs[c0 * SROW + ks * 8 + tig];
                b[j][1] = Bs[c0 * SROW + ks * 8 + 4 + tig];
            }
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j) {
                    mma_u4(p[i][j], a[i], b[j]);
                }
        }
        // per-group rescale
        float r0v[MT][2], r1v[MT][2], r2v[MT][2], r3v[MT][2];
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float4 rv = *reinterpret_cast<const float4*>(Rs + (wm * WM + i * 16 + gid + h * 8) * 4);
                r0v[i][h] = rv.x; r1v[i][h] = rv.y; r2v[i][h] = rv.z; r3v[i][h] = rv.w;
            }
        float c0v[NT][2], c1v[NT][2], c2v[NT][2];
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float4 cv = *reinterpret_cast<const float4*>(Cs + (wn * WN + j * 8 + 2 * tig + h) * 4);
                c0v[j][h] = cv.x; c1v[j][h] = cv.y; c2v[j][h] = cv.z;
            }
        __syncthreads();   // smem[cur] free for refill
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int j = 0; j < NT; ++j)
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int h = r >> 1, w = r & 1;
                    acc[i][j][r] += r0v[i][h] * c0v[j][w] * (float)p[i][j][r]
                                  - r1v[i][h] * c1v[j][w]
                                  - r2v[i][h] * c2v[j][w]
                                  + r3v[i][h] * c2v[j][w];
                }
    }

    // epilogue: add base, write fp16
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = m0 + i * 16 + gid + h * 8;
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int col = n0 + j * 8 + 2 * tig;
                const size_t off = (size_t)row * N + col;
                const __half2 bs = *reinterpret_cast<const __half2*>(base + off);
                const float2 bf = __half22float2(bs);
                float2 o;
                o.x = bf.x + acc[i][j][h * 2 + 0];
                o.y = bf.y + acc[i][j][h * 2 + 1];
                *reinterpret_cast<__half2*>(out + off) = __float22half2_rn(o);
            }
        }
}

torch::Tensor ade_recon_int4(torch::Tensor uq, torch::Tensor vq, torch::Tensor R, torch::Tensor C,
                             torch::Tensor base, int64_t G) {
    const int M = base.size(0), N = base.size(1);
    const int RANK = uq.size(1) * 8, NG = RANK / G;
    TORCH_CHECK(M % BM == 0 && N % BN == 0, "M must be a multiple of 64, N of 128");
    TORCH_CHECK(G == 128, "kernel is specialised for group size 128");
    auto out = torch::empty_like(base);
    dim3 grid(N / BN, M / BM);
    ade_recon_int4_kernel<<<grid, NTHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(uq.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(vq.data_ptr<int32_t>()), R.data_ptr<float>(), C.data_ptr<float>(),
        reinterpret_cast<const __half*>(base.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), M, N, RANK, NG);
    return out;
}
"""

_INT4_CPP = r"""
#include <torch/extension.h>
torch::Tensor ade_recon_int4(torch::Tensor uq, torch::Tensor vq, torch::Tensor R, torch::Tensor C,
                             torch::Tensor base, int64_t G);
"""

_int4_ext = None


def _load_int4(device):
    """
    Compile (on first use) and return the reconstruction extension.

    Parameters:
        device: CUDA device selecting the target architecture.

    Returns:
        Extension module exposing ade_recon_int4.
    """
    global _int4_ext
    if _int4_ext is None:
        arch, flags = _arch_flags(device)
        _int4_ext = load_inline(
            name=f"ade_recon_int4_sm{arch}", cpp_sources=_INT4_CPP, cuda_sources=_INT4_CUDA,
            functions=["ade_recon_int4"],
            extra_cuda_cflags=["-O3", "--use_fast_math", *flags], extra_cflags=["-O3"],
            build_directory=os.environ.get("ADE_KERNEL_BUILD_DIR"),
            verbose=os.environ.get("ADE_KERNEL_VERBOSE") == "1")
    return _int4_ext


def recon_int4(p4, base_fp16, group_size=128):
    """
    W_b + U^ V^^T of one module (FP16) with the INT4 Tensor Core kernel.

    Parameters:
        p4: Stored form (pack_int4) on the GPU.
        base_fp16: FP16 base weight W_b (d_o, d_i) on the same GPU.
        group_size: Group size G (128).

    Returns:
        W_b + U^ V^^T in FP16.
    """
    return _load_int4(base_fp16.device).ade_recon_int4(p4["uq"], p4["vq"], p4["R"], p4["C"],
                                                        base_fp16.contiguous(), group_size)


# ======================================================================
# Packed INT4 decode kernels (factored application)
# ======================================================================
_DECODE_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t h2u(__half2 v) { return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ __half2 u2h(uint32_t v) { return *reinterpret_cast<__half2*>(&v); }
// zero-point = rint(scale*zero / scale)
// (lo, hi) 4-bit codes -> half2 (1024 + lo, 1024 + hi), exact
__device__ __forceinline__ __half2 magic2(uint32_t lo, uint32_t hi) { return u2h(lo | (hi << 16) | 0x64006400u); }
// half2 (1024 + za, 1024 + zb) for integer zero-points (|z| < 1024), exact
__device__ __forceinline__ __half2 zmagic2(float za, float zb) { return __floats2half2_rn(1024.f + za, 1024.f + zb); }

// NTW consecutive words, one vector load
template <int NTW> __device__ __forceinline__ void ldw(uint32_t* d, const uint32_t* p);
template <> __device__ __forceinline__ void ldw<1>(uint32_t* d, const uint32_t* p) { d[0] = __ldg(p); }
template <> __device__ __forceinline__ void ldw<2>(uint32_t* d, const uint32_t* p) {
    const uint2 v = __ldg(reinterpret_cast<const uint2*>(p)); d[0] = v.x; d[1] = v.y; }
template <> __device__ __forceinline__ void ldw<4>(uint32_t* d, const uint32_t* p) {
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(p)); d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w; }

// Fragment order: each 32-wide k window [W, W+32) is issued as two m16n8k16 instructions;
// lane quad position t holds k = W+4t..W+4t+3 and W+16+4t..W+16+4t+3.
//
// ---------------------------------------------------------------------------------------------
// Kernel A: h[T, R] (fp32, atomics) += x[T, N] @ dq(V^T)^T over n in [z*per, (z+1)*per).
// Warp w owns ranks r0 + [0, 8*NTW); BN = 8*NTW*WARPS ranks per CTA (BN | 128); BT = 16*MT tokens.
template <int NTW, int WARPS, int MT>
__global__ void __launch_bounds__(32 * WARPS)
xv4_kernel(const __half* __restrict__ x, const uint32_t* __restrict__ vq, const float* __restrict__ Ct,
           float* __restrict__ h, int T, int N, int R, int NG, int per) {
    constexpr int BN = 8 * NTW * WARPS;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gq = lane >> 2, tq = lane & 3;
    const int rc = blockIdx.x * BN;                 // first rank of the CTA
    const int r0 = rc + warp * 8 * NTW;             // first rank of the warp
    const int g = rc >> 7;                          // group (G = 128)
    const int t0 = blockIdx.y * 16 * MT;
    const int kbeg = blockIdx.z * per;
    const int kend = min(kbeg + per, N);
    const int RW = R >> 3;
    // byte select / nibble shift of rank column gq
    const uint32_t psel = (uint32_t)(gq >> 1) | ((4u + (uint32_t)(gq >> 1)) << 8);
    const uint32_t nsh = 4u * (gq & 1);
    float acc[MT][NTW][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NTW; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;
    bool rowa[MT], rowb[MT];
    #pragma unroll
    for (int i = 0; i < MT; ++i) { rowa[i] = t0 + i * 16 + gq < T; rowb[i] = t0 + i * 16 + gq + 8 < T; }
    // per-lane base pointers
    const size_t cst = (size_t)NG * 4;                  // floats per n row of C
    const float* cp = Ct + ((size_t)(kbeg + 4 * tq) * NG + g) * 4;
    const uint32_t* vp = vq + (size_t)(kbeg + 4 * tq) * RW + (r0 >> 3);
    const __half* xp = x + (size_t)(t0 + gq) * N + kbeg + 4 * tq;
    const size_t x8 = (size_t)8 * N, x16 = (size_t)16 * N;

    for (int nb = kbeg; nb < kend; nb += 64) {
        #pragma unroll
        for (int wdw = 0; wdw < 2; ++wdw) {                 // two 32-wide k windows per 64-block
            // this lane's rows: n0..n0+3 and n0+16..n0+19, n0 = nb + 32*wdw + 4tq
            const float* cw0 = cp + (size_t)(wdw * 32) * cst;
            const uint32_t* vw0 = vp + (size_t)(wdw * 32) * RW;
            float4 cc[8];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                cc[e] = __ldg(reinterpret_cast<const float4*>(cw0 + e * cst));
                cc[4 + e] = __ldg(reinterpret_cast<const float4*>(cw0 + (16 + e) * cst));
            }
            uint32_t v[8][NTW];                              // the NTW words of each of the 8 rows
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                ldw<NTW>(v[e], vw0 + (size_t)e * RW);
                ldw<NTW>(v[4 + e], vw0 + (size_t)(16 + e) * RW);
            }
            __half2 sc[4], zc[4];                            // pairs (n0,n0+1) (n0+2,n0+3) (n0+16,+17) (n0+18,+19)
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float4 u = cc[2 * p], w = cc[2 * p + 1];
                sc[p] = __floats2half2_rn(u.x, w.x);
                zc[p] = zmagic2(rintf(__fdividef(u.z, u.x)), rintf(__fdividef(w.z, w.x)));
            }
            uint32_t bq[NTW][4];                             // b for (1st: pairs 0 | 2), (2nd: pairs 1 | 3)
            #pragma unroll
            for (int j = 0; j < NTW; ++j)
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const uint32_t t = __byte_perm(v[2 * p][j], v[2 * p + 1][j], psel) >> nsh;
                    bq[j][p] = h2u(__hsub2(u2h((t & 0x000F000Fu) | 0x64006400u), zc[p]));
                }
            #pragma unroll
            for (int i = 0; i < MT; ++i) {
                const __half* xw = xp + (size_t)i * x16 + wdw * 32;
                uint2 xa0 = make_uint2(0u, 0u), xa1 = xa0, xb0 = xa0, xb1 = xa0;
                if (rowa[i]) {
                    xa0 = *reinterpret_cast<const uint2*>(xw);
                    xa1 = *reinterpret_cast<const uint2*>(xw + 16);
                }
                if (rowb[i]) {
                    xb0 = *reinterpret_cast<const uint2*>(xw + x8);
                    xb1 = *reinterpret_cast<const uint2*>(xw + x8 + 16);
                }
                // 1st instruction: pairs 0 (a0/a1, b0) and 2 (a2/a3, b1); 2nd: pairs 1 and 3
                const uint32_t p0a = h2u(__hmul2(u2h(xa0.x), sc[0])), p0b = h2u(__hmul2(u2h(xb0.x), sc[0]));
                const uint32_t p1a = h2u(__hmul2(u2h(xa0.y), sc[1])), p1b = h2u(__hmul2(u2h(xb0.y), sc[1]));
                const uint32_t p2a = h2u(__hmul2(u2h(xa1.x), sc[2])), p2b = h2u(__hmul2(u2h(xb1.x), sc[2]));
                const uint32_t p3a = h2u(__hmul2(u2h(xa1.y), sc[3])), p3b = h2u(__hmul2(u2h(xb1.y), sc[3]));
                #pragma unroll
                for (int j = 0; j < NTW; ++j) {
                    mma16816(acc[i][j], p0a, p0b, p2a, p2b, bq[j][0], bq[j][2]);
                    mma16816(acc[i][j], p1a, p1b, p3a, p3b, bq[j][1], bq[j][3]);
                }
            }
        }
        cp += 64 * cst;
        vp += (size_t)64 * RW;
        xp += 64;
    }
    #pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int ra = t0 + i * 16 + gq, rb = ra + 8;
        #pragma unroll
        for (int j = 0; j < NTW; ++j) {
            const int c = r0 + 8 * j + 2 * tq;
            if (rowa[i]) { atomicAdd(h + (size_t)ra * R + c, acc[i][j][0]); atomicAdd(h + (size_t)ra * R + c + 1, acc[i][j][1]); }
            if (rowb[i]) { atomicAdd(h + (size_t)rb * R + c, acc[i][j][2]); atomicAdd(h + (size_t)rb * R + c + 1, acc[i][j][3]); }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel A, pipelined variant: 64-row blocks staged in smem with cp.async (STAGES in flight).
__device__ __forceinline__ void cp_async(void* smem, const void* gmem, int bytes, int src_bytes) {
    const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    if (bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(gmem), "r"(src_bytes));
    else if (bytes == 8)
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(sa), "l"(gmem), "r"(src_bytes));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(sa), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

template <int NTW, int WARPS, int MT, int STAGES>
__global__ void __launch_bounds__(32 * WARPS)
xv4p_kernel(const __half* __restrict__ x, const uint32_t* __restrict__ vq, const float* __restrict__ Ct,
            float* __restrict__ h, int T, int N, int R, int NG, int per) {
    constexpr int BN = 8 * NTW * WARPS, BT = 16 * MT, NTH = 32 * WARPS;
    constexpr int CB = BN / 2;                        // code bytes per n row of the CTA
    constexpr int CCH = CB >= 16 ? 16 : CB;           // cp.async chunk for codes
    constexpr int XS = 72;                            // padded x smem row stride (halves)
    constexpr int CODE_B = 64 * CB, C_B = 64 * 16, X_B = BT * XS * 2;
    constexpr int STAGE_B = CODE_B + C_B + X_B;
    extern __shared__ __align__(16) unsigned char smem[];   // [STAGES][code | C | x], then sz[2][64] half2
    __half* sz = reinterpret_cast<__half*>(smem + STAGES * STAGE_B);   // s16[64], zm[64]

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gq = lane >> 2, tq = lane & 3;
    const int rc = blockIdx.x * BN;
    const int g = rc >> 7;
    const int t0 = blockIdx.y * BT;
    const int kbeg = blockIdx.z * per;
    const int kend = min(kbeg + per, N);
    const int nblk = (kend - kbeg) >> 6;
    const int RW = R >> 3;
    const uint32_t psel = (uint32_t)(gq >> 1) | ((4u + (uint32_t)(gq >> 1)) << 8);
    const uint32_t nsh = 4u * (gq & 1);
    const int tv = min(BT, T - t0);                   // valid token rows of this tile

    auto issue = [&](int b) {
        if (b < nblk) {
            unsigned char* st = smem + (b % STAGES) * STAGE_B;
            const int n0 = kbeg + b * 64;
            constexpr int CPR = CB / CCH;             // code chunks per row
            for (int i = threadIdx.x; i < 64 * CPR; i += NTH) {
                const int row = i / CPR, cch = i - row * CPR;
                cp_async(st + row * CB + cch * CCH,
                         reinterpret_cast<const unsigned char*>(vq + (size_t)(n0 + row) * RW + (rc >> 3)) + cch * CCH,
                         CCH, CCH);
            }
            for (int i = threadIdx.x; i < 64; i += NTH)
                cp_async(st + CODE_B + i * 16, Ct + ((size_t)(n0 + i) * NG + g) * 4, 16, 16);
            for (int i = threadIdx.x; i < BT * 8; i += NTH) {
                const int row = i >> 3, cch = i & 7;
                const bool ok = row < tv;
                cp_async(st + CODE_B + C_B + (row * XS + cch * 8) * 2,
                         ok ? (const void*)(x + (size_t)(t0 + row) * N + n0 + cch * 8) : (const void*)x, 16, ok ? 16 : 0);
            }
        }
        cp_commit();
    };

    float acc[MT][NTW][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NTW; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

    #pragma unroll
    for (int b = 0; b < STAGES - 1; ++b) issue(b);

    for (int b = 0; b < nblk; ++b) {
        cp_wait<STAGES - 2>();
        __syncthreads();                                  // block b landed; everyone done with block b-1
        const unsigned char* st = smem + (b % STAGES) * STAGE_B;
        const float4* cs = reinterpret_cast<const float4*>(st + CODE_B);
        for (int i = threadIdx.x; i < 64; i += NTH) {    // fp16 scale and 1024 + zero of each n row
            const float4 c = cs[i];
            sz[i] = __float2half_rn(c.x);
            sz[64 + i] = __float2half_rn(1024.f + rintf(__fdividef(c.z, c.x)));
        }
        __syncthreads();
        issue(b + STAGES - 1);                            // refills the buffer block b-1 used
        const uint32_t* cw = reinterpret_cast<const uint32_t*>(st);
        const __half* xs_ = reinterpret_cast<const __half*>(st + CODE_B + C_B);
        #pragma unroll
        for (int wdw = 0; wdw < 2; ++wdw) {
            const int k = wdw * 32 + 4 * tq;              // local rows k..k+3 and k+16..k+19
            const uint2 s0 = *reinterpret_cast<const uint2*>(sz + k), s1 = *reinterpret_cast<const uint2*>(sz + k + 16);
            const uint2 z0 = *reinterpret_cast<const uint2*>(sz + 64 + k), z1 = *reinterpret_cast<const uint2*>(sz + 64 + k + 16);
            const __half2 sc[4] = {u2h(s0.x), u2h(s0.y), u2h(s1.x), u2h(s1.y)};
            const __half2 zc[4] = {u2h(z0.x), u2h(z0.y), u2h(z1.x), u2h(z1.y)};
            uint32_t v[8][NTW];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const uint32_t* r0p = cw + (k + e) * (CB / 4) + warp * NTW;
                const uint32_t* r1p = cw + (k + 16 + e) * (CB / 4) + warp * NTW;
                #pragma unroll
                for (int j = 0; j < NTW; ++j) { v[e][j] = r0p[j]; v[4 + e][j] = r1p[j]; }
            }
            uint32_t bq[NTW][4];
            #pragma unroll
            for (int j = 0; j < NTW; ++j)
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const uint32_t t = __byte_perm(v[2 * p][j], v[2 * p + 1][j], psel) >> nsh;
                    bq[j][p] = h2u(__hsub2(u2h((t & 0x000F000Fu) | 0x64006400u), zc[p]));
                }
            #pragma unroll
            for (int i = 0; i < MT; ++i) {
                const uint2 xa0 = *reinterpret_cast<const uint2*>(xs_ + (i * 16 + gq) * XS + k);
                const uint2 xa1 = *reinterpret_cast<const uint2*>(xs_ + (i * 16 + gq) * XS + k + 16);
                const uint2 xb0 = *reinterpret_cast<const uint2*>(xs_ + (i * 16 + gq + 8) * XS + k);
                const uint2 xb1 = *reinterpret_cast<const uint2*>(xs_ + (i * 16 + gq + 8) * XS + k + 16);
                const uint32_t p0a = h2u(__hmul2(u2h(xa0.x), sc[0])), p0b = h2u(__hmul2(u2h(xb0.x), sc[0]));
                const uint32_t p1a = h2u(__hmul2(u2h(xa0.y), sc[1])), p1b = h2u(__hmul2(u2h(xb0.y), sc[1]));
                const uint32_t p2a = h2u(__hmul2(u2h(xa1.x), sc[2])), p2b = h2u(__hmul2(u2h(xb1.x), sc[2]));
                const uint32_t p3a = h2u(__hmul2(u2h(xa1.y), sc[3])), p3b = h2u(__hmul2(u2h(xb1.y), sc[3]));
                #pragma unroll
                for (int j = 0; j < NTW; ++j) {
                    mma16816(acc[i][j], p0a, p0b, p2a, p2b, bq[j][0], bq[j][2]);
                    mma16816(acc[i][j], p1a, p1b, p3a, p3b, bq[j][1], bq[j][3]);
                }
            }
        }
    }
    cp_wait<0>();
    const int r0 = rc + warp * 8 * NTW;
    #pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int ra = t0 + i * 16 + gq, rb = ra + 8;
        #pragma unroll
        for (int j = 0; j < NTW; ++j) {
            const int c = r0 + 8 * j + 2 * tq;
            if (ra < T) { atomicAdd(h + (size_t)ra * R + c, acc[i][j][0]); atomicAdd(h + (size_t)ra * R + c + 1, acc[i][j][1]); }
            if (rb < T) { atomicAdd(h + (size_t)rb * R + c, acc[i][j][2]); atomicAdd(h + (size_t)rb * R + c + 1, acc[i][j][3]); }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel B: out[T, M] = yd + h[T, R] @ dq(U)^T. Warp w owns output columns o0 + [0, 8*NTW);
// BM = 8*NTW*WARPS per CTA; h staged in smem as fp16; the first token tile clears hz.
template <int NTW, int WARPS, int MT>
__global__ void __launch_bounds__(32 * WARPS)
hu4_kernel(const float* __restrict__ h, const uint32_t* __restrict__ uq, const float* __restrict__ Rt,
           const __half* __restrict__ yd, __half* __restrict__ out, float* __restrict__ hz,
           int T, int M, int R, int NG, int HN, int zb) {
    constexpr int BM = 8 * NTW * WARPS, BT = 16 * MT, NTH = 32 * WARPS;
    extern __shared__ __align__(16) __half hs[];     // [BT][R + 8]
    const int HS = R + 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gq = lane >> 2, tq = lane & 3;
    const int o0 = blockIdx.x * BM + warp * 8 * NTW;
    const int t0 = blockIdx.y * BT;
    const int RW = R >> 3;
    const int nrow = min(BT, T - t0);

    // prefetch group-0 codes
    uint4 cw[NTW][4];
    #pragma unroll
    for (int j = 0; j < NTW; ++j) {
        const int ob = o0 + 8 * j + gq;
        const uint4* p = reinterpret_cast<const uint4*>(uq + (size_t)min(ob, M - 1) * RW);
        #pragma unroll
        for (int q = 0; q < 4; ++q) cw[j][q] = __ldg(p + q);
    }
    // stage h tile as fp16 (zero rows beyond T)
    for (int idx = threadIdx.x; idx < BT * (R >> 2); idx += NTH) {
        const int row = idx / (R >> 2), c4 = (idx - row * (R >> 2)) * 4;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row < nrow) v = *reinterpret_cast<const float4*>(h + (size_t)(t0 + row) * R + c4);
        __half2* d = reinterpret_cast<__half2*>(hs + row * HS + c4);
        d[0] = __floats2half2_rn(v.x, v.y);
        d[1] = __floats2half2_rn(v.z, v.w);
    }
    __syncthreads();

    float acc[MT][NTW][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NTW; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

    for (int g = 0; g < NG; ++g) {
        __half2 z2[NTW];
        float s0[NTW], s1[NTW];
        #pragma unroll
        for (int j = 0; j < NTW; ++j) {
            const int ob = min(o0 + 8 * j + gq, M - 1);        // B-fragment column
            const int oc = min(o0 + 8 * j + 2 * tq, M - 2);    // accumulator columns oc, oc+1
            const float2 rz = __ldg(reinterpret_cast<const float2*>(Rt + ((size_t)ob * NG + g) * 4));
            const float zz = rintf(__fdividef(rz.y, rz.x));
            z2[j] = zmagic2(zz, zz);
            s0[j] = __half2float(__float2half_rn(__ldg(Rt + ((size_t)oc * NG + g) * 4)));
            s1[j] = __half2float(__float2half_rn(__ldg(Rt + ((size_t)(oc + 1) * NG + g) * 4)));
        }
        uint4 cn[NTW][4];                                      // next group's code words
        if (g + 1 < NG) {
            #pragma unroll
            for (int j = 0; j < NTW; ++j) {
                const int ob = min(o0 + 8 * j + gq, M - 1);
                const uint4* p = reinterpret_cast<const uint4*>(uq + (size_t)ob * RW + (g + 1) * 16);
                #pragma unroll
                for (int q = 0; q < 4; ++q) cn[j][q] = __ldg(p + q);
            }
        }
        float pa[MT][NTW][4];
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int j = 0; j < NTW; ++j)
                #pragma unroll
                for (int r = 0; r < 4; ++r) pa[i][j][r] = 0.f;
        #pragma unroll
        for (int wdw = 0; wdw < 4; ++wdw) {                    // 32-wide k windows of the group, in order
            const int k = g * 128 + wdw * 32 + 4 * tq;          // this lane's k: k..k+3 and k+16..k+19
            uint32_t a[MT][8];                                  // rows gq / gq+8, halves (k..k+3 | k+16..k+19)
            #pragma unroll
            for (int i = 0; i < MT; ++i) {
                const uint2 ha0 = *reinterpret_cast<const uint2*>(hs + (i * 16 + gq) * HS + k);
                const uint2 ha1 = *reinterpret_cast<const uint2*>(hs + (i * 16 + gq) * HS + k + 16);
                const uint2 hb0 = *reinterpret_cast<const uint2*>(hs + (i * 16 + gq + 8) * HS + k);
                const uint2 hb1 = *reinterpret_cast<const uint2*>(hs + (i * 16 + gq + 8) * HS + k + 16);
                a[i][0] = ha0.x; a[i][1] = hb0.x; a[i][2] = ha1.x; a[i][3] = hb1.x;   // 1st instruction
                a[i][4] = ha0.y; a[i][5] = hb0.y; a[i][6] = ha1.y; a[i][7] = hb1.y;   // 2nd instruction
            }
            #pragma unroll
            for (int j = 0; j < NTW; ++j) {
                // ranks wdw*32 + 4tq .. +3 = half (tq & 1) of word 4wdw + (tq >> 1); +16 -> word + 2
                const uint4 u = cw[j][wdw];
                const uint32_t w0 = (tq >> 1) ? u.y : u.x;
                const uint32_t w1 = (tq >> 1) ? u.w : u.z;
                const uint32_t y0 = w0 >> (16u * (tq & 1)), y1 = w1 >> (16u * (tq & 1));
                const uint32_t q00 = h2u(__hsub2(magic2(y0 & 15u, (y0 >> 4) & 15u), z2[j]));
                const uint32_t q01 = h2u(__hsub2(magic2((y0 >> 8) & 15u, (y0 >> 12) & 15u), z2[j]));
                const uint32_t q10 = h2u(__hsub2(magic2(y1 & 15u, (y1 >> 4) & 15u), z2[j]));
                const uint32_t q11 = h2u(__hsub2(magic2((y1 >> 8) & 15u, (y1 >> 12) & 15u), z2[j]));
                #pragma unroll
                for (int i = 0; i < MT; ++i) {
                    mma16816(pa[i][j], a[i][0], a[i][1], a[i][2], a[i][3], q00, q10);
                    mma16816(pa[i][j], a[i][4], a[i][5], a[i][6], a[i][7], q01, q11);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int j = 0; j < NTW; ++j) {
                acc[i][j][0] = fmaf(s0[j], pa[i][j][0], acc[i][j][0]);
                acc[i][j][1] = fmaf(s1[j], pa[i][j][1], acc[i][j][1]);
                acc[i][j][2] = fmaf(s0[j], pa[i][j][2], acc[i][j][2]);
                acc[i][j][3] = fmaf(s1[j], pa[i][j][3], acc[i][j][3]);
            }
        if (g + 1 < NG) {
            #pragma unroll
            for (int j = 0; j < NTW; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) cw[j][q] = cn[j][q];
        }
    }
    #pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int ra = t0 + i * 16 + gq, rb = ra + 8;
        #pragma unroll
        for (int j = 0; j < NTW; ++j) {
            const int oc = o0 + 8 * j + 2 * tq;
            if (oc < M) {                               // M % 8 == 0, so oc + 1 < M as well
                if (ra < T) {
                    const float2 yv = __half22float2(*reinterpret_cast<const __half2*>(yd + (size_t)ra * M + oc));
                    *reinterpret_cast<__half2*>(out + (size_t)ra * M + oc) =
                        __floats2half2_rn(acc[i][j][0] + yv.x, acc[i][j][1] + yv.y);
                }
                if (rb < T) {
                    const float2 yv = __half22float2(*reinterpret_cast<const __half2*>(yd + (size_t)rb * M + oc));
                    *reinterpret_cast<__half2*>(out + (size_t)rb * M + oc) =
                        __floats2half2_rn(acc[i][j][2] + yv.x, acc[i][j][3] + yv.y);
                }
            }
        }
    }
    if (blockIdx.y == 0) {
        const size_t base = (size_t)blockIdx.x * zb;
        for (int i = threadIdx.x; i < zb; i += NTH) {
            const size_t idx = base + i;
            if (idx < (size_t)HN) hz[idx] = 0.f;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// config tables: id -> (NTW, WARPS, MT)
#define A_LIST(X) X(0,1,1,1,0) X(1,1,2,1,0) X(2,1,4,1,0) X(3,2,1,1,0) X(4,2,2,1,0) X(5,2,4,1,0) X(6,4,1,1,0) X(7,4,2,1,0) X(8,4,4,1,0) \
                  X(9,1,1,2,0) X(10,1,2,2,0) X(11,1,4,2,0) X(12,2,1,2,0) X(13,2,2,2,0) X(14,2,4,2,0) X(15,4,1,2,0) X(16,4,2,2,0) X(17,4,4,2,0) \
                  X(18,2,1,4,0) X(19,2,2,4,0) X(20,2,4,4,0) X(21,4,1,4,0) X(22,4,2,4,0) X(23,4,4,4,0) \
                  X(24,1,4,1,3) X(25,2,2,1,3) X(26,2,4,1,3) X(27,4,1,1,3) X(28,4,2,1,3) X(29,4,4,1,3) \
                  X(30,2,2,2,3) X(31,2,4,2,3) X(32,4,1,2,3) X(33,4,2,2,3) X(34,4,4,2,3) \
                  X(35,2,2,4,2) X(36,4,1,4,2) X(37,4,2,4,2)
#define B_LIST(X) X(0,1,1,1) X(1,1,2,1) X(2,1,4,1) X(3,1,8,1) X(4,2,1,1) X(5,2,2,1) X(6,2,4,1) X(7,2,8,1) X(8,4,1,1) X(9,4,2,1) X(10,4,4,1) \
                  X(11,1,1,2) X(12,1,2,2) X(13,1,4,2) X(14,1,8,2) X(15,2,1,2) X(16,2,2,2) X(17,2,4,2) X(18,2,8,2) X(19,4,1,2) X(20,4,2,2) X(21,4,4,2) \
                  X(22,2,2,4) X(23,2,4,4) X(24,4,2,4) X(25,4,4,4)

template <int NTW, int WARPS, int MT, int STAGES>
static void launch_a_t(dim3 grid, cudaStream_t st, const __half* x, const uint32_t* vq, const float* C,
                       float* h, int T, int N, int R, int NG, int per) {
    if constexpr (STAGES == 0) {
        xv4_kernel<NTW, WARPS, MT><<<grid, 32 * WARPS, 0, st>>>(x, vq, C, h, T, N, R, NG, per);
    } else {
        constexpr int smem = STAGES * (64 * 4 * NTW * WARPS + 1024 + 16 * MT * 72 * 2) + 256;
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(xv4p_kernel<NTW, WARPS, MT, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        xv4p_kernel<NTW, WARPS, MT, STAGES><<<grid, 32 * WARPS, smem, st>>>(x, vq, C, h, T, N, R, NG, per);
    }
}
static void launch_a(int cfg, dim3 grid, cudaStream_t st, const __half* x, const uint32_t* vq, const float* C,
                     float* h, int T, int N, int R, int NG, int per) {
    switch (cfg) {
#define CASE_A(id, ntw, warps, mt, stages) case id: launch_a_t<ntw, warps, mt, stages>(grid, st, x, vq, C, h, T, N, R, NG, per); break;
        A_LIST(CASE_A)
#undef CASE_A
        default: TORCH_CHECK(false, "bad A config");
    }
}
static void launch_b(int cfg, dim3 grid, cudaStream_t st, const float* h, const uint32_t* uq, const float* Rt,
                     const __half* yd, __half* out, float* hz, int T, int M, int R, int NG, int HN, int zb) {
    switch (cfg) {
#define CASE_B(id, ntw, warps, mt) case id: { \
            const int smem = 16 * mt * (R + 8) * 2; \
            if (smem > 48 * 1024) cudaFuncSetAttribute(hu4_kernel<ntw, warps, mt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
            hu4_kernel<ntw, warps, mt><<<grid, 32 * warps, smem, st>>>(h, uq, Rt, yd, out, hz, T, M, R, NG, HN, zb); break; }
        B_LIST(CASE_B)
#undef CASE_B
        default: TORCH_CHECK(false, "bad B config");
    }
}

// raw-pointer entry points
void run_a(int64_t cfg, int64_t gx, int64_t gy, int64_t gz, int64_t x, int64_t vq, int64_t C, int64_t h,
           int64_t T, int64_t N, int64_t R, int64_t NG, int64_t per) {
    launch_a((int)cfg, dim3(gx, gy, gz), at::cuda::getCurrentCUDAStream(), (const __half*)x, (const uint32_t*)vq,
             (const float*)C, (float*)h, (int)T, (int)N, (int)R, (int)NG, (int)per);
}
void run_b(int64_t cfg, int64_t gx, int64_t gy, int64_t h, int64_t uq, int64_t Rt, int64_t yd, int64_t out,
           int64_t hz, int64_t T, int64_t M, int64_t R, int64_t NG, int64_t HN, int64_t zb) {
    launch_b((int)cfg, dim3(gx, gy, 1), at::cuda::getCurrentCUDAStream(), (const float*)h, (const uint32_t*)uq,
             (const float*)Rt, (const __half*)yd, (__half*)out, (float*)hz, (int)T, (int)M, (int)R, (int)NG,
             (int)HN, (int)zb);
}
void run_ab(int64_t cfga, int64_t gax, int64_t gay, int64_t gaz, int64_t cfgb, int64_t gbx, int64_t gby,
            int64_t x, int64_t vq, int64_t C, int64_t h, int64_t uq, int64_t Rt, int64_t yd, int64_t out, int64_t hz,
            int64_t T, int64_t N, int64_t M, int64_t R, int64_t NG, int64_t per, int64_t HN, int64_t zb) {
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    launch_a((int)cfga, dim3(gax, gay, gaz), st, (const __half*)x, (const uint32_t*)vq, (const float*)C, (float*)h,
             (int)T, (int)N, (int)R, (int)NG, (int)per);
    launch_b((int)cfgb, dim3(gbx, gby, 1), st, (const float*)h, (const uint32_t*)uq, (const float*)Rt,
             (const __half*)yd, (__half*)out, (float*)hz, (int)T, (int)M, (int)R, (int)NG, (int)HN, (int)zb);
}
"""

_DECODE_CPP = r"""
#include <cstdint>
void run_a(int64_t cfg, int64_t gx, int64_t gy, int64_t gz, int64_t x, int64_t vq, int64_t C, int64_t h,
           int64_t T, int64_t N, int64_t R, int64_t NG, int64_t per);
void run_b(int64_t cfg, int64_t gx, int64_t gy, int64_t h, int64_t uq, int64_t Rt, int64_t yd, int64_t out,
           int64_t hz, int64_t T, int64_t M, int64_t R, int64_t NG, int64_t HN, int64_t zb);
void run_ab(int64_t cfga, int64_t gax, int64_t gay, int64_t gaz, int64_t cfgb, int64_t gbx, int64_t gby,
            int64_t x, int64_t vq, int64_t C, int64_t h, int64_t uq, int64_t Rt, int64_t yd, int64_t out, int64_t hz,
            int64_t T, int64_t N, int64_t M, int64_t R, int64_t NG, int64_t per, int64_t HN, int64_t zb);
"""

# (NTW, WARPS, MT[, STAGES]) per config id, as A_LIST / B_LIST
A_CFGS = [(1, 1, 1), (1, 2, 1), (1, 4, 1), (2, 1, 1), (2, 2, 1), (2, 4, 1), (4, 1, 1), (4, 2, 1), (4, 4, 1),
          (1, 1, 2), (1, 2, 2), (1, 4, 2), (2, 1, 2), (2, 2, 2), (2, 4, 2), (4, 1, 2), (4, 2, 2), (4, 4, 2),
          (2, 1, 4), (2, 2, 4), (2, 4, 4), (4, 1, 4), (4, 2, 4), (4, 4, 4),
          (1, 4, 1, 3), (2, 2, 1, 3), (2, 4, 1, 3), (4, 1, 1, 3), (4, 2, 1, 3), (4, 4, 1, 3),
          (2, 2, 2, 3), (2, 4, 2, 3), (4, 1, 2, 3), (4, 2, 2, 3), (4, 4, 2, 3),
          (2, 2, 4, 2), (4, 1, 4, 2), (4, 2, 4, 2)]
B_CFGS = [(1, 1, 1), (1, 2, 1), (1, 4, 1), (1, 8, 1), (2, 1, 1), (2, 2, 1), (2, 4, 1), (2, 8, 1), (4, 1, 1),
          (4, 2, 1), (4, 4, 1),
          (1, 1, 2), (1, 2, 2), (1, 4, 2), (1, 8, 2), (2, 1, 2), (2, 2, 2), (2, 4, 2), (2, 8, 2), (4, 1, 2),
          (4, 2, 2), (4, 4, 2), (2, 2, 4), (2, 4, 4), (4, 2, 4), (4, 4, 4)]

_decode_ext = None


def _load_decode(device):
    """
    Compile (on first use) and return the decode extension.

    Parameters:
        device: CUDA device selecting the target architecture.

    Returns:
        Extension module exposing run_a, run_b, run_ab.
    """
    global _decode_ext
    if _decode_ext is None:
        arch, flags = _arch_flags(device)
        _decode_ext = load_inline(
            name=f"ade_decode_packed_sm{arch}", cpp_sources=_DECODE_CPP, cuda_sources=_DECODE_CUDA,
            functions=["run_a", "run_b", "run_ab"],
            extra_cuda_cflags=["-O3", *flags], extra_cflags=["-O3"],
            build_directory=os.environ.get("ADE_KERNEL_BUILD_DIR"),
            verbose=os.environ.get("ADE_KERNEL_VERBOSE") == "1")
    return _decode_ext


def _cand_a(T, R, N, G=128, split_k=False):
    """
    (config id, n-slice) candidates of kernel A.

    Parameters:
        T: Number of tokens.
        R: Rank.
        N: Input dimension d_i.
        G: Group size.
        split_k: Allow splitting the reduction over n.

    Returns:
        List of (config id, slice length).
    """
    full = _cdiv(N, 64) * 64
    out = []
    for cid, cfg in enumerate(A_CFGS):
        ntw, warps, mt = cfg[:3]
        bn, bt = 8 * ntw * warps, 16 * mt
        if G % bn or R % bn or (mt > 1 and T <= 16 * (mt // 2)):
            continue
        if not split_k:
            out.append((cid, full))
            continue
        nb, nt = R // bn, _cdiv(T, bt)
        pers = [p for p in (64, 128, 256, 512, 1024, 2048, 4096) if p <= max(64, full)]
        seen = set()
        for want in (384, 768, 1536):
            p = min(pers, key=lambda p: abs(nb * nt * _cdiv(N, p) - want))
            if p not in seen:
                seen.add(p)
                out.append((cid, p))
    return out


def _cand_b(T, M, R):
    """
    Config ids of kernel B whose grid stays within 32 .. 4096 CTAs (plus a fallback).

    Parameters:
        T: Number of tokens.
        M: Output dimension d_o.
        R: Rank.

    Returns:
        List of config ids.
    """
    out = []
    for cid, (ntw, warps, mt) in enumerate(B_CFGS):
        bm, bt = 8 * ntw * warps, 16 * mt
        if mt > 1 and T <= 16 * (mt // 2):
            continue
        n = _cdiv(M, bm) * _cdiv(T, bt)
        if 32 <= n <= 4096:
            out.append(cid)
    return out or [21]


def _time(fn, reps=16, rounds=3):
    """
    Time a kernel launch (best of `rounds` rounds of `reps` launches).

    Parameters:
        fn: Callable launching the kernel.
        reps: Launches per round.
        rounds: Number of rounds.

    Returns:
        Time per launch in microseconds.
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / reps * 1e3)
    return best


# chosen tile configurations per (d_o, d_i, r, T, split_k)
_PLAN_CONFIGS: dict = {}


class Plan:
    """
    Launch plan of the decode kernels for one (stored form, T): tile configurations and two
    FP32 T x r accumulators.
    """

    def __init__(self, p4, T, N, dev, G=128, split_k=False):
        """
        Parameters:
            p4: Stored form (pack_int4).
            T: Number of tokens.
            N: Input dimension d_i.
            dev: CUDA device.
            G: Group size (128).
            split_k: Allow split-K candidates for kernel A.
        """
        assert G == 128, "the kernels are specialised for group size 128"
        ext = _load_decode(dev)
        self.ext = ext
        self.uq, self.vq, self.rt, self.ct = p4["uq"], p4["vq"], p4["R"], p4["C"]
        M, RW = self.uq.shape
        R = RW * 8
        assert self.vq.shape == (N, RW) and R % G == 0 and N % 64 == 0 and M % 8 == 0
        assert all(t.is_contiguous() for t in (self.uq, self.vq, self.rt, self.ct))
        self.T, self.N, self.M, self.R, self.NG, self.HN = T, N, M, R, R // G, T * R
        self.bufs = [torch.zeros(T, R, device=dev, dtype=torch.float32) for _ in range(2)]
        self.i = 0
        x = torch.zeros(T, N, device=dev, dtype=torch.float16)
        yd = torch.zeros(T, M, device=dev, dtype=torch.float16)
        out = torch.empty(T, M, device=dev, dtype=torch.float16)
        h, hz = self.bufs
        key = (M, N, R, T, split_k)
        if key in _PLAN_CONFIGS:
            (self.cfg_a, self.per, self.ga), (self.cfg_b, self.gb, self.zb) = _PLAN_CONFIGS[key]
        else:
            # ---- kernel A
            best = (float("inf"), None)
            for cid, per in _cand_a(T, R, N, split_k=split_k):
                ntw, warps, mt = A_CFGS[cid][:3]
                ga = (R // (8 * ntw * warps), _cdiv(T, 16 * mt), _cdiv(N, per))
                args = (cid, *ga, x.data_ptr(), self.vq.data_ptr(), self.ct.data_ptr(), h.data_ptr(),
                        T, N, R, self.NG, per)
                t = _time(lambda a=args: ext.run_a(*a))
                if t < best[0]:
                    best = (t, (cid, per, ga))
            self.cfg_a, self.per, self.ga = best[1]
            # ---- kernel B
            best = (float("inf"), None)
            for cid in _cand_b(T, M, R):
                ntw, warps, mt = B_CFGS[cid]
                gb = (_cdiv(M, 8 * ntw * warps), _cdiv(T, 16 * mt))
                zb = _cdiv(self.HN, gb[0])
                args = (cid, *gb, h.data_ptr(), self.uq.data_ptr(), self.rt.data_ptr(), yd.data_ptr(),
                        out.data_ptr(), hz.data_ptr(), T, M, R, self.NG, self.HN, zb)
                t = _time(lambda a=args: ext.run_b(*a))
                if t < best[0]:
                    best = (t, (cid, gb, zb))
            self.cfg_b, self.gb, self.zb = best[1]
            _PLAN_CONFIGS[key] = ((self.cfg_a, self.per, self.ga), (self.cfg_b, self.gb, self.zb))
        h.zero_()
        hz.zero_()          # clear search partial sums
        torch.cuda.synchronize()
        self._head = (self.cfg_a, *self.ga, self.cfg_b, *self.gb)
        self._vq, self._ct, self._uq, self._rt = (self.vq.data_ptr(), self.ct.data_ptr(),
                                                  self.uq.data_ptr(), self.rt.data_ptr())
        self._hp = [b.data_ptr() for b in self.bufs]
        self._tail = (T, N, M, R, self.NG, self.per, self.HN, self.zb)

    def run(self, x, y_dense):
        """
        Launch kernels A and B.

        Parameters:
            x: Input (T, d_i), FP16, contiguous.
            y_dense: x W_b^T (T, d_o), FP16, contiguous.

        Returns:
            y_dense + U^ (V^^T x) in FP16.
        """
        i = self.i
        self.i = 1 - i
        out = torch.empty_like(y_dense)
        self.ext.run_ab(*self._head, x.data_ptr(), self._vq, self._ct, self._hp[i], self._uq, self._rt,
                        y_dense.data_ptr(), out.data_ptr(), self._hp[1 - i], *self._tail)
        return out


def decode_forward_packed(x, y_dense, p4, group_size=128, split_k=False):
    """
    Factored application y_dense + U^ (V^^T x) with the packed INT4 decode kernels.

    Parameters:
        x: Input (T, d_i), FP16, contiguous.
        y_dense: x W_b^T (T, d_o), FP16, contiguous.
        p4: Stored form (pack_int4) on the GPU.
        group_size: Group size G (128).
        split_k: Allow split-K plans for kernel A.

    Returns:
        (T, d_o) FP16 output.
    """
    key = ("_plan", x.shape[0], split_k)
    pl = p4.get(key)
    if pl is None:
        pl = p4[key] = Plan(p4, x.shape[0], x.shape[1], x.device, group_size, split_k=split_k)
    return pl.run(x, y_dense)
