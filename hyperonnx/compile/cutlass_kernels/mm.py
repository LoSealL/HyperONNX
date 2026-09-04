"""
Copyright (C) 2026 The HYPERONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CUTLASS GEMM config tuner using CuTe DSL.

Tunes a tiled Ampere-style tensor-core GEMM (adapted from the CuTe DSL
``ampere/kernel/dense_gemm/tensorop_gemm.py`` example) against a cuBLAS
baseline. ``tune_mm`` returns the best CUTLASS config (never ``None``) plus
a ``bench`` record; ``bench["winner"]`` decides at replay whether to run
CUTLASS or fall back to the cuBLAS/aten call scheme.
"""

# pyright: reportInvalidTypeArguments=none, reportIndexIssue=none, reportPrivateImportUsage=none

import math
from collections.abc import Callable
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute import Float16, Float32
from cutlass.cute.runtime import make_fake_stream

from .config import MM_CONFIGS, NAIVE_CONFIG, CutlassConfig

# 2% margin: only replace cuBLAS when CUTLASS is meaningfully faster,
# so timing noise never swaps in a slower kernel.
_WIN_MARGIN = 0.98
# Ampere-class max dynamic smem per block (99KB on Ada/Blackwell consumer).
_MAX_SMEM_BYTES = 99 * 1024
_MMA_SHAPE = (16, 8, 16)


class _AmpereGemm:
    """Tiled fp16 tensor-core GEMM, parameterized by a CutlassConfig.

    Adapted from the CuTe DSL Ampere ``tensorop_gemm.py`` example:
    cp.async multi-stage gmem→smem pipeline, ldmatrix smem→register
    copies, ``mma.sync`` 16x8x16 atoms, and an smem-buffered epilogue.
    """

    def __init__(self, cfg: CutlassConfig):
        self.bM, self.bN, self.bK = cfg.tile_m, cfg.tile_n, cfg.tile_k
        # The multistage cp.async pipeline below assumes >= 3 stages.
        self.num_stages = max(cfg.num_stages, 3)
        m_warps, n_warps = cfg.m_warps, cfg.n_warps
        self.atom_layout_mnk = (m_warps, n_warps, 1)
        self.num_threads = cfg.num_warps * 32
        mma_m, mma_n, mma_k = _MMA_SHAPE
        assert self.bM % (m_warps * mma_m) == 0, "tile_m must fit MMA atoms"
        assert self.bN % (n_warps * mma_n) == 0, "tile_n must fit MMA atoms"
        assert self.bK % mma_k == 0, "tile_k must be a multiple of 16"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        ab_copy_bits = 128
        sA_layout, sA_swizzle = self._make_smem_layout_AB(
            mA.element_type,
            self.a_major_mode,
            ab_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout, sB_swizzle = self._make_smem_layout_AB(
            mB.element_type,
            self.b_major_mode,
            ab_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = self._make_smem_layout_C(
            mC.element_type,
            self.c_major_mode,
            ab_copy_bits,
            (self.bM, self.bN),
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
            mA.element_type,
            num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits, self.bK
        )
        tiled_copy_B = self._make_gmem_tiled_copy(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits, self.bK
        )
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC.element_type,
            num_bits_per_copy=128,
        )
        tiled_copy_C = self._make_gmem_tiled_copy(
            atom_sync_copy, mC.element_type, self.c_major_mode, 128, self.bN
        )

        op = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, _MMA_SHAPE)
        permutation_mnk = (
            self.atom_layout_mnk[0] * _MMA_SHAPE[0],
            self.atom_layout_mnk[1] * _MMA_SHAPE[1] * 2,
            self.atom_layout_mnk[2] * _MMA_SHAPE[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sA_swizzle,
            sB_layout,
            sB_swizzle,
            sC_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_C,
            tiled_mma,
        ).launch(
            grid=cute.ceil_div(mC.shape, (self.bM, self.bN, 1)),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout,
        sA_swizzle: cute.Swizzle,
        sB_layout: cute.Layout,
        sB_swizzle: cute.Swizzle,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        tiler_coord = (bidx, bidy, None)

        gA = cute.local_tile(
            mA[None, None, bidz],
            tiler=(self.bM, self.bN, self.bK),
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB[None, None, bidz],
            tiler=(self.bM, self.bN, self.bK),
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        gC = cute.local_tile(
            mC[None, None, bidz],
            tiler=(self.bM, self.bN, self.bK),
            coord=tiler_coord,
            proj=(1, 1, None),
        )

        # Shift the pointer so irregular (non-multiple) K tails come first.
        residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(
            gA, mode=[2]
        )
        gA = cute.domain_offset((0, residual_k, 0), gA)
        gB = cute.domain_offset((0, residual_k, 0), gB)
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        mcA = cute.make_identity_tensor(mA.layout.shape)
        mcB = cute.make_identity_tensor(mB.layout.shape)
        cA = cute.local_tile(
            mcA[None, None, bidz],
            tiler=(self.bM, self.bN, self.bK),
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        cB = cute.local_tile(
            mcB[None, None, bidz],
            tiler=(self.bM, self.bN, self.bK),
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        cA = cute.domain_offset((0, residual_k, 0), cA)
        cB = cute.domain_offset((0, residual_k, 0), cB)

        @cute.struct
        class SharedStorageAB:
            a: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 16
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 16
            ]

        @cute.struct
        class SharedStorageC:
            c: cute.struct.Align[
                cute.struct.MemRange[mC.element_type, cute.cosize(sC_layout)], 16
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(
            max(
                SharedStorageAB.size_in_bytes(),  # type: ignore[attr-defined]
                SharedStorageC.size_in_bytes(),  # type: ignore[attr-defined]
            ),
            byte_alignment=16,
        )
        sA = SharedStorageAB(storage).a.get_tensor(  # type: ignore[call-arg]
            sA_layout, swizzle=sA_swizzle
        )
        sB = SharedStorageAB(storage).b.get_tensor(  # type: ignore[call-arg]
            sB_layout, swizzle=sB_swizzle
        )
        sC = SharedStorageC(storage).c.get_tensor(  # type: ignore[call-arg]
            sC_layout
        )

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        thr_copy_C = tiled_copy_C.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)
        tCsC_epilogue = thr_copy_C.partition_S(sC)
        tCgC_epilogue = thr_copy_C.partition_D(gC)
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)

        # Predicates for the M/N bounds (checked per copy atom).
        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAgA.shape[0][1],
                    cute.size(tAgA, mode=[1]),
                    cute.size(tAgA, mode=[2]),
                ),
                stride=(cute.size(tAgA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                )

        # Prologue: zero-fill smem (predicated-off loads), then prefetch.
        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()
        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        k_tile_index = cutlass.Int32(0)

        for k in range(tApA.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, k, k_tile_index],
                    tAsA[None, None, k, 0],
                    pred=tApA[None, None, k],
                )
        for k in range(tBpB.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, k, k_tile_index],
                    tBsB[None, None, k, 0],
                    pred=tBpB[None, None, k],
                )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        for k_tile in range(1, num_smem_stages - 1):
            if k_tile == k_tile_count:
                tApA.fill(0)
                tBpB.fill(0)
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, k_tile],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, k_tile_index],
                tBsB[None, None, None, k_tile],
                pred=tBpB,
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        # MMA partitions and accumulators.
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsC = thr_mma.partition_C(sC)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        # smem → register copies via ldmatrix.
        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mA.element_type,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mB.element_type,
        )
        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)
        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

        smem_pipe_read = 0
        smem_pipe_write = num_smem_stages - 1
        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

        # Register pipeline prefetch of the first k-block.
        num_k_block = cute.size(tCrA, mode=[2])
        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(
                tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0]
            )
            cute.copy(
                tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0]
            )

        # Mainloop: gmem→smem and smem→register double pipelines.
        for k_tile in range(k_tile_count):
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                if k_block == num_k_block - 1:
                    tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()

                k_block_next = (k_block + 1) % num_k_block  # static
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_p[None, None, k_block_next],
                    tCrA_copy_view[None, None, k_block_next],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_p[None, None, k_block_next],
                    tCrB_copy_view[None, None, k_block_next],
                )

                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(
                            tiled_copy_A,
                            tAgA[None, None, None, k_tile_index],
                            tAsA[None, None, None, smem_pipe_write],
                            pred=tApA,
                        )
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(
                            tiled_copy_B,
                            tBgB[None, None, None, k_tile_index],
                            tBsB[None, None, None, smem_pipe_write],
                            pred=tBpB,
                        )
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == num_smem_stages:
                        smem_pipe_read = 0

                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Epilogue via smem for coalesced gmem stores.
        c_dtype = mC.element_type
        tCrD = cute.make_fragment_like(tCrC, c_dtype)
        tCrD[None] = tCrC.load().to(c_dtype)
        cute.autovec_copy(tCrD, tCsC)

        ceilM, ceilN, _ = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        mcC = cute.make_identity_tensor((
            cute.size(ceilM) * self.bM,
            cute.size(ceilN) * self.bN,
            1,
        ))
        cC = cute.local_tile(
            mcC[None, None, bidz],
            tiler=(self.bM, self.bN, self.bK),
            coord=tiler_coord,
            proj=(1, 1, None),
        )
        tCcC = thr_copy_C.partition_S(cC)

        tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
        cute.arch.sync_threads()
        cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

        tCpC = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tCgC_epilogue.shape[0][1],
                    cute.size(tCgC_epilogue, mode=[1]),
                    cute.size(tCgC_epilogue, mode=[2]),
                ),
                stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tCpC.shape[0]):
            for m in range(tCpC.shape[1]):
                tCpC[rest_v, m, 0] = cute.elem_less(
                    tCcC[(0, rest_v), m, 0][0], mC.shape[0]
                )
        for rest_v in range(tCpC.shape[0]):
            for n in range(tCpC.shape[2]):
                if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                    cute.copy(
                        tiled_copy_C,
                        tCrC_epilogue[None, None, n],
                        tCgC_epilogue[None, None, n],
                        pred=tCpC[None, None, n],
                    )
        return

    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        max_elems = 128 * 8 // dtype.width
        major_mode_size = min(major_mode_size, max_elems)

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        base_bits = int(math.log2(copy_bits // 8))
        shift_bits = int(math.log2(copy_bits // dtype.width))
        swizzle = cute.make_swizzle(swizzle_bits, base_bits, shift_bits)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout = cute.tile_to_shape(layout_atom_outer, smem_tiler, (0, 1, 2))
        return layout, swizzle

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4), 0, layout_atom_outer
        )
        if major_mode == utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(
                cute.make_swizzle(0, 3, 4), 0, layout_atom_outer
            )
        layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1))
        return layout

    def _make_gmem_tiled_copy(self, atom_copy, dtype, major_mode, copy_bits, tile_dim):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(tile_dim) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)


@cute.jit
def _tiled_bmm(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (l, m, k)
    b: cute.Tensor,  # (l, k, n)
    c: cute.Tensor,  # (l, m, n)
    stream: cuda.CUstream,
):
    """bmm-style wrapper matching torch (l,m,k)x(l,k,n)->(l,m,n) layout."""
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c, stream)


def is_tiled_gemm_eligible(M: int, N: int, K: int, cfg: CutlassConfig) -> bool:
    """Whether the tiled tensor-core kernel can run this GEMM shape+config.

    The cp.async 128-bit path requires 16-byte-contiguous operands: A is
    (M,K) row-major so K must be 8-aligned; B/C are N-contiguous so N must
    be 8-aligned. tile_k must fit the mma atom, and the staged smem buffers
    must fit the per-block smem budget.
    """
    smem_bytes = (
        (cfg.tile_m * cfg.tile_k + cfg.tile_n * cfg.tile_k) * max(cfg.num_stages, 3) * 2
    )
    return (
        K % 8 == 0
        and N % 8 == 0
        # Validated tile family only: tile_n < 128 trips a CuTe DSL
        # composed-layout bug in the epilogue autovec_copy; tile_m > 128
        # produces out-of-bounds MMA partitions (illegal memory access).
        and cfg.tile_m <= 128
        and cfg.tile_n >= 128
        and cfg.tile_m % (cfg.m_warps * _MMA_SHAPE[0]) == 0
        and cfg.tile_n % (cfg.n_warps * _MMA_SHAPE[1]) == 0
        and cfg.tile_k % _MMA_SHAPE[2] == 0
        and smem_bytes <= _MAX_SMEM_BYTES
        and cfg.num_warps * 32 <= 1024
    )


@lru_cache(maxsize=256)
def compile_tiled_gemm(
    M: int,
    N: int,
    K: int,
    arch: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    num_warps: int,
) -> Callable:
    """Compile the tiled GEMM for (M,N,K)+config. Call as fn(a3, b3, c3, stream)."""
    cfg = CutlassConfig(tile_m, tile_n, tile_k, num_stages, num_warps)
    if not is_tiled_gemm_eligible(M, N, K, cfg):
        raise ValueError(f"tiled GEMM ineligible: M={M} N={N} K={K} cfg={cfg}")
    gemm = _AmpereGemm(cfg)
    # Compile against real (empty) CUDA tensors: from_dlpack'd tensors
    # support mark_compact_shape_dynamic, whose divisibility=8 marking lets
    # the 128-bit cp.async / epilogue copy atoms prove pointer alignment.
    # The marked mode is dynamic, so the cubin itself is shape-generic.
    import torch  # noqa: PLC0415  (linux/cutlass-only path)
    from cutlass.cute.runtime import from_dlpack  # noqa: PLC0415

    ta = torch.empty((1, M, K), dtype=torch.float16, device="cuda")
    tb = torch.empty((1, K, N), dtype=torch.float16, device="cuda")
    tc = torch.empty((1, M, N), dtype=torch.float16, device="cuda")
    try:
        cute_tensors = []
        for t in (ta, tb, tc):
            ct = from_dlpack(t, assumed_align=16)
            ct.mark_layout_dynamic(leading_dim=2)
            ct.mark_compact_shape_dynamic(
                mode=2, stride_order=(0, 1, 2), divisibility=8
            )
            cute_tensors.append(ct)
        return cute.compile(
            _tiled_bmm,
            gemm,
            *cute_tensors,
            make_fake_stream(),
            options=f"--gpu-arch {arch}",
        )
    finally:
        del ta, tb, tc


def _bench_gpu(fn, warmup: int = 10, iters: int = 50) -> float:
    """Benchmark fn() on the current stream with CUDA events. Returns avg ms."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _bench_cublas_mm(M: int, N: int, K: int) -> float:
    """cuBLAS baseline: torch.mm on the same shape/dtype the step will run."""
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")
    return _bench_gpu(lambda: torch.mm(a, b, out=c))


def _bench_tiled_mm(M: int, N: int, K: int, arch: str, cfg: CutlassConfig) -> float:
    """Benchmark one tiled-GEMM config. Raises on compile/launch failure."""
    compiled = compile_tiled_gemm(
        M,
        N,
        K,
        arch,
        cfg.tile_m,
        cfg.tile_n,
        cfg.tile_k,
        cfg.num_stages,
        cfg.num_warps,
    )
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")
    a3, b3, c3 = a.view(1, M, K), b.view(1, K, N), c.view(1, M, N)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Correctness spot-check against cuBLAS before trusting the timing.
    compiled(a3, b3, c3, stream)
    torch.cuda.synchronize()
    ref = torch.mm(a, b)
    if not torch.allclose(c.float(), ref.float(), atol=2e-2):
        raise RuntimeError(f"tiled GEMM mismatch for cfg={cfg}")

    ms = _bench_gpu(lambda: compiled(a3, b3, c3, stream))
    return ms


def _extract_matmul_shapes(
    args: list[dict], buffers: dict
) -> tuple[int, int, int, str]:
    """Extract M, N, K dimensions and dtype from manifest args."""
    tensor_args = [a for a in args if a.get("kind") == "tensor"]
    if len(tensor_args) < 2:
        raise ValueError(f"Expected >=2 tensor args for mm, got {len(tensor_args)}")

    def _shape_of(arg: dict) -> list[int]:
        shape = arg.get("shape")
        if shape:
            return [int(s) for s in shape]
        bid = arg.get("buffer_id")
        if bid is not None:
            for buf_meta in buffers.values():
                if buf_meta.get("buffer_id") == bid and buf_meta.get("shape"):
                    return [int(s) for s in buf_meta["shape"]]
        name = arg.get("name")
        if name and name in buffers:
            meta = buffers[name]
            if meta.get("shape"):
                return [int(s) for s in meta["shape"]]
        raise ValueError(f"Cannot determine shape for arg {arg}")

    shape_a = _shape_of(tensor_args[0])
    shape_b = _shape_of(tensor_args[1])

    # addmm(bias, mat1, mat2): a 1D leading arg is the bias — skip it so the
    # next two tensors (mat1, mat2) are treated as the GEMM operands.
    if len(shape_a) == 1 and len(tensor_args) >= 3:
        shape_a = _shape_of(tensor_args[1])
        shape_b = _shape_of(tensor_args[2])

    if len(shape_a) == 2:
        M, K = shape_a
    elif len(shape_a) == 3:
        M, K = shape_a[-2], shape_a[-1]
    else:
        raise ValueError(f"Unexpected A shape: {shape_a}")

    if len(shape_b) == 2:
        K2, N = shape_b
    elif len(shape_b) == 3:
        K2, N = shape_b[-2], shape_b[-1]
    else:
        raise ValueError(f"Unexpected B shape: {shape_b}")

    if K != K2:
        raise ValueError(f"K mismatch: A has {K}, B has {K2}")

    dtype = (args[-1].get("dtype") if args else None) or "float16"
    return M, N, K, dtype


def tune_mm(
    args: list[dict],
    buffers: dict,
    arch: str,
    configs: list[CutlassConfig] | None = None,
    kwargs: list[str] | None = None,  # noqa: ARG001 — uniform tuner signature
) -> tuple[CutlassConfig, dict]:
    """Autotune GEMM configs against a cuBLAS baseline.

    Args:
        args: KernelArgDescriptor list from the manifest step.
        buffers: buffer table from the manifest.
        arch: GPU arch string (e.g. "sm_120").
        configs: list of CutlassConfig to benchmark. Defaults to MM_CONFIGS.
        kwargs: step kwargs strings; unused for plain GEMMs (shape comes
            from the args). Accepted so all tuners share one signature.

    Returns:
        ``(config, bench)``. ``config`` is always the best CUTLASS config
        found (the fastest tiled one for fp16, or a naive fallback for
        fp32/unaligned shapes) — never ``None``. ``bench`` records both
        timings and ``winner`` ("cutlass" or "cublas"); the runner decides
        which kernel to use from ``bench["winner"]``.
    """
    if configs is None:
        configs = MM_CONFIGS

    M, N, K, dtype = _extract_matmul_shapes(args, buffers)

    # ponytail: fp16-only v1 — fp32 GEMM without tensor-core paths can't
    # beat cuBLAS, so don't waste compile time benching it. Keep a naive
    # config so the manifest is never empty; cuBLAS is the recorded winner.
    if dtype != "float16":
        return NAIVE_CONFIG, {
            "winner": "cublas",
            "reason": f"dtype {dtype} unsupported",
        }

    best: CutlassConfig | None = None
    best_ms = float("inf")
    for cfg in configs:
        if not is_tiled_gemm_eligible(M, N, K, cfg):
            continue
        try:
            ms = _bench_tiled_mm(M, N, K, arch, cfg)
        except Exception:
            continue
        if ms < best_ms:
            best_ms = ms
            best = cfg

    cublas_ms = _bench_cublas_mm(M, N, K)
    winner = best is not None and best_ms < cublas_ms * _WIN_MARGIN
    return (best if best is not None else NAIVE_CONFIG), {
        "winner": "cutlass" if winner else "cublas",
        "cutlass_ms": best_ms if best is not None else None,
        "cublas_ms": cublas_ms,
    }
