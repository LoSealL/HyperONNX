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
"""

import ctypes
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_NP_DTYPES: dict[str, type] = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "int16": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
    "bfloat16": np.float16,
}


def _np_dtype(dtype_str: str) -> type:
    return _NP_DTYPES.get(dtype_str, np.float32)


def _check(ret: Any, label: str) -> None:
    status = ret[0] if isinstance(ret, tuple) else ret
    if status != 0:
        raise RuntimeError(f"{label} failed: {status}")


def replay(
    bundle_dir: str | Path,
    input_arrays: list[np.ndarray],
) -> np.ndarray:
    """Replay compiled cubins from a kernel bundle — no PyTorch required.

    Reads ``manifest.json`` from the bundle, allocates device buffers
    (inputs from ``input_arrays``, parameters from ``.bin`` files,
    intermediates zeroed), loads each cubin via the CUDA Driver API,
    and launches kernels in manifest order.

    Args:
        bundle_dir: path to the ``<type>.kernels/`` directory.
        input_arrays: positional input numpy arrays, matching the
            manifest's ``io.inputs`` order.

    Returns:
        The output numpy array.
    """
    drv = importlib.import_module("cuda.bindings.driver")
    drv.cuInit(0)
    _, dev = drv.cuDeviceGet(0)
    _, ctx = drv.cuDevicePrimaryCtxRetain(dev)
    drv.cuCtxPushCurrent(ctx)

    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    buffers_meta: list[dict] = manifest.get("buffers", [])
    kernels_meta: list[dict] = manifest["kernels"]

    dev_ptrs: dict[int, int] = {}
    keep_alive: list[Any] = []

    for buf in buffers_meta:
        bid = buf["id"]
        np_dt = _np_dtype(buf["dtype"])
        shape = tuple(buf["shape"])
        nbytes = int(np.prod(shape)) * np.dtype(np_dt).itemsize

        if buf["kind"] == "input":
            name = buf.get("name", "")
            idx = int(name.split("_")[-1]) if "_" in name else bid
            arr = np.ascontiguousarray(
                input_arrays[min(idx, len(input_arrays) - 1)], dtype=np_dt
            )
            _, dev_ptr = drv.cuMemAlloc(nbytes)
            drv.cuMemcpyHtoD(dev_ptr, arr.ctypes.data, nbytes)
            dev_ptrs[bid] = int(dev_ptr)
            keep_alive.append(arr)

        elif buf["kind"] == "parameter" and "file" in buf:
            raw = (bundle_dir / buf["file"]).read_bytes()
            arr = np.frombuffer(raw, dtype=np_dt).reshape(shape)
            _, dev_ptr = drv.cuMemAlloc(nbytes)
            drv.cuMemcpyHtoD(dev_ptr, arr.ctypes.data, nbytes)
            dev_ptrs[bid] = int(dev_ptr)
            keep_alive.append(arr)

        else:
            _, dev_ptr = drv.cuMemAlloc(nbytes)
            drv.cuMemsetD32(dev_ptr, 0, nbytes // 4)
            dev_ptrs[bid] = int(dev_ptr)

    functions: dict[str, Any] = {}
    for kern in kernels_meta:
        sym = kern["symbol"]
        if sym not in functions:
            cubin_bytes = (bundle_dir / kern["cubin"]).read_bytes()
            ret, mod = drv.cuModuleLoadData(cubin_bytes)
            _check(ret, f"cuModuleLoadData for {sym}")
            ret, func = drv.cuModuleGetFunction(mod, sym.encode())
            _check(ret, f"cuModuleGetFunction for {sym}")
            functions[sym] = func

    _, stream = drv.cuStreamCreate(0)

    for kern in kernels_meta:
        func = functions[kern["symbol"]]
        grid = kern["launch"].get("captured_grid") or [1, 1, 1]
        gx, gy, gz = int(grid[0]), int(grid[1]), int(grid[2])
        shared = int(kern["launch"].get("shared_mem_bytes", 0))
        nw = int(kern["launch"].get("num_warps", 4))
        block_x = nw * int(kern["device_target"]["warp_size"])

        if shared > 0:
            drv.cuFuncSetAttribute(
                func,
                drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            )

        param_values: list[Any] = []
        param_types: list[type] = []
        for arg in kern["args"]:
            if arg["kind"] == "tensor":
                bid = arg["buffer_id"]
                ptr = dev_ptrs.get(bid)
                if ptr is None:
                    raise RuntimeError(f"buffer {bid} not allocated")
                param_values.append(ptr)
                param_types.append(ctypes.c_void_p)
            elif arg["dtype"] == "float32":
                param_values.append(float(arg["value"]))
                param_types.append(ctypes.c_float)
            else:
                param_values.append(int(arg["value"]))
                param_types.append(ctypes.c_int32)

        for _ in range(
            kern.get("num_scratch_args", kern["launch"].get("num_scratch_args", 2))
        ):
            param_values.append(0)
            param_types.append(ctypes.c_void_p)

        kernel_params = (tuple(param_values), tuple(param_types))
        keep_alive.append(param_values)

        ret = drv.cuLaunchKernel(
            func,
            gx,
            gy,
            gz,
            block_x,
            1,
            1,
            shared,
            stream,
            kernel_params,
            0,
        )
        _check(ret, f"cuLaunchKernel for {kern['symbol']}")

    drv.cuStreamSynchronize(stream)

    output_bid: int | None = None
    for buf in buffers_meta:
        if buf["kind"] == "output":
            output_bid = buf["id"]
            break

    if output_bid is None:
        raise RuntimeError("no output buffer in manifest")

    out_buf = next(b for b in buffers_meta if b["id"] == output_bid)
    np_dt = _np_dtype(out_buf["dtype"])
    shape = tuple(out_buf["shape"])
    nbytes = int(np.prod(shape)) * np.dtype(np_dt).itemsize
    result = np.empty(shape, dtype=np_dt)
    drv.cuMemcpyDtoH(result.ctypes.data, dev_ptrs[output_bid], nbytes)

    for ptr in dev_ptrs.values():
        drv.cuMemFree(ptr)

    return result


def verify(
    bundle_dir: str | Path,
    input_arrays: list[np.ndarray],
    expected_output: np.ndarray,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Replay from bundle and compare against expected output."""
    out = replay(bundle_dir, input_arrays)
    ok = np.allclose(out, expected_output, atol=atol, rtol=rtol)
    if not ok:
        diff = np.abs(out.astype(np.float64) - expected_output.astype(np.float64))
        print(f"MISMATCH: max abs diff = {diff.max()}")
    return ok
