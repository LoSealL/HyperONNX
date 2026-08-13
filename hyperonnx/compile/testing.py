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
# pyright: reportMissingImports=none

import ast as pyast
import ctypes
import importlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from onnxifier.logger import debug


def _check(ret: Any, label: str) -> None:
    status = ret[0] if isinstance(ret, tuple) else ret
    if status != 0:
        raise RuntimeError(f"{label} failed: {status}")


def _bid_of(table: dict, name: str | None, _depth: int = 0) -> int | None:
    """Resolve a static buffer name to a runtime buffer_id via the graph's
    buffer table (following alias/view links)."""
    if not name or _depth > 8:
        return None
    meta = table.get(name)
    if meta is None:
        return None
    if "buffer_id" in meta:
        return meta["buffer_id"]
    for key in ("alias_of", "view_of"):
        if key in meta:
            return _bid_of(table, meta[key], _depth + 1)
    return None


def _parse_extern_kwargs(kwarg_strs: list[str]) -> dict:
    """Parse ``["stride=(2, 2)", "bias=None"]`` kwarg strings into values."""
    out: dict[str, Any] = {}
    for kw in kwarg_strs:
        key, _, value = kw.partition("=")
        try:
            out[key] = pyast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def replay(
    bundle_dir: str | Path,
    input_arrays: list[torch.Tensor],
    *,
    cutlass: bool = False,
) -> torch.Tensor:
    """Replay a recorded pipeline from a kernel bundle.

    Walks the manifest's ``pipeline`` steps in execution order: triton
    kernels launch their cubins via the CUDA Driver API; ``extern_kernel``
    steps (cuDNN/cuBLAS calls recorded without cubins) are executed
    through their ``torch.ops.aten`` counterparts — PyTorch is required
    for bundles containing extern steps. Steps whose launch was not
    traced (``launch_missing``) launch best-effort with table-resolved
    buffers and raise when any buffer stays unresolved.

    Args:
        bundle_dir: path to the ``<type>.kernels/`` directory.
        input_arrays: positional input tensors, matching the
            manifest's ``io.inputs`` order.
        cutlass: when ``True``, extern_kernel steps that carry a
            ``cutlass_config`` and have a registered CUTLass runner
            (mm / bmm / addmm) are executed via CuTe DSL GEMM kernels
            instead of ``torch.ops.aten``. Requires Linux + CuTe DSL.
            Unsupported ops and steps without config fall through to aten.

    Returns:
        The output tensor.
    """

    drv = importlib.import_module("cuda.bindings.driver")
    drv.cuInit(0)
    _, dev = drv.cuDeviceGet(0)
    _, ctx = drv.cuDevicePrimaryCtxRetain(dev)
    drv.cuCtxPushCurrent(ctx)

    arch: str | None = None
    if cutlass:
        from .cutlass_kernels.extract import detect_gpu_arch

        arch = detect_gpu_arch()

    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    buffers_meta: list[dict] = manifest.get("buffers", [])

    tensors: dict[int, Any] = {}
    storages: dict[int, Any] = {}
    last_out_layout: dict[int, tuple] = {}

    # Pre-scan: compute max storage span per buffer and build name→bid hints.
    max_span: dict[int, int] = {}
    name_bid_hint: dict[str, int] = {}
    name_layout_hint: dict[str, tuple] = {}
    for graph in manifest["pipeline"]:
        for step in graph["steps"]:
            for arg in step.get("args", []):
                if not isinstance(arg, dict) or arg.get("kind") != "tensor":
                    continue
                bid = arg.get("buffer_id")
                shape, stride = arg.get("shape"), arg.get("stride")
                if shape and stride:
                    if arg.get("name") and arg["name"] not in name_layout_hint:
                        name_layout_hint[arg["name"]] = (
                            [int(s) for s in shape],
                            [int(s) for s in stride],
                        )
                    if bid is None:
                        continue
                    span = (
                        sum((int(s) - 1) * int(st) for s, st in zip(shape, stride)) + 1
                    )
                    max_span[bid] = max(max_span.get(bid, 0), span)
                if (
                    bid is not None
                    and arg.get("name")
                    and arg["name"] not in name_bid_hint
                ):
                    name_bid_hint[arg["name"]] = bid

    def _register(bid, shape, stride, torch_dt, src=None):
        """Allocate flat storage sized for the largest layout, then view it."""
        shape = [int(s) for s in shape]
        if stride:
            stride = [int(s) for s in stride]
            n = sum((s - 1) * st for s, st in zip(shape, stride)) + 1
        else:
            n = math.prod(shape) if shape else 1
        n = max(1, n, max_span.get(bid, 0))
        flat = storages.get(bid)
        if flat is None or flat.numel() < n:
            flat = torch.zeros(n, dtype=torch_dt, device="cuda")
            storages[bid] = flat
        view = flat.as_strided(shape, stride) if stride else flat.view(shape)
        if src is not None:
            view.copy_(src)
        tensors[bid] = view
        return view

    for buf in buffers_meta:
        bid = buf["id"]
        torch_dt = getattr(torch, buf["dtype"], torch.float32)
        shape = buf["shape"]
        stride = buf.get("stride")
        if not stride or len(stride) != len(shape):
            stride = None

        if buf["kind"] == "input":
            name = buf.get("name", "")
            idx = int(name.split("_")[-1]) if "_" in name else bid
            src = (
                input_arrays[min(idx, len(input_arrays) - 1)]
                .contiguous()
                .to(device="cuda", dtype=torch_dt)
            )
            _register(bid, shape, stride, torch_dt, src)
        elif buf["kind"] == "parameter" and "file" in buf:
            raw = (bundle_dir / buf["file"]).read_bytes()
            disk_dt = torch.float16 if buf["dtype"] == "bfloat16" else torch_dt
            src = (
                torch
                .frombuffer(bytearray(raw), dtype=disk_dt)
                .reshape(shape)
                .to(device="cuda", dtype=torch_dt)
            )
            _register(bid, shape, stride, torch_dt, src)
        else:
            _register(bid, shape, stride, torch_dt)

    functions: dict[str, Any] = {}
    modules: dict[str, Any] = {}
    name_tensors: dict[str, Any] = {}

    def _resolve_base(
        name: str | None, table: dict, bid: int | None
    ) -> tuple[int | None, int]:
        """Follow ``view_of`` links to the base allocation, accumulating offset.

        Reinterpret views get a distinct runtime id (pointer dedup hands
        non-zero-offset views their own id); the launch-trace buffer_id is
        the view's id, not the storage the data lives in. Reads
        (``tensor_for``) and writes (``run_extern``) must agree on storage,
        so both route through here.
        """
        cur_meta = table.get(name or "", {})
        cur_bid = bid
        view_offset = 0
        _seen: set[int] = set()
        while cur_meta.get("view_of") and cur_bid is not None and cur_bid not in _seen:
            _seen.add(cur_bid)
            view_offset += int(cur_meta.get("offset", 0))
            cur_meta = table.get(cur_meta["view_of"], {})
            cur_bid = cur_meta.get("buffer_id")
        base_bid = cur_bid if cur_bid is not None else bid
        return base_bid, view_offset

    def tensor_for(arg: dict, table: dict) -> Any:
        """Resolve a step arg to a device tensor.

        Resolution order: explicit buffer_id → table link (alias/view) →
        name-based lookup → lazy allocation from the static table.
        Raises only when nothing resolves.
        """
        name = arg.get("name")
        bid = arg.get("buffer_id")
        meta = table.get(name or "", {})
        if bid is None:
            # Guessed ids are trusted for inputs but not intermediates.
            if not meta.get("guessed") or meta.get("kind") == "input":
                bid = _bid_of(table, name)
        if bid is None and name is not None:
            bid = name_bid_hint.get(name)
        base_bid, view_offset = _resolve_base(name, table, bid)
        shape, stride = arg.get("shape"), arg.get("stride")
        if base_bid is not None and base_bid in storages:
            if shape and stride:
                return storages[base_bid].as_strided(
                    [int(s) for s in shape], [int(s) for s in stride], view_offset
                )
            # No per-arg layout: fall back to the table's recorded layout.
            meta_shape, meta_stride = meta.get("shape"), meta.get("stride")
            if meta_shape and meta_stride and len(meta_shape) == len(meta_stride):
                shape = [int(s) for s in meta_shape]
                stride = [int(s) for s in meta_stride]
                span = sum((s - 1) * st for s, st in zip(shape, stride)) + 1
                if storages[base_bid].numel() >= span + view_offset:
                    return storages[base_bid].as_strided(shape, stride, view_offset)
            return tensors[base_bid]
        if name is not None and name in name_tensors:
            return name_tensors[name]
        meta_shape = meta.get("shape")
        if name is not None and meta_shape is not None:
            torch_dt = getattr(torch, meta.get("dtype", "float32"), torch.float32)
            shape = [int(s) for s in meta_shape]
            meta_stride = meta.get("stride")
            if meta_stride and len(meta_stride) == len(shape):
                n = max(
                    1,
                    sum(
                        (s - 1) * st
                        for s, st in zip(shape, [int(x) for x in meta_stride])
                    )
                    + 1,
                )
                flat = torch.zeros(n, dtype=torch_dt, device="cuda")
                t = flat.as_strided(shape, [int(x) for x in meta_stride])
            else:
                t = torch.zeros(shape, dtype=torch_dt, device="cuda")
            name_tensors[name] = t
            return t
        raise RuntimeError(
            f"unresolved buffer {name or bid} — replay cannot cover this "
            "step; fall back to the ONNX function"
        )

    def load_function(cubin: str, sym: str) -> Any:
        if sym not in functions:
            cubin_bytes = (bundle_dir / cubin).read_bytes()
            ret, mod = drv.cuModuleLoadData(cubin_bytes)
            _check(ret, f"cuModuleLoadData for {sym}")
            modules[sym] = mod
            ret, func = drv.cuModuleGetFunction(mod, sym.encode())
            _check(ret, f"cuModuleGetFunction for {sym}")
            functions[sym] = func
        return functions[sym]

    _, stream = drv.cuStreamCreate(0)
    keep_alive: list[Any] = []

    def launch_triton(step: dict, table: dict) -> None:
        func = load_function(step["cubin"], step["kernel"])
        launch = step["launch"]
        grid = launch.get("captured_grid") or [1, 1, 1]
        gx, gy, gz = int(grid[0]), int(grid[1]), int(grid[2])
        shared = int(launch.get("shared_mem_bytes", 0))
        nw = int(launch.get("num_warps", 4))
        block_x = nw * int(step["device_target"]["warp_size"])
        if shared > 0:
            drv.cuFuncSetAttribute(
                func,
                drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            )

        param_values: list[Any] = []
        param_types: list[type] = []
        for arg in step["args"]:
            if arg["kind"] == "tensor":
                try:
                    t = tensor_for(arg, table)
                except RuntimeError as exc:
                    raise RuntimeError(f"{exc} (in {step['kernel']})") from exc
                if arg.get("direction") == "out":
                    out_bid_la = arg.get("buffer_id")
                    if arg.get("shape") and arg.get("stride"):
                        last_out_layout[out_bid_la] = (arg["shape"], arg["stride"])
                    elif out_bid_la is not None:
                        out_name_la = arg.get("name")
                        out_meta_la = table.get(out_name_la or "", {})
                        if out_meta_la.get("shape") and out_meta_la.get("stride"):
                            last_out_layout[out_bid_la] = (
                                [int(s) for s in out_meta_la["shape"]],
                                [int(s) for s in out_meta_la["stride"]],
                            )
                param_values.append(t.data_ptr())
                param_types.append(ctypes.c_void_p)
            elif arg.get("dtype") == "float32":
                param_values.append(float(arg["value"]))
                param_types.append(ctypes.c_float)
            else:
                param_values.append(int(arg["value"]))
                param_types.append(ctypes.c_int32)

        for _ in range(int(launch.get("num_scratch_args", 2))):
            param_values.append(0)
            param_types.append(ctypes.c_void_p)

        kernel_params = (tuple(param_values), tuple(param_types))
        keep_alive.append(param_values)
        ret = drv.cuLaunchKernel(
            func, gx, gy, gz, block_x, 1, 1, shared, stream, kernel_params, 0
        )
        _check(ret, f"cuLaunchKernel for {step['kernel']}")

    def run_extern(step: dict, table: dict) -> None:
        op_name = step["kernel"].rsplit(".", 1)[-1]
        args: list[Any] = []
        for a in step["args"]:
            if a["kind"] == "tensor":
                try:
                    args.append(tensor_for(a, table))
                except RuntimeError as exc:
                    raise RuntimeError(f"{exc} (in {step['kernel']})") from exc
            elif "value" in a:
                args.append(a["value"])
        kwargs = _parse_extern_kwargs(step.get("kwargs", []))
        config = step.get("cutlass_config")
        use_cutlass = cutlass and bool(config)
        if use_cutlass:
            from .cutlass_kernels.run import run_cutlass_extern

            debug(f"extern_kernel {op_name} → cutlass")
            assert config is not None and arch is not None
            result = run_cutlass_extern(op_name, args, config, arch, kwargs=kwargs)
        else:
            debug(f"extern_kernel {op_name} → aten")
            op = getattr(torch.ops.aten, op_name)
            result = op(*args, **kwargs)

        out_name = step["output"].get("name")
        out_bid = (
            step["output"].get("buffer_id")
            or _bid_of(table, out_name)
            or name_bid_hint.get(out_name)
        )
        # Writes must land on the same storage reads (tensor_for) pull from:
        # resolve the base allocation via the view_of chain, not the raw
        # launch-trace buffer_id (a view id, not the underlying allocation).
        base_bid, view_offset = _resolve_base(out_name, table, out_bid)
        # Layout priority: downstream consumer → static table → alias/view → contiguous.
        meta = table.get(out_name or "", {})
        hint_name = out_name
        if not meta:
            for cand_name, cand in table.items():
                if cand.get("alias_of") == out_name or cand.get("view_of") == out_name:
                    meta, hint_name = cand, cand_name
                    break
        layout = name_layout_hint.get(hint_name)
        if layout is None and meta.get("shape") and meta.get("stride"):
            layout = ([int(s) for s in meta["shape"]], [int(s) for s in meta["stride"]])
        # Manifest buffer stride as last-resort layout source.
        if layout is None and out_bid is not None:
            buf_meta = next((b for b in buffers_meta if b["id"] == out_bid), None)
            if buf_meta and buf_meta.get("stride"):
                bs = [int(s) for s in buf_meta["stride"]]
                if len(bs) == len(result.shape):
                    layout = (list(result.shape), bs)
        if base_bid is not None and base_bid in storages:
            storage = storages[base_bid]
            if layout is not None:
                shape, stride = layout
                span = sum((s - 1) * st for s, st in zip(shape, stride)) + 1
                if (
                    storage.numel() >= span + view_offset
                    and list(result.shape) == shape
                ):
                    view = storage.as_strided(shape, stride, view_offset)
                    view.copy_(result)
                    tensors[base_bid] = view
                    if out_bid is not None and out_bid != base_bid:
                        tensors[out_bid] = view
                    return
            n = int(result.numel())
            if storage.numel() >= n + view_offset:
                r_stride = tuple(int(s) for s in result.stride())
                r_shape = [int(s) for s in result.shape]
                span_r = sum((s - 1) * st for s, st in zip(r_shape, r_stride)) + 1
                if (
                    not result.is_contiguous()
                    and len(r_stride) == len(r_shape)
                    and storage.numel() >= span_r + view_offset
                ):
                    view = storage.as_strided(r_shape, r_stride, view_offset)
                else:
                    view = storage[view_offset : view_offset + n].view(r_shape)
                view.copy_(result)
                tensors[base_bid] = view
                if out_bid is not None and out_bid != base_bid:
                    tensors[out_bid] = view
                return
        # No runtime buffer identity: result becomes the named buffer.
        if out_name:
            name_tensors[out_name] = result.contiguous()

    def run_as_strided(step: dict, table: dict) -> None:
        src_arg = step["args"][0]
        try:
            src = tensor_for(src_arg, table)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} (in as_strided)") from exc
        shape = [int(s) for s in step["shape"]]
        stride = [int(s) for s in step["stride"]]
        offset = int(step.get("offset", 0))
        result = src.as_strided(shape, stride, offset)
        name_tensors[step["output"]["name"]] = result

    for graph in manifest["pipeline"]:
        table = graph.get("buffers", {})
        name_tensors.clear()
        for step in graph["steps"]:
            if step["type"] == "triton_kernel":
                launch_triton(step, table)
            elif step["type"] == "extern_kernel":
                run_extern(step, table)
            elif step["type"] == "as_strided":
                run_as_strided(step, table)

    drv.cuStreamSynchronize(stream)

    # Select the output buffer matching io.outputs[0]'s declared shape.
    # For multi-output modules the registry tags every tuple element as
    # kind="output" in buffer-creation order, which need not match the
    # return tuple's order — picking manifest-first returns the wrong tensor.
    io_outputs = manifest.get("io", {}).get("outputs", [])
    output_bufs = [b for b in buffers_meta if b["kind"] == "output"]
    if not output_bufs:
        raise RuntimeError("no output buffer in manifest")

    if io_outputs:
        expected_shape = [int(s) for s in io_outputs[0]["shape"]]
        expected_numel = math.prod(expected_shape)
        chosen = next(
            (b for b in output_bufs if [int(s) for s in b["shape"]] == expected_shape),
            None,
        )
        if chosen is None:
            chosen = next(
                (
                    b
                    for b in output_bufs
                    if math.prod(int(s) for s in b["shape"]) == expected_numel
                ),
                output_bufs[0],
            )
    else:
        expected_shape = None
        chosen = output_bufs[0]

    output_bid = chosen["id"]

    # Prefer the final writer's layout over the registry's first-seen view.
    out_t = tensors[output_bid]
    layout = last_out_layout.get(output_bid)
    if layout is not None and output_bid in storages:
        shape, stride = layout
        out_t = storages[output_bid].as_strided(
            [int(s) for s in shape], [int(s) for s in stride]
        )
    elif output_bid in storages:
        n = out_t.numel()
        out_t = storages[output_bid][:n].view(out_t.shape)

    # Apply the graph's output reshape (inductor's return expression wraps
    # the output buffer with reinterpret_tensor to give it the model's
    # declared output shape/stride). io.outputs carries the declared shape.
    if expected_shape is not None and list(out_t.shape) != expected_shape:
        if out_t.is_contiguous():
            out_t = out_t.view(expected_shape)
    return out_t


def verify(
    bundle_dir: str | Path,
    input_arrays: list[torch.Tensor],
    expected_output: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    cutlass: bool = False,
) -> bool:
    """Replay from bundle and compare against expected output."""
    out = replay(bundle_dir, input_arrays, cutlass=cutlass)
    ok = torch.allclose(out, expected_output, atol=atol, rtol=rtol)
    if not ok:
        diff = (out.float() - expected_output.float()).abs()
        print(f"MISMATCH: max abs diff = {diff.max().item()}")
    return ok
