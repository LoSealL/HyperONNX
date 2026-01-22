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

# pylint: disable=missing-class-docstring,missing-function-docstring

import random
import unittest

import onnx
import onnxscript
import torch

from hyperonnx.exporter.dynamo import build_onnxscript
from hyperonnx.hyper_export import trace_module_spec
from hyperonnx.typing import default_module_spec


@torch.library.custom_op("test::single_tensor_to_tensor", mutates_args=())
def single_tensor_to_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@single_tensor_to_tensor.register_fake
def single_tensor_to_tensor_fake(tensor_x):
    return torch.empty_like(tensor_x)


class TestSingleT2T(torch.nn.Module):
    def forward(self, x):
        return single_tensor_to_tensor(x)


@torch.library.custom_op("test::dual_tensor_to_tuple", mutates_args=())
def dual_tensor_to_tuple(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.add(x, y), torch.mul(x, y).mean(dim=1)


@dual_tensor_to_tuple.register_fake
def dual_tensor_to_tuple_fake(tensor_x, tensor_y):
    return (
        torch.empty_like(tensor_x + tensor_y),
        torch.empty_like((tensor_x * tensor_y).mean(dim=1)),
    )


class TestDualT2T(torch.nn.Module):
    def forward(self, x, y):
        return dual_tensor_to_tuple(x, y)


class TestBuildOnnxScript(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = default_module_spec()
        random.seed(42)

    def tearDown(self) -> None:
        self.spec.clear()

    def random_shape(self, rank: int, min_val: int = 1, max_val: int = 32) -> list[int]:
        return [random.randint(min_val, max_val) for _ in range(rank)]

    def trace(self, model: torch.nn.Module, example_inputs: tuple, **kwargs):
        return trace_module_spec(
            model,
            example_inputs,
            kwargs=kwargs,
            opset_version=19,
            hiera=[model.__class__],
            module_spec=self.spec,
        )

    def test_tutorial(self):
        test_op = onnxscript.values.Opset(domain="TestOp", version=1)
        rank = random.randint(1, 6)
        shape = self.random_shape(rank)
        model = TestSingleT2T()
        example_inputs = (torch.rand(shape, device="cpu"),)

        @onnxscript.script(test_op)
        def onnx_single_t2t(tensor_x: onnxscript.FLOAT) -> onnxscript.FLOAT:
            # Note: the attribute name for `test_op` can't be the same as
            # the function name.
            # E.g. `test_op.onnx_single_t2t` would cause infinite recursion.
            return test_op.myop(tensor_x)  # type: ignore

        f = torch.onnx.export(
            model,
            example_inputs,
            None,
            dynamo=True,
            custom_translation_table={  # type: ignore
                "test::single_tensor_to_tensor": onnx_single_t2t,
            },
        )
        assert f is not None
        onnx.checker.check_model(f.model_proto, full_check=True)
        assert len(f.model.graph.inputs) == 1
        assert len(f.model.graph.outputs) == 1

    def test_tutorial_dual2dual(self):
        test_op = onnxscript.values.Opset(domain="TestOp", version=1)
        rank = random.randint(1, 6)
        shape = self.random_shape(rank)
        model = TestDualT2T()
        example_inputs = (
            torch.rand(shape, device="cpu"),
            torch.rand(shape, device="cpu"),
        )

        @onnxscript.script(test_op)
        def onnx_dual_d2d(x: onnxscript.FLOAT, y: onnxscript.FLOAT) -> tuple[
            onnxscript.FLOAT,
            onnxscript.FLOAT,
        ]:
            sum_result, mean_mul_result = test_op.myop1(x, y)  # type: ignore
            return sum_result, mean_mul_result  # type: ignore

        f = torch.onnx.export(
            model,
            example_inputs,
            None,
            dynamo=True,
            custom_translation_table={  # type: ignore
                "test::dual_tensor_to_tuple": onnx_dual_d2d,
            },
        )
        assert f is not None
        onnx.checker.check_model(f.model_proto, full_check=True)
        assert len(f.model.graph.inputs) == 2
        assert len(f.model.graph.outputs) == 2

    def test_single_tensor_to_tensor(self):
        rank = random.randint(1, 6)
        shape = self.random_shape(rank)
        model = TestSingleT2T()
        example_inputs = (torch.rand(shape, device="cpu"),)
        spec = self.trace(model, example_inputs)

        onnx_fn = build_onnxscript(spec[model])

        f = torch.onnx.export(
            model,
            example_inputs,
            None,
            dynamo=True,
            input_names=["x"],
            output_names=["y"],
            custom_translation_table={  # type: ignore
                "test::single_tensor_to_tensor": onnx_fn,
            },
        )
        assert f is not None
        try:
            onnx.checker.check_model(f.model_proto, full_check=True)
            assert len(f.model.graph.inputs) == 1
            assert len(f.model.graph.outputs) == 1
        except Exception:
            f.save("error_single_tensor_to_tensor.onnx")
            raise

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_single_tensor_to_tensor_cuda_half(self):
        rank = random.randint(1, 6)
        shape = self.random_shape(rank)
        model = TestSingleT2T().cuda().half()
        example_inputs = (torch.rand(shape, device="cuda").half(),)
        spec = self.trace(model, example_inputs)

        onnx_fn = build_onnxscript(spec[model])

        f = torch.onnx.export(
            model,
            example_inputs,
            None,
            dynamo=True,
            input_names=["x"],
            output_names=["y"],
            custom_translation_table={  # type: ignore
                "test::single_tensor_to_tensor": onnx_fn,
            },
        )
        assert f is not None
        try:
            onnx.checker.check_model(f.model_proto, full_check=True)
            assert len(f.model.graph.inputs) == 1
            assert len(f.model.graph.outputs) == 1
        except Exception:
            f.save("error_single_tensor_to_tensor_cuda_half.onnx")
            raise

    def test_dual_tensor_to_tuple(self):
        rank = random.randint(1, 6)
        shape = self.random_shape(rank)
        model = TestDualT2T()
        example_inputs = (
            torch.rand(shape, device="cpu"),
            torch.rand(shape, device="cpu"),
        )
        spec = self.trace(model, example_inputs)

        onnx_fn = build_onnxscript(spec[model])
        f = torch.onnx.export(
            model,
            example_inputs,
            None,
            dynamo=True,
            input_names=["x", "y"],
            output_names=["sum", "mean_mul"],
            custom_translation_table={  # type: ignore
                "test::dual_tensor_to_tuple": onnx_fn,
            },
        )
        assert f is not None
        try:
            onnx.checker.check_model(f.model_proto, full_check=True)
            assert len(f.model.graph.inputs) == 2
            assert len(f.model.graph.outputs) == 2
        except Exception:
            f.save("error_dual_tensor_to_tuple.onnx")
            raise
