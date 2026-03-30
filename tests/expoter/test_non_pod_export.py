"""
Copyright Wenyi Tang 2026

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

These tests show how to trick torch.export to deal with non-POD
structures.
It should be traced frequently to verify the functionality.
"""

from typing import Protocol

import pytest
import torch


class CustomData(Protocol):
    """A custom structure that has custom method to be called."""

    def update(self) -> torch.Tensor:
        """For now it'd better return supported data"""
        ...


class ModuleWithCustomArgs(torch.nn.Module):
    """A module that consumes an argument of non pod data."""

    def __init__(self):
        super().__init__()

    def forward(self, x: CustomData):
        """Call a custom method defined in the custom data."""
        return x.update()


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_custom_data_directly_fail(dynamo):
    """Trivial user data is not supported and fail in flatten args check"""

    class TrivialUserData:
        """Custom object is not supported."""

        def __init__(self, x, y):
            self.x = x
            self.y = y

        def update(self):
            return self.x + self.y

    model = ModuleWithCustomArgs()
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])
    # normal forward should pass
    model(TrivialUserData(x, y))
    with pytest.raises(RuntimeError):
        torch.onnx.export(model, (TrivialUserData(x, y),), dynamo=dynamo)


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_custom_data_as_pod_tuple_fail(dynamo):
    """Custom POD like tuple/list/dict can pass flatten args check but failed
    because of lost class info when go into torch._C extension.
    """

    class TupleUserData(tuple):
        """update method won't be preserved and this object falls back to
        normal tuple."""

        def update(self):
            return self[0] + self[1]

    model = ModuleWithCustomArgs()
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])
    # normal forward should pass
    model(TupleUserData([x, y]))
    with pytest.raises(Exception if dynamo else AttributeError):
        torch.onnx.export(model, (TupleUserData([x, y]),), dynamo=dynamo)


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_custom_data_as_tensor(dynamo, tmp_path):
    """Wrap custom data as a subclass of torch.Tensor can trick the export
    procedure and custom method can be preserved in this way.
    """

    class CustomTensor(torch.Tensor):
        """__init__/__new__ methods are not recommended to change"""

        def update(self):
            """act like customed tuple"""
            return self[0] + self[1]

    model = ModuleWithCustomArgs()
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])
    # normal forward should pass
    model(CustomTensor(torch.stack([x, y])))
    torch.onnx.export(
        model,
        (CustomTensor(torch.stack([x, y])),),
        tmp_path / "model.onnx",
        dynamo=dynamo,
    )
