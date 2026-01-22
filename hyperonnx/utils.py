"""
Copyright (C) 2025 The HYPERONNX Authors.

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

HYPER_DOMAIN = "hyper"

OPTIMIZER_PASSES = (
    "constantofshape_to_constant",
    "fold_constant",
    "eliminate_identity",
    "eliminate_dead_nodes",
    "fold_constant",
    "erase_output_types",
    "onnx_simplifier",
)
