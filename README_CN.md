# 🚀 HYPER-ONNX

[中文](./README_CN.md)|[EN](./README.md)

Hyper-ONNX 可以以层级化方式导出 PyTorch 模型（`nn.Module`）。它能保留模块层级信息，并生成嵌套的 ONNX 图。✨


## 📦 安装

直接从 PyPI 安装：

```bash
pip install hyperonnx
```

或从源码安装：

```bash
git clone https://github.com/LoSealL/hyperonnx.git
pip install -e hyperonnx[test]
```

## 🧪 使用示例

### 1) 导出带指定层级信息的 `nn.Module`

```python
import torch
import torchvision as tv
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from hyperonnx import export_hyper_onnx

model = tv.models.resnet18()
export_hyper_onnx(
    resnet,
    (torch.randn(1, 3, 224, 224),),
    "hyper-resnet18.onnx",
    input_names=["img"],
    output_names=["features"],
    hiera=[ResNet, BasicBlock, Bottleneck],
    do_optimization=False,
    dynamo=False,
)
```

![r18-sample](docs/assets/r18-sample.gif)

### 2) 通过自动追踪导出模型的任意调用

```python
from hyperonnx import auto_trace_method
from hyperonnx.transformers import patch_transformers
from transformers import (
    GenerationConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5OmniAudioEncoderLayer,
    Qwen2_5OmniDecoderLayer,
    Qwen2_5OmniPatchMerger,
    Qwen2_5OmniVisionBlock,
)

thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    dtype="float16",
    device_map="cuda",
)
with (
    patch_transformers(),
    auto_trace_method(thinker.model.forward) as text_tracer,
    auto_trace_method(thinker.visual.forward) as visual_tracer,
    auto_trace_method(thinker.audio_tower.forward) as audio_tracer,
):
    try:
        outputs = thinker.generate(
            **inputs,  # 你的任意输入数据
            max_new_tokens=2048,
            generation_config=GenerationConfig(use_cache=False),
        )
    except StopIteration:
        pass
    text_tracer.export(
        "qwen-omni-2.5-3b-text.onnx",
        input_names=["input_ids"],
        output_names=["hidden_states"],
        hiera=[
            Qwen2_5OmniDecoderLayer,
        ],
        external_data=True,
        external_directory="qwen25_omni/text",
        do_optimization=True,
    )
    visual_tracer.export(
        "qwen-omni-2.5-3b-vision.onnx",
        input_names=["hidden_states"],
        output_names=["last_hidden_state"],
        hiera=[
            Qwen2_5_VisionPatchEmbed,
            Qwen2_5_VisionRotaryEmbedding,
            Qwen2_5OmniVisionBlock,
            Qwen2_5OmniPatchMerger,
        ],
        external_data=True,
        external_directory="qwen25_omni/vision",
        do_optimization=True,
    )
    audio_tracer.export(
        "qwen-omni-2.5-3b-audio.onnx",
        input_names=["hidden_states"],
        output_names=["last_hidden_state"],
        hiera=[
            Qwen2_5OmniAudioEncoderLayer,
        ],
        external_data=True,
        external_directory="qwen25_omni/audio",
        do_optimization=True,
    )
```

![qwen2](docs/assets/qwen2_omni_vision.gif)

### 3) 导出编译模块及其 CUDA kernel 包

通过 `compile=` 标记需要编译的模块，导出时会对其执行 `torch.compile`。
每个被编译模块的 ONNX function 旁边会写出一个 `<TypeName>.kernels/`
sidecar 目录，其中包含 cubin 文件和一个 `manifest.json`。

```python
export_hyper_onnx(
    model,
    (torch.randn(8, 768),),
    "model.onnx",
    hiera=[DecoderLayer, Attention],
    compile=[Attention],            # Attention gets a kernel bundle
    dynamo=True,
    external_data=True,
    external_directory="out/",
)
```

ONNX function 主体仍然是可移植的回退实现。kernel 包是纯 sidecar ——
删除它后，模型行为与 `compile=None` 完全一致。

注意：`torch.compile` 在同一个 Python 进程中会被 dynamo 缓存。如果在
同一进程中对同一模型调用两次 `export_hyper_onnx(..., compile=...)`，
请在两次调用之间执行 `torch._dynamo.reset()`，以便重新触发编译捕获。
---

如果你在使用中遇到问题或希望贡献代码，欢迎提 Issue 或 PR。💡
