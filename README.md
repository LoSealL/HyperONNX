# 🚀 HYPER-ONNX

[中文](./README_CN.md)|[EN](./README.md)

Hyper-ONNX can export pytorch models (`nn.Module`) in a hierarchical manner. It can keep the module hier information and make a nested onnx graph. ✨


## 📦 Install

Simply install from pypi:

```bash
pip install hyperonnx
```

Or you may install from source:

```bash
git clone https://github.com/LoSealL/hyperonnx.git
pip install -e hyperonnx[test]
```

## 🧪 Usage Example

### 1) Export `nn.Module` with specified hier info

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

### 2) Export any call to a model by auto tracing

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
            **inputs,  # your any input data
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

### 3) Export compiled modules with CUDA kernel bundle

Mark specific modules with `compile=` to `torch.compile` them during export.
A `<TypeName>.kernels/` sidecar directory is written next to each compiled
module's ONNX function, containing the cubin files and a `manifest.json`.

```python
export_hyper_onnx(
    model,
    (torch.randn(8, 768),),
    "model.onnx",
    hiera=[DecoderLayer, Attention],
    compile=[Attention],  # Attention gets a kernel bundle
    compile_static_grid=False,  # set True to skip grid AST extraction
    dynamo=True,
    external_data=True,
    external_directory="out/",
)
```

The ONNX function body remains the portable fallback. The kernel bundle is
a pure sidecar — deleting it makes the model behave as if `compile=None`.

Note: `torch.compile` is cached per Python process by dynamo. If you call
`export_hyper_onnx(..., compile=...)` twice on the same model in the same
process, call `torch._dynamo.reset()` between calls so the compile capture
fires again.
---

If you run into issues or want to contribute, feel free to open an Issue or PR. 💡
