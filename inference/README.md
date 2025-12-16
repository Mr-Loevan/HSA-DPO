# HSA-DPO Inference

This directory contains inference code and examples for the HSA-DPO model.

## Setup

1. Make sure you've installed HSA-DPO with its dependencies:
```bash
pip install -e ..
```

2. Download the model weights:
```bash
# Download base LLaVA model
huggingface-cli download liuhaotian/llava-v1.5-13b --local-dir ../models/llava-v1.5-13b

# Download HSA-DPO LoRA weights
modelscope download --model xiaowenyi/HSA-DPO --local-dir ../checkpoints
```

## Quick Start

### Single Image Inference

```bash
python inference_example.py \
    --model-base ../models/llava-v1.5-13b \
    --lora-path ../checkpoints/HSA-DPO_llava_v1.5-13B-lora \
    --image path/to/image.jpg \
    --prompt "Describe this image in detail."
```


