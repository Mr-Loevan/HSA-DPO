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

### Batch Processing

For batch processing of multiple images, you can modify the `inference_example.py` script or use it in a loop:

```python
import json
from pathlib import Path

# Load your dataset
with open("test_data.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Process each item
for item in data:
    output = run_inference(
        model, tokenizer, image_processor,
        item["image"], item["prompt"], args
    )
    item["response"] = output
```

## Parameters

- `--model-base`: Path to the base LLaVA-v1.5 model
- `--lora-path`: Path to HSA-DPO LoRA weights
- `--image`: Path or URL to the input image
- `--prompt`: Text prompt for the model
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.95)
- `--num-beams`: Number of beams for beam search (default: 1)
- `--max-new-tokens`: Maximum tokens to generate (default: 1024)
- `--device`: CUDA device to use (default: "0")

## Evaluation Benchmarks

The HSA-DPO model can be evaluated on various hallucination benchmarks:

- **CHAIR**: Object hallucination metrics
- **POPE**: Binary hallucination detection
- **MMHal-Bench**: Comprehensive hallucination evaluation
- **LLaVA-Bench**: General capability evaluation

## Notes

- The model automatically merges LoRA weights during loading
- Ensure your CUDA environment is properly configured
- For large-scale evaluation, consider using batch processing to improve efficiency