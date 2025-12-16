#!/usr/bin/env python3
"""
HSA-DPO Inference Example
This script demonstrates how to load and use the HSA-DPO model for inference.
"""

import os
import sys
import argparse
from io import BytesIO

# Add the LLaVA module to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hsa_dpo/models/llava-v1_5'))

import torch
from PIL import Image
import requests

def load_model(model_base, lora_path):
    """Load HSA-DPO model with LoRA weights"""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=lora_path,
        model_base=model_base,
        model_name=get_model_name_from_path(lora_path)
    )
    
    return tokenizer, model, image_processor, context_len

def load_image(image_path):
    """Load image from file or URL"""
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image

def run_inference(model, tokenizer, image_processor, image_path, prompt, args):
    """Run inference on a single image with prompt"""
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    
    # Load and process image
    image = load_image(image_path)
    image_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    # Prepare conversation
    if model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
    else:
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()
    
    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def main():
    parser = argparse.ArgumentParser(description="HSA-DPO Inference")
    parser.add_argument("--model-base", type=str, required=True,
                        help="Path to base LLaVA model")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="Path to HSA-DPO LoRA weights")
    parser.add_argument("--image", type=str, required=True,
                        help="Path or URL to image")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for the model")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p sampling parameter")
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device to use")
    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # Load model
    print("Loading model...")
    tokenizer, model, image_processor, context_len = load_model(
        args.model_base, args.lora_path
    )
    
    # Run inference
    print("Running inference...")
    output = run_inference(
        model, tokenizer, image_processor,
        args.image, args.prompt, args
    )
    
    print("\nModel Output:")
    print(output)

if __name__ == "__main__":
    main()