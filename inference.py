"""
Qwen3-VL Inference Script

Load a fine-tuned LoRA adapter and run inference on images.

Usage:
    python inference.py --image path/to/image.png
    python inference.py --image path/to/image.png --prompt "Describe this image"
    python inference.py --lora-dir qwen_lora --image path/to/image.png
"""

import argparse
import torch
from unsloth import FastVisionModel
from transformers import TextStreamer
from PIL import Image


def load_model(lora_dir: str, load_in_4bit: bool = True):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=lora_dir,
        load_in_4bit=load_in_4bit,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    image_path: str,
    prompt: str = "Write the LaTeX representation for this image.",
    max_new_tokens: int = 128,
    temperature: float = 1.5,
    min_p: float = 0.1,
):
    image = Image.open(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        min_p=min_p,
    )


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write the LaTeX representation for this image.",
        help="Instruction prompt",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="qwen_lora",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--min-p", type=float, default=0.1)
    parser.add_argument(
        "--no-4bit", action="store_true", help="Load in 16bit instead of 4bit"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.lora_dir}...")
    model, tokenizer = load_model(args.lora_dir, load_in_4bit=not args.no_4bit)

    print(f"Running inference on {args.image}...")
    run_inference(
        model,
        tokenizer,
        args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        min_p=args.min_p,
    )


if __name__ == "__main__":
    main()
