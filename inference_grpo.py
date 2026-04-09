"""
Qwen3-VL GRPO Inference Script

Load a GRPO-trained LoRA adapter and run inference on images with
structured reasoning/solution output format.

Usage:
    python inference_grpo.py --image path/to/image.png
    python inference_grpo.py --image-dir path/to/images/
    python inference_grpo.py --image path/to/image.png --question "What is the value of x?"
    python inference_grpo.py --lora-dir grpo_lora --image path/to/image.png
"""

import argparse
import os
import torch
from unsloth import FastVisionModel
from transformers import TextStreamer
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Must match the delimiters used during GRPO training
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

DEFAULT_QUESTION = (
    "What is shown in this image? Also first provide your reasoning or working out"
    f" on how you would go about solving the question between {REASONING_START} and {REASONING_END}"
    f" and then your final answer between {SOLUTION_START} and (put a single float here) {SOLUTION_END}"
)


def load_model(lora_dir: str, load_in_4bit: bool = True):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=lora_dir,
        load_in_4bit=load_in_4bit,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def build_prompt(question: str) -> list[dict]:
    """Build the chat prompt with GRPO reasoning/solution delimiters."""
    text = (
        f"{question}. Also first provide your reasoning or working out"
        f" on how you would go about solving the question between {REASONING_START} and {REASONING_END}"
        f" and then your final answer between {SOLUTION_START} and (put a single float here) {SOLUTION_END}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]


def run_inference(
    model,
    tokenizer,
    image_path: str,
    question: str = "What is shown in this image?",
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    min_p: float = 0.1,
):
    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(question)

    input_text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
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


def collect_images(image_dir: str) -> list[str]:
    paths = []
    for entry in sorted(os.listdir(image_dir)):
        if os.path.splitext(entry)[1].lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(image_dir, entry))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL GRPO Inference")
    parser.add_argument("--image", type=str, help="Path to a single input image")
    parser.add_argument(
        "--image-dir", type=str, help="Path to a directory of images"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is shown in this image?",
        help="Question to ask about the image (reasoning/solution delimiters are added automatically)",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="grpo_lora",
        help="Path to GRPO LoRA adapter directory",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.1)
    parser.add_argument(
        "--no-4bit", action="store_true", help="Load in 16bit instead of 4bit"
    )
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir is required")

    if args.image and args.image_dir:
        parser.error("Use --image or --image-dir, not both")

    image_paths = []
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = collect_images(args.image_dir)
        if not image_paths:
            print(f"No images found in {args.image_dir}")
            return
        print(f"Found {len(image_paths)} images in {args.image_dir}")

    print(f"Loading model from {args.lora_dir}...")
    model, tokenizer = load_model(args.lora_dir, load_in_4bit=not args.no_4bit)

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Running inference on {image_path}...")
        run_inference(
            model,
            tokenizer,
            image_path,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
        )


if __name__ == "__main__":
    main()
