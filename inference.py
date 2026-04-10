"""
Qwen3-VL Inference Script

Load a fine-tuned LoRA adapter and run inference on images.

Usage:
    python inference.py --image path/to/image.png
    python inference.py --image-dir path/to/images/
    python inference.py --image path/to/image.png --prompt "Describe this image"
    python inference.py --lora-dir qwen_lora --image path/to/image.png
"""

import argparse

from transformers import TextStreamer
from PIL import Image

from core.model import load_inference_model
from core.inference_utils import collect_images


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


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Inference")
    parser.add_argument("--image", type=str, help="Path to a single input image")
    parser.add_argument(
        "--image-dir", type=str, help="Path to a directory of images to run inference on"
    )
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
    model, tokenizer = load_inference_model(args.lora_dir, load_in_4bit=not args.no_4bit)

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Running inference on {image_path}...")
        run_inference(
            model,
            tokenizer,
            image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
        )


if __name__ == "__main__":
    main()
