"""
Qwen3-VL 8B Vision Fine-tuning with Unsloth

SFT training script for Qwen3-VL using LoRA adapters.
Based on: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb

Usage:
    python train.py
    python train.py --config configs/sft_qwen3_vl_8b.yaml
"""

import argparse
import json
import os

import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from PIL import Image

from core.config import load_config
from core.model import build_sft_model, save_model
from core.dataset import resolve_image_path, load_and_split, prepare_sft_dataset
from core.eval import evaluate_sft
from core.gpu import log_gpu_stats, log_training_stats
from core.snapshot import visualize_distribution, save_split_snapshot


def train(model, tokenizer, dataset, cfg: dict):
    train_cfg = cfg["training"]

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            warmup_steps=train_cfg["warmup_steps"],
            max_steps=train_cfg["max_steps"],
            learning_rate=train_cfg["learning_rate"],
            logging_steps=train_cfg["logging_steps"],
            optim=train_cfg["optim"],
            weight_decay=train_cfg["weight_decay"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            seed=train_cfg["seed"],
            output_dir=train_cfg["output_dir"],
            report_to=train_cfg["report_to"],
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=train_cfg["max_length"],
        ),
    )

    start_gpu_memory = log_gpu_stats()
    trainer_stats = trainer.train()
    log_training_stats(trainer_stats, start_gpu_memory)

    return trainer_stats


def run_inference(model, tokenizer, dataset_cfg: dict, inference_cfg: dict):
    """Run a sample inference using the first entry from the training dataset."""
    FastVisionModel.for_inference(model)

    dataset_root = dataset_cfg["dataset_root"]
    with open(dataset_cfg["path"]) as f:
        raw_data = json.load(f)

    first_entry = raw_data[0]["messages"]
    image_ref = None
    instruction = None
    for item in first_entry[0]["content"]:
        if item["type"] == "image":
            image_ref = item["image"]
        elif item["type"] == "text":
            instruction = item["text"]

    abs_path = resolve_image_path(image_ref, dataset_root)
    image = Image.open(abs_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
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
        max_new_tokens=inference_cfg["max_new_tokens"],
        use_cache=True,
        temperature=inference_cfg["temperature"],
        min_p=inference_cfg["min_p"],
    )


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_qwen3_vl_8b.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip training, run inference only (requires saved LoRA)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("Loading model...")
    model, tokenizer = build_sft_model(cfg)

    if args.inference_only:
        print("Running inference...")
        run_inference(model, tokenizer, cfg["dataset"], cfg["inference"])
        return

    print("Preparing dataset...")
    raw_data, train_idx, test_idx = load_and_split(cfg)
    visualize_distribution(raw_data, train_idx, test_idx)
    snapshot_dir = save_split_snapshot(raw_data, train_idx, test_idx)

    train_dataset, test_dataset = prepare_sft_dataset(cfg)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    print("Starting training...")
    train(model, tokenizer, train_dataset, cfg)

    print("Saving model...")
    save_model(model, tokenizer, cfg)

    print("Running test set evaluation...")
    metrics = evaluate_sft(model, tokenizer, test_dataset, cfg["inference"])

    if metrics:
        metrics_path = os.path.join(cfg["training"]["output_dir"], "eval_metrics.json")
        os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    print("Running post-training inference...")
    run_inference(model, tokenizer, cfg["dataset"], cfg["inference"])

    print("Done.")


if __name__ == "__main__":
    main()
