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
import yaml
import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from PIL import Image


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg["model"]["name"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        use_gradient_checkpointing=cfg["model"]["use_gradient_checkpointing"],
    )

    lora_cfg = cfg["lora"]
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora_cfg["finetune_vision_layers"],
        finetune_language_layers=lora_cfg["finetune_language_layers"],
        finetune_attention_modules=lora_cfg["finetune_attention_modules"],
        finetune_mlp_modules=lora_cfg["finetune_mlp_modules"],
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        random_state=lora_cfg["random_state"],
        use_rslora=lora_cfg["use_rslora"],
        loftq_config=None,
    )
    return model, tokenizer


def resolve_image_path(image_ref: str, dataset_root: str) -> str:
    """Resolve a file:// image reference to an absolute path."""
    if image_ref.startswith("file://"):
        image_ref = image_ref[len("file://"):]
    return os.path.join(dataset_root, image_ref)


def prepare_dataset(cfg: dict):
    """Load a local JSON dataset in Unsloth conversation format.

    Each entry has {"messages": [{"role": "user", "content": [...]}, ...]}.
    Image references like file://images/foo.jpg are resolved relative to dataset_root
    and loaded as PIL Images for Unsloth's data collator.
    """
    ds_cfg = cfg["dataset"]
    dataset_path = ds_cfg["path"]
    dataset_root = ds_cfg["dataset_root"]

    with open(dataset_path) as f:
        raw_data = json.load(f)

    dataset = []
    for entry in raw_data:
        messages = entry["messages"]
        resolved_messages = []
        for msg in messages:
            new_msg = {"role": msg["role"], "content": []}
            for item in msg["content"]:
                if item["type"] == "image":
                    abs_path = resolve_image_path(item["image"], dataset_root)
                    new_msg["content"].append({
                        "type": "image",
                        "image": Image.open(abs_path).convert("RGB"),
                    })
                else:
                    new_msg["content"].append(item)
            resolved_messages.append(new_msg)
        dataset.append({"messages": resolved_messages})

    return dataset


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

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']:.1f}s training time.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")

    return trainer_stats


def save_model(model, tokenizer, cfg: dict):
    save_cfg = cfg["save"]
    lora_dir = save_cfg["lora_dir"]
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"LoRA adapters saved to {lora_dir}/")


def run_inference(model, tokenizer, dataset_cfg: dict, inference_cfg: dict):
    """Run a sample inference using the first entry from the training dataset."""
    FastVisionModel.for_inference(model)

    dataset_root = dataset_cfg["dataset_root"]
    with open(dataset_cfg["path"]) as f:
        raw_data = json.load(f)

    # Extract the first sample's image and prompt
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

    from transformers import TextStreamer

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
    model, tokenizer = build_model(cfg)

    if args.inference_only:
        print("Running inference...")
        run_inference(model, tokenizer, cfg["dataset"], cfg["inference"])
        return

    print("Preparing dataset...")
    dataset = prepare_dataset(cfg)
    print(f"Dataset size: {len(dataset)} samples")

    print("Starting training...")
    train(model, tokenizer, dataset, cfg)

    print("Saving model...")
    save_model(model, tokenizer, cfg)

    print("Running post-training inference...")
    run_inference(model, tokenizer, cfg["dataset"], cfg["inference"])

    print("Done.")


if __name__ == "__main__":
    main()
