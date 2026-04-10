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
import random
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


def _resolve_entry(entry: dict, dataset_root: str) -> dict:
    """Resolve image paths in a single dataset entry."""
    resolved_messages = []
    for msg in entry["messages"]:
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
    return {"messages": resolved_messages}


def prepare_dataset(cfg: dict):
    """Load a local JSON dataset and split into train/test sets.

    Each entry has {"messages": [{"role": "user", "content": [...]}, ...]}.
    Image references like file://images/foo.jpg are resolved relative to dataset_root
    and loaded as PIL Images for Unsloth's data collator.

    Returns (train_dataset, test_dataset). If test_split_ratio is 0 or not set,
    test_dataset will be an empty list.
    """
    ds_cfg = cfg["dataset"]
    dataset_path = ds_cfg["path"]
    dataset_root = ds_cfg["dataset_root"]
    test_split_ratio = ds_cfg.get("test_split_ratio", 0.1)
    split_seed = ds_cfg.get("split_seed", 42)

    with open(dataset_path) as f:
        raw_data = json.load(f)

    # Shuffle and split
    indices = list(range(len(raw_data)))
    random.Random(split_seed).shuffle(indices)
    n_test = int(len(raw_data) * test_split_ratio)

    test_indices = set(indices[:n_test])
    train_entries = [raw_data[i] for i in indices if i not in test_indices]
    test_entries = [raw_data[i] for i in indices if i in test_indices]

    train_dataset = [_resolve_entry(e, dataset_root) for e in train_entries]
    test_dataset = [_resolve_entry(e, dataset_root) for e in test_entries]

    return train_dataset, test_dataset


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


def extract_assistant_text(messages: list) -> str:
    """Extract the text content from the assistant message."""
    for msg in messages:
        if msg["role"] == "assistant":
            for item in msg["content"]:
                if item.get("type") == "text":
                    return item["text"].strip()
                if isinstance(item, str):
                    return item.strip()
    return ""


def evaluate(model, tokenizer, test_dataset: list, inference_cfg: dict):
    """Run evaluation on the test set and print metrics.

    Compares model predictions against ground-truth assistant responses.
    Reports accuracy and per-class precision/recall/F1 for classification tasks.
    """
    if not test_dataset:
        print("No test samples — skipping evaluation.")
        return {}

    FastVisionModel.for_inference(model)

    predictions = []
    ground_truths = []

    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    for i, sample in enumerate(test_dataset):
        messages = sample["messages"]
        expected = extract_assistant_text(messages)
        ground_truths.append(expected)

        # Build input from user message only
        user_messages = [{"role": msg["role"], "content": msg["content"]}
                         for msg in messages if msg["role"] == "user"]

        input_text = tokenizer.apply_chat_template(
            user_messages, add_generation_prompt=True
        )

        # Collect images from user content
        images = []
        for msg in messages:
            if msg["role"] == "user":
                for item in msg["content"]:
                    if item.get("type") == "image" and "image" in item:
                        images.append(item["image"])

        image = images[0] if images else None
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=inference_cfg["max_new_tokens"],
                use_cache=True,
                temperature=1.0,
                do_sample=False,
            )

        # Decode only generated tokens (skip prompt)
        prompt_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        predictions.append(generated)

        if (i + 1) % 10 == 0 or (i + 1) == len(test_dataset):
            print(f"  [{i + 1}/{len(test_dataset)}] evaluated")

    # Compute metrics
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / len(ground_truths)

    # Per-class stats for precision/recall/F1
    labels = sorted(set(ground_truths))
    per_class = {}
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": tp + fn}

    # Macro averages
    macro_precision = sum(c["precision"] for c in per_class.values()) / len(per_class) if per_class else 0
    macro_recall = sum(c["recall"] for c in per_class.values()) / len(per_class) if per_class else 0
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class) if per_class else 0

    # Print report
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Test samples:  {len(ground_truths)}")
    print(f"Accuracy:      {accuracy:.4f} ({correct}/{len(ground_truths)})")
    print(f"\n{'Class':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print("-" * 60)
    for label in labels:
        c = per_class[label]
        display_label = label if len(label) <= 28 else label[:25] + "..."
        print(f"{display_label:<30} {c['precision']:>6.4f} {c['recall']:>6.4f} {c['f1']:>6.4f} {c['support']:>8}")
    print("-" * 60)
    total_support = sum(c["support"] for c in per_class.values())
    print(f"{'Macro avg':<30} {macro_precision:>6.4f} {macro_recall:>6.4f} {macro_f1:>6.4f} {total_support:>8}")
    print("=" * 60)

    # Print misclassified samples
    misclassified = [(i, p, g) for i, (p, g) in enumerate(zip(predictions, ground_truths)) if p != g]
    if misclassified:
        print(f"\nMisclassified samples ({len(misclassified)}):")
        for idx, pred, truth in misclassified[:20]:
            print(f"  Sample {idx}: predicted='{pred}', expected='{truth}'")
        if len(misclassified) > 20:
            print(f"  ... and {len(misclassified) - 20} more")

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "total_test_samples": len(ground_truths),
    }
    return metrics


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
    train_dataset, test_dataset = prepare_dataset(cfg)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    print("Starting training...")
    train(model, tokenizer, train_dataset, cfg)

    print("Saving model...")
    save_model(model, tokenizer, cfg)

    print("Running test set evaluation...")
    metrics = evaluate(model, tokenizer, test_dataset, cfg["inference"])

    # Save metrics to file
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
