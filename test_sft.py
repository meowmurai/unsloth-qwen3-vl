"""
Qwen3-VL 8B SFT Test Script

Standalone evaluation script that mirrors the evaluate function in train.py.
Loads a saved LoRA adapter and runs evaluation on a held-out test set.

Usage:
    python test_sft.py
    python test_sft.py --config configs/sft_qwen3_vl_8b.yaml --test-ratio 0.2
"""

import argparse
import json
import os
import random
import yaml
import torch
from unsloth import FastVisionModel
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


def prepare_test_dataset(cfg: dict, test_ratio: float):
    """Load dataset and return only the test split.

    Uses the same seed and splitting logic as train.py so the test set
    is identical for a given ratio and seed.
    """
    ds_cfg = cfg["dataset"]
    dataset_path = ds_cfg["path"]
    dataset_root = ds_cfg["dataset_root"]
    split_seed = ds_cfg.get("split_seed", 42)

    with open(dataset_path) as f:
        raw_data = json.load(f)

    indices = list(range(len(raw_data)))
    random.Random(split_seed).shuffle(indices)
    n_test = int(len(raw_data) * test_ratio)

    test_indices = indices[:n_test]
    test_entries = [raw_data[i] for i in test_indices]

    test_dataset = [_resolve_entry(e, dataset_root) for e in test_entries]
    return test_dataset


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


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL SFT Test Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_qwen3_vl_8b.yaml",
        help="Path to SFT config YAML",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Test split ratio (overrides config value, default: use config's test_split_ratio)",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default=None,
        help="Path to saved LoRA adapters (overrides config value)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    test_ratio = args.test_ratio if args.test_ratio is not None else cfg["dataset"].get("test_split_ratio", 0.1)
    split_seed = cfg["dataset"].get("split_seed", 42)

    print(f"Test ratio: {test_ratio}, Split seed: {split_seed}")

    print("Loading model...")
    model, tokenizer = build_model(cfg)

    # Load LoRA adapters if available
    lora_dir = args.lora_dir or cfg["save"]["lora_dir"]
    if os.path.exists(lora_dir):
        print(f"Loading LoRA adapters from {lora_dir}/...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
    else:
        print(f"Warning: LoRA dir '{lora_dir}' not found — evaluating base model.")

    print("Preparing test dataset...")
    test_dataset = prepare_test_dataset(cfg, test_ratio)
    print(f"Test samples: {len(test_dataset)}")

    print("Running evaluation...")
    metrics = evaluate(model, tokenizer, test_dataset, cfg["inference"])

    # Save metrics
    if metrics:
        output_dir = cfg["training"]["output_dir"]
        metrics_path = os.path.join(output_dir, "test_eval_metrics.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")

    print("Done.")


if __name__ == "__main__":
    main()
