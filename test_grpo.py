"""
Qwen3-VL 8B GRPO Test Script

Standalone evaluation script that loads a saved LoRA adapter and runs
evaluation on a held-out test set.

Usage:
    python test_grpo.py
    python test_grpo.py --config configs/grpo_qwen3_vl_8b.yaml --test-ratio 0.2
"""

import argparse
import json
import os

from core.config import load_config
from core.model import build_grpo_model
from core.dataset import prepare_grpo_test_dataset
from core.eval import evaluate_grpo


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL GRPO Test Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_qwen3_vl_8b.yaml",
        help="Path to GRPO config YAML",
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
    model, tokenizer = build_grpo_model(cfg)

    lora_dir = args.lora_dir or cfg["save"]["lora_dir"]
    if os.path.exists(lora_dir):
        print(f"Loading LoRA adapters from {lora_dir}/...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
    else:
        print(f"Warning: LoRA dir '{lora_dir}' not found — evaluating base model.")

    print("Preparing test dataset...")
    test_dataset = prepare_grpo_test_dataset(cfg, test_ratio)
    print(f"Test samples: {len(test_dataset)}")

    print("Running evaluation...")
    metrics = evaluate_grpo(model, tokenizer, test_dataset, cfg["inference"])

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
