"""
Qwen3-VL 8B GRPO Test Script

Standalone evaluation script that mirrors the evaluate function in train_grpo.py.
Loads a saved LoRA adapter and runs evaluation on a held-out test set.

Usage:
    python test_grpo.py
    python test_grpo.py --config configs/grpo_qwen3_vl_8b.yaml --test-ratio 0.2
"""

import argparse
import json
import os
import random
import re
import yaml
import torch
from PIL import Image
from unsloth import FastVisionModel
from datasets import Dataset


# Delimiter tokens for structured output
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    model_cfg = cfg["model"]
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        fast_inference=model_cfg["fast_inference"],
        gpu_memory_utilization=model_cfg["gpu_memory_utilization"],
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
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def _resolve_image(image_ref: str, dataset_root: str) -> str:
    """Resolve a file:// image reference to an absolute path."""
    if image_ref.startswith("file://"):
        image_ref = image_ref[len("file://"):]
    return os.path.join(dataset_root, image_ref)


def _parse_entry(entry: dict, dataset_root: str, image_size: int) -> dict:
    """Parse an Unsloth conversation entry into GRPO format."""
    messages = entry["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assistant_msg = next(m for m in messages if m["role"] == "assistant")

    question_text = ""
    image = None
    for item in user_msg["content"]:
        if item["type"] == "text":
            question_text = item["text"]
        elif item["type"] == "image":
            abs_path = _resolve_image(item["image"], dataset_root)
            image = Image.open(abs_path).convert("RGB").resize((image_size, image_size))

    answer_text = ""
    for item in assistant_msg["content"]:
        if item["type"] == "text":
            answer_text = item["text"]

    text_content = (
        f"{question_text}. Also first provide your reasoning or working out"
        f" on how you would go about solving the question between {REASONING_START} and {REASONING_END}"
        f" and then your final answer between {SOLUTION_START} and {SOLUTION_END}"
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_content},
            ],
        },
    ]

    return {"prompt": prompt, "image": image, "answer": answer_text}


def prepare_test_dataset(cfg: dict, test_ratio: float):
    """Load dataset and return only the test split.

    Uses the same seed and splitting logic as train_grpo.py so the test set
    is identical for a given ratio and seed.
    """
    ds_cfg = cfg["dataset"]
    dataset_path = ds_cfg["path"]
    dataset_root = ds_cfg["dataset_root"]
    split_seed = ds_cfg.get("split_seed", 42)
    image_size = ds_cfg.get("image_size", 512)

    with open(dataset_path) as f:
        raw_data = json.load(f)

    indices = list(range(len(raw_data)))
    random.Random(split_seed).shuffle(indices)
    n_test = int(len(raw_data) * test_ratio)

    test_indices = indices[:n_test]
    test_entries = [raw_data[i] for i in test_indices]

    test_records = [_parse_entry(e, dataset_root, image_size) for e in test_entries]
    test_dataset = Dataset.from_list(test_records)
    return test_dataset


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def evaluate(model, tokenizer, test_dataset, inference_cfg: dict):
    """Run evaluation on the test set and print metrics.

    Generates responses for each test sample, extracts the answer from
    <SOLUTION> tags, and compares against the ground truth.
    Reports exact-match accuracy, numeric-close accuracy (within 1% tolerance),
    and formatting compliance rate.
    """
    if test_dataset is None or len(test_dataset) == 0:
        print("No test samples — skipping evaluation.")
        return {}

    FastVisionModel.for_inference(model)

    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"

    exact_matches = 0
    close_matches = 0
    format_correct = 0
    total = len(test_dataset)
    results = []

    print(f"\nEvaluating on {total} test samples...")
    for i in range(total):
        image = test_dataset[i]["image"]
        prompt = test_dataset[i]["prompt"]
        expected = test_dataset[i]["answer"]

        inputs = tokenizer(
            image,
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=inference_cfg["max_new_tokens"],
                use_cache=True,
                temperature=1.0,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

        # Check formatting
        has_reasoning = len(re.findall(f"{REASONING_START}(.*?){REASONING_END}", generated, re.DOTALL)) == 1
        answer_matches = re.findall(answer_pattern, generated, re.DOTALL)
        has_answer = len(answer_matches) == 1
        if has_reasoning and has_answer:
            format_correct += 1

        # Check correctness
        extracted = answer_matches[0].replace("\n", "").strip() if has_answer else ""
        is_exact = extracted == expected

        is_close = False
        if _is_numeric(extracted) and _is_numeric(expected):
            pred_val = float(extracted)
            exp_val = float(expected)
            if exp_val != 0:
                is_close = abs(pred_val - exp_val) / abs(exp_val) <= 0.01
            else:
                is_close = abs(pred_val) <= 0.01

        if is_exact:
            exact_matches += 1
        if is_exact or is_close:
            close_matches += 1

        results.append({
            "index": i,
            "expected": expected,
            "extracted": extracted,
            "exact_match": is_exact,
            "close_match": is_exact or is_close,
            "format_ok": has_reasoning and has_answer,
        })

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i + 1}/{total}] evaluated")

    exact_acc = exact_matches / total
    close_acc = close_matches / total
    format_rate = format_correct / total

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Test samples:          {total}")
    print(f"Exact-match accuracy:  {exact_acc:.4f} ({exact_matches}/{total})")
    print(f"Close-match accuracy:  {close_acc:.4f} ({close_matches}/{total})  (within 1% tolerance)")
    print(f"Format compliance:     {format_rate:.4f} ({format_correct}/{total})")
    print("=" * 60)

    # Show some misses
    misses = [r for r in results if not r["exact_match"]]
    if misses:
        print(f"\nIncorrect predictions ({len(misses)}):")
        for r in misses[:20]:
            print(f"  Sample {r['index']}: predicted='{r['extracted']}', expected='{r['expected']}', format_ok={r['format_ok']}")
        if len(misses) > 20:
            print(f"  ... and {len(misses) - 20} more")

    metrics = {
        "exact_match_accuracy": exact_acc,
        "close_match_accuracy": close_acc,
        "format_compliance": format_rate,
        "total_test_samples": total,
        "exact_matches": exact_matches,
        "close_matches": close_matches,
        "format_correct": format_correct,
    }
    return metrics


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
