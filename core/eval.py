import re

import torch
from unsloth import FastVisionModel

from .constants import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END


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


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def evaluate_sft(model, tokenizer, test_dataset: list, inference_cfg: dict):
    """Run SFT evaluation on the test set.

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

        user_messages = [{"role": msg["role"], "content": msg["content"]}
                         for msg in messages if msg["role"] == "user"]

        # TODO: testing
        user_messages[0]['content']['text'] = "Does this image contain body horror defects such as distorted limbs, extra or missing body parts, backwards joints, merged or fused body parts, bad hands, bad feet, or unnatural body proportions? Answer YES/NO and reason why you choose that."

        input_text = tokenizer.apply_chat_template(
            user_messages, add_generation_prompt=True
        )

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

        prompt_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        predictions.append(generated)

        if (i + 1) % 10 == 0 or (i + 1) == len(test_dataset):
            print(f"  [{i + 1}/{len(test_dataset)}] evaluated")

    # Compute metrics
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / len(ground_truths)

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

    macro_precision = sum(c["precision"] for c in per_class.values()) / len(per_class) if per_class else 0
    macro_recall = sum(c["recall"] for c in per_class.values()) / len(per_class) if per_class else 0
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class) if per_class else 0

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


def evaluate_grpo(model, tokenizer, test_dataset, inference_cfg: dict):
    """Run GRPO evaluation on the test set.

    Generates responses, extracts answers from <SOLUTION> tags, and compares
    against ground truth. Reports exact-match accuracy, numeric-close accuracy
    (within 1% tolerance), and formatting compliance rate.
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

        has_reasoning = len(re.findall(f"{REASONING_START}(.*?){REASONING_END}", generated, re.DOTALL)) == 1
        answer_matches = re.findall(answer_pattern, generated, re.DOTALL)
        has_answer = len(answer_matches) == 1
        if has_reasoning and has_answer:
            format_correct += 1

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
