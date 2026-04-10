"""Dataset snapshot: visualize distribution and save train/test splits for review."""

import csv
import json
import os
from collections import Counter
from datetime import datetime


def _extract_label(entry: dict) -> str:
    """Extract the assistant's answer text from a raw dataset entry."""
    for msg in entry["messages"]:
        if msg["role"] == "assistant":
            for item in msg["content"]:
                if item["type"] == "text":
                    return item["text"].strip()
    return "<no_label>"


def _extract_image_path(entry: dict) -> str:
    """Extract the image file reference from a raw dataset entry."""
    for msg in entry["messages"]:
        if msg["role"] == "user":
            for item in msg["content"]:
                if item["type"] == "image":
                    ref = item["image"]
                    if ref.startswith("file://"):
                        ref = ref[len("file://"):]
                    return ref
    return "<no_image>"


def visualize_distribution(raw_data: list, train_idx: list, test_idx: list):
    """Print a text-based table of label distribution for train/test splits."""
    train_labels = [_extract_label(raw_data[i]) for i in train_idx]
    test_labels = [_extract_label(raw_data[i]) for i in test_idx]

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    all_labels = sorted(set(train_counts.keys()) | set(test_counts.keys()))

    print("\n" + "=" * 60)
    print("DATASET DISTRIBUTION")
    print("=" * 60)
    print(f"{'Label':<20} {'Train':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    for label in all_labels:
        tr = train_counts.get(label, 0)
        te = test_counts.get(label, 0)
        print(f"{label:<20} {tr:>8} {te:>8} {tr + te:>8}")
    print("-" * 60)
    print(f"{'TOTAL':<20} {len(train_idx):>8} {len(test_idx):>8} {len(train_idx) + len(test_idx):>8}")
    print("=" * 60 + "\n")


def save_split_snapshot(raw_data: list, train_idx: list, test_idx: list,
                        output_base: str = "snapshots"):
    """Save train/test split details to a timestamped folder.

    Creates:
        <output_base>/<timestamp>/train.csv  - file_name, label for each train sample
        <output_base>/<timestamp>/test.csv   - file_name, label for each test sample
        <output_base>/<timestamp>/summary.json - distribution summary
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join(output_base, timestamp)
    os.makedirs(snapshot_dir, exist_ok=True)

    def _write_csv(path: str, indices: list):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "file_name", "label"])
            for idx in indices:
                entry = raw_data[idx]
                writer.writerow([idx, _extract_image_path(entry), _extract_label(entry)])

    _write_csv(os.path.join(snapshot_dir, "train.csv"), train_idx)
    _write_csv(os.path.join(snapshot_dir, "test.csv"), test_idx)

    # Save summary
    train_labels = [_extract_label(raw_data[i]) for i in train_idx]
    test_labels = [_extract_label(raw_data[i]) for i in test_idx]
    summary = {
        "timestamp": timestamp,
        "total_samples": len(raw_data),
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "train_distribution": dict(Counter(train_labels)),
        "test_distribution": dict(Counter(test_labels)),
    }
    with open(os.path.join(snapshot_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset snapshot saved to: {snapshot_dir}")
    return snapshot_dir
