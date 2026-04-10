import json
import os
import random

from PIL import Image

from .constants import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END


def resolve_image_path(image_ref: str, dataset_root: str) -> str:
    """Resolve a file:// image reference to an absolute path."""
    if image_ref.startswith("file://"):
        image_ref = image_ref[len("file://"):]
    return os.path.join(dataset_root, image_ref)


def _split_indices(total: int, test_ratio: float, seed: int):
    """Shuffle indices and return (train_indices, test_indices)."""
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    n_test = int(total * test_ratio)
    test_set = set(indices[:n_test])
    train_idx = [i for i in indices if i not in test_set]
    test_idx = indices[:n_test]
    return train_idx, test_idx


# --- SFT dataset helpers ---

def resolve_sft_entry(entry: dict, dataset_root: str) -> dict:
    """Resolve image paths in a single SFT dataset entry."""
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


def load_and_split(cfg: dict):
    """Load the raw JSON dataset and compute train/test index split.

    Returns (raw_data, train_idx, test_idx) without resolving images,
    so callers can inspect or snapshot the split before building datasets.
    """
    ds_cfg = cfg["dataset"]
    test_ratio = ds_cfg.get("test_split_ratio", 0.1)
    seed = ds_cfg.get("split_seed", 42)

    with open(ds_cfg["path"]) as f:
        raw_data = json.load(f)

    train_idx, test_idx = _split_indices(len(raw_data), test_ratio, seed)
    return raw_data, train_idx, test_idx


def prepare_sft_dataset(cfg: dict):
    """Load a local JSON dataset and split into train/test for SFT.

    Returns (train_dataset, test_dataset). If test_split_ratio is 0,
    test_dataset will be an empty list.
    """
    ds_cfg = cfg["dataset"]
    dataset_root = ds_cfg["dataset_root"]

    raw_data, train_idx, test_idx = load_and_split(cfg)
    train_dataset = [resolve_sft_entry(raw_data[i], dataset_root) for i in train_idx]
    test_dataset = [resolve_sft_entry(raw_data[i], dataset_root) for i in test_idx]
    return train_dataset, test_dataset


def prepare_sft_test_dataset(cfg: dict, test_ratio: float):
    """Load dataset and return only the SFT test split."""
    ds_cfg = cfg["dataset"]
    dataset_root = ds_cfg["dataset_root"]
    seed = ds_cfg.get("split_seed", 42)

    with open(ds_cfg["path"]) as f:
        raw_data = json.load(f)

    _, test_idx = _split_indices(len(raw_data), test_ratio, seed)
    return [resolve_sft_entry(raw_data[i], dataset_root) for i in test_idx]


# --- GRPO dataset helpers ---

def parse_grpo_entry(entry: dict, dataset_root: str, image_size: int) -> dict:
    """Parse a conversation entry into GRPO format.

    Extracts the user question, image, and assistant answer, then builds
    the GRPO prompt with reasoning instructions.
    """
    messages = entry["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assistant_msg = next(m for m in messages if m["role"] == "assistant")

    question_text = ""
    image = None
    for item in user_msg["content"]:
        if item["type"] == "text":
            question_text = item["text"]
        elif item["type"] == "image":
            abs_path = resolve_image_path(item["image"], dataset_root)
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


def prepare_grpo_dataset(cfg: dict):
    """Load a local JSON dataset and prepare it for GRPO training.

    Returns (train_dataset, test_dataset). If test_split_ratio is 0,
    test_dataset will be None.
    """
    from datasets import Dataset

    ds_cfg = cfg["dataset"]
    dataset_root = ds_cfg["dataset_root"]
    test_ratio = ds_cfg.get("test_split_ratio", 0.1)
    image_size = ds_cfg.get("image_size", 512)

    raw_data, train_idx, test_idx = load_and_split(cfg)
    train_records = [parse_grpo_entry(raw_data[i], dataset_root, image_size) for i in train_idx]
    train_dataset = Dataset.from_list(train_records)

    test_dataset = None
    if test_ratio > 0 and test_idx:
        test_records = [parse_grpo_entry(raw_data[i], dataset_root, image_size) for i in test_idx]
        test_dataset = Dataset.from_list(test_records)

    return train_dataset, test_dataset


def prepare_grpo_test_dataset(cfg: dict, test_ratio: float):
    """Load dataset and return only the GRPO test split."""
    from datasets import Dataset

    ds_cfg = cfg["dataset"]
    dataset_root = ds_cfg["dataset_root"]
    seed = ds_cfg.get("split_seed", 42)
    image_size = ds_cfg.get("image_size", 512)

    with open(ds_cfg["path"]) as f:
        raw_data = json.load(f)

    _, test_idx = _split_indices(len(raw_data), test_ratio, seed)
    test_records = [parse_grpo_entry(raw_data[i], dataset_root, image_size) for i in test_idx]
    return Dataset.from_list(test_records)
