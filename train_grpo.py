"""
Qwen3-VL 8B Vision GRPO (Reinforcement Learning) Training

GRPO training script for Qwen3-VL using LoRA adapters and reward functions.
Based on: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb

Usage:
    python train_grpo.py
    python train_grpo.py --config configs/grpo_qwen3_vl_8b.yaml
"""

import argparse
import json
import os
import re

from unsloth import FastVisionModel
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer

from core.config import load_config
from core.constants import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END
from core.model import build_grpo_model, save_model
from core.dataset import prepare_grpo_dataset
from core.eval import evaluate_grpo
from core.gpu import log_gpu_stats, log_training_stats


# --- Reward functions ---

def formatting_reward_func(completions, **kwargs):
    """Reward for correct use of reasoning/solution delimiters."""
    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"

    scores = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""
        score = 0.0
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        if len(thinking_matches) == 1:
            score += 1.0
        if len(answer_matches) == 1:
            score += 1.0

        # Penalize addCriterion gibberish (known Qwen VL issue)
        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0

        scores.append(score)
    return scores


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward for matching the correct numeric answer."""
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"

    completions = [
        (c[0]["content"] if c else "") if isinstance(c, list) else c
        for c in completions
    ]
    responses = [re.findall(answer_pattern, c, re.DOTALL) for c in completions]

    q = prompts[0]
    print("-" * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:{completions[0]}")

    return [
        2.0 if len(r) == 1 and a == r[0].replace("\n", "") else 0.0
        for r, a in zip(responses, answer)
    ]


def train(model, tokenizer, dataset, cfg: dict):
    train_cfg = cfg["training"]

    training_args = GRPOConfig(
        learning_rate=train_cfg["learning_rate"],
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optim"],
        logging_steps=train_cfg["logging_steps"],
        log_completions=train_cfg["log_completions"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_generations=train_cfg["num_generations"],
        max_prompt_length=train_cfg["max_prompt_length"],
        max_completion_length=train_cfg["max_completion_length"],
        num_train_epochs=train_cfg["num_train_epochs"],
        save_steps=train_cfg["save_steps"],
        max_grad_norm=train_cfg["max_grad_norm"],
        report_to=train_cfg["report_to"],
        output_dir=train_cfg["output_dir"],
        importance_sampling_level=train_cfg["importance_sampling_level"],
        mask_truncated_completions=train_cfg["mask_truncated_completions"],
        loss_type=train_cfg["loss_type"],
    )

    if train_cfg.get("max_steps"):
        training_args.max_steps = train_cfg["max_steps"]

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[
            formatting_reward_func,
            correctness_reward_func,
        ],
        train_dataset=dataset,
    )

    start_gpu_memory = log_gpu_stats()
    trainer_stats = trainer.train()
    log_training_stats(trainer_stats, start_gpu_memory)

    return trainer_stats


def run_inference(model, tokenizer, dataset, cfg: dict, sample_idx: int = 0):
    """Run a sample inference on a dataset entry."""
    FastVisionModel.for_inference(model)

    image = dataset[sample_idx]["image"]
    prompt = dataset[sample_idx]["prompt"]

    inputs = tokenizer(
        image,
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=cfg["inference"]["max_new_tokens"],
        use_cache=True,
        temperature=cfg["inference"]["temperature"],
        min_p=cfg["inference"]["min_p"],
    )


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL GRPO Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_qwen3_vl_8b.yaml",
        help="Path to GRPO training config YAML",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip training, run inference only (requires saved LoRA)",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Dataset sample index for inference demo",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("Loading model...")
    model, tokenizer = build_grpo_model(cfg)

    print("Preparing dataset...")
    train_dataset, test_dataset = prepare_grpo_dataset(cfg)
    test_count = len(test_dataset) if test_dataset is not None else 0
    print(f"Train samples: {len(train_dataset)}, Test samples: {test_count} (numeric answers only)")

    if args.inference_only:
        print("Running inference...")
        run_inference(model, tokenizer, train_dataset, cfg, sample_idx=args.sample_idx)
        return

    print("Starting GRPO training...")
    train(model, tokenizer, train_dataset, cfg)

    print("Saving model...")
    save_model(model, tokenizer, cfg)

    print("Running test set evaluation...")
    metrics = evaluate_grpo(model, tokenizer, test_dataset, cfg["inference"])

    if metrics:
        metrics_path = os.path.join(cfg["training"]["output_dir"], "eval_metrics.json")
        os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    print("Running post-training inference...")
    run_inference(model, tokenizer, train_dataset, cfg, sample_idx=args.sample_idx)

    print("Done.")


if __name__ == "__main__":
    main()
