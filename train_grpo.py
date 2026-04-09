"""
Qwen3-VL 8B Vision GRPO (Reinforcement Learning) Training

GRPO training script for Qwen3-VL using LoRA adapters and reward functions.
Based on: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb

Usage:
    python train_grpo.py
    python train_grpo.py --config configs/grpo_qwen3_vl_8b.yaml
"""

import argparse
import re
import yaml
import torch
from unsloth import FastVisionModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


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


def prepare_dataset(cfg: dict):
    """Load and prepare the MathVista dataset for GRPO training."""
    ds_cfg = cfg["dataset"]
    dataset = load_dataset(ds_cfg["name"], split=ds_cfg["split"])

    # Keep only numeric answers
    dataset = dataset.filter(lambda ex: _is_numeric(ex["answer"]))

    image_size = ds_cfg.get("image_size", 512)

    # Resize and convert images
    def preprocess_image(example):
        image = example["decoded_image"]
        image = image.resize((image_size, image_size))
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["decoded_image"] = image
        return example

    dataset = dataset.map(preprocess_image)

    # Build conversational prompts
    def make_conversation(example):
        text_content = (
            f"{example['question']}. Also first provide your reasoning or working out"
            f" on how you would go about solving the question between {REASONING_START} and {REASONING_END}"
            f" and then your final answer between {SOLUTION_START} and (put a single float here) {SOLUTION_END}"
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
        return {"prompt": prompt, "image": example["decoded_image"], "answer": example["answer"]}

    dataset = dataset.map(make_conversation)

    # Fix column naming: remove original 'image', rename 'decoded_image' -> 'image'
    dataset = dataset.remove_columns("image")
    dataset = dataset.rename_column("decoded_image", "image")

    return dataset


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


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

    # Override max_steps if set in config
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

    from transformers import TextStreamer
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
    model, tokenizer = build_model(cfg)

    print("Preparing dataset...")
    dataset = prepare_dataset(cfg)
    print(f"Dataset size: {len(dataset)} samples (numeric answers only)")

    if args.inference_only:
        print("Running inference...")
        run_inference(model, tokenizer, dataset, cfg, sample_idx=args.sample_idx)
        return

    print("Starting GRPO training...")
    train(model, tokenizer, dataset, cfg)

    print("Saving model...")
    save_model(model, tokenizer, cfg)

    print("Running post-training inference...")
    run_inference(model, tokenizer, dataset, cfg, sample_idx=args.sample_idx)

    print("Done.")


if __name__ == "__main__":
    main()
