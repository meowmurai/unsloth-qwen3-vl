from unsloth import FastVisionModel


def build_sft_model(cfg: dict):
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


def build_grpo_model(cfg: dict):
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


def save_model(model, tokenizer, cfg: dict):
    lora_dir = cfg["save"]["lora_dir"]
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"LoRA adapters saved to {lora_dir}/")


def load_inference_model(lora_dir: str, load_in_4bit: bool = True):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=lora_dir,
        load_in_4bit=load_in_4bit,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer
