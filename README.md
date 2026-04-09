# Qwen3-VL Fine-Tuning with Unsloth

Fine-tune Qwen3-VL 8B vision model using Unsloth with QLoRA (4-bit) adapters for body defect detection in AI-generated images.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (24GB+ VRAM recommended, 48GB comfortable)
- CUDA toolkit installed
- Training dataset at `/paperclip/defect_training_dataset/`

## Step 1: Set Up Environment

```bash
cd unsloth-qwen-fine-tune

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Step 2: Verify Dataset

The training dataset should be at `/paperclip/defect_training_dataset/` with this structure:

```
/paperclip/defect_training_dataset/
├── share_binary_unsloth.json    # 3,627 samples in Unsloth format
├── images/                       # Image files referenced by the JSON
│   ├── image_0001.jpg
│   └── ...
```

The dataset contains binary defect classification (YES/NO) for body defects like distorted limbs, bad hands/feet, extra/missing parts, etc.

## Step 3: Review the Training Config

The config is at `configs/sft_qwen3_vl_8b.yaml`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| Model | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` | 4-bit quantized Qwen3-VL 8B |
| Dataset | `/paperclip/defect_training_dataset/share_binary_unsloth.json` | Local JSON in Unsloth format |
| LoRA rank (r) | 16 | Higher = more capacity, risk of overfitting |
| Learning rate | 2e-4 | AdamW 8-bit optimizer |
| Batch size | 2 | Per-device batch size |
| Gradient accumulation | 4 | Effective batch size = 8 |
| Max steps | 30 | Set to `null` and use `num_train_epochs: 1` for full training |
| Max sequence length | 2048 | Increase for longer outputs |

Edit the YAML to change any parameter before training.

## Step 4: Train the Model

```bash
python train.py
```

Or with a custom config:

```bash
python train.py --config configs/sft_qwen3_vl_8b.yaml
```

This will:
1. Download the Qwen3-VL 8B model (4-bit quantized)
2. Apply LoRA adapters to both vision and language layers
3. Load the local defect detection dataset (3,627 labeled images)
4. Resolve `file://images/...` references to absolute paths and load images
5. Run SFT training
6. Save LoRA adapters to `qwen_lora/`
7. Run a sample inference to verify the fine-tune

Training output and checkpoints are saved to `outputs/`.

## Step 5: Run Inference

After training, run inference on any image:

```bash
python inference.py --image path/to/image.png
```

For defect detection with the trained model:

```bash
python inference.py \
    --image path/to/image.png \
    --prompt "Does this image contain body horror defects such as distorted limbs, extra or missing body parts, backwards joints, merged or fused body parts, bad hands, bad feet, or unnatural body proportions? Answer YES or NO."
```

### Inference Options

```
--image          Path to input image (required)
--prompt         Instruction prompt (default: "Write the LaTeX representation for this image.")
--lora-dir       Path to LoRA adapter directory (default: qwen_lora)
--max-new-tokens Maximum tokens to generate (default: 128)
--temperature    Sampling temperature (default: 1.5)
--min-p          Min-p sampling threshold (default: 0.1)
--no-4bit        Load model in 16-bit instead of 4-bit
```

## Step 6: Export the Model (Optional)

To export to GGUF for use with llama.cpp or Ollama, run interactively:

```python
# Save to q8_0 GGUF
model.save_pretrained_gguf("qwen_finetune", tokenizer)

# Save to q4_k_m GGUF
model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method="q4_k_m")
```

To push to HuggingFace Hub:

```python
model.push_to_hub("your-username/qwen3-vl-finetune", token="YOUR_HF_TOKEN")
tokenizer.push_to_hub("your-username/qwen3-vl-finetune", token="YOUR_HF_TOKEN")
```

## Project Structure

```
.
├── configs/
│   └── sft_qwen3_vl_8b.yaml   # Training hyperparameters + dataset path
├── train.py                     # Main training script
├── inference.py                 # Standalone inference script
├── requirements.txt             # Python dependencies
└── README.md
```

## Using a Custom Dataset

The training script expects a local JSON file in Unsloth conversation format:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "file://images/example.jpg"},
          {"type": "text", "text": "Your prompt here"}
        ]
      },
      {
        "role": "assistant",
        "content": [
          {"type": "text", "text": "Expected answer"}
        ]
      }
    ]
  }
]
```

Update the config dataset section:

```yaml
dataset:
  path: "/path/to/your/dataset.json"
  dataset_root: "/path/to/your/dataset_folder"
```

Image paths in the JSON (`file://images/...`) are resolved relative to `dataset_root`.

## Troubleshooting

- **Out of memory**: Reduce `per_device_train_batch_size` to 1 or lower `max_length`
- **Slow download**: Install `hf_transfer` for faster HuggingFace downloads: `pip install hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1`
- **Multi-GPU**: Unsloth supports single-GPU training. For multi-GPU, see [Unsloth docs](https://unsloth.ai/docs)
- **Image not found**: Ensure `dataset_root` in config points to the directory containing the `images/` folder
