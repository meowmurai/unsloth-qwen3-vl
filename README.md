# Qwen3-VL Fine-Tuning with Unsloth

Fine-tune Qwen3-VL 8B vision model using Unsloth with QLoRA (4-bit) adapters. Trains the model on image-to-LaTeX OCR using the Unsloth LaTeX_OCR dataset.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (16GB+ VRAM recommended, e.g. T4 or better)
- CUDA toolkit installed

## Step 1: Clone and Set Up Environment

```bash
git clone <this-repo-url>
cd unsloth-qwen-fine-tune

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Step 2: Review the Training Config

The default config is at `configs/sft_qwen3_vl_8b.yaml`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| Model | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` | 4-bit quantized Qwen3-VL 8B |
| LoRA rank (r) | 16 | Higher = more capacity, risk of overfitting |
| Learning rate | 2e-4 | AdamW 8-bit optimizer |
| Batch size | 2 | Per-device batch size |
| Gradient accumulation | 4 | Effective batch size = 8 |
| Max steps | 30 | Set to `null` and use `num_train_epochs: 1` for full training |
| Max sequence length | 2048 | Increase for longer outputs |

Edit the YAML to change any parameter before training.

## Step 3: Train the Model

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
3. Download and prepare the LaTeX OCR dataset
4. Run SFT training
5. Save LoRA adapters to `qwen_lora/`
6. Run a sample inference to verify the fine-tune

Training output and checkpoints are saved to `outputs/`.

## Step 4: Run Inference

After training, run inference on any image:

```bash
python inference.py --image path/to/image.png
```

With a custom prompt:

```bash
python inference.py --image path/to/image.png --prompt "Describe this image in detail"
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

## Step 5: Export the Model (Optional)

To export to GGUF for use with llama.cpp or Ollama, add the following to your training script or run interactively:

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
│   └── sft_qwen3_vl_8b.yaml   # Training hyperparameters
├── train.py                     # Main training script
├── inference.py                 # Standalone inference script
├── requirements.txt             # Python dependencies
└── README.md
```

## Using a Custom Dataset

Edit `configs/sft_qwen3_vl_8b.yaml` and change the dataset section:

```yaml
dataset:
  name: "your-org/your-dataset"
  split: "train"
  instruction: "Your task instruction here"
```

The dataset must have `image` and `text` columns. For multi-image training, modify `train.py` to use list comprehension instead of `dataset.map()` (already the default).

## Troubleshooting

- **Out of memory**: Reduce `per_device_train_batch_size` to 1 or lower `max_length`
- **Slow download**: Install `hf_transfer` for faster HuggingFace downloads: `pip install hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1`
- **Multi-GPU**: Unsloth supports single-GPU training. For multi-GPU, see [Unsloth docs](https://unsloth.ai/docs)
