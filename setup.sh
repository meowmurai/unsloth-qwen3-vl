#!/usr/bin/env bash
# Set up environment on a new machine to run the fine-tuned model.
#
# Usage:
#   ./setup.sh              # creates venv and installs deps
#   ./setup.sh --no-venv    # install into current Python environment
set -euo pipefail

USE_VENV=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-venv) USE_VENV=false; shift ;;
    -h|--help)
      echo "Usage: $0 [--no-venv]"
      echo "  --no-venv  Skip virtual environment creation"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check prerequisites
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  echo "Error: Python 3.10+ is required but not found."
  exit 1
fi

PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PY_VERSION at $PYTHON"

if ! $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Warning: CUDA not detected. GPU is required for inference."
fi

# Set up virtual environment
if $USE_VENV; then
  if [[ ! -d "venv" ]]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
  fi
  source venv/bin/activate
  echo "Activated venv at $SCRIPT_DIR/venv"
fi

# Install dependencies
echo ""
echo "=== [1/5] Upgrading pip ==="
pip install --upgrade pip

# Detect CUDA driver version and choose the right PyTorch index
CUDA_INDEX=""
if command -v nvidia-smi &>/dev/null; then
  DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  # Get CUDA version from nvidia-smi header output
  CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
  if [[ -n "$CUDA_VER" ]]; then
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    echo ""
    echo "Detected CUDA driver version: $CUDA_VER (driver: $DRIVER_CUDA)"
    if [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -lt 4 ]]; then
      echo "CUDA $CUDA_VER detected — installing PyTorch with cu121 support for compatibility."
      CUDA_INDEX="--index-url https://download.pytorch.org/whl/cu121"
    elif [[ "$CUDA_MAJOR" -lt 12 ]]; then
      echo "CUDA $CUDA_VER detected — installing PyTorch with cu118 support for compatibility."
      CUDA_INDEX="--index-url https://download.pytorch.org/whl/cu118"
    fi
  fi
else
  echo "Warning: nvidia-smi not found. Installing default (CPU) PyTorch."
  echo "GPU training requires NVIDIA drivers. See: https://www.nvidia.com/Download/index.aspx"
fi

# Install PyTorch first with the correct CUDA index, then remaining deps
if [[ -n "$CUDA_INDEX" ]]; then
  echo ""
  echo "=== [2/5] Installing PyTorch (from: $CUDA_INDEX) ==="
  echo "This may take several minutes for the first download (~2GB)..."
  pip install --progress-bar on $CUDA_INDEX torch
else
  echo ""
  echo "=== [2/5] Installing PyTorch (default index) ==="
  echo "This may take several minutes for the first download (~2GB)..."
fi

echo ""
echo "=== [3/5] Installing remaining dependencies ==="
pip install --progress-bar on -r requirements.txt

echo ""
echo "=== [4/5] Installing Unsloth (after TRL for compatibility) ==="
echo "Installing unsloth and unsloth_zoo compatible with installed TRL version..."
pip install --progress-bar on unsloth

echo ""
echo "=== [5/5] Verifying installation ==="

# Verify CUDA is working with installed PyTorch
if command -v nvidia-smi &>/dev/null; then
  if $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    TORCH_CUDA=$($PYTHON -c "import torch; print(torch.version.cuda)")
    echo "PyTorch CUDA verification passed (CUDA $TORCH_CUDA)"
  else
    echo ""
    echo "ERROR: PyTorch cannot access CUDA. Possible fixes:"
    echo "  1. Update your NVIDIA driver: https://www.nvidia.com/Download/index.aspx"
    echo "  2. Install matching PyTorch manually:"
    echo "     pip install torch --index-url https://download.pytorch.org/whl/cu121"
    echo "  3. Check driver compatibility: nvidia-smi"
    echo ""
  fi
fi

# Verify LoRA adapter exists
if [[ -d "qwen_lora" ]]; then
  echo ""
  echo "Setup complete! LoRA adapters found at qwen_lora/"
  echo ""
  echo "Run inference:"
  echo "  source venv/bin/activate  # if using venv"
  echo "  python inference.py --image path/to/image.png"
  echo ""
  echo "Run inference with custom prompt:"
  echo "  python inference.py --image path/to/image.png \\"
  echo "    --prompt 'Does this image contain body horror defects? Answer YES or NO.'"
else
  echo ""
  echo "Warning: qwen_lora/ not found. You may need to train first:"
  echo "  python train.py --config configs/sft_qwen3_vl_8b.yaml"
fi
