#!/usr/bin/env bash
# Pack trained LoRA adapters + inference code into a portable archive.
#
# Usage:
#   ./pack.sh                          # uses defaults
#   ./pack.sh --lora-dir qwen_lora     # custom adapter path
#   ./pack.sh --output my_model.tar.gz # custom output name
set -euo pipefail

LORA_DIR="qwen_lora"
OUTPUT=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora-dir)  LORA_DIR="$2"; shift 2 ;;
    --output)    OUTPUT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--lora-dir DIR] [--output FILE]"
      echo "  --lora-dir  Path to LoRA adapter directory (default: qwen_lora)"
      echo "  --output    Output archive name (default: qwen_lora_pack_YYYYMMDD.tar.gz)"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Resolve lora dir relative to script dir if not absolute
if [[ "$LORA_DIR" != /* ]]; then
  LORA_DIR="$SCRIPT_DIR/$LORA_DIR"
fi

if [[ ! -d "$LORA_DIR" ]]; then
  echo "Error: LoRA adapter directory not found: $LORA_DIR"
  echo "Run training first: python train.py"
  exit 1
fi

# Default output name with date stamp
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="qwen_lora_pack_$(date +%Y%m%d).tar.gz"
fi

STAGING_DIR=$(mktemp -d)
PACK_NAME="qwen_lora_pack"
PACK_DIR="$STAGING_DIR/$PACK_NAME"
mkdir -p "$PACK_DIR"

echo "Packing trained model for portable deployment..."
echo "  LoRA adapters: $LORA_DIR"
echo "  Output: $OUTPUT"

# Copy LoRA adapter weights
cp -r "$LORA_DIR" "$PACK_DIR/qwen_lora"

# Copy inference and training scripts
cp "$SCRIPT_DIR/inference.py" "$PACK_DIR/"
cp "$SCRIPT_DIR/train.py" "$PACK_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$PACK_DIR/"
cp -r "$SCRIPT_DIR/configs" "$PACK_DIR/"

# Copy setup script
cp "$SCRIPT_DIR/setup.sh" "$PACK_DIR/"

# Write a manifest with metadata
cat > "$PACK_DIR/MANIFEST.json" <<MANIFEST
{
  "packed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "base_model": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
  "adapter_dir": "qwen_lora",
  "task": "body defect detection (binary classification)",
  "pack_version": "1.0"
}
MANIFEST

# Create the archive
tar -czf "$OUTPUT" -C "$STAGING_DIR" "$PACK_NAME"
rm -rf "$STAGING_DIR"

SIZE=$(du -h "$OUTPUT" | cut -f1)
echo ""
echo "Done! Archive: $OUTPUT ($SIZE)"
echo ""
echo "To deploy on another machine:"
echo "  1. Copy $OUTPUT to the target machine"
echo "  2. tar xzf $OUTPUT"
echo "  3. cd $PACK_NAME"
echo "  4. ./setup.sh"
echo "  5. python inference.py --image path/to/image.png"
