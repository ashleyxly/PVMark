#!/usr/bin/env bash
# PVMark Data and Model Download Script
# Downloads required models and datasets for reproducing PVMark experiments.
set -euo pipefail

PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
HF_MODEL_DIR="${HF_MODEL_DIR:-$HOME/.cache/huggingface/hub}"
DATA_DIR="${DATA_DIR:-$PVMARK_ROOT/experiment_data}"

echo "=== PVMark Data Download Script ==="
echo "PVMARK_ROOT: $PVMARK_ROOT"
echo "HF_MODEL_DIR: $HF_MODEL_DIR"
echo "DATA_DIR: $DATA_DIR"
echo ""

mkdir -p "$HF_MODEL_DIR"
mkdir -p "$DATA_DIR/prompts"

download_model() {
    local model_id=$1
    if python -c "from transformers import AutoModel; AutoModel.from_pretrained('${model_id}', cache_dir='${HF_MODEL_DIR}')" 2>/dev/null; then
        echo "[OK] Model $model_id"
    else
        echo "[DOWNLOAD] $model_id..."
        python -c "
from transformers import AutoModel, AutoTokenizer
m = '${model_id}'
c = '${HF_MODEL_DIR}'
AutoModel.from_pretrained(m, cache_dir=c)
AutoTokenizer.from_pretrained(m, cache_dir=c)
print(f'Done: {m}')
" || echo "[WARN] Failed to download $model_id"
    fi
}

download_c4_prompts() {
    local output_file="$DATA_DIR/prompts/num_100.json"
    if [ -f "$output_file" ]; then
        echo "[SKIP] $output_file already exists"
        return
    fi
    echo "[DOWNLOAD] C4 prompts -> $output_file"
    python -c "
from datasets import load_dataset
import json

ds = load_dataset('allenai/c4', 'en', split='train[:200]', trust_remote_code=True)
records = []
for item in ds:
    text = item.get('text', '')
    if len(text) < 200:
        continue
    # Truncate to ~512 tokens worth of text
    text_short = text[:2048]
    records.append({
        'text_shortened': text_short,
        'text_removed': '',
        'text_full': text,
    })
    if len(records) >= 100:
        break

with open('$output_file', 'w') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f'Saved {len(records)} prompts to $output_file')
" || echo "[WARN] Failed to download C4 prompts"
}

echo "=== Step 1: Downloading Models ==="
download_model "facebook/opt-1.3b"
download_model "facebook/opt-2.7b"
download_model "openai-community/gpt2"
download_model "google-bert/bert-base-uncased"

echo ""
echo "=== Step 2: Downloading Prompts ==="
download_c4_prompts

echo ""
echo "=== Download Complete ==="
echo "Next steps:"
echo "  bash scripts/download_ptau.sh"
echo "  bash scripts/run_smoke_test.sh"
