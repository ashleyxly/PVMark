# PVMark Experiment Data

This directory contains prompt subsets used in PVMark experiments. Models and full datasets are downloaded via `download_data.sh`.

## Directory Structure

```
experiment_data/
├── README.md              # This file
└── prompts/               # Prompt subsets (populated by download_data.sh)
```

## Data Sources

### Prompts

The prompt subsets used in the paper are downloaded by `download_data.sh`:

- **C4 news-like 100 prompts**: 100 texts with >300 tokens from the news-like subset of C4
- **ELI5 1000 questions**: 1000 questions from ELI5's test set

### Models (Download Required)

Models are NOT included in the artifact due to size. Use the download script:

```bash
bash download_data.sh
```

Required models:
- OPT-1.3B, OPT-2.7B (for KGW experiments and PPL evaluation)
- T5-Large (for KGW experiments)
- Bloom-1.1B (for KGW experiments)
- GPT-2 (for SynthID experiments)
- Gemma-2B-IT (for SynthID experiments)
- BERT-base-uncased (for context-aware attacks)

### Datasets (Download Required)

Full datasets are NOT included. The download script will fetch the prompt subsets.

## Notes

- Results may vary slightly due to random seeds, GPU/CPU performance, and model versions
- The paper reports averaged results over multiple runs
- Your results should be within 5% of the reported values
