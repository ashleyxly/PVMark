# Baseline Comparison Harness

This directory contains adapter scripts for running baseline watermark experiments
under the SynthID ELI5 setting.

The intended main comparison fixes:

- dataset: `$PVMark_ELI5_SELECT_TEST`
- backbone LM: GPT2 by default
- generation length: 200 new tokens where the method supports it
- PPL evaluator: OPT-2.7B by default
- attack strength: word deletion 0.3, synonym substitution 0.5, BERT context-aware substitution 0.5

Decoding parameters follow each baseline's original code by default.

## Environment

Use `setup_env.sh` to create `baseline_wm` by cloning the existing `synthid`
conda environment and installing only the extra crypto dependencies required by
`publicly-detectable-watermark`.

```bash
bash notebooks/baseline_compare/setup_env.sh
```

If you only run UPV, the existing `synthid` environment is usually enough.

GPU access in the Codex sandbox requires escalated execution. On this machine,
`nvidia-smi` shows 4 idle RTX 4090 GPUs when run outside the sandbox, and
`baseline_wm` sees CUDA. Use `CUDA_VISIBLE_DEVICES=<id>` to pin a run.

Use `PYTHONDONTWRITEBYTECODE=1` when running scripts from the two `$EXTERNAL_PATH \
  --mode full \
  --limit 1 \
  --output-dir tests/baseline_comparison/smoke/pdw
```

The PDW asymmetric method does not naturally generate exactly 200 tokens. The
WM side emits one message-signature pair and records the actual token count;
the UWM/plain side uses `--num-tokens 200`.

Run a small UPV test:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 conda run -n baseline_wm python notebooks/baseline_compare/upv_experiment.py \
  --mode full \
  --limit 5 \
  --output-dir tests/baseline_comparison/smoke/upv
```

Run attacks on a generation file:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 conda run -n baseline_wm python notebooks/baseline_compare/run_attacks.py \
  --input tests/baseline_comparison/smoke/upv/generations.json \
  --output tests/baseline_comparison/smoke/upv/attacks.json \
  --limit 5
```

Detect UPV attacks:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 conda run -n baseline_wm python notebooks/baseline_compare/upv_detect_attacks.py \
  --input tests/baseline_comparison/smoke/upv/attacks.json \
  --output tests/baseline_comparison/smoke/upv/attack_detection.json
```

Detect PDW attacks. This requires the `pdw_key` metadata written by
`pdw_experiment.py`:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 conda run -n baseline_wm python notebooks/baseline_compare/pdw_detect_attacks.py \
  --input tests/baseline_comparison/smoke/pdw/attacks.json \
  --output tests/baseline_comparison/smoke/pdw/attack_detection.json
```

Summarize attacked detection:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n baseline_wm python notebooks/baseline_compare/summarize_robustness.py \
  --detection tests/baseline_comparison/smoke/upv/attack_detection.json \
  --output-json tests/baseline_comparison/smoke/upv/robustness_summary.json \
  --output-csv tests/baseline_comparison/smoke/upv/robustness_summary.csv
```

Run PPL:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 conda run -n baseline_wm python notebooks/baseline_compare/run_ppl.py \
  --input tests/baseline_comparison/smoke/upv/generations.json \
  --output tests/baseline_comparison/smoke/upv/ppl.json \
  --text-key completion_text
```

Summarize detection:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n baseline_wm python notebooks/baseline_compare/summarize.py \
  --detection tests/baseline_comparison/smoke/upv/detection.json \
  --generation tests/baseline_comparison/smoke/upv/generations.json \
  --output tests/baseline_comparison/smoke/upv/summary.json
```

## Full GPT2 Runs

Parallel 4-GPU wrapper with shard-level resume:

```bash
PYTHONDONTWRITEBYTECODE=1 bash notebooks/baseline_compare/run_parallel_gpt2_baseline.sh upv 1000 $PVMark_RESULT_DIR/baseline_comparison 0,1,2,3
PYTHONDONTWRITEBYTECODE=1 bash notebooks/baseline_compare/run_parallel_gpt2_baseline.sh pdw 1000 $PVMark_RESULT_DIR/baseline_comparison 0,1,2,3
```

Each GPU writes to its own shard directory under `*_shards/shard_XX`, with
per-step `.records.jsonl` checkpoints. If a shard fails, inspect
`*_shards/logs/shard_XX.log` and re-run the same command; completed records are
skipped. After all shards finish, `merge_shards.py` writes the merged final
JSON files under `upv_gpt2/` or `pdw_gpt2/`.

For PDW, the parallel wrapper first creates one shared asymmetric key directory
`pdw_gpt2_pdw_shared_key/` and passes it to every shard, so all shards use the
same `sk/pk/params`.

Run both baselines inside tmux:

```bash
tmux new-session -d -s baseline_compare \
  'cd $PVMark_SYNTHID_ROOT && LIMIT=1000 GPUS_CSV=0,1,2,3 PYTHONDONTWRITEBYTECODE=1 bash notebooks/baseline_compare/run_all_baseline_experiments_tmux.sh'
```

Attach and monitor:

```bash
tmux attach -t baseline_compare
tail -f tests/baseline_comparison/logs/baseline_full_latest.log
tail -f tests/baseline_comparison/upv_gpt2_shards/logs/shard_00.log
find tests/baseline_comparison/upv_gpt2_shards -name '*.records.jsonl' -exec wc -l {} +
```

One-command wrapper:

```bash
GPU_ID=0 PYTHONDONTWRITEBYTECODE=1 bash notebooks/baseline_compare/run_full_gpt2_baseline.sh upv 1000
GPU_ID=1 PYTHONDONTWRITEBYTECODE=1 bash notebooks/baseline_compare/run_full_gpt2_baseline.sh pdw 1000
```

Equivalent manual commands:

UPV:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 conda run -n baseline_wm python $PVMark_UPV_ROOT \
  --mode full \
  --limit 1000 \
  --output-dir $PVMark_RESULT_DIR/baseline_comparison
```

PDW asymmetric:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 conda run -n baseline_wm python $PVMark_PDW_ROOT \
  --mode full \
  --limit 1000 \
  --output-dir $PVMark_RESULT_DIR/baseline_comparison
```

Then run `run_attacks.py`, method-specific attack detection, `run_ppl.py`,
`summarize.py`, and `summarize_robustness.py` on each method output.

## Efficiency Helpers

UPV network-based WET/WDT at 200 tokens:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n baseline_wm python notebooks/baseline_compare/time_upv_wet_wdt.py \
  --output-dir tests/baseline_comparison/upv_network_detector_gpt2_eli5_200 \
  --wet-token-length 200 \
  --wdt-token-length 200
```

PDW records timing during `pdw_experiment.py`. Summarize length-aware WET/WDT
from the completed generation and detection JSON files:

```bash
conda run -n baseline_wm python notebooks/baseline_compare/time_pdw_efficiency_from_records.py \
  --generations tests/baseline_comparison/pdw_gpt2/generations.json \
  --detection tests/baseline_comparison/pdw_gpt2/detection.json \
  --output tests/baseline_comparison/pdw_gpt2/pdw_efficiency_from_records.json
```
