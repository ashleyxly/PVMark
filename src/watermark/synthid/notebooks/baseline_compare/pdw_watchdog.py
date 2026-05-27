from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_DATASET_PATH,
    ensure_dir,
    load_eli5_prompts,
    read_json,
    shard_items,
)


REPO_ROOT = Path(os.environ.get("PVMark_SYNTHID_ROOT", "."))
SCRIPT_DIR = REPO_ROOT / "notebooks/baseline_compare"
PIPELINE_FILES = [
    "generations.json",
    "detection.json",
    "attacks.json",
    "attack_detection.json",
    "ppl.json",
    "attack_ppl.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor and recover the PDW baseline shard run.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-base", default=str(REPO_ROOT / "tests/baseline_comparison"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--gpus-csv", default="0,1,2,3")
    parser.add_argument("--env-name", default="baseline_wm")
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--stalled-intervals", type=int, default=3)
    parser.add_argument("--log-path", default=str(REPO_ROOT / "tests/baseline_comparison/logs/pdw_watchdog.log"))
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str, log_path: Path) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    ensure_dir(log_path.parent)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def payload_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(read_json(path).get("records", []))
    except Exception:
        return 0


def expected_generation_counts(dataset_path: str, limit: int, num_shards: int) -> dict[int, int]:
    prompts = load_eli5_prompts(dataset_path, limit)
    return {
        idx: 2 * len(shard_items(prompts, shard_index=idx, num_shards=num_shards))
        for idx in range(num_shards)
    }


def stage_expectations(expected_generation: int) -> dict[str, int]:
    return {
        "generations": expected_generation,
        "detection": expected_generation,
        "attacks": expected_generation,
        "attack_detection": expected_generation * 3,
        "ppl": expected_generation,
        "attack_ppl": expected_generation * 3,
    }


def stage_jsonl_path(shard_dir: Path, stage: str) -> Path:
    if stage == "generations":
        return shard_dir / "generations.records.jsonl"
    return shard_dir / f"{stage}.json.records.jsonl"


def stage_json_path(shard_dir: Path, stage: str) -> Path:
    if stage == "generations":
        return shard_dir / "generations.json"
    return shard_dir / f"{stage}.json"


def shard_counts(shard_dir: Path, expected_generation: int) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for stage, expected in stage_expectations(expected_generation).items():
        jsonl = count_jsonl(stage_jsonl_path(shard_dir, stage))
        payload = payload_count(stage_json_path(shard_dir, stage))
        counts[stage] = {
            "jsonl": jsonl,
            "payload": payload,
            "expected": expected,
            "complete": payload >= expected,
        }
    return counts


def shard_complete(shard_dir: Path, expected_generation: int) -> bool:
    counts = shard_counts(shard_dir, expected_generation)
    return all(counts[stage]["complete"] for stage in counts)


def compact_counts(counts: dict[str, dict[str, Any]]) -> str:
    parts = []
    for stage in ["generations", "detection", "attacks", "attack_detection", "ppl", "attack_ppl"]:
        data = counts[stage]
        parts.append(f"{stage}={data['payload']}/{data['jsonl']}/{data['expected']}")
    return " ".join(parts)


def tmux_session_exists(name: str) -> bool:
    result = subprocess.run(["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def process_exists_for_shard(shard_dir: Path, shard_index: int) -> bool:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,stat=,etime=,cmd="],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    shard_marker = str(shard_dir)
    shard_arg = f"--shard-index {shard_index}"
    for line in result.stdout.splitlines():
        if "pdw_watchdog.py" in line:
            continue
        if shard_marker in line and any(
            marker in line
            for marker in [
                "pdw_experiment.py",
                "run_attacks.py",
                "pdw_detect_attacks.py",
                "run_ppl.py",
                "conda run",
                "bash",
            ]
        ):
            return True
        if shard_arg in line and "pdw_experiment.py" in line:
            return True
    return False


def run_checked(command: list[str], log_path: Path, description: str) -> int:
    log(f"running {description}: {' '.join(shlex.quote(x) for x in command)}", log_path)
    result = subprocess.run(command, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.stdout:
        for line in result.stdout.splitlines()[-40:]:
            log(f"{description}: {line}", log_path)
    log(f"{description} exit_code={result.returncode}", log_path)
    return result.returncode


def start_recovery(
    *,
    shard_index: int,
    gpu_id: str,
    shard_dir: Path,
    key_dir: Path,
    limit: int,
    num_shards: int,
    env_name: str,
    log_path: Path,
) -> None:
    session = f"pdw_shard{shard_index:02d}_recover"
    recovery_log = shard_dir.parent / "logs" / f"shard_{shard_index:02d}_recovery.log"
    command = f"""
set -euo pipefail
cd {shlex.quote(str(REPO_ROOT))}
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_id))}
echo "[$(date -Is)] watchdog recovery shard={shard_index} gpu={gpu_id} start" >> {shlex.quote(str(recovery_log))}
conda run -n {shlex.quote(env_name)} python {shlex.quote(str(SCRIPT_DIR / "pdw_experiment.py"))} --mode full --limit {limit} --num-shards {num_shards} --shard-index {shard_index} --output-dir {shlex.quote(str(shard_dir))} --key-dir {shlex.quote(str(key_dir))} >> {shlex.quote(str(recovery_log))} 2>&1
conda run -n {shlex.quote(env_name)} python {shlex.quote(str(SCRIPT_DIR / "run_attacks.py"))} --input {shlex.quote(str(shard_dir / "generations.json"))} --output {shlex.quote(str(shard_dir / "attacks.json"))} >> {shlex.quote(str(recovery_log))} 2>&1
conda run -n {shlex.quote(env_name)} python {shlex.quote(str(SCRIPT_DIR / "pdw_detect_attacks.py"))} --input {shlex.quote(str(shard_dir / "attacks.json"))} --output {shlex.quote(str(shard_dir / "attack_detection.json"))} >> {shlex.quote(str(recovery_log))} 2>&1
conda run -n {shlex.quote(env_name)} python {shlex.quote(str(SCRIPT_DIR / "run_ppl.py"))} --input {shlex.quote(str(shard_dir / "generations.json"))} --output {shlex.quote(str(shard_dir / "ppl.json"))} --text-key completion_text >> {shlex.quote(str(recovery_log))} 2>&1
conda run -n {shlex.quote(env_name)} python {shlex.quote(str(SCRIPT_DIR / "run_ppl.py"))} --input {shlex.quote(str(shard_dir / "attacks.json"))} --output {shlex.quote(str(shard_dir / "attack_ppl.json"))} --text-key missing >> {shlex.quote(str(recovery_log))} 2>&1
echo "[$(date -Is)] watchdog recovery shard={shard_index} done" >> {shlex.quote(str(recovery_log))}
"""
    result = subprocess.run(["tmux", "new-session", "-d", "-s", session, command], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log(f"started recovery session={session} gpu={gpu_id} shard={shard_index} exit_code={result.returncode}", log_path)
    if result.stdout:
        log(result.stdout.strip(), log_path)


def merge_if_ready(output_base: Path, shard_root: Path, num_shards: int, env_name: str, log_path: Path) -> bool:
    final_out = output_base / "pdw_gpt2"
    if (final_out / "robustness_summary.json").exists() and (final_out / "summary.json").exists():
        log(f"merged outputs already exist: {final_out}", log_path)
        return True
    code = run_checked(
        [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            str(SCRIPT_DIR / "merge_shards.py"),
            "--shard-root",
            str(shard_root),
            "--output-dir",
            str(final_out),
            "--num-shards",
            str(num_shards),
        ],
        log_path,
        "merge_shards",
    )
    return code == 0


def main() -> None:
    args = parse_args()
    output_base = Path(args.output_base)
    log_path = Path(args.log_path)
    gpus = [gpu.strip() for gpu in args.gpus_csv.split(",") if gpu.strip()]
    num_shards = len(gpus)
    shard_root = output_base / "pdw_gpt2_shards"
    key_dir = output_base / "pdw_gpt2_pdw_shared_key"
    expected = expected_generation_counts(args.dataset_path, args.limit, num_shards)
    log(
        f"watchdog start output_base={output_base} limit={args.limit} gpus={args.gpus_csv} "
        f"expected_generation={json.dumps(expected, sort_keys=True)}",
        log_path,
    )

    last_jsonl_totals: dict[int, int] = {}
    stalled_counts: dict[int, int] = {}
    merge_done = False

    while True:
        all_complete = True
        for shard_index, gpu_id in enumerate(gpus):
            shard_dir = shard_root / f"shard_{shard_index:02d}"
            counts = shard_counts(shard_dir, expected[shard_index])
            log(f"shard={shard_index} gpu={gpu_id} {compact_counts(counts)}", log_path)
            complete = shard_complete(shard_dir, expected[shard_index])
            all_complete = all_complete and complete
            if complete:
                stalled_counts[shard_index] = 0
                continue

            current_total = sum(data["jsonl"] for data in counts.values())
            previous_total = last_jsonl_totals.get(shard_index)
            if previous_total is not None and current_total <= previous_total:
                stalled_counts[shard_index] = stalled_counts.get(shard_index, 0) + 1
            else:
                stalled_counts[shard_index] = 0
            last_jsonl_totals[shard_index] = current_total

            session = f"pdw_shard{shard_index:02d}_recover"
            running = process_exists_for_shard(shard_dir, shard_index) or tmux_session_exists(session)
            if running:
                continue
            if stalled_counts.get(shard_index, 0) < args.stalled_intervals:
                log(
                    f"shard={shard_index} incomplete but not running; waiting "
                    f"stalled={stalled_counts.get(shard_index, 0)}/{args.stalled_intervals}",
                    log_path,
                )
                continue
            log(f"shard={shard_index} incomplete and stalled; starting recovery", log_path)
            start_recovery(
                shard_index=shard_index,
                gpu_id=gpu_id,
                shard_dir=shard_dir,
                key_dir=key_dir,
                limit=args.limit,
                num_shards=num_shards,
                env_name=args.env_name,
                log_path=log_path,
            )
            stalled_counts[shard_index] = 0

        if all_complete and not merge_done:
            log("all shards complete; merging", log_path)
            merge_done = merge_if_ready(output_base, shard_root, num_shards, args.env_name, log_path)
            if merge_done:
                log("PDW baseline merge complete; watchdog exiting", log_path)
                return

        time.sleep(max(args.interval_sec, 10))


if __name__ == "__main__":
    main()
