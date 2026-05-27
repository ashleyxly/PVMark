from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from tqdm import tqdm

from baseline_eval.common import (
    DEFAULT_PDW_ROOT,
    DEFAULT_UPV_ROOT,
    WallTimer,
    add_repo_to_path,
    append_resume_shard_args,
    bool_from_score,
    ensure_dir,
    get_device,
    make_failure,
    read_json,
    read_json_dir,
    remove_checkpoint_if_exists,
    select_shard,
    write_checkpoint,
    write_json,
)
from baseline_eval.upv_network import UpvNetworkDetector


ATTACK_FIELDS = [
    "word_deletion",
    "synonym_substitution",
    "context_aware_synonym_substitution",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-detect watermark after robustness attacks.")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pdw-root", default=DEFAULT_PDW_ROOT)
    parser.add_argument("--upv-root", default=DEFAULT_UPV_ROOT)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None, help="Explicit torch device for supported detectors, e.g. cuda:2.")
    append_resume_shard_args(parser)
    return parser.parse_args()


def make_publicly_detectable_detector(generations: dict[str, Any], pdw_root: str) -> Callable[[str], dict[str, Any]]:
    add_repo_to_path(pdw_root)
    import pickle
    from petlib.pack import decode as petlib_decode
    import detect

    meta = generations["metadata"]["scheme_metadata"]
    with open(meta["pk_path"], "rb") as f:
        pk = petlib_decode(pickle.load(f))
    with open(meta["params_path"], "rb") as f:
        G = petlib_decode(pickle.load(f))
        params = (G, G.order(), G.gen1(), G.gen2(), G.pair)

    def _detector(text: str) -> dict[str, Any]:
        pred = bool(
            detect.search_for_asymmetric_watermark(
                pk,
                params,
                text,
                meta["message_length"],
                meta["signature_segment_length"],
                meta["bit_size"],
                meta["max_planted_errors"],
            )
        )
        return {"score": None, "prediction": pred}

    return _detector


def make_unforgeable_detector(
    generations: dict[str, Any],
    upv_root: str,
    use_gpu: bool,
) -> Callable[[str], dict[str, Any]]:
    add_repo_to_path(upv_root)
    from transformers import AutoTokenizer
    from watermark_model import Watermark

    meta = generations["metadata"]["scheme_metadata"]
    args = generations["metadata"]["args"]
    device = get_device(use_gpu)
    tokenizer = AutoTokenizer.from_pretrained(args["model"], use_fast=False)
    watermark = Watermark(
        bit_number=meta["bit_number"],
        window_size=meta["window_size"],
        layers=meta["layers"],
        delta=meta["delta"],
        model_dir=meta["generator_model"],
        beam_size=args.get("beam_size", 0),
    )
    threshold = meta["z_threshold"]

    def _detector(text: str) -> dict[str, Any]:
        if not text:
            return {"score": None, "prediction": False}
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        if ids.numel() <= watermark.min_prefix_len:
            return {"score": None, "prediction": False}
        _, _, z_score = watermark.green_token_mask_and_stats(ids.cpu())
        z = float(z_score)
        return {"score": z, "prediction": bool_from_score(z, threshold)}

    return _detector


def make_unforgeable_network_detector(
    generations: dict[str, Any],
    upv_root: str,
    use_gpu: bool,
    device_override: Optional[str] = None,
) -> Callable[[str], dict[str, Any]]:
    meta = generations["metadata"]["scheme_metadata"]
    args = generations["metadata"]["args"]
    device = device_override or get_device(use_gpu)
    detector = UpvNetworkDetector(
        upv_root=upv_root,
        detector_model=meta["network_detector_model"],
        tokenizer_path=args["model"],
        bit_number=meta["bit_number"],
        layers=meta["layers"],
        fixed_length=meta.get("detector_fixed_length", 200),
        threshold=meta.get("network_threshold", 0.5),
        device=device,
    )

    def _detector(text: str) -> dict[str, Any]:
        score, pred = detector.score_text(text)
        return {"score": score, "prediction": pred}

    return _detector


def make_kgw_detector(generations: dict[str, Any], use_gpu: bool) -> Callable[[str], dict[str, Any]]:
    from transformers import AutoTokenizer
    from watermark_processor_org_scheme import WatermarkDetector

    meta = generations["metadata"]["scheme_metadata"]
    device = get_device(use_gpu)
    tokenizer = AutoTokenizer.from_pretrained(meta["model"], use_fast=False)
    detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=meta["gamma"],
        seeding_scheme=meta["seeding_scheme"],
        device=device,
        tokenizer=tokenizer,
        z_threshold=meta["z_threshold"],
        normalizers=meta.get("normalizers", []),
        ignore_repeated_bigrams=meta.get("ignore_repeated_bigrams", False),
        select_green_tokens=meta.get("select_green_tokens", True),
    )

    def _detector(text: str) -> dict[str, Any]:
        if not text:
            return {"score": None, "prediction": False}
        result = detector.detect(text)
        z = float(result["z_score"])
        return {"score": z, "prediction": bool(result["prediction"])}

    return _detector


def main() -> None:
    args = parse_args()
    generations = read_json(args.generations)
    attacks = read_json(args.attacks)
    scheme = generations["metadata"]["scheme"]

    if scheme == "publicly_detectable":
        detector = make_publicly_detectable_detector(generations, args.pdw_root)
    elif scheme == "unforgeable":
        detector = make_unforgeable_detector(generations, args.upv_root, args.use_gpu)
    elif scheme == "unforgeable_network":
        detector = make_unforgeable_network_detector(generations, args.upv_root, args.use_gpu, args.device)
    elif scheme == "kgw":
        detector = make_kgw_detector(generations, args.use_gpu)
    else:
        raise ValueError(f"Unsupported scheme for robustness detection: {scheme}")

    attack_rows = select_shard(attacks["attacks"], args.shard_index, args.num_shards)
    out = args.output or str(Path(args.attacks).with_name("robustness_detection.json"))
    out_dir = ensure_dir(Path(out).parent)
    sample_dir = ensure_dir(out_dir / "robustness_samples")
    failure_dir = ensure_dir(out_dir / "robustness_failures")

    rows_by_id = read_json_dir(sample_dir, "sample_id") if args.resume else {}
    failures_by_id = read_json_dir(failure_dir, "id") if args.resume else {}
    for row in tqdm(attack_rows, desc="Robustness detection"):
        sample_id = int(row["sample_id"])
        if args.resume and sample_id in rows_by_id:
            continue
        if args.resume and sample_id in failures_by_id and not args.retry_failures:
            continue

        try:
            result: dict[str, Any] = {"sample_id": sample_id, "attacks": {}}
            with WallTimer() as t0:
                original = detector(row["original_text"])
            result["original"] = {**original, "detection_time_sec": t0.elapsed}
            for field in ATTACK_FIELDS:
                with WallTimer() as timer:
                    det = detector(row[field])
                result["attacks"][field] = {**det, "detection_time_sec": timer.elapsed}
            write_checkpoint(sample_dir, sample_id, result)
            remove_checkpoint_if_exists(failure_dir, sample_id)
            rows_by_id[sample_id] = result
            failures_by_id.pop(sample_id, None)
        except Exception as exc:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            failure = make_failure(sample_id, "robustness_detection", exc)
            write_checkpoint(failure_dir, sample_id, failure)
            failures_by_id[sample_id] = failure
            if args.fail_fast:
                raise

    rows = [rows_by_id[k] for k in sorted(rows_by_id)]
    failures = [failures_by_id[k] for k in sorted(failures_by_id)]

    summary: dict[str, Any] = {"num_attacked": len(rows)}
    for field in ATTACK_FIELDS:
        detected = sum(1 for r in rows if r["attacks"][field]["prediction"])
        total = len(rows)
        summary[field] = {
            "post_attack_tpr": detected / total if total else None,
            "attack_success_rate": 1 - (detected / total) if total else None,
            "detected_after_attack": detected,
            "total": total,
        }

    write_json(
        out,
        {
            "metadata": {
                "scheme": scheme,
                "source_generations": args.generations,
                "source_attacks": args.attacks,
                "failures": failures,
                "shard": {"index": args.shard_index, "num_shards": args.num_shards},
            },
            "summary": summary,
            "results": rows,
        },
    )


if __name__ == "__main__":
    main()
