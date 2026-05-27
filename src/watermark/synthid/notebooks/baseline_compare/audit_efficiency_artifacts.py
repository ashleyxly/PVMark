from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ORIGINAL = Path(
    "tests/baseline_comparison/original_synthid_efficiency_lcg_warm_seq200/"
    "efficiency_original_synthid_timing.json"
)
DEFAULT_HASH = Path(
    "tests/baseline_comparison/pvmark_poseidon2_t4_efficiency_warm_seq200/"
    "efficiency_hash_synthid_timing.json"
)
DEFAULT_HASH_TYPE3 = Path(
    "tests/baseline_comparison/pvmark_poseidon_t3_efficiency_warm_seq200/"
    "efficiency_hash_synthid_timing.json"
)
DEFAULT_HASH_TYPE5 = Path(
    "tests/baseline_comparison/pvmark_mimc_t5_efficiency_warm_seq200/"
    "efficiency_hash_synthid_timing.json"
)
DEFAULT_UPV = Path(
    "tests/baseline_comparison/upv_network_detector_gpt2_eli5_200_rerun/"
    "wet_wdt_network_z1_wet200_wdt200.json"
)
DEFAULT_PDW = Path("tests/baseline_comparison/pdw_gpt2/pdw_efficiency_from_records.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit saved WET/WDT artifacts for the four watermark schemes."
    )
    parser.add_argument("--original", default=str(DEFAULT_ORIGINAL))
    parser.add_argument("--hash", default=str(DEFAULT_HASH), help="PVMark Type 4 / Poseidon2 artifact.")
    parser.add_argument("--hash-type3", default=str(DEFAULT_HASH_TYPE3), help="PVMark Type 3 / Poseidon artifact.")
    parser.add_argument("--hash-type5", default=str(DEFAULT_HASH_TYPE5), help="PVMark Type 5 / MiMC artifact.")
    parser.add_argument("--upv", default=str(DEFAULT_UPV))
    parser.add_argument("--pdw", default=str(DEFAULT_PDW))
    parser.add_argument("--token-length", type=int, default=200)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any audit finding is an error.",
    )
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.error(message)


def check_synthid(
    audit: Audit,
    payload: dict[str, Any],
    *,
    label: str,
    token_length: int,
    expected_hash: bool,
    expected_hash_type: int | None = None,
) -> None:
    metadata = payload.get("metadata") or {}
    wet = (payload.get("wet") or {}).get(str(token_length)) or {}
    wdt = (payload.get("wdt") or {}).get(str(token_length)) or {}

    if expected_hash:
        audit.require(
            metadata.get("rust_lib") is True and metadata.get("is_lcg") is False,
            f"{label}: expected hash backend (rust_lib=true, is_lcg=false).",
        )
        if expected_hash_type is not None:
            audit.require(
                metadata.get("hash_type") == expected_hash_type,
                f"{label}: expected hash_type={expected_hash_type}.",
            )
    else:
        audit.require(
            metadata.get("is_lcg") is True,
            f"{label}: expected Original SynthID LCG backend.",
        )
        audit.require(
            metadata.get("rust_lib") is False or metadata.get("backend") == "original-rust-lcg",
            f"{label}: Original backend metadata is ambiguous.",
        )
        if metadata.get("hash_type") not in (None, "unused"):
            audit.error(
                f"{label}: hash_type={metadata.get('hash_type')} is present in "
                "an Original SynthID artifact; this artifact likely used the "
                "hash backend or predates the self-configuring timing script."
            )

    audit.require(bool(wet), f"{label}: missing wet[{token_length}] section.")
    audit.require(bool(wdt), f"{label}: missing wdt[{token_length}] section.")
    if wet:
        timed_calls = wet.get("timed_embedding_calls")
        if timed_calls is None:
            audit.error(
                f"{label}: WET artifact lacks timed_embedding_calls; old "
                "single-call SynthID WET artifacts should be rerun."
            )
        else:
            audit.require(
                int(timed_calls) == token_length,
                f"{label}: WET timed_embedding_calls={timed_calls}, expected {token_length}.",
            )
        audit.require(
            finite_number(wet.get("mean_ms_per_sample")),
            f"{label}: WET mean_ms_per_sample is missing or non-finite.",
        )
    if wdt:
        audit.require(
            finite_number(wdt.get("mean_ms_per_sample")),
            f"{label}: WDT mean_ms_per_sample is missing or non-finite.",
        )


def check_upv(audit: Audit, payload: dict[str, Any], *, token_length: int) -> None:
    metadata = payload.get("metadata") or {}
    wet_key = f"wet_{token_length}_tokens"
    wdt_key = f"wdt_{token_length}_tokens"
    wet = payload.get(wet_key) or {}
    wdt = payload.get(wdt_key) or {}

    audit.require(
        metadata.get("wet_token_length") == token_length,
        f"UPV: metadata wet_token_length should be {token_length}.",
    )
    audit.require(
        metadata.get("wdt_token_length") == token_length,
        f"UPV: metadata wdt_token_length should be {token_length}.",
    )
    audit.require(bool(wet), f"UPV: missing {wet_key}.")
    audit.require(bool(wdt), f"UPV: missing {wdt_key}.")
    if wet:
        audit.require(
            finite_number((wet.get("warm_cached") or {}).get("mean_sec_per_token")),
            "UPV: warm_cached mean_sec_per_token is missing or non-finite.",
        )
        audit.require(
            int(wet.get("num_prefix_samples") or 0) > 0,
            "UPV: no prefix samples were timed for WET.",
        )
    if wdt:
        audit.require(
            finite_number(wdt.get("mean_sec_per_text")),
            "UPV: WDT mean_sec_per_text is missing or non-finite.",
        )
        audit.require(
            int(wdt.get("num_texts_per_run") or 0) > 0,
            "UPV: WDT num_texts_per_run must be positive.",
        )


def check_pdw(audit: Audit, payload: dict[str, Any], *, token_length: int) -> None:
    wet = ((payload.get("wet") or {}).get("watermarked") or {})
    wdt = ((payload.get("wdt") or {}).get("unwatermarked_plain") or {})
    avg_wm_len = ((wet.get("completion_token_count") or {}).get("mean"))
    avg_uwm_len = ((wdt.get("sec_per_token") or {}).get("count"))

    audit.require(
        int(wet.get("num_valid_records") or 0) > 0,
        "PDW: WET has no valid watermarked generation records.",
    )
    audit.require(
        finite_number((wet.get("generation_time_sec") or {}).get("mean")),
        "PDW: WET mean generation_time_sec is missing or non-finite.",
    )
    audit.require(
        finite_number(avg_wm_len) and float(avg_wm_len) > token_length,
        "PDW: expected variable-length WM outputs longer than the 200-token comparison point.",
    )
    audit.require(
        int(wdt.get("num_valid_records") or 0) > 0,
        "PDW: WDT has no valid unwatermarked detection records.",
    )
    audit.require(
        finite_number((wdt.get("detection_time_sec") or {}).get("mean")),
        "PDW: WDT mean detection_time_sec is missing or non-finite.",
    )
    audit.require(
        int(avg_uwm_len or 0) > 0,
        "PDW: WDT per-token denominator count is missing.",
    )


def main() -> None:
    args = parse_args()
    audit = Audit()

    check_synthid(
        audit,
        read_json(args.original),
        label="Original SynthID",
        token_length=args.token_length,
        expected_hash=False,
    )
    check_synthid(
        audit,
        read_json(args.hash),
        label="Hash-based SynthID",
        token_length=args.token_length,
        expected_hash=True,
        expected_hash_type=4,
    )
    check_synthid(
        audit,
        read_json(args.hash_type3),
        label="PVMark Poseidon Type 3",
        token_length=args.token_length,
        expected_hash=True,
        expected_hash_type=3,
    )
    check_synthid(
        audit,
        read_json(args.hash_type5),
        label="PVMark MiMC Type 5",
        token_length=args.token_length,
        expected_hash=True,
        expected_hash_type=5,
    )
    check_upv(audit, read_json(args.upv), token_length=args.token_length)
    check_pdw(audit, read_json(args.pdw), token_length=args.token_length)

    result = {
        "errors": audit.errors,
        "warnings": audit.warnings,
        "notes": audit.notes,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict and audit.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
