from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

from common import DEFAULT_PUBLICLY_DETECTABLE_ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one shared PDW asymmetric key directory.")
    parser.add_argument("--pdw-root", default=str(DEFAULT_PUBLICLY_DETECTABLE_ROOT))
    parser.add_argument("--key-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def import_crypto(root: str | Path) -> Any:
    sys.path.insert(0, str(root))
    import crypto  # type: ignore

    return crypto


def main() -> None:
    args = parse_args()
    key_dir = ensure_dir(args.key_dir)
    sk_path = key_dir / "shared_sk.pickle"
    pk_path = key_dir / "shared_pk.pickle"
    params_path = key_dir / "shared_params.pickle"
    if not args.force and sk_path.exists() and pk_path.exists() and params_path.exists():
        print(f"PDW shared key already exists: {key_dir}")
        return

    crypto = import_crypto(args.pdw_root)
    from petlib.pack import encode  # type: ignore

    sk, pk, params = crypto.bls_generate_openssl()
    G, _o, _g1, _g2, _e = params
    with open(sk_path, "wb") as f:
        pickle.dump(encode(sk), f)
    with open(pk_path, "wb") as f:
        pickle.dump(encode(pk), f)
    with open(params_path, "wb") as f:
        pickle.dump(encode(G), f)
    print(f"Prepared PDW shared key: {key_dir}")


if __name__ == "__main__":
    main()
