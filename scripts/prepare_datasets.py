from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import ensure_processed_dataset


DATASET_ROOT = ROOT / "dataset"
SUPPORTED_SOURCES = ("math_data", "gsm8k", "mmlu_pro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets under the project's dataset/ directory.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SUPPORTED_SOURCES),
        choices=list(SUPPORTED_SOURCES),
        help="Dataset sources to prepare.",
    )
    parser.add_argument("--math-data-src", default="/GenSIvePFS/users/yfwang/code/verl/data/math_data")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--gsm8k-val-size", type=int, default=748)
    parser.add_argument("--mmlu-train-ratio", type=float, default=0.8)
    parser.add_argument("--mmlu-val-ratio", type=float, default=0.1)
    parser.add_argument("--mmlu-test-ratio", type=float, default=0.1)
    return parser.parse_args()


def _copy_math_data(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("dapo-math-17k.parquet", "validation.parquet", "test.parquet"):
        src_path = src_dir / name
        if not src_path.is_file():
            raise FileNotFoundError(f"missing math_data source file: {src_path}")
        dst_path = dst_dir / name
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)


def _ensure_dataset_layout(source_root: Path) -> None:
    for subdir in ("raw", "processed", "processed_exports"):
        (source_root / subdir).mkdir(parents=True, exist_ok=True)


def _write_manifest(source_root: Path, *, source: str) -> None:
    manifest = {
        "source": source,
        "root": str(source_root),
        "raw_dir": str(source_root / "raw"),
        "processed_dir": str(source_root / "processed"),
        "processed_exports_dir": str(source_root / "processed_exports"),
        "manifest_path": str(source_root / "manifest.json"),
        "splits": ["train", "val", "test"],
        "format": {
            "processed": "datasets.DatasetDict saved via save_to_disk",
            "processed_exports": "train/val/test parquet with unified project schema",
        },
    }
    (source_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def prepare_math_data(args: argparse.Namespace) -> None:
    source_root = DATASET_ROOT / "math_data"
    _ensure_dataset_layout(source_root)
    _copy_math_data(Path(args.math_data_src), source_root / "raw")
    ensure_processed_dataset(
        raw_path=source_root / "raw",
        processed_path=source_root / "processed",
        split_seed=args.split_seed,
        val_size=0,
        source="math_data",
    )
    _write_manifest(source_root, source="math_data")
    print(f"DATASET_READY source=math_data root={source_root}", flush=True)


def prepare_gsm8k(args: argparse.Namespace) -> None:
    source_root = DATASET_ROOT / "gsm8k"
    _ensure_dataset_layout(source_root)
    ensure_processed_dataset(
        raw_path=source_root / "raw",
        processed_path=source_root / "processed",
        split_seed=args.split_seed,
        val_size=args.gsm8k_val_size,
        source="gsm8k",
    )
    _write_manifest(source_root, source="gsm8k")
    print(f"DATASET_READY source=gsm8k root={source_root}", flush=True)


def prepare_mmlu_pro(args: argparse.Namespace) -> None:
    source_root = DATASET_ROOT / "mmlu_pro"
    _ensure_dataset_layout(source_root)
    ensure_processed_dataset(
        raw_path=source_root / "raw",
        processed_path=source_root / "processed",
        split_seed=args.split_seed,
        val_size=0,
        source="mmlu_pro",
        mmlu_pro_raw_splits=("validation", "test"),
        mmlu_pro_train_ratio=args.mmlu_train_ratio,
        mmlu_pro_val_ratio=args.mmlu_val_ratio,
        mmlu_pro_test_ratio=args.mmlu_test_ratio,
    )
    _write_manifest(source_root, source="mmlu_pro")
    print(f"DATASET_READY source=mmlu_pro root={source_root}", flush=True)


def main() -> None:
    args = parse_args()
    for source in args.sources:
        if source == "math_data":
            prepare_math_data(args)
        elif source == "gsm8k":
            prepare_gsm8k(args)
        elif source == "mmlu_pro":
            prepare_mmlu_pro(args)
        else:
            raise ValueError(f"unsupported source: {source}")


if __name__ == "__main__":
    main()
