# Dataset Layout

All project datasets live under `dataset/` and follow the same structure:

- `dataset/<source>/raw/`
- `dataset/<source>/processed/`
- `dataset/<source>/processed_exports/`
- `dataset/<source>/manifest.json`

Config interface:

- `data.root_dir`
- `data.raw_dir`
- `data.processed_dir`
- `data.processed_exports_dir`
- `data.manifest_path`

For backward compatibility, old configs using `data.cache_dir` are normalized to `data.raw_dir`.

Meaning:

- `raw/`: source dataset files or the original Hugging Face dataset cache.
- `processed/`: the canonical `datasets.DatasetDict` saved with `save_to_disk`.
- `processed_exports/`: exported `train.parquet`, `val.parquet`, and `test.parquet` using the project's unified record schema.
- `manifest.json`: lightweight metadata describing paths and split conventions.

Current sources:

- `math_data`
- `gsm8k`
- `mmlu_pro`

Use `python scripts/prepare_datasets.py` to materialize all supported sources into this layout.
