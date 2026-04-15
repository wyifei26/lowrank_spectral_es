#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SRC_DIR:-/GenSIvePFS/users/yfwang/code/verl/data/math_data}"
DST_DIR="${DST_DIR:-${ROOT_DIR}/dataset/math_data/raw}"

mkdir -p "${DST_DIR}"

for name in dapo-math-17k.parquet validation.parquet test.parquet; do
  if [[ ! -f "${SRC_DIR}/${name}" ]]; then
    echo "missing source file: ${SRC_DIR}/${name}" >&2
    exit 1
  fi
  cp -n "${SRC_DIR}/${name}" "${DST_DIR}/${name}"
done

echo "math_data is ready at ${DST_DIR}"
