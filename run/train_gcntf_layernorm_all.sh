#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for dataset in la2a cl1b alesis3630; do
    "$script_dir/train_gcntf_layernorm_dataset.sh" "$dataset"
done
