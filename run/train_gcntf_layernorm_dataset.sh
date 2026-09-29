#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {la2a|cl1b|alesis3630}" >&2
    exit 2
fi

dataset="$1"
case "$dataset" in
    la2a|cl1b|alesis3630) ;;
    *)
        echo "Unsupported dataset: $dataset" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

configs=(
    "configs/release/gcntfilm/gcntf_250.yaml"
    "configs/release/gcntfilm/gcntf_2500_extended.yaml"
    "configs/release/ablation/mamba2_mag_phase_no_input_layernorm.yaml"
    "configs/release/ablation/mamba2_mag_phase_no_layernorm.yaml"
)

cd "$repo_root"
for config in "${configs[@]}"; do
    "$python_bin" train.py \
        --config_path "$config" \
        --dataset "$dataset"
done
