"""Benchmark actual VST3s and compare latency-aligned renders with PyTorch."""
import csv
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "gcntf" / "results"
BUILD = ROOT / "juce" / "GCNTFBenchmarks" / "build"
MODELS = [
    ("gcntf_250_release", "GCNTF250", "GCNTF 250"),
    (
        "gcntf_2500_extended_release",
        "GCNTF2500Extended",
        "GCNTF 2500 Extended",
    ),
]


def run_host(plugin, channels, warmup, iterations, csv_path, **extra_args):
    command = [
        str(BUILD / "gcntf_host"),
        "--plugin",
        str(plugin),
        "--sample-rate",
        "44100",
        "--block-size",
        str(extra_args.pop("block_size", 512)),
        "--channels",
        str(channels),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--csv",
        str(csv_path),
    ]
    for option, value in extra_args.items():
        command.extend([f"--{option.replace('_', '-')}", str(value)])
    subprocess.run(command, check=True)


def main():
    rows, validation = [], []
    for stem, target, name in MODELS:
        plugin = (
            BUILD
            / f"{target}_artefacts"
            / "Release"
            / "VST3"
            / f"{name} CPU benchmark.vst3"
        )
        for channels in [1, 2]:
            for repeat in range(3):
                path = OUT / f"host_{stem}_{channels}_{repeat}.csv"
                run_host(plugin, channels, 300, 500, path)
                row = next(csv.DictReader(path.open()))
                assert row["status"] == "ok", row
                row.update(model=stem, repeat=repeat)
                rows.append(row)

        for block in [512, 173]:
            render = OUT / f"{stem}_host_{block}.f32"
            run_host(
                plugin,
                channels=2,
                warmup=5,
                iterations=10,
                csv_path=OUT / "validation_host.csv",
                block_size=block,
                render_out=render,
            )
            actual = np.fromfile(render, dtype=np.float32)
            reference = np.fromfile(OUT / f"{stem}_reference.f32", dtype=np.float32)
            assert actual.shape == reference.shape
            error = actual.astype(np.float64) - reference
            result = {
                "model": stem,
                "host_block": block,
                "max_abs": float(np.max(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
            }
            print("VST3 PARITY", result, flush=True)
            assert np.allclose(actual, reference, atol=2e-5, rtol=2e-4), result
            validation.append(result)

    with (OUT / "host.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "host_validation.json").write_text(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
