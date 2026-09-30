"""Export, verify and benchmark trained GCNTF checkpoints with persistent states."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
from runtime import StreamingGCNTF

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
BLOCK_DURATION_MS = BLOCK_SIZE / SAMPLE_RATE * 1000
MODEL_STRATEGIES = {
    "gcntf_250_release": "conv",
    "gcntf_2500_extended_release": "gather",
}


def load_source(path):
    package = types.ModuleType("models")
    package.__path__ = [str(ROOT / "models")]
    sys.modules.setdefault("models", package)
    from models.gcntfilm.gcntfilm import GCNTFModel

    checkpoint = torch.load(path, map_location="cpu")
    source = GCNTFModel(**checkpoint["hyper_parameters"]).eval().float()
    source.load_state_dict(checkpoint["state_dict"], strict=True)
    return source


def checkpoint_paths():
    checkpoints = []
    for name in MODEL_STRATEGIES:
        model_dir = ROOT / "experiments" / "alesis3630" / name
        matches = sorted(model_dir.glob("**/*.ckpt"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one checkpoint for {name}, found {len(matches)}"
            )
        checkpoints.append(matches[0])
    return checkpoints


def signal(count, channel=0):
    frequency = 997 if channel == 0 else 1499
    t = torch.arange(count, dtype=torch.float64) / SAMPLE_RATE
    waveform = 0.1 * torch.sin(2 * torch.pi * frequency * t)
    waveform += 0.03 * torch.sin(2 * torch.pi * 71 * t)
    return waveform.float().reshape(1, 1, -1)


def measure(model, count, warmup, block):
    model.reset()
    audio = signal((count + warmup) * block)
    params = torch.tensor([[-0.18, 0.4, 0.01, 0.5]])
    times = []
    with torch.no_grad():
        for i in range(count + warmup):
            x = audio[:, :, i * block : (i + 1) * block]
            start = time.perf_counter_ns()
            output = model(x, params)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if i >= warmup:
                times.append(elapsed)
    assert torch.isfinite(output).all()
    return np.asarray(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks" / "gcntf" / "results"
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    rows, validation = [], []
    checkpoints = checkpoint_paths()
    for ckpt in checkpoints:
        name = ckpt.parts[-5]
        source = load_source(ckpt)
        source.pad_input_to_receptive_field = False
        n = 262144  # exceeds the largest receptive field and wraps every ring
        x = torch.randn(1, 1, n) * .1
        params = torch.tensor([[-0.18, 0.4, 0.01, 0.5]])
        with torch.no_grad():
            reference = source(x, params)

        strategy = MODEL_STRATEGIES[name]
        runtime = StreamingGCNTF(source, strategy).eval()
        scripted = torch.jit.script(runtime)

        for block in [128, BLOCK_SIZE]:
            scripted.reset()
            chunks = []
            with torch.no_grad():
                for start in range(0, n, block):
                    chunk = x[:, :, start : start + block]
                    chunks.append(scripted(chunk, params))
            actual = torch.cat(chunks, dim=2)
            error = (actual - reference).abs()
            max_abs = error.max().item()
            rmse = error.square().mean().sqrt().item()
            assert torch.allclose(actual, reference, atol=2e-5, rtol=2e-4), (
                name,
                strategy,
                block,
                max_abs,
            )
            result = {
                "name": name,
                "strategy": strategy,
                "block": block,
                "max_abs": max_abs,
                "rmse": rmse,
            }
            validation.append(result)
            print("PARITY", result, flush=True)

        # Save a clean state, not the state remaining after the tolerance test.
        scripted.reset()
        scripted.save(str(args.output / f"{name}_{strategy}.pt"))

        backends = [("python_eager", runtime), ("torchscript", scripted)]
        for backend, model in backends:
            for repeat in range(args.repeats):
                times = measure(model, args.iterations, args.warmup, BLOCK_SIZE)
                mean_ms = times.mean()
                row = {
                    "name": name,
                    "backend": backend,
                    "strategy": strategy,
                    "repeat": repeat,
                    "sample_rate": SAMPLE_RATE,
                    "block": BLOCK_SIZE,
                    "channels": 1,
                    "threads": 1,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "mean_ms": mean_ms,
                    "std_ms": times.std(),
                    "p99_ms": np.quantile(times, 0.99),
                    "max_ms": times.max(),
                    "cpu_percent": mean_ms / BLOCK_DURATION_MS * 100,
                    "realtime_speed": BLOCK_DURATION_MS / mean_ms,
                    "deadline_misses": int((times > BLOCK_DURATION_MS).sum()),
                }
                rows.append(row)
                print(
                    "BENCH",
                    name,
                    backend,
                    strategy,
                    repeat,
                    round(mean_ms, 4),
                    flush=True,
                )
        # Files for the real VST3 host accuracy check, with the same defaults as its parameters.
        test_input = torch.cat([signal(262144, ch) for ch in range(2)], dim=0)
        with torch.no_grad():
            test_output = torch.cat(
                [source(test_input[ch : ch + 1], params) for ch in range(2)], dim=0
            )
        test_input.numpy().tofile(args.output / f"{name}_input.f32")
        test_output.numpy().tofile(args.output / f"{name}_reference.f32")

    with (args.output / "cpu.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (args.output / "validation.json").write_text(
        json.dumps(validation, indent=2)
    )
    checkpoint_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoints
    }
    metadata = {
        "torch": torch.__version__,
        "threads": 1,
        "cpu": subprocess.getoutput("sysctl -n machdep.cpu.brand_string"),
        "git": subprocess.getoutput("git rev-parse HEAD"),
        "checkpoints": checkpoint_hashes,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
