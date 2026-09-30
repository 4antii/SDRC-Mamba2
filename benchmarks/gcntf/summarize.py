"""Aggregate repeated CPU and VST3 host measurements using explicit RT conventions."""
import csv
from collections import defaultdict
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent / "results"
STRATEGIES = {
    "gcntf_250_release": "conv",
    "gcntf_2500_extended_release": "gather",
}


def main():
    groups = defaultdict(list)
    for row in csv.DictReader((OUT / "cpu.csv").open()):
        if row["strategy"] == STRATEGIES[row["name"]]:
            groups[row["name"], row["backend"], 1].append(row)
    for row in csv.DictReader((OUT / "host.csv").open()):
        groups[row["model"], "juce_vst3", int(row["channels"])].append(row)

    rows = []
    for (model, backend, channels), repeats in sorted(groups.items()):
        means = [float(row["mean_ms"]) for row in repeats]
        mean_ms = statistics.mean(means)
        latency_samples = 512 if backend == "juce_vst3" else 0
        warmup = int(repeats[0].get("warmup", 300))
        iterations = int(repeats[0].get("iterations", 500))
        block_duration_ms = 512 / 44.1

        row = {
            "model": model,
            "backend": backend,
            "strategy": STRATEGIES[model],
            "sample_rate": 44100,
            "block_samples": 512,
            "output_samples_per_step": 512,
            "channels": channels,
            "threads": 1,
            "repeats": len(repeats),
            "warmup_blocks": warmup,
            "measured_blocks_per_repeat": iterations,
            "mean_ms": mean_ms,
            "repeat_std_ms": statistics.stdev(means) if len(means) > 1 else 0.0,
            "mean_within_run_std_ms": statistics.mean(
                float(repeat["std_ms"]) for repeat in repeats
            ),
            "worst_p99_ms": max(float(repeat["p99_ms"]) for repeat in repeats),
            "max_ms": max(float(repeat["max_ms"]) for repeat in repeats),
            "cpu_percent": 100 * mean_ms / block_duration_ms,
            "rtf": mean_ms / block_duration_ms,
            "irtf": block_duration_ms / mean_ms,
            "deadline_misses": sum(
                int(repeat["deadline_misses"]) for repeat in repeats
            ),
            "adapter_latency_samples": latency_samples,
            "adapter_latency_ms": latency_samples / 44.1,
        }
        rows.append(row)

    with (OUT / "summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            row["model"],
            row["backend"],
            row["channels"],
            f"{row['mean_ms']:.3f} +/- {row['repeat_std_ms']:.3f} ms",
            f"CPU {row['cpu_percent']:.1f}%",
            f"iRTF {row['irtf']:.2f}",
            f"misses {row['deadline_misses']}",
        )


if __name__ == "__main__":
    main()
