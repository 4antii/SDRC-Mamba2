# Single figure with 4 heatmaps (2x2):
#   - Fixed threshold: -40 dB
#   - Ratios: 1, 2, 4, 100 (labeled "inf" when 100)
#   - x-axis: release (ms)
#   - y-axis: attack (ms)
#   - value: mean STFT loss
#
# Usage:
#   python temporal_analysis.py \
#       --config_path ./configs/release/mamba2_mag_phase_mask.yaml
#       --dataset alesis3630 \
#       --model_dir ./experiments/alesis3630/mamba2_mag_phase_mask_release \
#       --display_name SDRC-Mamba2
#
# Output:
#   ./eval_heatmaps/<dataset>/<display_name>_thr-40_ratio-panels_stft.png

import os
import yaml
import torch
import auraloss
import numpy as np
from matplotlib import colors
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Dict, Tuple

from data import datasets_map
from models import models_map
from utils import Config
from models.utils import causal_crop, center_crop

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

def truncate_cmap(cmap_name="YlGnBu", minval=0.10, maxval=0.90, n=256):
    base = cm.get_cmap(cmap_name, n)
    new_colors = base(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", new_colors)

pl.seed_everything(42)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def denorm_params_ms(p_norm: np.ndarray):
    """
    Dataset returns normalized params:
      thr/100, ratio/10, attack/1000, release/1000
    """
    thr_db = round(float(p_norm[0] * 100.0))
    ratio = round(float(p_norm[1] * 10.0))
    atk_ms = round(float(p_norm[2] * 1000.0), 1)
    rel_ms = int(round(float(p_norm[3] * 1000.0)))
    return thr_db, ratio, atk_ms, rel_ms


def pick_latest_checkpoint(model_dir: str):
    logs_root = Path(model_dir) / "lightning_logs"
    versions = sorted(logs_root.glob("version_*"))
    if not versions:
        raise FileNotFoundError(f"No lightning_logs/version_* found in: {model_dir}")

    def vnum(p: Path):
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return -1

    version_dir = sorted(versions, key=vnum)[-1]
    ckpt_dir = version_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        ckpts = sorted(ckpt_dir.glob("*"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    def epoch_num(p: Path):
        try:
            return int(p.name.split("-")[0].split("=")[-1])
        except Exception:
            return -1

    ckpt = sorted(ckpts, key=epoch_num)[-1]
    hparams = version_dir / "hparams.yaml"
    if not hparams.exists():
        raise FileNotFoundError(f"hparams.yaml not found: {hparams}")
    return str(ckpt), str(hparams)


def build_grid(
    stft_by_ar: Dict[Tuple[int, int], List[float]],
    attacks: List[int],
    releases: List[int],
) -> np.ndarray:
    grid = np.full((len(attacks), len(releases)), np.nan, dtype=np.float32)
    a2i = {a: i for i, a in enumerate(attacks)}
    r2i = {r: i for i, r in enumerate(releases)}
    for (a, r), vals in stft_by_ar.items():
        if a not in a2i or r not in r2i or len(vals) == 0:
            continue
        grid[a2i[a], r2i[r]] = float(np.mean(vals))
    return grid


def ratio_label(r: float) -> str:
    return "inf" if int(round(r)) == 100 else f"{int(round(r))}"


def annotate_grid(ax, grid: np.ndarray, fmt: str = "{:.3f}", fontsize: int = 7):
    """
    Draw white numbers in each finite cell, with black stroke for readability.
    """
    ny, nx = grid.shape
    for y in range(ny):
        for x in range(nx):
            v = grid[y, x]
            if not np.isfinite(v):
                continue
            ax.text(
                x, y, fmt.format(float(v)),
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
                path_effects=[pe.withStroke(linewidth=1.0, foreground="black")],
            )


def main():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="alesis3630")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--display_name", type=str, default=None)

    parser.add_argument("--thr_db", type=float, default=-40.0)
    parser.add_argument("--thr_tol", type=float, default=1e-3)

    parser.add_argument("--ratios", type=float, nargs="+", default=[1, 2, 4, 100])
    parser.add_argument("--ratio_tol", type=float, default=1e-3)

    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    config = Config(cfg)

    dataset_config = datasets_map[args.dataset]
    test_dataset = dataset_config["dataset_class"](
        dataset_config["test_source"] if config.eval_subset == "test" else dataset_config["val_source"],
        dataset_config["test_targets"] if config.eval_subset == "test" else dataset_config["val_targets"],
        subset=config.eval_subset,
        half=False,
        preload=config.preload,
        length=config.eval_length,
        params_num=dataset_config["nparams"],
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=max(1, config.batch_size // 2),
        num_workers=config.num_workers,
    )

    # Load model
    model_dir = str(Path(args.model_dir).resolve())
    ckpt_path, hparams_path = pick_latest_checkpoint(model_dir)

    with open(hparams_path, "r") as f:
        hparams = yaml.safe_load(f)
    model_name = hparams["model_type"]

    model = models_map[model_name].load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location="cuda:0",
    ).cuda().eval()

    if int(config.eval_precision) == 16:
        model.half()

    stft = auraloss.freq.STFTLoss()

    display_name = args.display_name if args.display_name else os.path.basename(model_dir)

    out_root = Path(args.out_dir) if args.out_dir else (Path("./eval_heatmaps") / args.dataset)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{display_name}_thr-40_ratio-panels_stft.png"

    ratios = [float(r) for r in args.ratios]

    # ratio -> (atk,rel) -> [stft]
    stft_by_ratio = {r: defaultdict(list) for r in ratios}
    attacks_union = set()
    releases_union = set()

    print(f"Model: {display_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Fixed threshold: {args.thr_db} dB")
    print(f"Ratios: {ratios}")
    print("Evaluating...")

    for _, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        inp, tgt, params = batch
        inp = inp.to("cuda:0")
        tgt = tgt.to("cuda:0")
        params = params.to("cuda:0")

        with torch.no_grad():
            if int(config.eval_precision) == 16:
                with torch.cuda.amp.autocast():
                    out = model(inp, params)
            else:
                out = model(inp, params)

            if isinstance(out, dict):
                out = out["waveform"]

            if model.hparams.causal:
                tgt_crop = causal_crop(tgt, out.shape[-1])
            else:
                tgt_crop = center_crop(tgt, out.shape[-1])

        for o, t, p in zip(
            torch.split(out, 1, dim=0),
            torch.split(tgt_crop, 1, dim=0),
            torch.split(params, 1, dim=0),
        ):
            p_norm = p.squeeze().detach().cpu().numpy()
            thr_db, ratio_val, atk_ms, rel_ms = denorm_params_ms(p_norm)

            if abs(thr_db - args.thr_db) > args.thr_tol:
                continue

            matched = None
            for r in ratios:
                if abs(ratio_val - r) <= args.ratio_tol:
                    matched = r
                    break
            if matched is None:
                continue

            stft_loss = float(stft(o, t).detach().cpu().numpy())
            stft_by_ratio[matched][(atk_ms, rel_ms)].append(stft_loss)

            attacks_union.add(atk_ms)
            releases_union.add(rel_ms)

    attacks = sorted(attacks_union)
    releases = sorted(releases_union)

    if len(attacks) == 0 or len(releases) == 0:
        raise RuntimeError("No samples matched the requested threshold/ratios. Check thr/ratio values and tolerances.")

    # Build grids + global color scale
    grids = {}
    all_vals = []
    for r in ratios:
        grid = build_grid(stft_by_ratio[r], attacks, releases)
        grids[r] = grid
        vals = grid[np.isfinite(grid)]
        if vals.size:
            all_vals.append(vals)

    if not all_vals:
        raise RuntimeError("All grids are empty (no finite values).")

    all_vals = np.concatenate(all_vals)
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()

    cmap = truncate_cmap("viridis", 0.12, 0.70)

    last_im = None
    for ax, r in zip(axes, ratios):
        grid = grids[r]
        last_im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_xticks(range(len(releases)))
        ax.set_xticklabels([f"{x} ms" for x in releases], rotation=45, ha="right")

        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels([f"{y} ms" for y in attacks])

        ax.set_xlabel("Release", fontsize=12)
        ax.set_ylabel("Attack", fontsize=12)
        ax.set_title(f"SDRC-Mamba2-A | Thr = {int(args.thr_db)}db | Ratio = {ratio_label(r)}", fontsize=12)


        if np.isfinite(grid).any():
            annotate_grid(ax, grid, fmt="{:.3f}", fontsize=14)
        else:
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes,
                ha="center", va="center",
                color="white", fontsize=16,
            )

    for k in range(len(ratios), 4):
        axes[k].axis("off")

    cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.95)
    cbar.set_label("STFT loss")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()