#!/usr/bin/env python
import os
import sys
import glob
import yaml
import torch
import torchaudio
from pathlib import Path
from argparse import ArgumentParser

from data import datasets_map
from models import models_map
from utils import Config

from models.utils import causal_crop, center_crop

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

torchaudio.set_audio_backend("sox_io")

INFER_ITEMS = {
    "cl1b": [
        {
            "wav": "your_wav",
            # [threshold, ratio, attack, release]
            "params": [0.25, 0.0, 0.50, 0.0],
        },
        {
            "wav": "your_wav",
            "params": [0.5, 0.0, 1.00, 0.5],
        },
        {
            "wav": "your_wav",
            "params": [0.25, 0.50, 0.50, 0.00],
        },        
    ],
    "la2a": [
        {
            "wav": "your_wav",
            # [peak_reduction, gain]
            "params": [0.0, 65.0],
        },
        {
            "wav": "your_wav",
            "params": [1.0, 65.0],
        },
        {
            "wav": "your_wav",
            "params": [1.0, 80.0],
        },
    ],
    "alesis3630": [
        {
            "wav": "your_wav",
            # [threshold, ratio, attack, release]
            "params": [-20.0, 4.0, 50.0, 500.0],
        },
        {
            "wav": "your_wav",
            "params": [-40.0, 2.0, 0.1, 3000.0],
        },
        {
            "wav": "your_wav",
            "params": [-20.0, 4.0, 200.0, 500.0],
        },
    ],
}

def make_param_tensor(raw_params, dataset_cls_name, device, dtype):
    params = torch.tensor(raw_params, dtype=dtype, device=device).unsqueeze(0)

    if dataset_cls_name == "SignalTrainLA2ADataset":
        if params.shape[-1] != 2:
            raise ValueError("LA2A expects 2 params (peak_reduction, gain).")
        params[:, 1] /= 100.0

    elif dataset_cls_name == "VCADataset":
        if params.shape[-1] != 4:
            raise ValueError("VCADataset expects 4 params [th, rt, at, rl].")
        params[:, 0] /= 100.0
        params[:, 1] /= 10.0
        params[:, 2] /= 1000.0
        params[:, 3] /= 1000.0

    elif dataset_cls_name == "CL1BDataset":
        pass

    params = params.unsqueeze(1)

    return params

def infer_file(
    model,
    wav_path: str,
    raw_params,
    dataset_cls_name: str,
    chunk_length: int,
    sr_expected: int,
    device: torch.device,
    precision: int,
    out_dir: Path,
    model_id: str,
):
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    # Load audio
    audio, sr = torchaudio.load(str(wav_path), normalize=True)
    if sr != sr_expected:
        audio = torchaudio.functional.resample(audio, sr, sr_expected)
        sr = sr_expected

    C, T = audio.shape
    dtype = torch.float16 if precision == 16 else torch.float32

    # Param tensor (1, P), scaled like dataset
    params = make_param_tensor(raw_params, dataset_cls_name, device=device, dtype=dtype)

    num_chunks = T // chunk_length
    if num_chunks == 0:
        num_chunks = 1  # handle very short files

    out_channels = []

    use_amp = (precision == 16 and device.type == "cuda")

    for ch_idx in range(C):
        ch = audio[ch_idx]
        chunks_out = []

        offset = 0
        while offset < T:
            end = min(offset + chunk_length, T)
            x_seg = ch[offset:end]
            seg_len = x_seg.numel()
            if seg_len == 0:
                break

            if seg_len < chunk_length:
                pad = chunk_length - seg_len
                x_seg = torch.nn.functional.pad(x_seg, (0, pad))

            x_seg = x_seg.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)

            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        y = model(x_seg, params)
                        if y.size()[-1] != x_seg.size()[-1]:
                            delta = x_seg.size()[-1] - y.size()[-1]
                            x_seg = torch.nn.functional.pad(x_seg, (delta, 0))
                            y = model(x_seg, params)
                else:               
                    y = model(x_seg, params)
                    if not isinstance(y, dict) and y.size()[-1] != x_seg.size()[-1]:
                        delta = x_seg.size()[-1] - y.size()[-1]
                        x_seg = torch.nn.functional.pad(x_seg, (delta, 0))
                        y = model(x_seg, params)

            if isinstance(y, dict):
                y = y["waveform"]

            y = y.squeeze(0).squeeze(0).cpu()

            y = y[:seg_len]

            chunks_out.append(y)
            offset += chunk_length

        ch_out = torch.cat(chunks_out, dim=-1) if len(chunks_out) > 0 else torch.zeros(0)
        out_channels.append(ch_out)

    max_len = max(ch.size(-1) for ch in out_channels)
    padded = []
    for ch in out_channels:
        if ch.size(-1) < max_len:
            pad = max_len - ch.size(-1)
            ch = torch.nn.functional.pad(ch, (0, pad))
        padded.append(ch.unsqueeze(0))
    out_audio = torch.cat(padded, dim=0)

    out_dir.mkdir(parents=True, exist_ok=True)

    params_tag = "_".join(f"{p}" for p in raw_params)
    out_name = f"{wav_path.stem}__p_{params_tag}__{model_id}.wav"
    out_path = out_dir / out_name

    torchaudio.save(str(out_path), out_audio.float(), sr)
    print(f"Saved: {out_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to model config (YAML)")
    parser.add_argument("--dataset", type=str, default="alesis3630")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    config = Config(cfg_dict)

    dataset_config = datasets_map[args.dataset]
    dataset_cls = dataset_config["dataset_class"]
    dataset_cls_name = dataset_cls.__name__

    if args.dataset == "cl1b":
        sr = 48000
    else:
        sr = 44100

    chunk_length = int(getattr(config, "eval_length", 16384))

    precision = int(getattr(config, "eval_precision", 32))
    print(f"Using dataset: {args.dataset} ({dataset_cls_name})")
    print(f"Sample rate: {sr}  |  chunk length: {chunk_length}  |  precision: {precision}")

    models = [
        f"./experiments/{args.dataset}/lstm_raw_32_release",
        f"./experiments/{args.dataset}/uTCN-300_4-10-13_release",
        f"./experiments/{args.dataset}/uTCN-100_4-10-5_release",
        f"./experiments/{args.dataset}/s4_c32_f4_release",
        f"./experiments/{args.dataset}/s6_16d_release", 
        f"./experiments/{args.dataset}/mamba2_mag_mask_release",
        f"./experiments/{args.dataset}/mamba2_mag_phase_mask_release",
        
        # Ablation
        f"./experiments/{args.dataset}/mamba2_phase_mask_film_fix_no_add_losses",
        f"./experiments/{args.dataset}/mamba2_phase_mask_film_fix_only_phase_circ",
        f"./experiments/{args.dataset}/mamba2_mag_phase_mask_fix_phase_circ_and_consistency",
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset not in INFER_ITEMS:
        raise KeyError(
            f"No hard-coded inference items for dataset '{args.dataset}'. "
            f"Add them to INFER_ITEMS at the top of this script."
        )
    infer_items = INFER_ITEMS[args.dataset]

    for model_dir in models:
        model_dir = Path(model_dir).resolve()
        if not model_dir.exists():
            print(f"Skipping missing model dir: {model_dir}")
            continue

        ckpts = glob.glob(str(model_dir / "lightning_logs" / "version_0" / "checkpoints" / "*"))
        if not ckpts:
            print(f"No checkpoints found in {model_dir}, skipping.")
            continue
        checkpoint_path = ckpts[0]

        hparams_file = model_dir / "lightning_logs" / "version_0" / "hparams.yaml"
        if not hparams_file.exists():
            raise FileNotFoundError(f"hparams.yaml not found at {hparams_file}")

        with open(hparams_file, "r") as f:
            hparams = yaml.safe_load(f)

        model_name = hparams["model_type"]
        model_id = model_dir.name

        print("-" * 80)
        print(f"Model dir: {model_dir}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"model_type: {model_name}")
        print("-" * 80)

        model = models_map[model_name].load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            map_location=device,
        )
        model.to(device)
        model.eval()

        if precision == 16:
            model.half()

        print("Precision: ", precision)
        print(sr)

        out_dir = Path("./experiments") / args.dataset / "inference" / model_id

        for item in infer_items:
            wav = item["wav"]
            raw_params = item["params"]
            print(f"Infer: {wav}  |  params={raw_params}")
            infer_file(
                model=model,
                wav_path=wav,
                raw_params=raw_params,
                dataset_cls_name=dataset_cls_name,
                chunk_length=chunk_length,
                sr_expected=sr,
                device=device,
                precision=precision,
                out_dir=out_dir,
                model_id=model_id,
            )


if __name__ == "__main__":
    main()