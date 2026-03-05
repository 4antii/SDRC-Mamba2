import os
import sys
import time
import yaml
import glob
import json
import yaml
import torch
import pickle
import librosa
import auraloss
import torchaudio
import numpy as np
import torchsummary
from pathlib import Path
from tqdm import tqdm
from thop import profile
import pyloudnorm as pyln
import pytorch_lightning as pl
from argparse import ArgumentParser

from data import datasets_map
from models import models_map
from utils import Config


from models.utils import causal_crop, center_crop

pl.seed_everything(42)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

parser = ArgumentParser()
parser.add_argument('--config_path', type=str, help='path to model config', required=True)
parser.add_argument('--dataset', type=str, default='alesis3630')

args = parser.parse_args()

with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

config = Config(config)

print(f"Using {args.dataset} Dataset")
dataset_config = datasets_map[args.dataset]

test_dataset = dataset_config['dataset_class'](dataset_config['test_source'] if config.eval_subset == 'test' else dataset_config['val_source'], 
                                dataset_config['test_targets'] if config.eval_subset == 'test' else dataset_config['val_targets'],
                                subset=config.eval_subset,
                                half=False,
                                preload=config.preload,
                                length=config.eval_length,
                                params_num=dataset_config['nparams'])


test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                               shuffle=False,
                                               batch_size=config.batch_size // 2,
                                               num_workers=config.num_workers)

overall_results = {}

if config.eval_audo_save_dir is not None:
    if not os.path.isdir(config.eval_audo_save_dir):
        os.makedirs(config.eval_audo_save_dir)

if args.dataset == 'cl1b':
    sr = 48000
else: 
    sr = 44100

l1   = torch.nn.L1Loss()
mse = torch.nn.MSELoss()
stft = auraloss.freq.STFTLoss()
meter = pyln.Meter(sr)

default_root_dir = Path(f'./experiments/{args.dataset}/').resolve()

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

def spectral_flux(y_true, y_pred, sr, w=48000):
    step = w // 4
    y_true = y_true.squeeze().detach().cpu().numpy()
    y_pred = y_pred.squeeze().detach().cpu().numpy()
    T = min(len(y_true), len(y_pred))
    vals = []
    for i in range(0, T - w, step):
        yt = librosa.onset.onset_strength(y=y_true[i:i+w], sr=sr)
        yp = librosa.onset.onset_strength(y=y_pred[i:i+w], sr=sr)
        vals.append(np.mean(np.abs(yt - yp)))
    return float(np.mean(vals)) if vals else 0.0

for idx, model_dir in enumerate(models):

    results = {}

    model_dir = str(Path(model_dir).resolve())

    checkpoint_path = glob.glob(os.path.join(model_dir,
                                             "lightning_logs",
                                             "version_0",
                                             "checkpoints",
                                             "*"
                                            ))[0]
    
    hparams_file = os.path.join(model_dir, 
                                "lightning_logs", 
                                "version_0", 
                                "hparams.yaml")
    
    with open(hparams_file, 'r') as f:
        hparams = yaml.safe_load(f)
    
    model_name = hparams["model_type"]
    model_id = os.path.basename(model_dir)
    epoch = int(os.path.basename(checkpoint_path).split('-')[0].split('=')[-1])

    print('checkpoint path: ', checkpoint_path)

    model = models_map[model_name].load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location="cuda:0"
            )

    model.cuda()
    model.eval()

    print('Precision: ', config.eval_precision)

    if config.eval_precision == 16:
        model.half()

    print("Evaluating the model...")

    for bidx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        input, target, params = batch

        input = input.to("cuda:0")
        target = target.to("cuda:0")
        params = params.to("cuda:0")

        with torch.no_grad():
            if config.eval_precision == 16:
                with torch.cuda.amp.autocast():
                   output = model(input, params) 
            else:
                output = model(input, params)

            
            if isinstance(output, dict):
                output = output["waveform"]

            if model.hparams.causal:
                input_crop = causal_crop(input, output.shape[-1])
                target_crop = causal_crop(target, output.shape[-1])
            else:
                input_crop = center_crop(input, output.shape[-1])
                target_crop = center_crop(target, output.shape[-1])


        for idx, (i, o, t, p) in enumerate(zip(
                                            torch.split(input_crop, 1, dim=0),
                                            torch.split(output, 1, dim=0),
                                            torch.split(target_crop, 1, dim=0),
                                            torch.split(params, 1, dim=0))):

            l1_loss = l1(o, t).cpu().numpy()
            stft_loss = stft(o, t).cpu().numpy()
            rms_loss = torch.sqrt(mse(o, t)).cpu().numpy()
            aggregate_loss = l1_loss + stft_loss 

            t_np = t.squeeze().cpu().numpy()
            o_np = o.squeeze().cpu().numpy()

            target_lufs = meter.integrated_loudness(t_np)
            output_lufs = meter.integrated_loudness(o_np)

            flux_loss = spectral_flux(t, o, sr=sr, w=sr)

            if (target_lufs is None) or (output_lufs is None) or (not np.isfinite(target_lufs)) or (not np.isfinite(output_lufs)):
                continue
            
            l1_lufs = np.abs(output_lufs - target_lufs)

            l1i_loss = (l1(i, t) - l1(o, t)).cpu().numpy()
            stfti_loss = (stft(i, t) - stft(o, t)).cpu().numpy()

            params = p.squeeze().cpu().numpy()
            if args.dataset == "la2a":
                params_key = f"{params[0]}-{params[1]}"
            else:
                params_key = f"{params[0]}-{params[1]}-{params[2]}-{params[3]}"

            if params_key not in list(results.keys()):
                results[params_key] = {
                    "L1" : [l1_loss],
                    "L1i" : [l1i_loss],
                    "STFT" : [stft_loss],
                    "STFTi" : [stfti_loss],
                    "RMS": [rms_loss],
                    "LUFS" : [l1_lufs],
                    "Flux": [flux_loss],
                    "Agg" : [aggregate_loss]
                }
            else:
                results[params_key]["L1"].append(l1_loss)
                results[params_key]["L1i"].append(l1i_loss)
                results[params_key]["STFT"].append(stft_loss)
                results[params_key]["STFTi"].append(stfti_loss)
                results[params_key]["RMS"].append(rms_loss)
                results[params_key]["LUFS"].append(l1_lufs)
                results[params_key]["Flux"].append(flux_loss)
                results[params_key]["Agg"].append(aggregate_loss)

    l1_scores = []
    lufs_scores = []
    rms_scores = []
    stft_scores = []
    flux_scores = []
    agg_scores = []
    print("-" * 64)
    print("Config         L1         STFT      RMS       LUFS      Flux")
    print("-" * 64)
    for key, val in results.items():
        l1_scores += val["L1"]
        stft_scores += val["STFT"]
        rms_scores += val["RMS"]
        lufs_scores += val["LUFS"]
        flux_scores += val["Flux"]
        agg_scores += val["Agg"]

    print(f"Mean Error {np.mean(l1_scores):0.2e}    {np.mean(stft_scores):0.3f}    {np.mean(rms_scores):0.4f}      {np.mean(lufs_scores):0.3f}      {np.mean(flux_scores):0.3f}")
    overall_results[model_id] = results