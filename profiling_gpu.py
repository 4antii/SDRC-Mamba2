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

os.environ["MAMBA_DISABLE_FUSED"] = "1"
os.environ["MAMBA_USE_TRITON"] = "0"

from data import datasets_map
from models import models_map
from utils import Config

from models.utils import causal_crop, center_crop

pl.seed_everything(42)

os.environ["OMP_NUM_THREADS"]       = "1"
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
]

def spectral_flux(y_true, y_pred, sr, w=48000):
    # print(w)
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

    i = torch.rand(1,1,65536).to("cuda")
    p = torch.rand(1,1,4).to("cuda")
    model.to("cuda").eval()
    macs, params = profile(model, inputs=(i, p))

    print(f" {idx+1}/{len(models)} : epoch: {epoch} {os.path.basename(model_dir)}")
    print(   f"MACs: {macs/10**9:0.2f} G     Params: {params/1e3:0.2f} k")

    iterations = 10000

    with torch.no_grad():
        # warm‑up
        for _ in range(10):
            _ = model(i, p)

        # benchmark
        start = time.perf_counter()
        for _ in tqdm(range(iterations)):
            _ = model(i, p)
        end = time.perf_counter()

    avg_time = (end - start) / iterations
    audio_duration = 65536 / sr
    rt_factor = avg_time / audio_duration

    print(f"Average inference time: {avg_time:.6f}s")
    print(f"Realtime factor: {rt_factor:.6f}")
    print(f"Inverse Realtime factor: {1 / rt_factor:.6f}")