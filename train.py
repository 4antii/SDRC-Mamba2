import os as _os
_DEFAULT_SEED = int(_os.environ.get("SEED", "42"))
_os.environ["PYTHONHASHSEED"] = str(_DEFAULT_SEED)
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import yaml
import glob
import torch
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor

from data import datasets_map
from models import models_map
from utils import Config

parser = ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--dataset', type=str, default='alesis3630')
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
config = Config(config)

seed = int(getattr(config, "seed", _DEFAULT_SEED))

random.seed(seed)
numpy_random_state = np.random.RandomState(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

pl.seed_everything(seed, workers=True)

dl_generator = torch.Generator()
dl_generator.manual_seed(seed)

def _seed_worker(worker_id: int):
    worker_seed = (seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

print(f"Using {args.dataset} Dataset")
dataset_config = datasets_map[args.dataset]

train_dataset = dataset_config['dataset_class'](
    dataset_config['train_source'],
    dataset_config['train_targets'],
    subset=config.train_subset,
    half=True if config.precision == 16 else False,
    preload=config.preload,
    length=config.train_length,
    params_num=dataset_config['nparams']
)

val_dataset = dataset_config['dataset_class'](
    dataset_config['val_source'],
    dataset_config['val_targets'],
    preload=config.preload,
    half=True if config.precision == 16 else False,
    subset=config.val_subset,
    length=config.eval_length,
    params_num=dataset_config['nparams']
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=config.shuffle,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    worker_init_fn=_seed_worker,
    generator=dl_generator,
    drop_last=False
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=8,
    num_workers=config.num_workers,
    pin_memory=True,
    worker_init_fn=_seed_worker,
    generator=dl_generator,
    drop_last=False
)

config.nparams = dataset_config['nparams']

model = models_map[config.model_type](**config.to_dict())

default_root_dir = _os.path.join('experiments', args.dataset, config.exp_name)

trainer = pl.Trainer(
    max_epochs=config.max_epochs,
    precision=config.precision,
    default_root_dir=default_root_dir,
    limit_train_batches=10000,
    limit_val_batches=1000,
    callbacks=[LearningRateMonitor(logging_interval="step")],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    deterministic=False,
)

trainer.fit(model, train_dataloader, val_dataloader)