"""
This file contains code adapted from the micro-TCN repository.
Repository:
https://github.com/csteinmetz1/micro-tcn

If you use this code or derivatives in academic work, please cite the paper below.

Reference:
Steinmetz, C. J., & Reiss, J. D. (2022).
Efficient neural networks for real-time modeling of analog dynamic range compression.
Proceedings of the 152nd AES Convention.
"""

import os
import torch
import auraloss
import torchaudio
import numpy as np
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

from .utils import center_crop, causal_crop

class Base(pl.LightningModule):
    def __init__(self, 
                    lr = 3e-4,
                    train_loss = "l1+stft",
                    save_dir = None,
                    num_examples = 4,
                    scheduler = "CosineAnnealingLR",
                    **kwargs):
        super(Base, self).__init__()
        self.save_hyperparameters()

        self.l1      = torch.nn.L1Loss()
        self.stft    = auraloss.freq.STFTLoss(output="full")

    def forward(self, x, p):
        pass

    @torch.jit.unused   
    def training_step(self, batch, batch_idx):
        input, target, params = batch

        pred = self(input, params)

        # crop the input and target signals if their shapes are different
        if self.hparams.causal:
            target = causal_crop(target, pred.shape[-1])
        else:
            target = center_crop(target, pred.shape[-1])
    
        if   self.hparams.train_loss == "l1":
            loss = self.l1(pred, target)
        elif self.hparams.train_loss == "stft":
            loss = self.stft(pred, target)
        elif self.hparams.train_loss == "l1+stft":
            l1_loss = self.l1(pred, target)
            stft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = self.stft(pred, target)
            loss = l1_loss + stft_loss

        else:
            raise NotImplementedError(f"Invalid loss fn: {self.hparams.train_loss}")

        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)

        return loss

    @torch.jit.unused
    def validation_step(self, batch, batch_idx):
        input, target, params = batch

        pred = self(input, params)

        if self.hparams.causal:
            input_crop = causal_crop(input, pred.shape[-1])
            target_crop = causal_crop(target, pred.shape[-1])
        else:
            input_crop = center_crop(input, pred.shape[-1])
            target_crop = center_crop(target, pred.shape[-1])

        l1_loss      = self.l1(pred, target_crop)
        stft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = self.stft(pred, target_crop)
        aggregate_loss = l1_loss + stft_loss 

        self.log('val_loss', aggregate_loss)
        self.log('val_loss/L1', l1_loss)
        self.log('val_loss/STFT', stft_loss)
        self.log('val_loss/Sc_Mag', sc_mag_loss)
        self.log('val_loss/Log_Mag', log_mag_loss)
        self.log('val_loss/Lin_Mag', lin_mag_loss)
        self.log('val_loss/Phs_Mag', phs_loss)

        outputs = {
            "input" : input_crop.cpu().numpy(),
            "target": target_crop.cpu().numpy(),
            "pred"  : pred.cpu().numpy(),
            "params" : params.cpu().numpy()}

        return outputs

    @torch.jit.unused
    def validation_epoch_end(self, validation_step_outputs):
        outputs = {
            "input" : [],
            "target" : [],
            "pred" : [],
            "params" : []}

        for out in validation_step_outputs:
            for key, val in out.items():
                bs = val.shape[0]
                for bidx in np.arange(bs):
                    outputs[key].append(val[bidx,...])

        example_indices = np.arange(len(outputs["input"]))
        rand_indices = np.random.choice(example_indices,
                                        replace=False,
                                        size=np.min([len(outputs["input"]), self.hparams.num_examples]))

        for idx, rand_idx in enumerate(list(rand_indices)):
            i = outputs["input"][rand_idx].squeeze()
            t = outputs["target"][rand_idx].squeeze()
            p = outputs["pred"][rand_idx].squeeze()
            prm = outputs["params"][rand_idx].squeeze()

            # self.logger.experiment.add_audio(f"input/{idx}",  
            #                                  i, self.global_step, 
            #                                  sample_rate=self.hparams.sample_rate)
            # self.logger.experiment.add_audio(f"target/{idx}", 
            #                                  t, self.global_step, 
            #                                  sample_rate=self.hparams.sample_rate)
            # self.logger.experiment.add_audio(f"pred/{idx}",   
            #                                  p, self.global_step, 
            #                                  sample_rate=self.hparams.sample_rate)

            if self.hparams.save_dir is not None:
                if not os.path.isdir(self.hparams.save_dir):
                    os.makedirs(self.hparams.save_dir)

                input_filename = os.path.join(self.hparams.save_dir, f"{idx}-input-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")
                target_filename = os.path.join(self.hparams.save_dir, f"{idx}-target-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")

                if not os.path.isfile(input_filename):
                    torchaudio.save(input_filename, 
                                    torch.tensor(i).view(1,-1).float(),
                                    sample_rate=self.hparams.sample_rate)

                if not os.path.isfile(target_filename):
                    torchaudio.save(target_filename,
                                    torch.tensor(t).view(1,-1).float(),
                                    sample_rate=self.hparams.sample_rate)

                torchaudio.save(os.path.join(self.hparams.save_dir, 
                                f"{idx}-pred-{self.hparams.train_loss}-{int(prm[0]):1d}-{prm[1]:0.2f}.wav"), 
                                torch.tensor(p).view(1,-1).float(),
                                sample_rate=self.hparams.sample_rate)

    @torch.jit.unused
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @torch.jit.unused
    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        if self.hparams.scheduler == "CosineAnnealingLR":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=0
            )
            return {"optimizer": optimizer, "lr_scheduler": sched}
        elif self.hparams.scheduler == "ReduceLROnPlateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,
                },
            }
        else:
            raise TypeError("Unsupported scheduler type")
    
    @torch.jit.unused
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """Handle both CosineAnnealingLR and ReduceLROnPlateau correctly."""
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        if isinstance(scheduler, ReduceLROnPlateau):
            if metric is None and getattr(self, "trainer", None) is not None:
                metric = self.trainer.callback_metrics.get('val_loss', None)
            if metric is not None:
                scheduler.step(metric)
        else:
            scheduler.step()