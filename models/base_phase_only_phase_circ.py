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
import numpy as np
import torch
import torch.nn.functional as F
import auraloss
import torchaudio
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils import center_crop, causal_crop


class Base(pl.LightningModule):
    def __init__(self,
                 lr=3e-4,
                 save_dir=None,
                 num_examples=4,
                 scheduler="CosineAnnealingLR",
                 sample_rate=16000,
                 max_epochs=100,
                 causal=False,
                 n_fft=1024,
                 hop_length=256,
                 win_length=None,
                 window=None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = torch.nn.L1Loss()
        self.stft = auraloss.freq.STFTLoss(output="full")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else self.n_fft
        if window is None:
            window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    @staticmethod
    def _phase_loss(Shat_c: torch.Tensor, S_c: torch.Tensor):
        T = min(Shat_c.shape[-1], S_c.shape[-1])
        dphi = torch.angle(Shat_c[..., :T]) - torch.angle(S_c[..., :T])
        return (1.0 - torch.cos(dphi)).mean()

    def _stft_consistency_loss(self, Shat_c: torch.Tensor):
        n_fft = self.n_fft
        hop = self.hop_length
        win = self.window.to(device=Shat_c.device, dtype=Shat_c.real.dtype)
        y_recon = torch.istft(Shat_c, n_fft=n_fft, hop_length=hop, window=win, center=False)
        X_re = torch.stft(y_recon, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, center=False)
        T = min(X_re.shape[-1], Shat_c.shape[-1])
        return (X_re[..., :T] - Shat_c[..., :T]).abs().mean()

    @staticmethod
    def _framed_rms(x: torch.Tensor, win: int = 1024, hop: int = 256, eps: float = 1e-8):
        frames = x.unfold(dimension=-1, size=win, step=hop)
        rms = torch.sqrt(frames.pow(2).mean(dim=-1) + eps)
        return rms

    def _gain_db_loss(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor, win: int = 1024, hop: int = 256, eps: float = 1e-6):
        rx = self._framed_rms(x, win, hop, eps)
        ry = self._framed_rms(y, win, hop, eps)
        rhy = self._framed_rms(y_hat, win, hop, eps)
        n = min(rx.shape[-1], ry.shape[-1], rhy.shape[-1])
        rx, ry, rhy = rx[..., :n], ry[..., :n], rhy[..., :n]
        g = (ry + eps) / (rx + eps)
        gh = (rhy + eps) / (rx + eps)
        thr = (rx.mean(dim=-1, keepdim=True) * 0.1)
        mask = (rx > thr).float()
        diff_db = (20.0 * torch.log10(gh) - 20.0 * torch.log10(g)).abs()
        num = (mask * diff_db).sum()
        den = mask.sum().clamp_min(1.0)
        return num / den

    def forward(self, x, p):
        raise NotImplementedError

    @torch.jit.unused
    def training_step(self, batch, batch_idx):
        input, target, params = batch
        pred_out = self(input, params)
        if isinstance(pred_out, dict):
            pred = pred_out["waveform"]
            Yhat_c = pred_out["pred_stft"]
        else:
            pred = pred_out
            Yhat_c = pred_out

        if self.hparams.causal:
            target_crop = causal_crop(target, pred.shape[-1])
            input_crop = causal_crop(input, pred.shape[-1])
        else:
            target_crop = center_crop(target, pred.shape[-1])
            input_crop = center_crop(input, pred.shape[-1])

        l1_loss = self.l1(pred, target_crop)
        stft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_mag_loss = self.stft(pred, target_crop)

        n_fft = self.n_fft
        hop = self.hop_length
        win = self.window.to(device=pred.device, dtype=pred.dtype)
        pad_left = n_fft - hop
        tgt_pad = F.pad(target_crop.squeeze(1), (pad_left, 0))
        S_c = torch.stft(tgt_pad, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, center=False)

        phase_loss = self._phase_loss(Yhat_c, S_c)
        stft_cons = self._stft_consistency_loss(Yhat_c)
        gain_db = self._gain_db_loss(input_crop.squeeze(1), pred.squeeze(1), target_crop.squeeze(1), win=1024, hop=256)

        loss = 0.46 * l1_loss + 0.36 * stft_loss + 0.18 * phase_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/L1', l1_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train/STFT', stft_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train/STFT_sc', sc_mag_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train/STFT_logmag', log_mag_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train/STFT_linmag', lin_mag_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train/STFT_phase', phs_mag_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train/PhaseCirc', phase_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train/STFT_consistency', stft_cons, on_step=True, on_epoch=True, logger=True)
        self.log('train/Gain_dB', gain_db, on_step=True, on_epoch=True, logger=True)

        return loss

    @torch.jit.unused
    def validation_step(self, batch, batch_idx):
        input, target, params = batch
        pred_out = self(input, params)
        if isinstance(pred_out, dict):
            pred = pred_out["waveform"]
            Yhat_c = pred_out["pred_stft"]
        else:
            pred = pred_out
            Yhat_c = pred_out

        if self.hparams.causal:
            input_crop = causal_crop(input, pred.shape[-1])
            target_crop = causal_crop(target, pred.shape[-1])
        else:
            input_crop = center_crop(input, pred.shape[-1])
            target_crop = center_crop(target, pred.shape[-1])

        l1_loss = self.l1(pred, target_crop)
        stft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = self.stft(pred, target_crop)
        aggregate_loss = l1_loss + stft_loss

        n_fft = self.n_fft
        hop = self.hop_length
        win = self.window.to(device=pred.device, dtype=pred.dtype)
        pad_left = n_fft - hop
        tgt_pad = F.pad(target_crop.squeeze(1), (pad_left, 0))
        S_c = torch.stft(tgt_pad, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, center=False)
        phase_loss = self._phase_loss(Yhat_c, S_c)
        stft_cons = self._stft_consistency_loss(Yhat_c)
        gain_db = self._gain_db_loss(input_crop.squeeze(1), pred.squeeze(1), target_crop.squeeze(1), win=1024, hop=256)

        self.log('val_loss', aggregate_loss)
        self.log('val_loss/L1', l1_loss)
        self.log('val_loss/STFT', stft_loss)
        self.log('val_loss/Sc_Mag', sc_mag_loss)
        self.log('val_loss/Log_Mag', log_mag_loss)
        self.log('val_loss/Lin_Mag', lin_mag_loss)
        self.log('val_loss/Phs_Mag', phs_loss)
        self.log('val_loss/PhaseCirc', phase_loss)
        self.log('val_loss/STFT_consistency', stft_cons)
        self.log('val_loss/Gain_dB', gain_db)

        outputs = {
            "input": input_crop.cpu().numpy(),
            "target": target_crop.cpu().numpy(),
            "pred": pred.cpu().numpy(),
            "params": params.cpu().numpy()
        }
        return outputs

    @torch.jit.unused
    def validation_epoch_end(self, validation_step_outputs):
        outputs = {"input": [], "target": [], "pred": [], "params": []}
        for out in validation_step_outputs:
            for key, val in out.items():
                bs = val.shape[0]
                for bidx in np.arange(bs):
                    outputs[key].append(val[bidx, ...])

        example_indices = np.arange(len(outputs["input"]))
        rand_indices = np.random.choice(example_indices, replace=False, size=np.min([len(outputs["input"]), self.hparams.num_examples]))

        for idx, rand_idx in enumerate(list(rand_indices)):
            i = outputs["input"][rand_idx].squeeze()
            t = outputs["target"][rand_idx].squeeze()
            p = outputs["pred"][rand_idx].squeeze()
            prm = outputs["params"][rand_idx].squeeze()

            try:
                self.logger.experiment.add_audio(f"input/{idx}", i, self.global_step, sample_rate=self.hparams.sample_rate)
                self.logger.experiment.add_audio(f"target/{idx}", t, self.global_step, sample_rate=self.hparams.sample_rate)
                self.logger.experiment.add_audio(f"pred/{idx}", p, self.global_step, sample_rate=self.hparams.sample_rate)
            except Exception:
                pass

            if self.hparams.save_dir is not None:
                os.makedirs(self.hparams.save_dir, exist_ok=True)
                input_filename = os.path.join(self.hparams.save_dir, f"{idx}-input-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")
                target_filename = os.path.join(self.hparams.save_dir, f"{idx}-target-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")
                if not os.path.isfile(input_filename):
                    torchaudio.save(input_filename, torch.tensor(i).view(1, -1).float(), sample_rate=self.hparams.sample_rate)
                if not os.path.isfile(target_filename):
                    torchaudio.save(target_filename, torch.tensor(t).view(1, -1).float(), sample_rate=self.hparams.sample_rate)
                torchaudio.save(os.path.join(self.hparams.save_dir, f"{idx}-pred-va_complex-{int(prm[0]):1d}-{prm[1]:0.2f}.wav"), torch.tensor(p).view(1, -1).float(), sample_rate=self.hparams.sample_rate)

    @torch.jit.unused
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @torch.jit.unused
    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        if self.hparams.scheduler == "CosineAnnealingLR":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=0)
            return {"optimizer": optimizer, "lr_scheduler": sched}
        elif self.hparams.scheduler == "ReduceLROnPlateau":
            sched = ReduceLROnPlateau(optimizer, patience=10, verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss", "interval": "epoch", "frequency": 1, "strict": False}}
        else:
            return optimizer

    @torch.jit.unused
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if isinstance(scheduler, ReduceLROnPlateau):
            if metric is None and getattr(self, "trainer", None) is not None:
                metric = self.trainer.callback_metrics.get('val_loss', None)
            if metric is not None:
                scheduler.step(metric)
        else:
            scheduler.step()