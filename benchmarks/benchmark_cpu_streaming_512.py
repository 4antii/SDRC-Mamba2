#!/usr/bin/env python3
"""CPU streaming benchmark for SDRC model checkpoints.

The benchmark is intentionally plugin-like:
  - batch size = 1
  - CPU only, single PyTorch intra/inter-op thread by default
  - 512 host samples per process block
  - TCN receives a rolling context buffer matching the micro-tcn plugin:
    [receptive_field - 1 past samples, current 512-sample block]
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


def s6_selective_scan_vectorized(
    conv_inner: torch.Tensor,
    z_seq: torch.Tensor,
    dt: torch.Tensor,
    b_part: torch.Tensor,
    c_part: torch.Tensor,
    a: torch.Tensor,
    d_skip: torch.Tensor,
) -> torch.Tensor:
    """Vectorized CPU fallback for the zero-initial-state S6 selective scan.

    This uses the same cumsum trick as the original TensorFlow implementation,
    avoiding a Python/TorchScript loop over all 512 samples. It is intended for
    block benchmark parity, not for stateful sample-by-sample plugin execution.
    """
    log_da = dt.unsqueeze(-1) * a.view(1, 1, a.shape[0], a.shape[1])
    prefix = torch.cumsum(log_da, dim=1)
    exp_prefix = torch.exp(prefix)
    dbu = dt.unsqueeze(-1) * conv_inner.unsqueeze(-1) * b_part.unsqueeze(2)
    state = torch.cumsum(dbu / (exp_prefix + 1.0e-12), dim=1) * exp_prefix
    y = torch.sum(state * c_part.unsqueeze(2), dim=-1)
    y = y + d_skip.view(1, 1, -1) * conv_inner
    return y * (z_seq * torch.sigmoid(z_seq))


@torch.jit.script
def scripted_s6_selective_scan(
    conv_inner: torch.Tensor,
    z_seq: torch.Tensor,
    dt: torch.Tensor,
    b_part: torch.Tensor,
    c_part: torch.Tensor,
    a: torch.Tensor,
    d_skip: torch.Tensor,
) -> torch.Tensor:
    batch = conv_inner.size(0)
    seqlen = conv_inner.size(1)
    d_inner = conv_inner.size(2)
    d_state = b_part.size(2)
    state = torch.zeros((batch, d_inner, d_state), dtype=conv_inner.dtype, device=conv_inner.device)
    y = torch.empty((batch, seqlen, d_inner), dtype=conv_inner.dtype, device=conv_inner.device)

    for t in range(seqlen):
        dt_t = dt[:, t, :]
        x_t = conv_inner[:, t, :]
        b_t = b_part[:, t, :]
        c_t = c_part[:, t, :]
        d_a = torch.exp(dt_t.unsqueeze(-1) * a.unsqueeze(0))
        d_b = dt_t.unsqueeze(-1) * b_t.unsqueeze(1)
        state = state * d_a + x_t.unsqueeze(-1) * d_b
        y_t = torch.sum(state * c_t.unsqueeze(1), dim=-1)
        y_t = y_t + d_skip.view(1, -1) * x_t
        y_t = y_t * (z_seq[:, t, :] * torch.sigmoid(z_seq[:, t, :]))
        y[:, t, :] = y_t

    return y


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def configure_cpu(num_threads: int) -> None:
    os.environ.setdefault("MAMBA_DISABLE_FUSED", "1")
    os.environ.setdefault("MAMBA_USE_TRITON", "0")
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)


def sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return ""


def cpu_info(num_threads: int) -> dict[str, str]:
    return {
        "cpu_model": sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine(),
        "physical_cores": sysctl("hw.physicalcpu"),
        "logical_cores": sysctl("hw.logicalcpu"),
        "thread_assumption": f"torch_num_threads={num_threads}; interop_threads=1",
        "framework_backend": f"PyTorch {torch.__version__}; CPU; float32",
        "platform": platform.platform(),
    }


def discover_checkpoints(root: Path) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(root / "**" / "*.ckpt"), recursive=True))]


def dataset_from_path(ckpt: Path, root: Path) -> str:
    rel = ckpt.relative_to(root)
    return rel.parts[0] if rel.parts else ""


def model_name_from_path(ckpt: Path, root: Path) -> str:
    rel = ckpt.relative_to(root)
    return rel.parts[1] if len(rel.parts) > 1 else ckpt.stem


def load_models_map() -> dict[str, type]:
    try:
        import lightning.pytorch
    except Exception:
        import pytorch_lightning as pl

        lightning_mod = types.ModuleType("lightning")
        lightning_mod.pytorch = pl
        sys.modules.setdefault("lightning", lightning_mod)
        sys.modules.setdefault("lightning.pytorch", pl)

    pkg = sys.modules.get("models")
    if pkg is None:
        pkg = types.ModuleType("models")
        pkg.__path__ = [str(REPO_ROOT / "models")]
        sys.modules["models"] = pkg

    models_map: dict[str, type] = {}

    def safe_register(model_type: str, module_name: str, class_name: str) -> None:
        try:
            module = importlib.import_module(module_name)
            models_map[model_type] = getattr(module, class_name)
        except Exception as exc:
            print(f"  optional model unavailable: {model_type} ({type(exc).__name__}: {exc})", flush=True)

    safe_register("tcn", "models.tcn.tcn", "TCNModel")
    safe_register("lstm", "models.raw.lstm", "LSTMModel")
    safe_register("gru", "models.raw.gru", "GRUModel")
    safe_register("s4_raw", "models.raw.s4_raw", "S4Model")
    safe_register("mamba_raw", "models.raw.mamba_raw", "MambaRaw")

    return models_map


def instantiate_model(ckpt_path: Path, models_map: dict[str, type]) -> tuple[torch.nn.Module, dict[str, Any], str]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = dict(ckpt.get("hyper_parameters", {}))
    model_type = hparams.get("model_type")
    if str(model_type).startswith("mamba2_"):
        return (
            SdrcMamba2CPUFallback(hparams, ckpt["state_dict"]),
            hparams,
            "generic CPU Mamba2 fallback inferred from checkpoint tensors",
        )
    if model_type not in models_map:
        raise KeyError(f"model_type {model_type!r} is not registered")

    model = models_map[model_type](**hparams)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    note = ""
    if missing or unexpected:
        note = f"load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}"
    model.eval()
    model.to("cpu")
    return model, hparams, note


def output_tensor(y: Any) -> torch.Tensor:
    if isinstance(y, dict):
        return y["waveform"]
    return y


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def layer_norm_affine(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    y = x * silu(gate)
    var = y.pow(2).mean(dim=-1, keepdim=True)
    return y * torch.rsqrt(var + eps) * weight


def dense(sd: dict[str, torch.Tensor], prefix: str, x: torch.Tensor, bias: bool = True) -> torch.Tensor:
    b = sd.get(prefix + ".bias") if bias else None
    return F.linear(x, sd[prefix + ".weight"], b)


class TorchMamba2CPU:
    """Generic CPU Mamba2 step matching the local JUCE implementation.

    It supports the checkpoint tensor layout used by the SDRC Mamba2 STFT models.
    """

    def __init__(self, sd: dict[str, torch.Tensor], prefix: str):
        self.sd = sd
        self.prefix = prefix
        self.in_w = sd[prefix + "in_proj.weight"]
        self.conv_w = sd[prefix + "conv1d.weight"].squeeze(1)
        self.conv_b = sd[prefix + "conv1d.bias"]
        self.dt_bias = sd[prefix + "dt_bias"]
        self.a_log = sd[prefix + "A_log"]
        self.d_skip = sd[prefix + "D"]
        self.norm_w = sd[prefix + "norm.weight"]
        self.out_w = sd[prefix + "out_proj.weight"]

        self.inner_dim = int(self.out_w.shape[1])
        self.conv_dim = int(self.conv_w.shape[0])
        self.n_heads = int(self.dt_bias.numel())
        self.state_dim = int((self.conv_dim - self.inner_dim) // 2)
        self.head_dim = int(self.inner_dim // self.n_heads)
        self.conv_kernel = int(self.conv_w.shape[1])
        self.reset()

    def reset(self) -> None:
        self.conv_state = torch.zeros(self.conv_dim, self.conv_kernel)
        self.ssm_state = torch.zeros(self.n_heads, self.head_dim, self.state_dim)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        zxbcdt = F.linear(x, self.in_w)
        z = zxbcdt[: self.inner_dim]
        xbcdt = zxbcdt[self.inner_dim :]
        xbc_in = xbcdt[: self.conv_dim]
        dt_in = xbcdt[self.conv_dim : self.conv_dim + self.n_heads]

        self.conv_state = torch.roll(self.conv_state, shifts=-1, dims=-1)
        self.conv_state[:, -1] = xbc_in
        conv = silu((self.conv_state * self.conv_w).sum(dim=-1) + self.conv_b)

        x_part = conv[: self.inner_dim].reshape(self.n_heads, self.head_dim)
        b_part = conv[self.inner_dim : self.inner_dim + self.state_dim]
        c_part = conv[self.inner_dim + self.state_dim : self.inner_dim + (2 * self.state_dim)]

        dt = F.softplus(dt_in + self.dt_bias)
        d_a = torch.exp(dt * (-torch.exp(self.a_log)))
        scaled_b = dt[:, None] * b_part[None, :]
        self.ssm_state = self.ssm_state * d_a[:, None, None] + x_part[:, :, None] * scaled_b[:, None, :]

        scan = (self.ssm_state * c_part[None, None, :]).sum(dim=-1)
        scan = scan + self.d_skip[:, None] * x_part
        scan = scan.reshape(self.inner_dim)

        normed = rms_norm_gated(scan, z, self.norm_w)
        return F.linear(normed, self.out_w)


class SdrcMamba2CPUFallback(torch.nn.Module):
    def __init__(self, hparams: dict[str, Any], state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.hparams_dict = hparams
        self.model_type = str(hparams.get("model_type", ""))
        self.sd = {k: v.detach().float().cpu() for k, v in state_dict.items()}
        self.n_fft = int(hparams.get("n_fft", 512))
        self.hop_length = int(hparams.get("hop_length", 256))
        self.nparams = int(hparams.get("nparams", 0))
        self.d_model = int(hparams.get("d_model", self.sd.get("post_norm.weight", torch.empty(0)).numel()))
        self.register_buffer("window", torch.hamming_window(self.n_fft, periodic=True), persistent=False)
        self.reset()

    def reset(self) -> None:
        self._cores: dict[str, TorchMamba2CPU] = {}

    def _core(self, prefix: str) -> TorchMamba2CPU:
        if prefix not in self._cores:
            self._cores[prefix] = TorchMamba2CPU(self.sd, prefix)
        return self._cores[prefix]

    @staticmethod
    def _squeeze_params(p: torch.Tensor | None) -> torch.Tensor | None:
        if p is None:
            return None
        if p.dim() == 4:
            p = p.squeeze(0).squeeze(1)
        elif p.dim() == 3:
            p = p.squeeze(1)
        return p

    def _film(self, prefix: str, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        hidden = silu(dense(self.sd, prefix + "net.0", p))
        gb = dense(self.sd, prefix + "net.2", hidden)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta

    def _block_step(self, prefix: str, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        h = layer_norm_affine(x, self.sd[prefix + "norm.weight"], self.sd[prefix + "norm.bias"])
        h = self._film(prefix + "film.", h, p)
        h = self._core(prefix + "mamba.").step(h)
        return x + h

    def _run_stack(self, h_in: torch.Tensor, p: torch.Tensor, block_prefix: str, depth: int) -> torch.Tensor:
        outs = []
        for t in range(h_in.shape[0]):
            h = h_in[t]
            for layer in range(depth):
                h = self._block_step(f"{block_prefix}.{layer}.", h, p)
            outs.append(h)
        return torch.stack(outs, dim=0)

    def _head(self, prefix: str, h: torch.Tensor) -> torch.Tensor:
        h = dense(self.sd, prefix + ".0", h)
        alpha = self.sd[prefix + ".1.weight"].reshape(-1)
        if alpha.numel() == 1:
            h = torch.where(h >= 0.0, h, h * alpha[0])
        else:
            h = torch.where(h >= 0.0, h, h * alpha)
        return dense(self.sd, prefix + ".2", h)

    def _forward_mag_only(self, mag_lin: torch.Tensor, phase: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        feats = torch.log1p(mag_lin).transpose(0, 1)
        h_in = dense(self.sd, "in_proj", feats)
        h = self._run_stack(h_in, p, "blocks", int(self.hparams_dict.get("depth", 1)))
        h = layer_norm_affine(h, self.sd["post_norm.weight"], self.sd["post_norm.bias"])
        h = h_in + self.sd["stack_gate"] * h
        h = torch.stack([self._film("film_before_head.", frame, p) for frame in h], dim=0)
        logits = self._head("head", h).transpose(0, 1)
        out_scale = self.sd["out_scale"]
        mask = torch.sigmoid(logits) * out_scale
        return torch.polar(mag_lin * mask, phase)

    def _forward_phase_mask(self, mag_lin: torch.Tensor, phase: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        depth = int(self.hparams_dict.get("depth", 1))
        mag_feats = torch.log1p(mag_lin).transpose(0, 1)
        phase_feats = torch.cat([torch.cos(phase).transpose(0, 1), torch.sin(phase).transpose(0, 1)], dim=-1)

        h_in_mag = dense(self.sd, "in_proj_mag", mag_feats)
        h_mag = self._run_stack(h_in_mag, p, "blocks_mag", depth)
        h_mag = layer_norm_affine(h_mag, self.sd["post_norm_mag.weight"], self.sd["post_norm_mag.bias"])
        h_mag = h_in_mag + self.sd["stack_gate_mag"] * h_mag
        h_mag = torch.stack([self._film("film_before_head_mag.", frame, p) for frame in h_mag], dim=0)
        logits_mag = self._head("head_mag", h_mag).transpose(0, 1)
        mask = torch.sigmoid(logits_mag) * self.sd["out_scale_mag"]

        h_in_ph = dense(self.sd, "in_proj_phase", phase_feats)
        h_ph = self._run_stack(h_in_ph, p, "blocks_ph", depth)
        h_ph = layer_norm_affine(h_ph, self.sd["post_norm_ph.weight"], self.sd["post_norm_ph.bias"])
        h_ph = h_in_ph + self.sd["stack_gate_ph"] * h_ph
        h_ph = torch.stack([self._film("film_before_head_ph.", frame, p) for frame in h_ph], dim=0)
        logits_ph = self._head("head_ph", h_ph).transpose(0, 1)
        dphi = torch.pi * torch.tanh(logits_ph)

        return torch.polar(mag_lin * mask, phase + dphi)

    def forward(self, x: torch.Tensor, p: torch.Tensor | None) -> torch.Tensor | dict[str, torch.Tensor]:
        p = self._squeeze_params(p)
        if p is None:
            p = torch.zeros(1, self.nparams, dtype=x.dtype)
        p0 = p[0].float().cpu()
        x1 = x[0, 0].float().cpu()

        # For fair streaming benchmark timings, reset state at each host block.
        self.reset()

        pad_left = self.n_fft - self.hop_length
        x_pad = F.pad(x1, (pad_left, 0))
        x_stft = torch.stft(
            x_pad,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=False,
        )
        mag_lin = torch.abs(x_stft)
        phase = torch.angle(x_stft)

        if self.model_type == "mamba2_base_causal_film":
            y_stft = self._forward_mag_only(mag_lin, phase, p0)
        else:
            y_stft = self._forward_phase_mask(mag_lin, phase, p0)

        recon_len = (y_stft.shape[-1] - 1) * self.hop_length + self.n_fft
        y_full = torch.istft(
            y_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=False,
            length=recon_len,
        )
        y = y_full[pad_left : pad_left + x1.numel()].view(1, 1, -1)
        if self.model_type == "mamba2_base_causal_film":
            return y
        return {"waveform": y, "pred_stft": y_stft.unsqueeze(0), "mix_stft": x_stft.unsqueeze(0)}


def run_mamba_block_stepwise(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run raw Mamba block with its CPU-safe recurrent step path."""
    core = block.core
    batch = x.shape[0]
    conv_state, ssm_state = core.allocate_inference_cache(batch_size=batch, max_seqlen=1, dtype=x.dtype)
    ys = []
    for idx in range(x.shape[1]):
        y, conv_state, ssm_state = core.step(x[:, idx : idx + 1, :], conv_state, ssm_state)
        ys.append(y)
    return torch.cat(ys, dim=1)


def run_mamba_block_scan_cpu(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run raw S6/Mamba block with a full-sequence CPU scan.

    This avoids calling the recurrent `step()` method once per sample. The scan
    itself is still sequential, but all dense/projection/conv work is batched and
    each recurrent step operates on full `(batch, d_inner, d_state)` tensors.
    """
    core = block.core
    batch, seqlen, _ = x.shape

    xz = F.linear(x, core.in_proj.weight, core.in_proj.bias)
    xz = xz.transpose(1, 2).contiguous()
    x_part, z_part = xz.chunk(2, dim=1)

    conv = core.conv1d(x_part)[..., :seqlen]
    conv = core.act(conv).transpose(1, 2).contiguous()

    x_dbl = core.x_proj(conv.reshape(batch * seqlen, core.d_inner))
    dt, b_part, c_part = torch.split(x_dbl, [core.dt_rank, core.d_state, core.d_state], dim=-1)
    dt = F.linear(dt, core.dt_proj.weight, None).reshape(batch, seqlen, core.d_inner)
    dt = F.softplus(dt + core.dt_proj.bias)
    b_part = b_part.reshape(batch, seqlen, core.d_state)
    c_part = c_part.reshape(batch, seqlen, core.d_state)

    a = -torch.exp(core.A_log.float()).to(dtype=x.dtype)
    d_skip = core.D.to(dtype=x.dtype)
    conv_inner = conv
    z_seq = z_part.transpose(1, 2).contiguous()
    y = s6_selective_scan_vectorized(conv_inner, z_seq, dt, b_part, c_part, a, d_skip)
    return core.out_proj(y)


def run_mamba_raw_stepwise(model: torch.nn.Module, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Equivalent to MambaRaw.forward, but avoids unavailable full-sequence scan kernels."""
    batch, _, samples = x.shape
    device = x.device
    dtype = x.dtype

    win_seq, x_cur = model._build_windows(x)
    x_proj = model.fc_in(win_seq)

    h = run_mamba_block_stepwise(model.mamba1, x_proj)
    h = model.fc_after_m1(h)
    features = model._fft_features(win_seq)

    if p is None:
        p_rep = torch.zeros(batch, samples, model.nparams, device=device, dtype=dtype)
    else:
        p_rep = p.squeeze(1)
        p_rep = p_rep.unsqueeze(1).expand(batch, samples, p_rep.shape[-1]).contiguous()

    cond = torch.cat([p_rep, features], dim=-1)
    h = model.film(h, cond)
    h = model.tfilm(h, cond)
    h = run_mamba_block_stepwise(model.mamba2, h)
    h = model.fc_after_m2(h)
    g = model.out_head(h)
    y = g * x_cur
    return y.transpose(1, 2).contiguous()


def run_mamba_raw_scan_cpu(model: torch.nn.Module, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Equivalent to MambaRaw.forward using a CPU selective-scan fallback."""
    batch, _, samples = x.shape
    device = x.device
    dtype = x.dtype

    win_seq, x_cur = model._build_windows(x)
    x_proj = model.fc_in(win_seq)
    h = run_mamba_block_scan_cpu(model.mamba1, x_proj)
    h = model.fc_after_m1(h)
    features = model._fft_features(win_seq)

    if p is None:
        p_rep = torch.zeros(batch, samples, model.nparams, device=device, dtype=dtype)
    else:
        p_rep = p.squeeze(1)
        p_rep = p_rep.unsqueeze(1).expand(batch, samples, p_rep.shape[-1]).contiguous()

    cond = torch.cat([p_rep, features], dim=-1)
    h = model.film(h, cond)
    h = model.tfilm(h, cond)
    h = run_mamba_block_scan_cpu(model.mamba2, h)
    h = model.fc_after_m2(h)
    g = model.out_head(h)
    y = g * x_cur
    return y.transpose(1, 2).contiguous()


def build_params(nparams: int) -> torch.Tensor:
    if nparams <= 0:
        return torch.empty(1, 1, 0)
    vals = torch.full((1, 1, nparams), 0.5, dtype=torch.float32)
    if nparams >= 1:
        vals[..., 0] = 0.25
    if nparams >= 2:
        vals[..., 1] = 0.5
    if nparams >= 3:
        vals[..., 2] = 0.5
    if nparams >= 4:
        vals[..., 3] = 0.5
    return vals


def parameter_count(model: torch.nn.Module) -> int:
    if hasattr(model, "sd"):
        return int(sum(v.numel() for v in getattr(model, "sd").values()))
    return int(sum(p.numel() for p in model.parameters()))


def infer_latency_and_context(model: torch.nn.Module, hparams: dict[str, Any], model_type: str, block: int) -> tuple[int, int, str]:
    if model_type == "tcn":
        rf = int(model.compute_receptive_field()) if hasattr(model, "compute_receptive_field") else 0
        return 0, max(0, rf - 1), "tcn_micro_plugin_context"

    if "mamba2_" in model_type:
        n_fft = int(hparams.get("n_fft", getattr(model, "n_fft", 512)))
        return n_fft, max(0, n_fft - int(hparams.get("hop_length", 256))), "stft_causal_pad"

    if model_type == "mamba_raw":
        win = int(hparams.get("window_size", getattr(model, "win", 1)))
        return 0, max(0, win - 1), "raw_mamba_window_context"

    return 0, 0, "causal_or_block_raw"


def make_runner(
    model: torch.nn.Module,
    hparams: dict[str, Any],
    block_samples: int,
) -> tuple[Callable[[], torch.Tensor], int, int, int, str]:
    model_type = str(hparams.get("model_type", ""))
    nparams = int(hparams.get("nparams", 0))
    params = build_params(nparams)

    latency_samples, context_samples, mode = infer_latency_and_context(model, hparams, model_type, block_samples)
    input_samples = block_samples + context_samples if model_type == "tcn" else block_samples

    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    x = torch.randn(1, 1, input_samples, generator=gen, dtype=torch.float32) * 0.1

    def run_once() -> torch.Tensor:
        if model_type == "mamba_raw":
            y = run_mamba_raw_scan_cpu(model, x, params)
        else:
            y = output_tensor(model(x, params))
        if model_type == "tcn":
            y = y[..., -block_samples:]
        return y

    with torch.inference_mode():
        y0 = run_once()
    output_samples = int(y0.shape[-1])
    return run_once, input_samples, output_samples, latency_samples, mode


def measure(run_once: Callable[[], torch.Tensor], warmup: int, iterations: int) -> tuple[list[float], tuple[int, ...]]:
    times_ms: list[float] = []
    shape: tuple[int, ...] = ()
    with torch.inference_mode():
        for _ in range(warmup):
            y = run_once()
            shape = tuple(y.shape)
        for _ in range(iterations):
            start = time.perf_counter()
            y = run_once()
            elapsed = (time.perf_counter() - start) * 1000.0
            shape = tuple(y.shape)
            times_ms.append(elapsed)
    return times_ms, shape


def benchmark_checkpoint(
    ckpt_path: Path,
    root: Path,
    models_map: dict[str, type],
    block_samples: int,
    sample_rate_override: int | None,
    warmup: int,
    iterations: int,
    num_threads: int,
    info: dict[str, str],
) -> dict[str, Any]:
    dataset = dataset_from_path(ckpt_path, root)
    model_name = model_name_from_path(ckpt_path, root)
    base: dict[str, Any] = {
        "dataset": dataset,
        "model_name": model_name,
        "checkpoint": str(ckpt_path),
        "block_samples": block_samples,
        "batch_size": 1,
        "warmup": warmup,
        "iterations": iterations,
        **info,
    }

    hparams_preview: dict[str, Any] = {}
    try:
        hparams_preview = dict(torch.load(ckpt_path, map_location="cpu").get("hyper_parameters", {}))
        base["model_type"] = hparams_preview.get("model_type", "")
        base["sample_rate"] = int(sample_rate_override or hparams_preview.get("sample_rate", 44100))
    except Exception:
        pass

    try:
        model, hparams, load_note = instantiate_model(ckpt_path, models_map)
        model_type = str(hparams.get("model_type", ""))
        sample_rate = int(sample_rate_override or hparams.get("sample_rate", 44100))
        run_once, input_samples, output_samples, latency_samples, mode = make_runner(model, hparams, block_samples)
        times_ms, out_shape = measure(run_once, warmup, iterations)

        mean_ms = statistics.fmean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        audio_ms = (output_samples / sample_rate) * 1000.0
        realtime_factor = audio_ms / mean_ms if mean_ms > 0 else float("inf")

        return {
            **base,
            "status": "ok",
            "error": "",
            "model_type": model_type,
            "parameter_count": parameter_count(model),
            "sample_rate": sample_rate,
            "input_samples_per_inference": input_samples,
            "output_samples_per_inference": output_samples,
            "output_shape": "x".join(str(v) for v in out_shape),
            "latency_samples": latency_samples,
            "latency_ms": latency_samples / sample_rate * 1000.0,
            "context_samples": max(0, input_samples - block_samples),
            "processing_mode": mode,
            "mean_block_ms": mean_ms,
            "std_block_ms": std_ms,
            "min_block_ms": min(times_ms),
            "max_block_ms": max(times_ms),
            "audio_ms_per_block": audio_ms,
            "real_time_factor": realtime_factor,
            "single_core_cpu_percent": 100.0 / realtime_factor if realtime_factor > 0 else float("inf"),
            "load_note": load_note,
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=4),
        }


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing model checkpoints")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "benchmarks" / "cpu_streaming_512_results.csv")
    parser.add_argument("--block-samples", type=int, default=512)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--only", type=str, default="", help="Substring filter for checkpoint paths")
    args = parser.parse_args()

    configure_cpu(args.threads)
    info = cpu_info(args.threads)
    models_map = load_models_map()
    checkpoints = discover_checkpoints(args.root)
    if args.only:
        checkpoints = [p for p in checkpoints if args.only in str(p)]

    rows: list[dict[str, Any]] = []
    for idx, ckpt in enumerate(checkpoints, start=1):
        rel = ckpt.relative_to(args.root)
        print(f"[{idx:02d}/{len(checkpoints):02d}] {rel}", flush=True)
        row = benchmark_checkpoint(
            ckpt,
            args.root,
            models_map,
            args.block_samples,
            args.sample_rate,
            args.warmup,
            args.iterations,
            args.threads,
            info,
        )
        rows.append(row)
        if row.get("status") == "ok":
            print(
                f"  ok {row['model_type']} mean={row['mean_block_ms']:.3f} ms "
                f"rtf={row['real_time_factor']:.2f} cpu={row['single_core_cpu_percent']:.1f}%",
                flush=True,
            )
        else:
            print(f"  error {row.get('error')}", flush=True)
        write_csv(rows, args.output)

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
