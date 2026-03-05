"""
This file contains model code inspired by the Optical-DRC-with-Selective-SSMs repository.
Repository:
https://github.com/RiccardoVib/Optical-DRC-with-Selective-SSMs

If you use this code or derivatives in academic work, please cite the paper below.

Reference:
Riccardo Simionato, Stefano Fasciani
"Modeling Time-Variant Responses of Optical Compressors with Selective State Space Models."
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import Base
from einops import rearrange, repeat
from typing import Optional as _Optional

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except Exception as _e:
    selective_scan_fn, mamba_inner_fn = None, None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except Exception:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except Exception:
    selective_state_update = None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def forward(self, hidden_states, inference_params: _Optional[object] = None):
        batch, seqlen, dim = hidden_states.shape
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A = -torch.exp(self.A_log.float())
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,
                None,
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            if conv_state is not None:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())
        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
        if initialize_states:
            conv_state.zero_()
            ssm_state.zero_()
        return conv_state, ssm_state


class InferenceParams:
    __slots__ = ("key_value_memory_dict", "seqlen_offset")
    def __init__(self):
        self.key_value_memory_dict = {}
        self.seqlen_offset = 0


class MambaBlockStateful(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 1,
        layer_idx: int = 0,
        use_fast_path: bool = True,
        bias: bool = False,
        conv_bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.core = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        self._ip: _Optional[InferenceParams] = None

    @torch.no_grad()
    def enable_streaming(self, batch_size: int, max_seqlen: int = 1, dtype=None):
        self._ip = InferenceParams()
        conv_state, ssm_state = self.core.allocate_inference_cache(
            batch_size=batch_size, max_seqlen=max_seqlen, dtype=dtype
        )
        self._ip.key_value_memory_dict[self.core.layer_idx] = (conv_state, ssm_state)
        self._ip.seqlen_offset = 0

    @torch.no_grad()
    def reset_streaming(self):
        if self._ip is None:
            return
        conv_state, ssm_state = self._ip.key_value_memory_dict[self.core.layer_idx]
        conv_state.zero_()
        ssm_state.zero_()
        self._ip.seqlen_offset = 0

    @torch.no_grad()
    def disable_streaming(self):
        self._ip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ip is None:
            return self.core(x)
        B, L, D = x.shape
        if L == 1:
            conv_state, ssm_state = self._ip.key_value_memory_dict[self.core.layer_idx]
            y, _, _ = self.core.step(x, conv_state, ssm_state)
            self._ip.seqlen_offset += 1
            return y
        y = self.core(x, inference_params=self._ip)
        self._ip.seqlen_offset += L
        return y


class FiLM(nn.Module):
    def __init__(self, d_x: int, d_c: int):
        super().__init__()
        self.affine = nn.Linear(d_c, 2 * d_x, bias=True)
        self.post = nn.Linear(d_x, 2 * d_x, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gb = self.affine(c)
        gamma, beta = gb.chunk(2, dim=-1)
        kf = gamma * x + beta
        kf1, kf2 = self.post(kf).chunk(2, dim=-1)
        return kf1 * F.softsign(kf2)


class TemporalFiLM(nn.Module):
    def __init__(self, d_x: int, d_c: int, hidden_bias: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_size=d_c, hidden_size=d_x, num_layers=1, batch_first=True, bias=hidden_bias)
        self.affine = nn.Linear(d_x, 2 * d_x, bias=True)
        self.post = nn.Linear(d_x, 2 * d_x, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(c)
        c_mod, d_mod = self.affine(h).chunk(2, dim=-1)
        t1, t2 = self.post(c_mod * x + d_mod).chunk(2, dim=-1)
        return t1 * F.softsign(t2)


class MambaRaw(Base):
    def __init__(
        self,
        nparams: int,
        ninputs: int = 1,
        noutputs: int = 1,
        *,
        window_size: int = 64,
        nfft: int = 128,
        d_model: int = 2,
        mamba_expand: int = 2,
        mamba_dconv: int = 4,
        mamba_states: int = 16,
        stateful: bool = True,
        max_streams: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert ninputs == 1 and noutputs == 1
        assert window_size > 0 and nfft >= window_size
        self.win = window_size
        self.nfft = nfft
        self.d_model = d_model
        self.nparams = nparams
        self._stateful_default = stateful
        self.fc_in = nn.Linear(self.win, self.d_model, bias=True)
        self.mamba1 = MambaBlockStateful(
            d_model=self.d_model, d_state=mamba_states, d_conv=mamba_dconv, expand=mamba_expand, dt_rank=1, layer_idx=0
        )
        self.fc_after_m1 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            nn.GELU(),
        )
        self.freq_conv = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=self.nfft, bias=True)
        self.film = FiLM(d_x=self.d_model, d_c=self.nparams + 2)
        self.tfilm = TemporalFiLM(d_x=self.d_model, d_c=self.nparams + 2)
        self.mamba2 = MambaBlockStateful(
            d_model=self.d_model, d_state=mamba_states, d_conv=mamba_dconv, expand=mamba_expand, dt_rank=1, layer_idx=1
        )
        self.fc_after_m2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            nn.GELU(),
        )
        self.out_head = nn.Linear(self.d_model, 1, bias=True)
        self._streaming_enabled = False

    @torch.no_grad()
    def enable_streaming(self, batch_size: int, dtype=None):
        self.mamba1.enable_streaming(batch_size=batch_size, dtype=dtype)
        self.mamba2.enable_streaming(batch_size=batch_size, dtype=dtype)
        self._streaming_enabled = True

    @torch.no_grad()
    def reset_streaming(self):
        if self._streaming_enabled:
            self.mamba1.reset_streaming()
            self.mamba2.reset_streaming()

    @torch.no_grad()
    def disable_streaming(self):
        if self._streaming_enabled:
            self.mamba1.disable_streaming()
            self.mamba2.disable_streaming()
            self._streaming_enabled = False

    def _build_windows(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, S = x.shape
        assert C == 1
        x_pad = F.pad(x, (self.win - 1, 0))
        win_seq = x_pad.unfold(dimension=-1, size=self.win, step=1).squeeze(1).contiguous()
        x_cur = win_seq[..., -1:].contiguous()
        return win_seq, x_cur

    def _fft_features(self, win_seq: torch.Tensor) -> torch.Tensor:
        B, L, W = win_seq.shape
        if self.nfft > W:
            x_fft = F.pad(win_seq, (0, self.nfft - W))
        else:
            x_fft = win_seq
        X = torch.fft.fft(x_fft, n=self.nfft, dim=-1)
        mag = X.abs()
        mag_ = mag.reshape(B * L, 1, self.nfft)
        feat = self.freq_conv(mag_)
        feat = feat.squeeze(-1).reshape(B, L, 2).contiguous()
        return feat

    def forward(self, x: torch.Tensor, p: Optional[torch.Tensor]) -> torch.Tensor:
        B, C, S = x.shape
        device = x.device
        dtype = x.dtype
        win_seq, x_cur = self._build_windows(x)
        x_proj = self.fc_in(win_seq)
        h = self.mamba1(x_proj)
        h = self.fc_after_m1(h)
        features = self._fft_features(win_seq)
        if p is None:
            p_rep = torch.zeros(B, S, self.nparams, device=device, dtype=dtype)
        else:
            p_rep = p.squeeze(1)
            p_rep = p_rep.unsqueeze(1).expand(B, S, p_rep.shape[-1]).contiguous()
        cond = torch.cat([p_rep, features], dim=-1)
        h = self.film(h, cond)
        h = self.tfilm(h, cond)
        h = self.mamba2(h)
        h = self.fc_after_m2(h)
        g = self.out_head(h)
        y = g * x_cur
        return y.transpose(1, 2).contiguous()