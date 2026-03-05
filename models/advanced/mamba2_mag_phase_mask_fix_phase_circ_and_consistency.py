import math
import torch
from torch import nn
from torch.nn import functional as Fn
from ..base_phase_phase_circ_and_consistency import Base

try:
    from mamba_ssm import Mamba2 as MambaLayer
except ImportError as e:
    raise ImportError("Mamba-2 not found. Install with `pip install mamba-ssm`.") from e


class FiLM(nn.Module):
    def __init__(self, p_dim: int, D: int, hidden: int = 256):
        super().__init__()
        self.p_dim = int(p_dim)
        self.D = int(D)
        if self.p_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(self.p_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2 * self.D),
            )
        else:
            self.net = None

    def forward(self, x: torch.Tensor, p: torch.Tensor):
        if self.net is None or p is None:
            return x
        gb = self.net(p)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta  = beta.unsqueeze(1)
        return x * (1.0 + gamma) + beta


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        p_dim: int,
        film_hidden: int = 256,
        m2_ngroups=None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.film = FiLM(p_dim=p_dim, D=d_model, hidden=film_hidden)
        self.dropout = nn.Dropout(dropout)
        kwargs = {}
        if m2_ngroups is not None:
            kwargs["ngroups"] = m2_ngroups
        self.mamba = MambaLayer(d_model=d_model, **kwargs)

    def forward(self, x, p):
        h = self.norm(x)
        h = self.film(h, p)
        h = self.mamba(h)
        h = self.dropout(h)
        return x + h


class Mamba2STFTCausalFilmPhaseMaskFixed(Base):
    def __init__(
        self,
        nparams: int,
        n_fft: int = 1024,
        hop_length: int = 512,
        d_model: int = 1024,
        depth: int = 2,
        m2_ngroups=None,
        dropout: float = 0.00,
        mlp_hidden: int = 512,
        out_scale_init: float = 2.0,
        use_log_mag: bool = True,
        film_hidden: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hamming_window(n_fft, periodic=True), persistent=False)

        self.F = self.n_fft // 2 + 1
        self.P = int(nparams)
        self.d_model = int(d_model)
        self.use_log_mag = bool(use_log_mag)

        # Inputs: mag -> F, phase (cos/sin) -> 2F
        self.in_proj_mag   = nn.Linear(self.F,        self.d_model)
        self.in_proj_phase = nn.Linear(2 * self.F,    self.d_model)

        # Magnitude branch
        self.blocks_mag = nn.ModuleList([
            MambaBlock(d_model=self.d_model, p_dim=self.P, film_hidden=film_hidden,
                       m2_ngroups=m2_ngroups, dropout=dropout)
            for _ in range(depth)
        ])
        self.post_norm_mag = nn.LayerNorm(self.d_model)
        self.stack_gate_mag = nn.Parameter(torch.tensor(1.0))
        self.film_before_head_mag = FiLM(p_dim=self.P, D=self.d_model, hidden=film_hidden)
        self.head_mag = nn.Sequential(
            nn.Linear(self.d_model, mlp_hidden),
            nn.PReLU(),
            nn.Linear(mlp_hidden, self.F),
        )
        self.out_scale_mag = nn.Parameter(torch.tensor(float(out_scale_init)))

        # Phase branch
        self.blocks_ph = nn.ModuleList([
            MambaBlock(d_model=self.d_model, p_dim=self.P, film_hidden=film_hidden,
                       m2_ngroups=m2_ngroups, dropout=dropout)
            for _ in range(depth)
        ])
        self.post_norm_ph = nn.LayerNorm(self.d_model)
        self.stack_gate_ph = nn.Parameter(torch.tensor(1.0))
        self.film_before_head_ph = FiLM(p_dim=self.P, D=self.d_model, hidden=film_hidden)
        self.head_ph = nn.Sequential(
            nn.Linear(self.d_model, mlp_hidden),
            nn.PReLU(),
            nn.Linear(mlp_hidden, self.F),
        )

        with torch.no_grad():
            self.head_mag[-1].weight.zero_()
            self.head_mag[-1].bias.zero_()
            self.head_ph[-1].weight.zero_()
            self.head_ph[-1].bias.zero_()

    @staticmethod
    def _squeeze_params(p):
        if p is None:
            return None
        if p.dim() == 4:
            p = p.squeeze(0).squeeze(1)
        elif p.dim() == 3:
            p = p.squeeze(1)
        return p

    def forward(self, x, p):
        p = self._squeeze_params(p)
        x = x.squeeze(1)

        pad_left = self.n_fft - self.hop_length
        if pad_left < 0:
            raise ValueError(f"hop_length ({self.hop_length}) must be <= n_fft ({self.n_fft}).")
        x_pad = Fn.pad(x, (pad_left, 0))
        window = self.window.to(device=x.device, dtype=x.dtype)

        X = torch.stft(
            x_pad,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=False,
        )

        mag_lin = torch.abs(X)
        mag = torch.log1p(mag_lin) if self.use_log_mag else mag_lin
        phase = torch.angle(X)
        B, F, TT = mag.shape

        mag_feats = mag.transpose(1, 2)

        cosφ = torch.cos(phase).transpose(1, 2)
        sinφ = torch.sin(phase).transpose(1, 2)
        phase_feats = torch.cat([cosφ, sinφ], dim=-1)

        h_in_mag   = self.in_proj_mag(mag_feats)
        h_in_phase = self.in_proj_phase(phase_feats)

        h_mag = h_in_mag
        for blk in self.blocks_mag:
            h_mag = blk(h_mag, p)
        h_mag = self.post_norm_mag(h_mag)
        h_mag = h_in_mag + self.stack_gate_mag * h_mag
        h_mag = self.film_before_head_mag(h_mag, p)
        logits_mag = self.head_mag(h_mag)
        logits_mag = logits_mag.transpose(1, 2)
        mask = torch.sigmoid(logits_mag) * self.out_scale_mag

        h_ph = h_in_phase
        for blk in self.blocks_ph:
            h_ph = blk(h_ph, p)
        h_ph = self.post_norm_ph(h_ph)
        h_ph = h_in_phase + self.stack_gate_ph * h_ph
        h_ph = self.film_before_head_ph(h_ph, p)
        logits_ph = self.head_ph(h_ph)
        logits_ph = logits_ph.transpose(1, 2)
        dphi = math.pi * torch.tanh(logits_ph)

        mag_hat = mag_lin * mask
        phi_hat = phase + dphi
        Y_real = mag_hat * torch.cos(phi_hat)
        Y_imag = mag_hat * torch.sin(phi_hat)
        Y_hat  = torch.complex(Y_real, Y_imag)

        recon_len = (TT - 1) * self.hop_length + self.n_fft
        y_full = torch.istft(
            Y_hat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=False,
            length=recon_len,
        )

        y = y_full[:, pad_left:pad_left + x.size(1)]
        return {
            "waveform": y.unsqueeze(1),
            "pred_stft": Y_hat,
            "mix_stft": X
        }