import torch
from torch import nn
from torch.nn import functional as Fn
from ..base import Base

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


class Mamba2STFTCausalFilm(Base):
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

        if self.d_model == self.F:
            self.in_proj = nn.Identity()
        else:
            self.in_proj = nn.Linear(self.F, self.d_model)

        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=self.d_model,
                p_dim=self.P,
                film_hidden=film_hidden,
                m2_ngroups=m2_ngroups,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.post_norm = nn.LayerNorm(self.d_model)

        self.stack_gate = nn.Parameter(torch.tensor(1.0))

        self.film_before_head = FiLM(p_dim=self.P, D=self.d_model, hidden=film_hidden)

        self.head = nn.Sequential(
            nn.Linear(self.d_model, mlp_hidden),
            nn.PReLU(),
            nn.Linear(mlp_hidden, self.F),
        )

        with torch.no_grad():
            self.head[-1].weight.zero_()
            self.head[-1].bias.zero_()

        self.out_scale = nn.Parameter(torch.tensor(float(out_scale_init)))

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
        """
        x: [B, 1, T]
        p: [B, P]
        """
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

        feats = mag.transpose(1, 2)

        h_in = self.in_proj(feats)

        h = h_in
        for blk in self.blocks:
            h = blk(h, p)
        h = self.post_norm(h)
        h = h_in + self.stack_gate * h

        h = self.film_before_head(h, p)

        logits = self.head(h)
        logits = logits.transpose(1, 2)

        mask = torch.sigmoid(logits) * self.out_scale
        real = (mag_lin * mask) * torch.cos(phase)
        imag = (mag_lin * mask) * torch.sin(phase)
        Y = torch.complex(real, imag)

        recon_len = (TT - 1) * self.hop_length + self.n_fft
        y_full = torch.istft(
            Y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=False,
            length=recon_len,
        )  # [B, recon_len]

        y = y_full[:, pad_left:pad_left + x.size(1)]
        return y.unsqueeze(1)