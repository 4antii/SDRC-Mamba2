"""
This file contains code adapted from the S4 Dynamic Range Compressor repository.
Repository:
https://github.com/ineffab1e-vista/s4-dynamic-range-compressor

If you use this code or derivatives in academic work, please cite the paper below.

Reference:
Yin, H., Cheng, G., Steinmetz, C. J., Yuan, R., Stern, R. M., & Dannenberg, R. B.
"Modeling Analog Dynamic Range Compressors using Deep Learning and
State-space Models."
"""

from typing import TypedDict

import lightning.pytorch as pl
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.optim import AdamW

from ..base import Base
from .module.film import FiLM
from .module.s4 import FFTConv as S4

class S4Block(nn.Module):
    def __init__(
        self,
        conditional_information_dimension: int,
        inner_audio_channel: int,
        s4_hidden_size: int,
    ):
        super().__init__()

        self.linear = nn.Linear(inner_audio_channel, inner_audio_channel)
        self.activation1 = nn.PReLU()
        self.s4 = S4(inner_audio_channel, activation='id', mode='s4d', d_state=s4_hidden_size)
        self.batchnorm = nn.BatchNorm1d(inner_audio_channel, affine=False)
        self.film = FiLM(inner_audio_channel, conditional_information_dimension)
        self.activation2 = nn.PReLU()
        self.residual_connection = nn.Conv1d(
            inner_audio_channel,
            inner_audio_channel,
            kernel_size=1,
            groups=inner_audio_channel,
            bias=False
        )

    def forward(self, x: Tensor, conditional_information: Tensor) -> Tensor:
        out = rearrange(x, 'B H L -> B L H')
        out = self.linear(out)
        out = rearrange(out, 'B L H -> B H L')

        out = self.activation1(out)
        s4_out = self.s4(out)
        out = s4_out[0] if isinstance(s4_out, tuple) else s4_out

        if self.batchnorm:
            out = self.batchnorm(out)
        out = self.film(out, conditional_information)
        out = self.activation2(out)

        if self.residual_connection:
            out += self.residual_connection(x)

        return out
    

class S4Model(Base):
    def __init__(
        self, 
        nparams: int,
        inner_audio_channel: int = 32,
        s4_hidden_size: int = 4,
        depth: int = 4,
        **kwargs
    ):
        if inner_audio_channel < 1:
            raise ValueError(f'The inner audio channel is expected to be one or greater, but got {inner_audio_channel}.')
        if s4_hidden_size < 1:
            raise ValueError(f'The S4 hidden size is expected to be one or greater, but got {s4_hidden_size}.')
        if depth < 0:
            raise ValueError(f'The model depth is expected to be zero or greater, but got {depth}.')
        if nparams < 1:
            raise ValueError(f'The number of conditioning parameters must be >= 1, got {nparams}.')

        super().__init__()
        self.save_hyperparameters()

        self.control_parameter_mlp = nn.Sequential( 
            nn.Linear(nparams, 16), 
            nn.ReLU(), 
            nn.Linear(16, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32), 
            nn.ReLU() 
        )

        self.expand = nn.Linear(1, inner_audio_channel)
        self.blocks = nn.ModuleList([
            S4Block(
                32,
                inner_audio_channel,
                s4_hidden_size,
            ) for _ in range(depth)
        ])
        self.contract = nn.Linear(inner_audio_channel, 1)

        self.tanh = nn.Tanh()

    def forward(
        self,
        x: Tensor,
        parameters: Tensor,
    ) -> Tensor:

        x = x.squeeze(dim=1)
        parameters = parameters.squeeze(dim=1)

        conditional_information = self.control_parameter_mlp(parameters)

        out = rearrange(x, 'B L -> B L 1')
        out = self.expand(out)
        out = rearrange(out, 'B L H -> B H L')

        for block in self.blocks:
            out = block(out, conditional_information)

        out = rearrange(out, 'B H L -> B L H')
        out = self.contract(out)

        out = rearrange(out, 'B L 1 -> B L')

        out = self.tanh(out)

        out = out.unsqueeze(dim=1)

        return out