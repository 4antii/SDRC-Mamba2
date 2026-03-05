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

import torch
import pytorch_lightning as pl

from ..base import Base

class LSTMModel(Base):
    def __init__(self, 
                 nparams,
                 ninputs=1,
                 noutputs=1,
                 hidden_size=32,
                 num_layers=1,
                 **kwargs):
        super(LSTMModel, self).__init__()
        self.save_hyperparameters()

        input_size = ninputs + nparams
        self.lstm = torch.nn.LSTM(input_size,
                                   self.hparams.hidden_size,
                                   self.hparams.num_layers,
                                   batch_first=False,
                                   bidirectional=False)
        
        self.linear = torch.nn.Linear(self.hparams.hidden_size, 
                                      self.hparams.noutputs)

    def forward(self, x, p):
        bs = x.size(0) 
        s = x.size(-1) 
        x = x.permute(2,0,1) # shape for LSTM (seq, batch, channel)

        if p is not None:
            p = p.permute(1,0,2)
            p = p.repeat(s,1,1) 
            x = torch.cat((x, p), dim=-1)

        out, _ = self.lstm(x)
        out = torch.tanh(self.linear(out))
        out = out.permute(1,2,0) # put shape back (batch, channel, seq)

        return out