import torch
import pytorch_lightning as pl

from ..base import Base

class GRUModel(Base):
    def __init__(self, 
                 nparams,
                 ninputs=1,
                 noutputs=1,
                 hidden_size=32,
                 num_layers=1,
                 **kwargs):
        super(GRUModel, self).__init__()
        self.save_hyperparameters()

        input_size = ninputs + nparams
        self.gru = torch.nn.GRU(input_size,
                                   self.hparams.hidden_size,
                                   self.hparams.num_layers,
                                   batch_first=False,
                                   bidirectional=False)
        
        self.linear = torch.nn.Linear(self.hparams.hidden_size, 
                                      self.hparams.noutputs)

    def forward(self, x, p):
        bs = x.size(0) 
        s = x.size(-1) 
        x_wp = x.permute(2,0,1)

        if p is not None:
            p = p.permute(1,0,2)
            p = p.repeat(s,1,1) 
            x_wp = torch.cat((x_wp, p), dim=-1)

        out, _ = self.gru(x_wp)
        out = torch.tanh(self.linear(out))
        out = out.permute(1,2,0)
        out = out + x
        
        return out