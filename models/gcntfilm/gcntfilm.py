"""Parametric GCNTF model adapted from mcomunita/gcn-tfilm.

Source: https://github.com/mcomunita/gcn-tfilm/tree/parametric
Source commit: 8e2f186344a8fd35ed5b12b64b891d24a5bab2f0
"""

import math

import torch

from ..base import Base


"""
Temporal FiLM layer - Conditional
"""
class TFiLM(torch.nn.Module):
    def __init__(self,
                 nchannels,
                 nparams,
                 block_size):
        super(TFiLM, self).__init__()
        self.nchannels = nchannels
        self.nparams = nparams
        self.block_size = block_size
        self.num_layers = 1
        self.hidden_state = None  # (hidden_state, cell_state)

        # used to downsample input
        self.maxpool = torch.nn.MaxPool1d(kernel_size=block_size,
                                          stride=None,
                                          padding=0,
                                          dilation=1,
                                          return_indices=False,
                                          ceil_mode=False)

        self.lstm = torch.nn.LSTM(input_size=nchannels+nparams,
                                  hidden_size=nchannels,
                                  num_layers=self.num_layers,
                                  batch_first=False,
                                  bidirectional=False)

    def forward(self, x, p=None):
        # x = [batch, nchannels, length]
        # p = [batch, nparams]
        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[2] % self.block_size) != 0:
            padding = x.new_zeros(x_in_shape[0], x_in_shape[1], self.block_size - (x_in_shape[2] % self.block_size))
            x = torch.cat((x, padding), dim=-1)

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)

        # downsample signal [batch, nchannels, nsteps]
        x_down = self.maxpool(x)

        if self.nparams > 0 and p is not None:
            p_up = p.unsqueeze(-1).repeat(1, 1, nsteps) # upsample params [batch, nparams, nsteps]
            x_down = torch.cat((x_down, p_up), dim=1) # concat along channel dim [batch, nchannels+nparams, nsteps]

        # shape for LSTM (length, batch, channels)
        x_down = x_down.permute(2, 0, 1)

        # modulation sequence
        if self.hidden_state == None:  # state was reset
            # init hidden and cell states with zeros
            h0 = x.new_zeros(self.num_layers, x.size(0), self.nchannels).requires_grad_()
            c0 = x.new_zeros(self.num_layers, x.size(0), self.nchannels).requires_grad_()
            x_norm, self.hidden_state = self.lstm(x_down, (h0.detach(), c0.detach()))  # detach for truncated BPTT
        else:
            x_norm, self.hidden_state = self.lstm(x_down, self.hidden_state)

        # put shape back (batch, channels, length)
        x_norm = x_norm.permute(1, 2, 0)

        # reshape input and modulation sequence into blocks
        x_in = torch.reshape(
            x, shape=(-1, self.nchannels, nsteps, self.block_size))
        x_norm = torch.reshape(
            x_norm, shape=(-1, self.nchannels, nsteps, 1))

        # multiply
        x_out = x_norm * x_in

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        # crop to original (input) shape
        x_out = x_out[..., :x_in_shape[2]]

        return x_out

    def reset_state(self):
        self.hidden_state = None


"""
Gated convolutional layer, zero pads and then applies a causal convolution to the input
"""
class GatedConv1d(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 dilation,
                 kernel_size,
                 nparams,
                 tfilm_block_size):
        super(GatedConv1d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.kernal_size = kernel_size
        self.nparams = nparams
        self.tfilm_block_size = tfilm_block_size

        # Layers: Conv1D -> Activations -> TFiLM -> Mix + Residual

        self.conv = torch.nn.Conv1d(in_channels=in_ch,
                                    out_channels=out_ch * 2,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=0,
                                    dilation=dilation)

        self.tfilm = TFiLM(nchannels=out_ch,
                           nparams=nparams,
                           block_size=tfilm_block_size)

        self.mix = torch.nn.Conv1d(in_channels=out_ch,
                                   out_channels=out_ch,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def forward(self, x, p=None):
        residual = x

        # dilated conv
        y = self.conv(x)

        # gated activation
        z = torch.tanh(y[:, :self.out_ch, :]) * \
            torch.sigmoid(y[:, self.out_ch:, :])

        # zero pad on the left side, so that z is the same length as x
        z = torch.cat((residual.new_zeros(residual.shape[0],
                                         self.out_ch,
                                         residual.shape[2] - z.shape[2]),
                       z),
                      dim=2)

        # modulation
        z = self.tfilm(z, p)

        x = self.mix(z) + residual

        return x, z


"""
Gated convolutional neural net block, applies successive gated convolutional layers to the input, a total of 'layers'
layers are applied, with the filter size 'kernel_size' and the dilation increasing by a factor of 'dilation_growth' for
each successive layer.
"""
class GCNBlock(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 nlayers,
                 kernel_size,
                 dilation_growth,
                 nparams,
                 tfilm_block_size):
        super(GCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nlayers = nlayers
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.nparams = nparams
        self.tfilm_block_size = tfilm_block_size

        dilations = [dilation_growth ** l for l in range(nlayers)]

        self.layers = torch.nn.ModuleList()

        for d in dilations:
            self.layers.append(GatedConv1d(in_ch=in_ch,
                                           out_ch=out_ch,
                                           dilation=d,
                                           kernel_size=kernel_size,
                                           nparams=nparams,
                                           tfilm_block_size=tfilm_block_size))
            in_ch = out_ch

    def forward(self, x, p=None):
        # [batch, channels, length]
        z = x.new_empty([x.shape[0],
                         self.nlayers * self.out_ch,
                         x.shape[2]])

        for n, layer in enumerate(self.layers):
            x, zn = layer(x, p)
            z[:, n * self.out_ch: (n + 1) * self.out_ch, :] = zn

        return x, z


"""
Gated Convolutional Neural Net class, based on the 'WaveNet' architecture, takes a single channel of audio as input and
produces a single channel of audio of equal length as output. one-sided zero-padding is used to ensure the network is
causal and doesn't reduce the length of the audio.

Made up of 'blocks', each one applying a series of dilated convolutions, with the dilation of each successive layer
increasing by a factor of 'dilation_growth'. 'layers' determines how many convolutional layers are in each block,
'kernel_size' is the size of the filters. Channels is the number of convolutional channels.

The output of the model is creating by the linear mixer, which sums weighted outputs from each of the layers in the
model
"""
class GCNTFModel(Base):
    def __init__(self,
                 nparams=0,
                 nblocks=2,
                 nlayers=9,
                 nchannels=8,
                 kernel_size=3,
                 dilation_growth=2,
                 tfilm_block_size=128,
                 pad_input_to_receptive_field=False,
                 **kwargs):
        super(GCNTFModel, self).__init__()
        self.save_hyperparameters()
        self.nparams = nparams
        self.nblocks = nblocks
        self.nlayers = nlayers
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.tfilm_block_size = tfilm_block_size
        self.pad_input_to_receptive_field = pad_input_to_receptive_field

        self.blocks = torch.nn.ModuleList()
        for b in range(nblocks):
            self.blocks.append(GCNBlock(in_ch=1 if b == 0 else nchannels,
                                        out_ch=nchannels,
                                        nlayers=nlayers,
                                        kernel_size=kernel_size,
                                        dilation_growth=dilation_growth,
                                        nparams=nparams,
                                        tfilm_block_size=tfilm_block_size))

        # output mixing layer
        self.blocks.append(
            torch.nn.Conv1d(in_channels=nchannels * nlayers * nblocks,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0))

    def forward(self, x, p=None):
        self.reset_states()

        input_length = x.shape[-1]
        if self.pad_input_to_receptive_field:
            padded_length = math.ceil(self.compute_receptive_field() / self.tfilm_block_size) * self.tfilm_block_size
            if input_length < padded_length:
                x = torch.nn.functional.pad(x, (padded_length - input_length, 0))

        if p is not None and p.ndim == 3:
            p = p.squeeze(1)

        z = x.new_empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]])

        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x, p)
            z[:,
              n * self.nchannels * self.nlayers:
              (n + 1) * self.nchannels * self.nlayers,
              :] = zn

        out = self.blocks[-1](z)
        self.reset_states()
        return out[..., -input_length:]

    # reset state for all TFiLM layers
    def reset_states(self):
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.reset_state()

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.nblocks * self.nlayers):
            dilation = self.dilation_growth ** (n % self.nlayers)
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf
