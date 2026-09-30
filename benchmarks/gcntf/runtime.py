"""Stateful CPU inference for the upstream parametric GCNTF architecture.

Preserves upstream valid-convolution then left-zero-padding startup semantics.
Each call processes an integer number of 128-sample TFiLM groups, batch size 1.
"""
import copy
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class StreamingLayer(nn.Module):
    def __init__(self, source, strategy: str, max_block: int = 512):
        super().__init__()
        self.conv = copy.deepcopy(source.conv)
        self.mix = copy.deepcopy(source.mix)
        self.lstm = copy.deepcopy(source.tfilm.lstm)
        self.channels = source.out_ch
        self.group = source.tfilm.block_size
        self.context = self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
        self.strategy = strategy
        self.max_block = max_block
        self.register_buffer(
            "history", torch.zeros(1, self.conv.in_channels, self.context)
        )
        self.register_buffer(
            "ring", torch.zeros(self.context + max_block, self.conv.in_channels)
        )
        # Tap order agrees with Conv1d's [out, in, kernel] flattened weights.
        taps = torch.arange(self.conv.kernel_size[0]) * self.conv.dilation[0]
        self.register_buffer(
            "indices", torch.arange(max_block)[:, None] + taps[None, :]
        )
        self.register_buffer("h", torch.zeros(1, 1, self.channels))
        self.register_buffer("c", torch.zeros(1, 1, self.channels))
        self.seen = 0
        self.write_pos = 0

    def reset(self):
        self.history.zero_()
        self.ring.zero_()
        self.h.zero_()
        self.c.zero_()
        self.seen = 0
        self.write_pos = 0

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> List[torch.Tensor]:
        count = x.size(2)
        if self.strategy == "conv":
            buf = torch.cat([self.history, x], dim=2)
            y = self.conv(buf)
            if self.context > 0:
                self.history.copy_(buf[:, :, -self.context:])
        else:
            values = x[0].transpose(0, 1)
            capacity = self.ring.size(0)
            first = min(count, capacity - self.write_pos)
            self.ring[self.write_pos : self.write_pos + first].copy_(values[:first])
            if first < count:
                self.ring[: count - first].copy_(values[first:])
            ids = (self.indices[:count] + self.write_pos - self.context).remainder(
                capacity
            )
            gathered = self.ring.index_select(0, ids.reshape(-1))
            features = gathered.reshape(count, -1, self.conv.in_channels)
            features = features.transpose(1, 2).reshape(count, -1)
            y = F.linear(features, self.conv.weight.flatten(1), self.conv.bias)
            y = y.transpose(0, 1).unsqueeze(0)
            self.write_pos = (self.write_pos + count) % capacity

        z = torch.tanh(y[:, : self.channels]) * torch.sigmoid(
            y[:, self.channels :]
        )
        invalid = min(count, max(0, self.context - self.seen))
        if invalid > 0:
            z[:, :, :invalid] = 0
        self.seen += count
        pooled = F.max_pool1d(z, self.group)
        cond = p.unsqueeze(2).expand(1, p.size(1), pooled.size(2))
        seq = torch.cat([pooled, cond], 1).permute(2, 0, 1)
        modulation, state = self.lstm(seq, (self.h, self.c))
        self.h, self.c = state
        modulation = modulation.permute(1, 2, 0).unsqueeze(-1)
        grouped = z.reshape(1, self.channels, -1, self.group)
        z = (grouped * modulation).reshape_as(z)
        return [self.mix(z) + x, z]


class StreamingGCNTF(nn.Module):
    def __init__(self, source, strategy="gather", max_block=512):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                StreamingLayer(layer, strategy, max_block)
                for block in source.blocks[:-1]
                for layer in block.layers
            ]
        )
        self.output = copy.deepcopy(source.blocks[-1])
        self.group = source.tfilm_block_size
        self.max_block = max_block

    @torch.jit.export
    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        valid_shape = x.size(0) == 1 and x.size(1) == 1
        valid_length = x.size(2) % self.group == 0 and x.size(2) <= self.max_block
        if not valid_shape or not valid_length:
            raise RuntimeError(
                "Expected mono batch=1 and a block divisible by TFiLM group, "
                "<= max_block"
            )
        skips = torch.jit.annotate(List[torch.Tensor], [])
        for layer in self.layers:
            result = layer(x, p)
            x = result[0]
            skips.append(result[1])
        return self.output(torch.cat(skips, 1))
