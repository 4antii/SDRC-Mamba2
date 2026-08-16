#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def causal_crop(x, length: int):
    stop = x.shape[-1] - 1
    start = stop - length
    return x[..., start:stop]


class FiLM(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        gb = self.adaptor(cond)
        g, b = torch.chunk(gb, 2, dim=-1)
        return self.bn(x) * g.permute(0, 2, 1) + b.permute(0, 2, 1)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0, dilation=dilation, bias=False)
        self.film = FiLM(out_ch, 32)
        self.relu = nn.PReLU(out_ch)
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=in_ch, bias=False)

    def forward(self, x, p):
        y = self.conv1(x)
        y = self.film(y, p)
        y = self.relu(y)
        return y + causal_crop(self.res(x), y.shape[-1])


class TCNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gen = nn.Sequential(
            nn.Linear(int(cfg["nparams"]), 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        blocks = []
        out_ch = None
        for n in range(int(cfg["nblocks"])):
            in_ch = out_ch if n > 0 else int(cfg["ninputs"])
            out_ch = int(cfg["channel_width"])
            dilation = int(cfg["dilation_growth"]) ** (n % int(cfg["stack_size"]))
            blocks.append(TCNBlock(in_ch, out_ch, int(cfg["kernel_size"]), dilation))
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv1d(out_ch, int(cfg["noutputs"]), kernel_size=1)

    def forward(self, x, p):
        cond = self.gen(p)
        for block in self.blocks:
            x = block(x, cond)
        return torch.tanh(self.output(x))

    def compute_receptive_field(self):
        rf = int(self.cfg["kernel_size"])
        for n in range(1, int(self.cfg["nblocks"])):
            dilation = int(self.cfg["dilation_growth"]) ** (n % int(self.cfg["stack_size"]))
            rf += (int(self.cfg["kernel_size"]) - 1) * dilation
        return rf


def load_archive(json_path: Path, bin_path: Path):
    meta = json.loads(json_path.read_text())
    blob = np.fromfile(bin_path, dtype=np.float32)
    state = {}
    for name, t in meta["tensors"].items():
        if "num_batches_tracked" in name:
            continue
        arr = blob[t["offset_floats"]:t["offset_floats"] + t["num_floats"]]
        arr = arr.reshape(t["shape"])
        state[name] = torch.tensor(arr, dtype=torch.float32)
    return meta, state


def export_one(stem: str, input_dir: Path, output_dir: Path):
    meta, state = load_archive(input_dir / f"{stem}.json", input_dir / f"{stem}.bin")
    model = TCNModel(meta["config"]).eval()
    model.load_state_dict(state, strict=False)
    rf = model.compute_receptive_field()
    proc_len = (rf - 1) + 2048
    example_x = torch.zeros(1, 1, proc_len, dtype=torch.float32)
    example_p = torch.zeros(1, 1, int(meta["config"]["nparams"]), dtype=torch.float32)
    traced = torch.jit.trace(model, (example_x, example_p), strict=False)
    traced = torch.jit.freeze(traced.eval())
    out_pt = output_dir / f"{stem}.pt"
    traced.save(str(out_pt))
    (output_dir / f"{stem}_torch_meta.json").write_text(json.dumps({
        "stem": stem,
        "receptive_field": rf,
        "context_samples": rf - 1,
        "nparams": int(meta["config"]["nparams"]),
        "kernel_size": int(meta["config"]["kernel_size"]),
        "nblocks": int(meta["config"]["nblocks"]),
        "dilation_growth": int(meta["config"]["dilation_growth"]),
    }, indent=2))
    print(out_pt, "rf", rf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("stems", nargs="+")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for stem in args.stems:
        export_one(stem, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
