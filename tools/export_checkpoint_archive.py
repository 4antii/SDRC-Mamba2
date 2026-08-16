#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import torch


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def export(ckpt_path: Path, json_out: Path, bin_out: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["state_dict"]
    hparams = {k: _jsonable(v) for k, v in ckpt.get("hyper_parameters", {}).items()}

    tensor_meta = {}
    offset_floats = 0

    json_out.parent.mkdir(parents=True, exist_ok=True)
    bin_out.parent.mkdir(parents=True, exist_ok=True)

    with open(bin_out, "wb") as fbin:
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            data = tensor.detach().cpu().contiguous().to(torch.float32).view(-1).tolist()
            if data:
                fbin.write(struct.pack(f"<{len(data)}f", *data))
            tensor_meta[key] = {
                "shape": list(tensor.shape),
                "offset_floats": offset_floats,
                "num_floats": len(data),
            }
            offset_floats += len(data)

    payload = {
        "format": "sdrc_cpu_benchmark_archive_v1",
        "config": hparams,
        "tensors": tensor_meta,
    }
    with open(json_out, "w") as fjson:
        json.dump(payload, fjson)

    print(f"Wrote {json_out}")
    print(f"Wrote {bin_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--bin-out", required=True, type=Path)
    args = parser.parse_args()
    export(args.ckpt, args.json_out, args.bin_out)


if __name__ == "__main__":
    main()
