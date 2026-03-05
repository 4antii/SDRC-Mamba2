import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
import numpy as np
import soundfile as sf

torchaudio.set_audio_backend("sox_io")

_CL1B_OUT_RE = re.compile(
    r"^output_(?P<idx>\d+)_"
    r"th_(?P<th>-?\d+(?:\.\d+)?)_"
    r"rt_(?P<rt>-?\d+(?:\.\d+)?)_"
    r"at_(?P<at>-?\d+(?:\.\d+)?)_"
    r"rl_(?P<rl>-?\d+(?:\.\d+)?)$"
)

def _num_frames(path: Path, use_soundfile: bool) -> int:
    return sf.info(str(path)).frames if use_soundfile else torchaudio.info(str(path)).num_frames

def _load_wav(path: Path, use_soundfile: bool, normalize: bool = True):
    if use_soundfile:
        x, sr = sf.read(str(path), always_2d=True)
        x = torch.from_numpy(x.T.astype(np.float32))
    else:
        x, sr = torchaudio.load(str(path), normalize=normalize)
    return x, sr


class CL1BDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        inputs_dir: str,
        outputs_dir: str,
        length: int = 16384,
        preload: bool = False,
        half: bool = True,
        subset="train",
        use_soundfile: bool = False,
        polarity_flip: bool = True,
        **kwargs
    ):
        self.inputs_dir = Path(inputs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.length = int(length)
        self.preload = preload
        self.half = half
        self.use_soundfile = use_soundfile
        self.polarity_flip = polarity_flip

        inputs_by_idx: Dict[int, Path] = {}
        for p in sorted(self.inputs_dir.glob("*.wav")):
            stem = p.stem
            if stem.startswith("input_"):
                try:
                    i = int(stem.split("_")[1])
                    inputs_by_idx[i] = p
                except Exception:
                    pass

        outputs_by_idx: Dict[int, Tuple[Path, Tuple[float, float, float, float]]] = {}
        for p in sorted(self.outputs_dir.glob("*.wav")):
            m = _CL1B_OUT_RE.match(p.stem)
            if not m:
                continue
            i = int(m.group("idx"))
            th = float(m.group("th"))
            rt = float(m.group("rt"))
            at = float(m.group("at"))
            rl = float(m.group("rl"))
            outputs_by_idx[i] = (p, (th, rt, at, rl))

        common = sorted(set(inputs_by_idx.keys()) & set(outputs_by_idx.keys()))
        if not common:
            raise RuntimeError("No matching input_{i} / output_{i}_*.wav pairs found.")

        self.pairs: List[Tuple[Path, Path, Tuple[float, float, float, float]]] = []
        for i in common:
            out_path, params = outputs_by_idx[i]
            in_path = inputs_by_idx[i]
            self.pairs.append((in_path, out_path, params))

        self._cache: Optional[Dict[Path, torch.Tensor]] = None
        if self.preload:
            self._cache = {}
            for in_path, out_path, _ in self.pairs:
                if in_path not in self._cache:
                    x, _ = _load_wav(in_path, self.use_soundfile, normalize=True)
                    if x.size(0) > 1: x = x[:1, ...]
                    self._cache[in_path] = x.contiguous()
                if out_path not in self._cache:
                    y, _ = _load_wav(out_path, self.use_soundfile, normalize=True)
                    if y.size(0) > 1: y = y[:1, ...]
                    self._cache[out_path] = y.contiguous()

        self.examples: List[Dict] = []
        for file_idx, (in_path, out_path, params) in enumerate(self.pairs):
            nf_in  = _num_frames(in_path,  self.use_soundfile)
            nf_out = _num_frames(out_path, self.use_soundfile)
            num_frames = int(min(nf_in, nf_out))
            for n in range(num_frames // self.length):
                self.examples.append({
                    "file_idx": file_idx,
                    "input_file":  in_path,
                    "target_file": out_path,
                    "params": params,
                    "offset": n * self.length,
                    "input_audio":  None,
                    "target_audio": None,
                })

        if self.preload and self._cache is not None:
            for ex in self.examples:
                ex["input_audio"]  = self._cache[ex["input_file"]]
                ex["target_audio"] = self._cache[ex["target_file"]]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        offset = int(ex["offset"])

        if self.preload and ex["input_audio"] is not None:
            input  = ex["input_audio"][:,  offset: offset + self.length]
            target = ex["target_audio"][:, offset: offset + self.length]
        else:
            input, _  = torchaudio.load(
                str(ex["input_file"]),
                frame_offset=offset,
                num_frames=self.length,
                normalize=True,
            )
            target, _ = torchaudio.load(
                str(ex["target_file"]),
                frame_offset=offset,
                num_frames=self.length,
                normalize=True,
            )
            if input.size(0)  > 1: input  = input[:1, ...]
            if target.size(0) > 1: target = target[:1, ...]

        if self.half:
            input  = input.half()
            target = target.half()
        else:
            input  = input.float()
            target = target.float()

        if self.polarity_flip and np.random.rand() > 0.5:
            input  = -input
            target = -target

        params = torch.tensor(ex["params"], dtype=input.dtype).unsqueeze(0)

        return input, target, params