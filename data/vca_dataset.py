import os
import sys
import glob
import torch 
import torchaudio
import numpy as np
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
torchaudio.set_audio_backend("sox_io")

class VCADataset(torch.utils.data.Dataset):
    def __init__(self, source_dir, device_dir, subset="train", length=16384, preload=False, half=True, use_soundfile=False, params_num=2):
        self.source_dir = Path(source_dir)
        self.device_dir = Path(device_dir)
        self.subset = subset
        self.length = length
        self.preload = preload
        self.half = half
        self.use_soundfile = use_soundfile
        self.params_num = params_num

        self.input_files = list(self.source_dir.glob("**/*.wav"))
        self.target_files = list(self.device_dir.glob("**/*.wav"))

        pairs_map = {}
        pairs_params = {}

        inp_map = {file.stem:file for file in self.input_files}

        for file in self.target_files:
            parts = [part.replace("3k", "3000").replace("2k", "2000").replace("inf", "100") if part != "01" else "0.1" for part in file.stem.split("_")]
            source_name = "_".join(parts[:-self.params_num])
            source_path = inp_map[source_name]
            params = tuple([float(pr) for pr in parts[-self.params_num:]])
            assert source_path.exists()
            pairs_map[file] = source_path
            pairs_params[file] = params

        self.examples = [] 
        self.minutes = 0
        
        self.target_files = list(pairs_map.keys())
        self.input_files = list(pairs_map.values())
        self.params = list(pairs_params.values())

        for idx, (tfile, ifile, params) in enumerate(zip(self.target_files, self.input_files, self.params)):

            if self.preload:
                sys.stdout.write(f"* Pre-loading... {idx+1:3d}/{len(self.target_files):3d} ...\r")
                sys.stdout.flush()
                input, sr  = self.load(ifile)
                target, sr = self.load(tfile)

                input = input[0]
                target = target[0]
                
                input = input.squeeze()
                target = target.squeeze()

                num_frames = int(np.min([input.shape, target.shape]))
                if self.half:
                    input = input.half()
                    target = target.half()
            else:
                input = None
                target = None

            self.file_examples = []

            for n in range((num_frames // self.length)):
                offset = int(n * self.length)
                end = offset + self.length
                
                self.file_examples.append({"idx": idx, 
                                        "target_file" : tfile,
                                        "input_file" : ifile,
                                        "input_audio" : input[offset:end] if input is not None else None,
                                        "target_audio" : target[offset:end] if input is not None else None,
                                        "params" : params,
                                        "offset": offset,
                                        "frames" : num_frames})

            self.examples += self.file_examples
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.preload:
            audio_idx = self.examples[idx]["idx"]
            offset = self.examples[idx]["offset"]
            input = self.examples[idx]["input_audio"]
            target = self.examples[idx]["target_audio"]
            input = input.unsqueeze(dim=0)
            target = target.unsqueeze(dim=0)
        else:
            offset = self.examples[idx]["offset"] 
            input, sr  = torchaudio.load(self.examples[idx]["input_file"], 
                                        num_frames=self.length, 
                                        frame_offset=offset, 
                                        normalize=True)
            target, sr = torchaudio.load(self.examples[idx]["target_file"], 
                                        num_frames=self.length, 
                                        frame_offset=offset, 
                                        normalize=True)

            if self.half:
                input = input.half()
                target = target.half()

        if np.random.rand() > 0.5:
            input *= -1
            target *= -1

        params = torch.tensor(self.examples[idx]["params"]).unsqueeze(0)
        params[:,0] /= 100
        params[:,1] /= 10
        params[:,2] /= 1000
        params[:,3] /= 1000
    
        return input, target, params

    def load(self, filename):
        if self.use_soundfile:
            x, sr = sf.read(filename, always_2d=True)
            x = torch.tensor(x.T)
        else:
            x, sr = torchaudio.load(filename, normalize=True)
        return x, sr