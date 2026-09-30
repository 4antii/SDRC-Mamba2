# GCNTF CPU and VST3 benchmarks

This package benchmarks the trained Alesis 3630 GCNTF-250 and GCNTF-2500
Extended models under streaming, batch-size-one conditions. It also verifies
the streaming implementation and VST3 output against the full PyTorch model.

## Models

| Model | Streaming implementation | TorchScript resource |
|---|---|---|
| GCNTF-250 | Native dilated convolution with retained histories | `gcntf_250_release_conv.pt` |
| GCNTF-2500 Extended | Circular tap gathering with retained histories | `gcntf_2500_extended_release_gather.pt` |

The repository does not contain checkpoints, generated TorchScript resources,
rendered audio, benchmark CSV files, or compiled plug-ins. Download the
checkpoints separately and place them under:

```text
experiments/alesis3630/gcntf_250_release/
experiments/alesis3630/gcntf_2500_extended_release/
```

Each directory must contain exactly one `.ckpt` file below it.

## Benchmark conditions

- Batch size 1
- CPU float32
- 44.1 kHz
- 512 output samples per inference step
- One CPU thread for PyTorch and LibTorch
- Mono Python measurements
- Independent mono and stereo VST3 measurements
- 300 warm-up blocks and 500 measured blocks by default

The plug-ins accumulate 512 input samples, process one frame, and report 512
samples of adapter latency. Stereo processing uses independent model state for
each channel. These benchmark plug-ins use LibTorch and are not strict
hard-real-time implementations because LibTorch may allocate or initialize
kernels during processing.

## Streaming implementation

`runtime.py` copies the trained convolution, TFiLM LSTM, mixing, and output
weights. It retains convolution histories and every TFiLM hidden/cell state
between calls, so previously processed receptive-field samples are not replayed.

GCNTF-250 uses LibTorch's convolution implementation. GCNTF-2500 Extended uses
circular tap gathering followed by a dense projection. `benchmark.py` validates
the selected implementation at 128- and 512-sample blocks before exporting the
TorchScript resources consumed by the plug-ins.

## Export and verify

From the repository root:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python benchmarks/gcntf/benchmark.py \
  --warmup 300 \
  --iterations 500 \
  --repeats 3

python benchmarks/gcntf/verify.py
```

Generated files are written to `benchmarks/gcntf/results/` and remain ignored
by Git.

## Build the VST3 benchmarks

The build requires JUCE and the LibTorch CMake package supplied by the active
PyTorch installation:

```bash
cmake -S juce/GCNTFBenchmarks \
  -B juce/GCNTFBenchmarks/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DJUCE_DIR="$JUCE_DIR" \
  -DTorch_DIR="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/Torch"

cmake --build juce/GCNTFBenchmarks/build --target \
  GCNTF250_VST3 \
  GCNTF2500Extended_VST3 \
  gcntf_host \
  -j 4
```

Build products remain under `juce/GCNTFBenchmarks/build/` and are ignored by
Git.

## Benchmark and validate the VST3s

```bash
python benchmarks/gcntf/run_host.py
python benchmarks/gcntf/summarize.py
```

`run_host.py` measures actual VST3 `processBlock` calls in mono and stereo. It
also renders both plug-ins with 512- and 173-sample host callbacks, compensates
the reported plug-in latency, and compares the results with the PyTorch
reference using `atol=2e-5` and `rtol=2e-4`.

`summarize.py` reports processing time, repeat variation, CPU deadline fraction,
RTF, inverse RTF, deadline misses, and adapter latency. CPU percentage is
defined as `100 * processing_time / block_duration`; it is not Activity Monitor
usage or a DAW meter reading.

The plug-ins do not resample. They are intended for the models' trained 44.1 kHz
sample rate.
