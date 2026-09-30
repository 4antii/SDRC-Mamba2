"""Check trained model state/reset, conditioning and short padded-segment parity."""
import torch
from benchmark import checkpoint_paths, load_source
from runtime import StreamingGCNTF


def run():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    parameter_sets = [
        [0.0, 0.1, 0.0001, 0.05],
        [-0.4, 10.0, 0.2, 3.0],
    ]

    with torch.no_grad():
        for path in checkpoint_paths():
            source = load_source(path)
            stream = torch.jit.script(StreamingGCNTF(source).eval())
            for knobs in parameter_sets:
                p = torch.tensor([knobs])
                length = 65536 if source.pad_input_to_receptive_field else 131072
                x = torch.randn(1, 1, length) * .1
                reference = source(x, p)
                prefix = 0
                if source.pad_input_to_receptive_field:
                    rf = source.compute_receptive_field()
                    prefix = ((rf + 127) // 128) * 128 - length
                padded = torch.nn.functional.pad(x, (prefix, 0))
                stream.reset()
                chunks = []
                # Use different valid block lengths to exercise ring wrap and group alignment.
                offset = 0
                sizes = [128, 256, 512, 384]
                i = 0
                while offset < padded.size(2):
                    count = min(sizes[i % len(sizes)], padded.size(2) - offset)
                    chunk = padded[:, :, offset : offset + count]
                    chunks.append(stream(chunk, p))
                    offset += count
                    i += 1
                y = torch.cat(chunks, 2)[:, :, prefix:]
                torch.testing.assert_close(y, reference, atol=2e-5, rtol=2e-4)
                stream.reset()
                a = stream(padded[:, :, :512], p)
                stream.reset()
                b = stream(padded[:, :, :512], p)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                max_abs = (y - reference).abs().max().item()
                print(
                    path.parts[-5],
                    knobs,
                    "max_abs=",
                    max_abs,
                    "PASS",
                    flush=True,
                )


if __name__ == "__main__":
    run()
