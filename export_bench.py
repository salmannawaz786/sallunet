"""Export the student to ONNX and measure CPU latency against rembg.

CPU is the honest place to measure this: it is where a "lightweight" claim
actually matters, and where a 4M model should separate from a 44M one. Both
models run through the same onnxruntime, on the same container, at the input
resolution each one actually uses in practice -- rembg/U2-Net runs at 320x320,
ours at 512x512. Forcing them to a common size would flatter one or the other;
reporting each at its real operating point is what a user experiences.

No GPU: measuring CPU latency on a GPU container would be paying for an idle
accelerator.

    modal run export_bench.py::main
"""
import modal

from common import CKPT, DATA, app, ckpt_vol, data_vol

VOLS = {str(DATA): data_vol, str(CKPT): ckpt_vol}
OUT = DATA / "export"

BENCH_CPU = 4     # fixed so the numbers mean something
WARMUP = 3
ITERS = 20

# Torch is needed to export; onnxruntime to time. Separate image, CPU only.
export_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("pillow", "numpy", "timm", "onnx", "onnxruntime",
                 "opencv-python-headless")
    .env({"HF_HOME": "/data/hf"})
    .add_local_python_source("common", "model", "losses", "data")
)


@app.function(image=export_image, volumes=VOLS, cpu=BENCH_CPU, timeout=3600)
def export_and_bench(run: str = "v1", size: int = 512):
    import time

    import numpy as np
    import onnxruntime as ort
    import torch

    from model import MattingStudent, count_params

    ck = torch.load(CKPT / run / "best.pt", map_location="cpu", weights_only=False)
    model = MattingStudent(pretrained=False).eval()
    sd = model.state_dict()
    if ck.get("ema"):
        shadow, buffers = ck["ema"]["shadow"], ck["ema"]["buffers"]
        model.load_state_dict({
            k: (shadow[k].to(sd[k].dtype) if k in shadow else buffers[k]).cpu()
            for k in sd})
    else:
        model.load_state_dict(ck["model"])
    n_params, mb = count_params(model)

    OUT.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT / f"student_{run}_{size}.onnx"

    class AlphaOnly(torch.nn.Module):
        """Export a single alpha output; the coarse head is training-only."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    torch.onnx.export(
        AlphaOnly(model), torch.randn(1, 3, size, size), str(onnx_path),
        input_names=["image"], output_names=["alpha"], opset_version=17,
        dynamo=False,
    )
    data_vol.commit()
    onnx_mb = onnx_path.stat().st_size / 1024 ** 2

    def bench(path, in_size, is_rvm=False):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = BENCH_CPU
        sess = ort.InferenceSession(str(path), opts,
                                    providers=["CPUExecutionProvider"])
        x = np.random.rand(1, 3, in_size, in_size).astype(np.float32)
        if is_rvm:
            # RVM is recurrent: four hidden states plus a downsample ratio,
            # zero-initialised exactly as RVM does for a video's first frame.
            z = np.zeros([1, 1, 1, 1], np.float32)
            feed = {"src": x, "r1i": z, "r2i": z, "r3i": z, "r4i": z,
                    "downsample_ratio": np.array([1.0], np.float32)}
        else:
            feed = {sess.get_inputs()[0].name: x}
        for _ in range(WARMUP):
            sess.run(None, feed)
        ts = []
        for _ in range(ITERS):
            t = time.perf_counter()
            sess.run(None, feed)
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        return {"mean_ms": sum(ts) / len(ts), "p50_ms": ts[len(ts) // 2],
                "p95_ms": ts[int(len(ts) * 0.95) - 1]}

    results = [{"name": f"ours ({n_params/1e6:.2f}M)", "size": size,
                "file_mb": round(onnx_mb, 1), **bench(onnx_path, size)}]

    # rembg's U2-Net was cached to the volume by baseline_eval.py.
    u2 = DATA / "rembg" / "u2net.onnx"
    if u2.exists():
        sess = ort.InferenceSession(str(u2), providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
        u2_size = shape[-1] if isinstance(shape[-1], int) else 320
        results.append({"name": "rembg/U2-Net (~44M)", "size": u2_size,
                        "file_mb": round(u2.stat().st_size / 1024 ** 2, 1),
                        **bench(u2, u2_size)})
    else:
        results.append({"name": "rembg/U2-Net", "err": f"not cached at {u2}"})

    # RVM: the closest competitor by size (14.3 MB vs our 15.4 MB), benchmarked
    # at the same 512x512 input so the comparison is like-for-like.
    rvm_path = DATA / "rvm" / "rvm_mobilenetv3_fp32.onnx"
    if rvm_path.exists():
        results.append({"name": "RVM mobilenetv3 (3.7M)", "size": size,
                        "file_mb": round(rvm_path.stat().st_size / 1024 ** 2, 1),
                        **bench(rvm_path, size, is_rvm=True)})
    else:
        results.append({"name": "RVM mobilenetv3",
                        "err": "not cached; run baseline_eval.py::rvm first"})

    return {"torch_mb": round(mb, 1), "onnx_mb": round(onnx_mb, 1),
            "cpu": BENCH_CPU, "results": results}


@app.local_entrypoint()
def main(run: str = "v1", size: int = 512):
    r = export_and_bench.remote(run, size)
    print(f"\nCPU latency, onnxruntime, {r['cpu']} threads, {ITERS} iters\n")
    print(f"{'model':<24}{'input':>8}{'file MB':>10}{'mean ms':>10}"
          f"{'p50 ms':>9}{'p95 ms':>9}")
    print("-" * 70)
    for x in r["results"]:
        if "err" in x:
            print(f"{x['name']:<24}  {x['err']}")
            continue
        print(f"{x['name']:<24}{x['size']:>8}{x['file_mb']:>10.1f}"
              f"{x['mean_ms']:>10.1f}{x['p50_ms']:>9.1f}{x['p95_ms']:>9.1f}")
    ok = [x for x in r["results"] if "err" not in x]
    if ok:
        a = ok[0]
        print()
        for b in ok[1:]:
            rel = b["mean_ms"] / a["mean_ms"]
            print(f"vs {b['name']:<26} ours {rel:.2f}x "
                  f"{'faster' if rel > 1 else 'slower'}, "
                  f"{b['file_mb']/a['file_mb']:.1f}x smaller file")
