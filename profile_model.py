"""Find where our CPU milliseconds actually go, per component.

We are 2.1x slower than RVM at the same size class (4.03M vs 3.7M params), so
the cost is not parameter count -- it is where those parameters sit. A channel
at stride 2 costs 16x the FLOPs of the same channel at stride 8.

Times encoder / decoder / detail-branch separately, and tries a few cheaper
decoder configurations, so the architecture change is aimed rather than guessed.

CPU only: that is where the latency claim lives.

    modal run profile_model.py::main
"""
import modal

from common import app

BENCH_CPU = 4
WARMUP = 3
ITERS = 10
SIZE = 512

profile_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("pillow", "numpy", "timm")
    .env({"HF_HOME": "/tmp/hf"})
    .add_local_python_source("common", "model", "losses", "data")
)


@app.function(image=profile_image, cpu=BENCH_CPU, timeout=3600)
def profile():
    import time

    import torch

    from model import MattingStudent, count_params

    torch.set_num_threads(BENCH_CPU)

    def timeit(fn, *a):
        with torch.no_grad():
            for _ in range(WARMUP):
                fn(*a)
            ts = []
            for _ in range(ITERS):
                t = time.perf_counter()
                fn(*a)
                ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    x = torch.randn(1, 3, SIZE, SIZE)
    out = {"size": SIZE, "components": [], "configs": []}

    # --- component breakdown of the current 224/64 model
    m = MattingStudent(pretrained=False).eval()
    feats = None
    with torch.no_grad():
        feats = m.encoder(x)
    t_enc = timeit(m.encoder, x)
    t_dec = timeit(m.decoder, feats)
    with torch.no_grad():
        sem = m.decoder(feats)
        coarse = m.sem_head(sem)
        up = torch.nn.functional.interpolate(coarse, size=x.shape[-2:],
                                             mode="bilinear", align_corners=False)
    t_det = timeit(m.detail, x, up, sem)
    t_all = timeit(m, x)
    out["components"] = [
        ("encoder", t_enc), ("decoder", t_dec), ("detail", t_det),
        ("full forward", t_all),
    ]
    out["feat_shapes"] = [tuple(f.shape) for f in feats]
    out["sem_shape"] = tuple(sem.shape)

    # --- candidates. `stop` is the decoder's finest stride level: 0 = stride 2
    # (the original, very expensive), 1 = stride 4, 2 = stride 8.
    # v1 = (224,64,0) at band_mad 136 but 374ms ONNX; v3 = (224,32,1) at 163
    # and ~142ms. Two things changed between them, so the middle rows isolate
    # the detail branch's cost at v3's cheap decoder stride.
    for dec, det, stop in ((224, 64, 0), (224, 64, 1), (224, 48, 1),
                           (224, 32, 1), (192, 64, 1), (160, 64, 1)):
        try:
            mm = MattingStudent(pretrained=False, decoder_ch=dec,
                                detail_ch=det, decoder_stop=stop).eval()
            n, mb = count_params(mm)
            out["configs"].append((dec, det, stop, n / 1e6, mb, timeit(mm, x)))
        except Exception as e:
            out["configs"].append((dec, det, stop, 0, 0, str(e)[:40]))
    return out


@app.local_entrypoint()
def main():
    r = profile.remote()
    print(f"\nCPU, {BENCH_CPU} threads, {SIZE}x{SIZE} input, median of {ITERS}\n")
    print("encoder feature shapes:")
    for s in r["feat_shapes"]:
        print(f"   {s}")
    print(f"decoder output (sem): {r['sem_shape']}   <- channels x spatial here"
          f" dominate cost\n")

    print(f"{'component':<18}{'ms':>9}{'share':>9}")
    print("-" * 36)
    total = [t for n, t in r["components"] if n == "full forward"][0]
    for name, t in r["components"]:
        share = "" if name == "full forward" else f"{t/total*100:>8.0f}%"
        print(f"{name:<18}{t:>9.1f}{share:>9}")

    # Eager PyTorch is slower than the ONNX runtime the benchmark uses; the
    # measured ratio on the current model was 374/601 = 0.62.
    ONNX_RATIO = 0.62
    print(f"\n{'dec':>5}{'det':>5}{'stop':>6}{'params(M)':>11}{'MB':>8}"
          f"{'eager ms':>10}{'~onnx ms':>10}{'vs RVM 178':>13}")
    print("-" * 68)
    for dec, det, stop, n, mb, t in r["configs"]:
        if isinstance(t, str):
            print(f"{dec:>5}{det:>5}{stop:>6}  {t}")
            continue
        onnx = t * ONNX_RATIO
        verdict = "FASTER" if onnx < 178 else f"{onnx/178:.2f}x slower"
        print(f"{dec:>5}{det:>5}{stop:>6}{n:>11.2f}{mb:>8.1f}{t:>10.1f}"
              f"{onnx:>10.1f}{verdict:>13}")
