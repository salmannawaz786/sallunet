"""INT8-quantize the exported student and measure the speed/quality trade.

The stride-2 decoder that gives us best-in-class hair is also what makes us ~2x
slower than RVM, and the v4 experiment showed those are the same computation --
not separable by tuning. Quantization attacks the cost from the other side:
it changes how the arithmetic is executed, not what the network computes, so it
is the one speed lever that does not trade away hair quality by construction.

Both dynamic and static (QDQ) quantization are tried. Dynamic needs no data but
in onnxruntime mainly accelerates MatMul, which a convnet barely uses; static
quantizes Conv too but needs calibration data, which we have. Measuring both
costs a few minutes and settles which applies here.

Quality is re-measured on the full held-out split through the quantized graph,
because a speedup that silently costs hair detail is not a win.

    modal run quantize_bench.py::main
"""
import modal

from common import CKPT, DATA, app, ckpt_vol, data_vol

VOLS = {str(DATA): data_vol, str(CKPT): ckpt_vol}
EVAL_TAR = DATA / "shards" / "p3m-eval" / "P3M_500_NP.tar"
OUT = DATA / "export"

BENCH_CPU = 4
WARMUP, ITERS = 3, 15
CALIB_N = 64

quant_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("pillow", "numpy", "timm", "onnx", "onnxruntime",
                 "opencv-python-headless")
    .env({"HF_HOME": "/data/hf"})
    .add_local_python_source("common", "model", "losses", "data")
)


@app.function(image=quant_image, volumes=VOLS, cpu=BENCH_CPU, timeout=5400)
def quantize_and_measure(run: str = "v1", size: int = 512):
    import time

    import numpy as np
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_dynamic,
                                          quantize_static)
    from torch.utils.data import DataLoader

    from data import EvalDataset
    from losses import matting_metrics
    from model import build_from_checkpoint, count_params

    ck = torch.load(CKPT / run / "best.pt", map_location="cpu", weights_only=False)
    model, arch, _ = build_from_checkpoint(ck)
    n_params, mb = count_params(model)

    OUT.mkdir(parents=True, exist_ok=True)
    fp32 = OUT / f"student_{run}_{size}.onnx"

    class AlphaOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    torch.onnx.export(AlphaOnly(model), torch.randn(1, 3, size, size), str(fp32),
                      input_names=["image"], output_names=["alpha"],
                      opset_version=17, dynamo=False)

    ds = EvalDataset(EVAL_TAR, size=size)

    class Calib(CalibrationDataReader):
        """Real held-out images, so activation ranges match deployment."""
        def __init__(self, n):
            self.items = iter([{"image": ds[i][0].numpy()[None]} for i in range(n)])

        def get_next(self):
            return next(self.items, None)

    dyn = OUT / f"student_{run}_{size}_int8_dynamic.onnx"
    quantize_dynamic(str(fp32), str(dyn), weight_type=QuantType.QUInt8)

    stat = OUT / f"student_{run}_{size}_int8_static.onnx"
    try:
        quantize_static(str(fp32), str(stat), Calib(CALIB_N),
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8)
        static_ok = True
    except Exception as e:
        static_ok = False
        static_err = str(e)[:140]

    data_vol.commit()

    def bench(path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = BENCH_CPU
        sess = ort.InferenceSession(str(path), opts,
                                    providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        x = np.random.rand(1, 3, size, size).astype(np.float32)
        for _ in range(WARMUP):
            sess.run(None, {name: x})
        ts = []
        for _ in range(ITERS):
            t = time.perf_counter()
            sess.run(None, {name: x})
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    def quality(path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = BENCH_CPU
        sess = ort.InferenceSession(str(path), opts,
                                    providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        loader = DataLoader(ds, batch_size=1, num_workers=2)
        agg, nb = {}, 0
        for ims, als in loader:
            out = sess.run(None, {name: ims.numpy()})[0]
            p = torch.from_numpy(out).float().clamp(0, 1)
            for k, v in matting_metrics(p, als).items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        return {k: v / nb for k, v in agg.items()}

    rows = []
    for label, path in (("fp32", fp32), ("int8 dynamic", dyn),
                        ("int8 static", stat if static_ok else None)):
        if path is None:
            rows.append({"name": label, "err": static_err})
            continue
        rows.append({"name": label, "file_mb": round(path.stat().st_size / 1024**2, 1),
                     "ms": bench(path), **quality(path)})

    return {"run": run, "arch": arch, "params_m": round(n_params / 1e6, 2),
            "torch_mb": round(mb, 1), "rows": rows}


@app.local_entrypoint()
def main(run: str = "v1", size: int = 512):
    r = quantize_and_measure.remote(run, size)
    print(f"\n{r['run']}  {r['params_m']}M params  arch={r['arch']}")
    print(f"500 held-out images @ {size}px, {BENCH_CPU} CPU threads\n")
    print(f"{'variant':<16}{'file MB':>9}{'ms':>9}{'sad':>9}{'mad':>9}"
          f"{'grad':>9}{'band_mad':>11}")
    print("-" * 74)
    for x in r["rows"]:
        if "err" in x:
            print(f"{x['name']:<16}  FAILED: {x['err']}")
            continue
        print(f"{x['name']:<16}{x['file_mb']:>9.1f}{x['ms']:>9.1f}{x['sad']:>9.2f}"
              f"{x['mad']:>9.2f}{x['grad']:>9.2f}{x['band_mad']:>11.2f}")
    print("\nRVM reference: 178 ms, band_mad 165.63, mad 11.72")
