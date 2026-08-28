"""Verify the demo's inference path on real images before publishing it.

The ONNX itself is already validated (it scored band_mad 136.04 across the full
held-out split). What is unverified is demo/app.py's own preprocessing: the
resize-to-square, the alpha resize back to original dimensions, and the
compositing. A demo that silently mangles aspect ratio or channel order would
be worse than no demo at all.

Runs on CPU: this is exactly the environment a Hugging Face Space free tier
provides, so the latency it reports is the latency visitors will see.

    modal run demo_test.py::main
"""
import modal

from common import DATA, app, data_vol

EVAL_TAR = DATA / "shards" / "p3m-eval" / "P3M_500_NP.tar"
ONNX = DATA / "export" / "student_v1_512.onnx"
OUT = DATA / "qa"

demo_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("pillow", "numpy", "onnxruntime")
    .add_local_python_source("common")
)


@app.function(image=demo_image, volumes={str(DATA): data_vol}, cpu=2,
              timeout=1800)
def check(n: int = 5):
    import io
    import tarfile
    import time

    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    SIZE = 512
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(ONNX), opts,
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def predict_alpha(image):
        """Mirrors demo/app.py exactly."""
        w, h = image.size
        small = image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        x = np.asarray(small, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        alpha = np.clip(sess.run(None, {name: x})[0][0, 0], 0.0, 1.0)
        return np.asarray(
            Image.fromarray((alpha * 255).astype(np.uint8)).resize(
                (w, h), Image.BILINEAR), dtype=np.float32) / 255.0

    rows, stats = [], []
    with tarfile.open(EVAL_TAR) as tar:
        keys = sorted({m[:-4] for m in tar.getnames() if m.endswith(".jpg")})
        for key in keys[:n]:
            img = Image.open(io.BytesIO(
                tar.extractfile(f"{key}.jpg").read())).convert("RGB")

            t = time.perf_counter()
            alpha = predict_alpha(img)
            ms = (time.perf_counter() - t) * 1000

            # Shape must match the input frame, or the demo is broken.
            assert alpha.shape == (img.size[1], img.size[0]), \
                f"alpha {alpha.shape} != image {img.size[::-1]}"

            stats.append({
                "key": key, "size": img.size, "ms": round(ms),
                "fg": round(float((alpha > 0.5).mean()), 3),
                "band": round(float(((alpha > 0.05) & (alpha < 0.95)).mean()), 4),
                "min": round(float(alpha.min()), 3),
                "max": round(float(alpha.max()), 3),
            })

            a3 = alpha[..., None]
            rgb = np.asarray(img, np.float32)
            comp = (rgb * a3 + 255 * (1 - a3)).astype(np.uint8)
            h = 320
            s = h / img.size[1]
            tw = max(1, int(img.size[0] * s))
            r = lambda arr: np.asarray(
                Image.fromarray(arr).resize((tw, h), Image.BILINEAR))
            rows.append(np.concatenate([
                r(np.asarray(img)),
                r(np.repeat((alpha * 255).astype(np.uint8)[..., None], 3, 2)),
                r(comp)], axis=1))

    width = max(x.shape[1] for x in rows)
    canvas = np.full((320 * len(rows), width, 3), 20, np.uint8)
    for i, x in enumerate(rows):
        canvas[i * 320:(i + 1) * 320, :x.shape[1]] = x
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(OUT / "demo_check.jpg", quality=90)
    data_vol.commit()
    return stats


@app.local_entrypoint()
def main(n: int = 5):
    stats = check.remote(n)
    print(f"\n{'key':<22}{'size':>12}{'ms':>6}{'fg':>7}{'band':>8}"
          f"{'min':>7}{'max':>7}")
    print("-" * 70)
    for s in stats:
        print(f"{s['key'][-20:]:<22}{str(s['size']):>12}{s['ms']:>6}"
              f"{s['fg']:>7.3f}{s['band']:>8.4f}{s['min']:>7.3f}{s['max']:>7.3f}")
    print("\nsheet: modal volume get bg-matting-data qa/demo_check.jpg .")
