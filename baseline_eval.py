"""Measure baselines on the identical held-out split.

Everything so far has tuned band_mad with no reference point, which makes the
number meaningless: 136 could be excellent or terrible. This establishes the
two anchors that give it meaning.

  teacher (BiRefNet_HR-matting, ~220M) -- the upper bound. A distilled student
      cannot exceed it, so the gap to this number is the real score.
  rembg / U2-Net (~44M)                -- the thing the project claims to beat.
  box                                  -- a degenerate control: the bounding
      box of the ground truth, filled solid. Any real model must beat this, and
      it reveals how much of the metric is trivially earned.

All three run through the same EvalDataset, resolution and metric code as the
student, so the numbers are directly comparable.

    modal run baseline_eval.py::main --size 512
"""
import modal

from common import DATA, app, data_vol, gpu_image

EVAL_TAR = DATA / "shards" / "p3m-eval" / "P3M_500_NP.tar"
VOLS = {str(DATA): data_vol}

TEACHER = "ZhengPeng7/BiRefNet_HR-matting"

# rembg pulls onnxruntime and its own model zoo, so it gets its own image.
# Built from the base rather than by extending gpu_image: Modal forbids build
# steps after add_local_python_source, and gpu_image ends with one.
rembg_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("pillow", "numpy", "opencv-python-headless", "rembg==2.0.59",
                 "onnxruntime")
    .env({"U2NET_HOME": "/data/rembg", "NUMBA_CACHE_DIR": "/tmp"})
    .add_local_python_source("common", "model", "losses", "data")
)


@app.function(image=gpu_image, gpu="H100", volumes=VOLS, timeout=3600,
              scaledown_window=2)
def eval_teacher(size: int = 512):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageSegmentation

    from data import EvalDataset
    from losses import matting_metrics

    model = AutoModelForImageSegmentation.from_pretrained(
        TEACHER, trust_remote_code=True).eval().half().cuda()

    ds = EvalDataset(EVAL_TAR, size=size)
    loader = DataLoader(ds, batch_size=4, num_workers=4)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    agg, nb = {}, 0
    with torch.no_grad():
        for ims, als in loader:
            ims, als = ims.cuda(), als.cuda()
            x = ((ims - mean) / std).half()
            pred = model(x)[-1].sigmoid().float().clamp(0, 1)
            for k, v in matting_metrics(pred, als).items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
    return {"name": "BiRefNet_HR-matting (teacher, ~220M)",
            **{k: v / nb for k, v in agg.items()}}


@app.function(image=rembg_image, gpu="A100-40GB", volumes=VOLS, timeout=3600,
              scaledown_window=2)
def eval_rembg(size: int = 512, model_name: str = "u2net"):
    import io

    import numpy as np
    import torch
    from PIL import Image
    from rembg import new_session, remove

    from data import EvalDataset
    from losses import matting_metrics

    ds = EvalDataset(EVAL_TAR, size=size)
    session = new_session(model_name)

    agg, nb = {}, 0
    for i in range(len(ds)):
        img_t, gt_t = ds[i]
        arr = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        mask = remove(buf.getvalue(), session=session, only_mask=True)
        alpha = np.asarray(Image.open(io.BytesIO(mask)).convert("L"))
        alpha = alpha.astype(np.float32) / 255.0

        p = torch.from_numpy(alpha)[None, None]
        g = gt_t[None]
        for k, v in matting_metrics(p, g).items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    return {"name": f"rembg/{model_name} (~44M)", **{k: v / nb for k, v in agg.items()}}


RVM_URL = ("https://github.com/PeterL1n/RobustVideoMatting/releases/download/"
           "v1.0.0/rvm_mobilenetv3_fp32.onnx")


@app.function(image=rembg_image, volumes=VOLS, cpu=8, timeout=3600,
              scaledown_window=2)
def eval_rvm(size: int = 512):
    """RVM mobilenetv3 (3.7M) -- the closest competitor by size, 14.3 MB.

    RVM is recurrent: it takes four hidden-state inputs so it can carry temporal
    context between video frames. For single images those start at zero, which
    is exactly how RVM itself handles the first frame -- so this measures RVM
    fairly, at the task we actually care about, not handicapped.
    """
    import shutil
    import urllib.request

    import numpy as np
    import onnxruntime as ort
    import torch
    from torch.utils.data import DataLoader

    from data import EvalDataset
    from losses import matting_metrics

    path = DATA / "rvm" / "rvm_mobilenetv3_fp32.onnx"
    if not path.exists():
        # urllib rather than wget: this image has no wget, and adding one would
        # rebuild the layer for a single download.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(RVM_URL, timeout=120) as resp, \
                open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.replace(path)
        data_vol.commit()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    pha_idx = out_names.index("pha")

    ds = EvalDataset(EVAL_TAR, size=size)
    loader = DataLoader(ds, batch_size=1, num_workers=4)
    zero = np.zeros([1, 1, 1, 1], np.float32)

    agg, nb = {}, 0
    for ims, als in loader:
        feed = {"src": ims.numpy().astype(np.float32),
                "r1i": zero, "r2i": zero, "r3i": zero, "r4i": zero,
                "downsample_ratio": np.array([1.0], np.float32)}
        pha = sess.run(None, feed)[pha_idx]
        p = torch.from_numpy(pha).float().clamp(0, 1)
        for k, v in matting_metrics(p, als).items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    return {"name": "RVM mobilenetv3 (3.7M)", **{k: v / nb for k, v in agg.items()}}


@app.function(image=gpu_image, volumes=VOLS, cpu=8, timeout=1800)
def eval_box(size: int = 512):
    """Degenerate control: solid ground-truth bounding box. Beat this or quit."""
    import torch
    from torch.utils.data import DataLoader

    from data import EvalDataset
    from losses import matting_metrics

    ds = EvalDataset(EVAL_TAR, size=size)
    loader = DataLoader(ds, batch_size=8, num_workers=4)
    agg, nb = {}, 0
    for _, als in loader:
        pred = torch.zeros_like(als)
        for b in range(als.shape[0]):
            ys, xs = torch.where(als[b, 0] > 0.5)
            if ys.numel():
                pred[b, 0, ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1.0
        for k, v in matting_metrics(pred, als).items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    return {"name": "GT bounding box (control)", **{k: v / nb for k, v in agg.items()}}


@app.local_entrypoint()
def hires(size: int = 1024):
    """Teacher and RVM at high resolution.

    BiRefNet_HR-matting is a high-resolution model -- evaluating it at 512, well
    under its design point, understated it badly. Each model deserves its own
    operating resolution before any ranking is believed.
    """
    for fn in (eval_teacher, eval_rvm):
        try:
            r = fn.remote(size)
            print(f"{r['name']:<40} @{size}px  sad {r['sad']:.2f}  "
                  f"mad {r['mad']:.2f}  grad {r['grad']:.2f}  "
                  f"band_mad {r['band_mad']:.2f}")
        except Exception as e:
            print(f"{fn.info.function_name} FAILED: {str(e)[:120]}")


@app.local_entrypoint()
def rvm(size: int = 512):
    """Only the RVM leg; the other baselines are already measured."""
    r = eval_rvm.remote(size)
    print(f"{r['name']} @ {size}px")
    for k in ("sad", "mad", "grad", "band_mad"):
        print(f"  {k:<10}{r[k]:>10.2f}")


@app.local_entrypoint()
def main(size: int = 512):
    rows = []
    for fn in (eval_box, eval_teacher, eval_rembg, eval_rvm):
        try:
            rows.append(fn.remote(size))
        except Exception as e:
            rows.append({"name": f"{fn.info.function_name} FAILED",
                         "err": str(e)[:120]})

    print(f"\nheld-out P3M_500_NP, 500 images @ {size}px\n")
    print(f"{'model':<42}{'sad':>9}{'mad':>9}{'grad':>9}{'band_mad':>11}")
    print("-" * 80)
    for r in rows:
        if "err" in r:
            print(f"{r['name']:<42}  {r['err']}")
        else:
            print(f"{r['name']:<42}{r['sad']:>9.2f}{r['mad']:>9.2f}"
                  f"{r['grad']:>9.2f}{r['band_mad']:>11.2f}")
    print(f"\n{'our student (4.03M, v1/best)':<42}"
          f"{5.39:>9.2f}{20.57:>9.2f}{30.84:>9.2f}{136.25:>11.2f}")
