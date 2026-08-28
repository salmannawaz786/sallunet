"""Full held-out evaluation of a trained checkpoint, plus visual head crops.

Training-time eval sampled only 100 of the 500 held-out images for speed, which
is enough to steer a run but too noisy to report. This runs the complete
P3M_500_NP split and writes zoomed hair crops, since hair quality is the point
of the project and is invisible at thumbnail scale.

Evaluates the EMA weights, matching how the checkpoint was selected.

    modal run evaluate.py::main                    # best.pt
    modal run evaluate.py::main --ckpt latest.pt
"""
import modal

from common import CKPT, DATA, app, ckpt_vol, data_vol, gpu_image

VOLS = {str(DATA): data_vol, str(CKPT): ckpt_vol}
EVAL_TAR = DATA / "shards" / "p3m-eval" / "P3M_500_NP.tar"
# Run dir is a parameter: v1 and ft1024 must be measured identically.
OUT = DATA / "qa"

CROP_H = 340
N_CROPS = 8


@app.function(image=gpu_image, gpu="A100-40GB", volumes=VOLS, cpu=8,
              timeout=3600, scaledown_window=2)
def evaluate(run: str = "v1", ckpt_name: str = "best.pt",
             size: int = 512, n_crops: int = N_CROPS):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    from data import EvalDataset
    from losses import matting_metrics
    from model import build_from_checkpoint, count_params

    path = CKPT / run / ckpt_name
    if not path.exists():
        raise RuntimeError(f"no checkpoint at {path}")

    ck = torch.load(path, map_location="cpu", weights_only=False)
    # Rebuild the architecture the checkpoint was trained with: v1 and v3 differ
    # (stride-2 vs stride-4 decoder), so a fixed constructor cannot load both.
    model, arch, which = build_from_checkpoint(ck)
    model = model.cuda().eval()
    n_params, mb = count_params(model)

    # All 500 held-out images at a caller-chosen resolution. SAD scales with
    # pixel count, so two models are only comparable at the SAME size.
    ds = EvalDataset(EVAL_TAR, size=size)
    loader = DataLoader(ds, batch_size=8, num_workers=4)

    agg, nb = {}, 0
    crops = []
    with torch.no_grad():
        for ims, als in loader:
            ims, als = ims.cuda(), als.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred, _ = model(ims)
            pred = pred.float().clamp(0, 1)
            m = matting_metrics(pred, als)
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1

            if len(crops) < n_crops:
                for i in range(ims.shape[0]):
                    if len(crops) >= n_crops:
                        break
                    img = (ims[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    pa = pred[i, 0].cpu().numpy()
                    ga = als[i, 0].cpu().numpy()
                    ys, xs = np.where(ga > 0.5)
                    if ys.size == 0:
                        continue
                    y0 = ys.min()
                    y1 = y0 + max(40, int((ys.max() - y0) * 0.33))
                    x0, x1 = xs.min(), xs.max()
                    sub, sp, sg = img[y0:y1, x0:x1], pa[y0:y1, x0:x1], ga[y0:y1, x0:x1]
                    if sub.shape[0] < 10 or sub.shape[1] < 10:
                        continue
                    s = CROP_H / sub.shape[0]
                    tw = max(1, int(sub.shape[1] * s))
                    rs = lambda a: cv2.resize(a, (tw, CROP_H), interpolation=cv2.INTER_CUBIC)
                    sub, sp, sg = rs(sub), rs(sp), rs(sg)
                    comp = (sub * sp[..., None] + 255 * (1 - sp[..., None])).astype(np.uint8)
                    # image | ground truth | prediction | composited prediction
                    crops.append(np.concatenate([
                        sub,
                        np.repeat((sg * 255).astype(np.uint8)[..., None], 3, 2),
                        np.repeat((sp * 255).astype(np.uint8)[..., None], 3, 2),
                        comp], axis=1))

    metrics = {k: v / max(nb, 1) for k, v in agg.items()}

    if crops:
        width = max(c.shape[1] for c in crops)
        canvas = np.full((CROP_H * len(crops), width, 3), 20, np.uint8)
        for i, c in enumerate(crops):
            canvas[i * CROP_H:(i + 1) * CROP_H, :c.shape[1]] = c
        OUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(OUT / f"eval_{run}_{size}.jpg", quality=90)
        data_vol.commit()

    return {"run": run, "ckpt": ckpt_name, "size": size, "arch": arch,
            "weights": which, "step": ck.get("step"),
            "n_images": len(ds), "params_m": round(n_params / 1e6, 2),
            "mb": round(mb, 1), **metrics}


@app.local_entrypoint()
def main(run: str = "v1", size: int = 512):
    r = evaluate.remote(run, "best.pt", size)
    print(f"\n{r['run']} step {r['step']} ({r['weights']} weights) @ {r['size']}px")
    print(f"model: {r['params_m']}M params, {r['mb']} MB fp32")
    print(f"held-out: {r['n_images']} images from P3M_500_NP\n")
    for k in ("sad", "mad", "mse", "grad", "band_mad"):
        print(f"  {k:<10}{r[k]:>10.3f}")
    print(f"\ncrops: modal volume get bg-matting-data qa/eval_{run}_{size}.jpg .")


@app.local_entrypoint()
def compare(size: int = 512, runs: str = "v1,v3"):
    """v1 vs the 1024 fine-tune through an identical pipeline.

    Both models must be measured at the same resolution: SAD is a sum over
    pixels, so a model evaluated at 1024 scores ~4x higher than the same model
    at 512 without being any worse.
    """
    runs = [r.strip() for r in runs.split(",")]
    rows = [evaluate.remote(r, "best.pt", size) for r in runs]
    print(f"\nboth models, {rows[0]['n_images']} held-out images @ {size}px\n")
    print(f"{'run':<10}{'step':>7}{'sad':>10}{'mad':>10}{'grad':>10}{'band_mad':>11}")
    print("-" * 58)
    for r in rows:
        print(f"{r['run']:<10}{r['step']:>7}{r['sad']:>10.3f}{r['mad']:>10.3f}"
              f"{r['grad']:>10.3f}{r['band_mad']:>11.3f}")
    a, b = rows[0], rows[-1]
    print()
    for key in ("band_mad", "mad", "grad", "sad"):
        d = (a[key] - b[key]) / a[key] * 100
        winner = b["run"] if d > 0 else a["run"]
        print(f"{key:>10}: {d:+6.1f}%   ({winner} better)")
