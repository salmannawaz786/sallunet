"""Compare how *soft* each teacher's alpha is, especially around hair.

stage2 produced alphas with only ~1% of pixels in the 0.05-0.95 transition band,
i.e. near-binary masks. Distilling those yields a segmentation model, not a
matting model, and hair is the whole differentiator. This measures whether
HR-matting genuinely produces softer hair than portrait before we pay for a
second full labeling pass.

Reports, per teacher:
  band_all   -- fraction of pixels in the soft transition band, whole image
  band_head  -- same, restricted to the top of the COCO person bbox (the hair)
  interior   -- mean alpha where the mask is confidently foreground; portrait
                should be ~1.0, HR-matting was suspected of being < 1

Also writes zoomed head crops so the difference can be judged by eye.

    modal run alpha_probe.py::main
"""
import modal

from common import DATA, app, data_vol, gpu_image, read_json

RAW_DIR = DATA / "raw" / "coco"
ANN = DATA / "coco" / "annotations" / "instances_train2017.json"
MANIFEST = DATA / "manifest_coco.json"
OUT = DATA / "alpha_probe"

CANDIDATES = [
    "ZhengPeng7/BiRefNet-portrait",
    "ZhengPeng7/BiRefNet_HR-matting",
]

N_IMAGES = 50
N_CROPS = 6
RES = 1024
BATCH = 8
CROP_H = 360


@app.function(image=gpu_image, gpu="H100", volumes={str(DATA): data_vol},
              timeout=1800, retries=modal.Retries(max_retries=2))
def probe(teacher: str):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from pycocotools.coco import COCO
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    data_vol.reload()
    slug = teacher.split("/")[-1]
    out_dir = OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    allr = read_json(MANIFEST, {}).get("images", [])
    # Single-subject, mid-size frames: big enough that the head is resolvable.
    band = [dict(r, _shard=i // 500) for i, r in enumerate(allr)
            if 0.20 <= r["person_frac"] <= 0.60 and r["n_people"] == 1]
    step = max(1, len(band) // N_IMAGES)
    records = band[::step][:N_IMAGES]

    coco = COCO(str(ANN))
    person_cat = coco.getCatIds(catNms=["person"])[0]

    model = AutoModelForImageSegmentation.from_pretrained(teacher, trust_remote_code=True)
    model.eval().half().to("cuda")
    prep = transforms.Compose([
        transforms.Resize((RES, RES)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    band_all, band_head, interiors = [], [], []
    crops = []

    for i in range(0, len(records), BATCH):
        loaded = []
        for rec in records[i:i + BATCH]:
            fp = RAW_DIR / f"{rec['_shard']:05d}" / rec["file"]
            if fp.exists():
                loaded.append((rec, Image.open(fp).convert("RGB")))
        if not loaded:
            continue

        with torch.no_grad():
            x = torch.stack([prep(im) for _, im in loaded]).half().to("cuda")
            preds = model(x)[-1].sigmoid().float().cpu().numpy()

        for (rec, im), pred in zip(loaded, preds):
            w, h = im.size
            alpha = cv2.resize(pred[0], (w, h), interpolation=cv2.INTER_LINEAR)

            soft = (alpha > 0.05) & (alpha < 0.95)
            band_all.append(float(soft.mean()))
            core = alpha > 0.95
            interiors.append(float(alpha[core].mean()) if core.any() else 0.0)

            # Head region: top 35% of the person's bounding box.
            anns = coco.loadAnns(coco.getAnnIds(imgIds=rec["id"], catIds=[person_cat]))
            anns = [a for a in anns if not a.get("iscrowd", 0)]
            if not anns:
                continue
            bx, by, bw, bh = max(anns, key=lambda a: a["area"])["bbox"]
            x0, y0 = int(max(0, bx)), int(max(0, by))
            x1, y1 = int(min(w, bx + bw)), int(min(h, by + bh * 0.35))
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            band_head.append(float(soft[y0:y1, x0:x1].mean()))

            if len(crops) < N_CROPS:
                sub = np.asarray(im)[y0:y1, x0:x1]
                sa = alpha[y0:y1, x0:x1]
                s = CROP_H / sub.shape[0]
                tw = max(1, int(sub.shape[1] * s))
                sub = cv2.resize(sub, (tw, CROP_H), interpolation=cv2.INTER_CUBIC)
                sa = cv2.resize(sa, (tw, CROP_H), interpolation=cv2.INTER_CUBIC)
                a3 = sa[..., None]
                comp = (sub * a3 + 255 * (1 - a3)).astype(np.uint8)
                crops.append(np.concatenate(
                    [sub, np.repeat((sa * 255).astype(np.uint8)[..., None], 3, 2), comp],
                    axis=1))

    if crops:
        width = max(c.shape[1] for c in crops)
        canvas = np.full((CROP_H * len(crops), width, 3), 20, np.uint8)
        for i, c in enumerate(crops):
            canvas[i * CROP_H:(i + 1) * CROP_H, :c.shape[1]] = c
        Image.fromarray(canvas).save(out_dir / "heads.jpg", quality=90)

    data_vol.commit()
    import numpy as _np
    return {
        "teacher": slug,
        "n": len(band_all),
        "band_all": round(float(_np.mean(band_all)), 4) if band_all else 0,
        "band_head": round(float(_np.mean(band_head)), 4) if band_head else 0,
        "interior": round(float(_np.mean(interiors)), 4) if interiors else 0,
    }


@app.local_entrypoint()
def main():
    rows = list(probe.map(CANDIDATES, order_outputs=True))
    print(f"\n{'teacher':<26}{'n':>5}{'band_all':>11}{'band_head':>12}{'interior':>11}")
    print("-" * 65)
    for r in rows:
        print(f"{r['teacher']:<26}{r['n']:>5}{r['band_all']:>11.4f}"
              f"{r['band_head']:>12.4f}{r['interior']:>11.4f}")
    if len(rows) == 2:
        a, b = rows
        if a["band_head"] > 0:
            print(f"\nhair softness ratio (HR-matting / portrait): "
                  f"{b['band_head'] / a['band_head']:.2f}x")
    print("\ncrops: modal volume get bg-matting-data alpha_probe/<teacher>/heads.jpg")
