"""Teacher bake-off: compare candidate teachers on identical images.

The teacher's output is a hard ceiling on the student's quality, so this picks
that teacher by eye rather than by reputation. Each candidate runs over the same
N images and writes a 4-panel sheet:

    original | raw teacher alpha | COCO-gated alpha | gated composite

The gate intersects the teacher's alpha with a dilated COCO person polygon.
COCO's polygons are coarse, so they are used only to answer "is this pixel part
of a person at all" -- the fine hair alpha inside still comes from the teacher.

    modal run stage2_bakeoff.py::main
"""
import pathlib

import modal

from common import DATA, app, data_vol, gpu_image, read_json

RAW_DIR = DATA / "raw" / "coco"
ANN = DATA / "coco" / "annotations" / "instances_train2017.json"
OUT = DATA / "bakeoff"
MANIFEST = DATA / "manifest_coco.json"

CANDIDATES = [
    "ZhengPeng7/BiRefNet",             # salient object -- current baseline
    "ZhengPeng7/BiRefNet-portrait",    # human portraits, trained on P3M
    "ZhengPeng7/BiRefNet_HR-matting",  # high-res matting, MIT
]

N_IMAGES = 100
N_PREVIEWS = 10
RES = 1024
BATCH = 8
DILATE_FRAC = 0.02  # gate dilation as a fraction of the image's long side


@app.function(image=gpu_image, gpu="H100", volumes={str(DATA): data_vol},
              timeout=3600, retries=modal.Retries(max_retries=2))
def run_teacher(teacher: str):
    """Run one candidate over the shared image set; write preview sheets."""
    import time

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

    # The manifest is sorted person_frac-descending, so its head is all extreme
    # close-ups -- useless for judging hair. Sample a mid-range band where the
    # head and hairline are actually visible, and prefer single-subject frames.
    allr = read_json(MANIFEST, {}).get("images", [])
    # Position in the manifest determines which stage-1 shard holds the file.
    band = [dict(r, _shard=i // 500) for i, r in enumerate(allr)
            if 0.25 <= r["person_frac"] <= 0.60 and r["n_people"] == 1]
    step = max(1, len(band) // N_IMAGES)
    records = band[::step][:N_IMAGES]
    print(f"{len(band)} in band -> sampling {len(records)}")
    coco = COCO(str(ANN))
    person_cat = coco.getCatIds(catNms=["person"])[0]

    model = AutoModelForImageSegmentation.from_pretrained(teacher, trust_remote_code=True)
    model.eval().half().to("cuda")
    prep = transforms.Compose([
        transforms.Resize((RES, RES)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def person_gate(rec, shape):
        """Dilated union of COCO person polygons, as a float mask in [0,1]."""
        h, w = shape
        anns = coco.loadAnns(coco.getAnnIds(imgIds=rec["id"], catIds=[person_cat]))
        m = np.zeros((h, w), np.uint8)
        for a in anns:
            if a.get("iscrowd", 0):
                continue
            m |= cv2.resize(coco.annToMask(a), (w, h), interpolation=cv2.INTER_NEAREST)
        k = max(3, int(DILATE_FRAC * max(h, w)) | 1)
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)

    t0, n, deltas, made = time.time(), 0, [], 0
    for i in range(0, len(records), BATCH):
        chunk = records[i:i + BATCH]
        loaded = []
        for rec in chunk:
            fp = RAW_DIR / f"{rec['_shard']:05d}" / rec["file"]
            if not fp.exists():
                continue
            loaded.append((rec, Image.open(fp).convert("RGB")))
        if not loaded:
            continue

        with torch.no_grad():
            batch = torch.stack([prep(im) for _, im in loaded]).half().to("cuda")
            preds = model(batch)[-1].sigmoid().float().cpu().numpy()

        for (rec, im), pred in zip(loaded, preds):
            w, h = im.size
            alpha = cv2.resize(pred[0], (w, h), interpolation=cv2.INTER_LINEAR)
            gate = person_gate(rec, (h, w))
            gated = alpha * gate
            # How much of the teacher's mass sat outside any person: the
            # held-object / background-blob rate.
            tot = float(alpha.sum())
            deltas.append(1.0 - float(gated.sum()) / tot if tot > 1e-6 else 0.0)
            n += 1

            if made < N_PREVIEWS:
                s = 512 / max(w, h)
                tw, th = max(1, round(w * s)), max(1, round(h * s))
                r = lambda a: cv2.resize(a, (tw, th), interpolation=cv2.INTER_LINEAR)
                small = np.asarray(im.resize((tw, th), Image.LANCZOS)).astype(np.float32)
                a_s, g_s = r(alpha)[..., None], r(gated)[..., None]
                comp = small * g_s + 255.0 * (1 - g_s)
                sheet = np.concatenate([
                    small,
                    np.repeat(a_s * 255, 3, axis=2),
                    np.repeat(g_s * 255, 3, axis=2),
                    comp,
                ], axis=1).astype(np.uint8)
                Image.fromarray(sheet).save(out_dir / f"{rec['id']:012d}.jpg", quality=90)
                made += 1

    data_vol.commit()
    secs = time.time() - t0
    return {"teacher": teacher, "n": n, "seconds": round(secs, 1),
            "img_per_s": round(n / secs, 2) if secs else 0,
            "outside_person_frac": round(float(np.mean(deltas)), 4) if deltas else 0.0}


@app.local_entrypoint()
def main():
    print(f"bake-off: {len(CANDIDATES)} teachers x {N_IMAGES} images\n")
    rows = list(run_teacher.map(CANDIDATES, order_outputs=True))

    print(f"\n{'teacher':<34}{'img/s':>8}{'secs':>8}{'outside-person':>16}")
    print("-" * 66)
    for r in rows:
        print(f"{r['teacher'].split('/')[-1]:<34}{r['img_per_s']:>8}"
              f"{r['seconds']:>8}{r['outside_person_frac']:>15.1%}")
    print("\n'outside-person' = share of the teacher's alpha mass falling outside any")
    print("COCO person polygon. Lower is better: it is pizzas, blankets, and blobs.")
    print("\npreviews:  modal volume ls bg-matting-data bakeoff/<teacher>")
