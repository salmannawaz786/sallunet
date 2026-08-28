"""Stage 2: pseudo-label human images with the BiRefNet-portrait teacher.

Teacher choice comes from stage2_bakeoff.py: portrait beat the salient-object
base (10.9% vs 16.0% of alpha mass outside any person polygon) and beat
HR-matting on interior opacity, which matters more than HR-matting's slightly
softer hair -- a semi-transparent torso shows background through the subject.

Held objects (a pizza, a book) are removed by intersecting the teacher's alpha
with a dilated COCO person polygon. The polygon is coarse, so it only answers
"is this pixel part of a person at all"; the fine hair alpha inside stays the
teacher's.

Model and COCO index load once per container via @modal.enter rather than once
per shard -- that setup is ~50s and would otherwise be paid on all 18 shards.

    modal run stage2_label.py::pilot   # 1 shard, ~$0.20
    modal run stage2_label.py::main    # all shards
"""
import modal

from common import (DATA, app, cpu_image, data_vol, gpu_image, is_done,
                    mark_done, read_json)

TEACHER = "ZhengPeng7/BiRefNet_HR-matting"
SLUG = TEACHER.split("/")[-1]

RAW_DIR = DATA / "raw" / "coco"
ANN = DATA / "coco" / "annotations" / "instances_train2017.json"
# Namespaced by teacher, so switching teachers never collides with an earlier
# run's markers or tars.
OUT_DIR = DATA / "shards" / SLUG
MARKERS = DATA / "markers" / "stage2" / SLUG
PREVIEW_DIR = DATA / "previews" / SLUG
MANIFEST = DATA / "manifest_coco.json"

RES = 1024
BATCH = 8
LONG_SIDE = 1024
SHARD_SIZE = 500
DILATE_FRAC = 0.02

# Raised from 0.02: the small-subject shards were already rejecting ~30%, and
# a person occupying under 15% of frame teaches nothing about hair detail.
MIN_FG_FRAC, MAX_FG_FRAC = 0.15, 0.98
MAX_BAND_FRAC = 0.35
MAX_OUTSIDE_FRAC = 0.40  # teacher mass outside the gate; high means wrong subject

# HR-matting leaves interiors at ~0.978 rather than 1.0, which shows as a faint
# background bleed through the torso when composited. A gentle gain restores
# opacity while leaving the soft hair band essentially untouched (a 2% scale).
INTERIOR_GAIN = 0.98
# Drop connected components smaller than this share of the mask -- the stray
# floating blobs seen in QA.
MIN_BLOB_FRAC = 0.01
MAX_PEOPLE = 2


def _drop_small_blobs(alpha):
    """Zero connected components smaller than MIN_BLOB_FRAC of the mask.

    The teacher occasionally emits a detached fragment near a hand or elbow.
    Keeping it teaches the student to hallucinate the same floating debris.
    """
    import cv2
    import numpy as np

    binary = (alpha > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:  # background plus at most one component: nothing to prune
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_min = MIN_BLOB_FRAC * float(areas.sum())
    drop = np.zeros(n, bool)
    for i, area in enumerate(areas, start=1):
        drop[i] = area < keep_min
    return np.where(drop[labels], 0.0, alpha)


# scaledown_window=2: release the GPU almost immediately once a container runs
# out of shards. The default 60s idle window would bill ~10 idle H100-minutes
# across the fan-out for no work done.
@app.cls(image=gpu_image, gpu="H100", volumes={str(DATA): data_vol},
         timeout=3600, max_containers=10, scaledown_window=2,
         retries=modal.Retries(max_retries=2, backoff_coefficient=2.0))
class Labeler:
    @modal.enter()
    def setup(self):
        import torch
        from pycocotools.coco import COCO
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        data_vol.reload()
        self.model = AutoModelForImageSegmentation.from_pretrained(
            TEACHER, trust_remote_code=True
        )
        self.model.eval().half().to("cuda")
        torch.set_float32_matmul_precision("high")

        self.coco = COCO(str(ANN))
        self.person_cat = self.coco.getCatIds(catNms=["person"])[0]
        self.prep = transforms.Compose([
            transforms.Resize((RES, RES)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _gate(self, img_id, h, w):
        """Dilated union of this image's COCO person polygons."""
        import cv2
        import numpy as np

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.person_cat])
        m = np.zeros((h, w), np.uint8)
        for a in self.coco.loadAnns(ann_ids):
            if not a.get("iscrowd", 0):
                m |= cv2.resize(self.coco.annToMask(a), (w, h),
                                interpolation=cv2.INTER_NEAREST)
        k = max(3, int(DILATE_FRAC * max(h, w)) | 1)
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    @modal.method()
    def label_shard(self, shard_idx: int, save_previews: bool = False):
        import io
        import json
        import tarfile
        import time

        import cv2
        import numpy as np
        import torch
        from PIL import Image

        marker = MARKERS / f"shard_{shard_idx:05d}.done"
        if is_done(marker):
            return shard_idx, 0, 0, 0.0

        data_vol.reload()
        records = read_json(MANIFEST, {}).get("images", [])
        chunk = records[shard_idx * SHARD_SIZE:(shard_idx + 1) * SHARD_SIZE]
        src = RAW_DIR / f"{shard_idx:05d}"
        if not chunk or not src.exists():
            raise RuntimeError(f"shard {shard_idx}: nothing at {src}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tar_path = OUT_DIR / f"train-{shard_idx:05d}.tar"
        tmp = tar_path.with_suffix(".part")

        kept = rejected = 0
        previews = []
        t0 = time.time()

        with tarfile.open(tmp, "w") as tar:
            for i in range(0, len(chunk), BATCH):
                loaded = []
                for rec in chunk[i:i + BATCH]:
                    fp = src / rec["file"]
                    if fp.exists():
                        try:
                            loaded.append((rec, Image.open(fp).convert("RGB")))
                        except Exception:
                            pass
                if not loaded:
                    continue

                with torch.no_grad():
                    batch = torch.stack([self.prep(im) for _, im in loaded])
                    preds = self.model(batch.half().to("cuda"))[-1]
                    preds = preds.sigmoid().float().cpu().numpy()

                for (rec, im), pred in zip(loaded, preds):
                    # Crowded frames give the teacher an ambiguous subject and
                    # produced the merged-heads contamination seen in QA. Two is
                    # the useful cutoff: requiring exactly one costs half the
                    # dataset (3371 vs 5213 images) for little quality gain,
                    # since the gate covers every person in the frame anyway.
                    if rec.get("n_people", 1) > MAX_PEOPLE:
                        rejected += 1
                        continue

                    w, h = im.size
                    raw = cv2.resize(pred[0], (w, h), interpolation=cv2.INTER_LINEAR)
                    gate = self._gate(rec["id"], h, w).astype(np.float32)
                    alpha = raw * gate
                    alpha = np.clip(alpha / INTERIOR_GAIN, 0.0, 1.0)
                    alpha = _drop_small_blobs(alpha)

                    total = float(raw.sum())
                    outside = 1.0 - float(alpha.sum()) / total if total > 1e-6 else 1.0
                    fg = float((alpha > 0.5).mean())
                    band = float(((alpha > 0.05) & (alpha < 0.95)).mean())
                    if (not (MIN_FG_FRAC <= fg <= MAX_FG_FRAC)
                            or band > MAX_BAND_FRAC
                            or outside > MAX_OUTSIDE_FRAC):
                        rejected += 1
                        continue

                    scale = LONG_SIDE / max(w, h)
                    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
                    im_o = im.resize((tw, th), Image.LANCZOS)
                    a_o = Image.fromarray((alpha * 255).astype(np.uint8), "L")
                    a_o = a_o.resize((tw, th), Image.BILINEAR)

                    key = rec["file"].rsplit(".", 1)[0]
                    for ext, img, kw in (
                        ("jpg", im_o, {"format": "JPEG", "quality": 92}),
                        ("png", a_o, {"format": "PNG", "optimize": True}),
                    ):
                        buf = io.BytesIO()
                        img.save(buf, **kw)
                        info = tarfile.TarInfo(f"{key}.{ext}")
                        info.size = buf.tell()
                        buf.seek(0)
                        tar.addfile(info, buf)

                    meta = json.dumps({
                        "key": key, "fg": round(fg, 4),
                        "band": round(band, 4), "outside": round(outside, 4),
                    }).encode()
                    info = tarfile.TarInfo(f"{key}.json")
                    info.size = len(meta)
                    tar.addfile(info, io.BytesIO(meta))

                    kept += 1
                    if save_previews and len(previews) < 8:
                        previews.append((key, im_o, a_o))

        # Rename only after the tar is fully written: a killed container leaves
        # a .part, never a truncated tar that looks complete.
        tmp.replace(tar_path)

        if previews:
            PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            for key, im_o, a_o in previews:
                a = np.asarray(a_o).astype(np.float32)[..., None] / 255.0
                comp = (np.asarray(im_o) * a + 255 * (1 - a)).astype(np.uint8)
                w2, h2 = im_o.size
                sheet = Image.new("RGB", (w2 * 3, h2))
                sheet.paste(im_o, (0, 0))
                sheet.paste(a_o.convert("RGB"), (w2, 0))
                sheet.paste(Image.fromarray(comp), (w2 * 2, 0))
                sheet.save(PREVIEW_DIR / f"{key}.jpg", quality=90)

        secs = time.time() - t0
        mark_done(marker, {"kept": kept, "rejected": rejected,
                           "seconds": round(secs, 1)}, vol=data_vol)
        return shard_idx, kept, rejected, secs


@app.function(image=cpu_image, volumes={str(DATA): data_vol}, timeout=300)
def n_shards():
    """Shard count, read on the Volume.

    This must run remotely: local entrypoints execute on the developer's
    machine, where /data does not exist -- reading the manifest there silently
    yields zero shards and the run does nothing.
    """
    import math
    data_vol.reload()
    n = len(read_json(MANIFEST, {}).get("images", []))
    if n == 0:
        raise RuntimeError(f"empty or missing manifest at {MANIFEST}; run stage 1")
    return math.ceil(n / SHARD_SIZE)


@app.local_entrypoint()
def pilot():
    idx, kept, rej, secs = Labeler().label_shard.remote(0, save_previews=True)
    total = kept + rej
    rate = kept / secs if secs else 0
    print(f"shard {idx}: kept {kept}, rejected {rej} "
          f"({rej / max(total, 1):.1%}) in {secs:.0f}s -> {rate:.1f} img/s")


@app.local_entrypoint()
def main():
    n = n_shards.remote()
    print(f"labeling {n} shards with {SLUG}", flush=True)
    kept = rej = 0
    for idx, k, r, secs in Labeler().label_shard.map(
        range(n), kwargs={"save_previews": False}, order_outputs=False
    ):
        kept += k
        rej += r
        print(f"  shard {idx:05d}: kept {k}, rejected {r} ({secs:.0f}s)")
    print(f"\nstage 2 complete: {kept} kept, {rej} rejected "
          f"({rej / max(kept + rej, 1):.1%})")
