"""Sample labeled pairs straight out of the packed tars for visual QA.

Deliberately samples across easy and hard shards. The manifest is sorted
person-fraction-descending, so shard 0 holds the biggest, easiest subjects and
the last shards hold the small, hard ones -- judging quality on shard 0 alone
would flatter the teacher.

    modal run inspect_quality.py::main
"""
import modal

from common import DATA, app, cpu_image, data_vol

SLUG = "BiRefNet-portrait"
SHARDS_DIR = DATA / "shards" / SLUG
OUT = DATA / "qa"

SAMPLE_SHARDS = [1, 8, 16]   # easy / medium / hard
PER_SHARD = 3
ROW_H = 340


@app.function(image=cpu_image, volumes={str(DATA): data_vol}, timeout=900)
def contact_sheet(seed: int = 0):
    """Build one contact sheet: image | alpha | composited on white."""
    import io
    import json
    import random
    import tarfile

    import numpy as np
    from PIL import Image

    data_vol.reload()
    rng = random.Random(seed)
    rows = []

    for shard_idx in SAMPLE_SHARDS:
        tar_path = SHARDS_DIR / f"train-{shard_idx:05d}.tar"
        if not tar_path.exists():
            print(f"missing {tar_path}")
            continue

        with tarfile.open(tar_path) as tar:
            names = tar.getnames()
            keys = sorted({n.rsplit(".", 1)[0] for n in names if n.endswith(".jpg")})
            picked = rng.sample(keys, min(PER_SHARD, len(keys)))

            for key in picked:
                jpg = tar.extractfile(f"{key}.jpg")
                png = tar.extractfile(f"{key}.png")
                meta_f = tar.extractfile(f"{key}.json")
                if not (jpg and png):
                    continue
                im = Image.open(io.BytesIO(jpg.read())).convert("RGB")
                al = Image.open(io.BytesIO(png.read())).convert("L")
                meta = json.loads(meta_f.read()) if meta_f else {}

                s = ROW_H / im.height
                tw = max(1, round(im.width * s))
                im = im.resize((tw, ROW_H), Image.LANCZOS)
                al = al.resize((tw, ROW_H), Image.BILINEAR)

                a = np.asarray(al).astype(np.float32)[..., None] / 255.0
                comp = (np.asarray(im) * a + 255 * (1 - a)).astype(np.uint8)
                row = np.concatenate(
                    [np.asarray(im), np.repeat(np.asarray(al)[..., None], 3, 2), comp],
                    axis=1,
                )
                rows.append((shard_idx, key, meta, row))

    if not rows:
        raise RuntimeError("no samples found -- did stage 2 run?")

    width = max(r[3].shape[1] for r in rows)
    canvas = np.full((ROW_H * len(rows), width, 3), 20, np.uint8)
    for i, (_, _, _, row) in enumerate(rows):
        canvas[i * ROW_H:(i + 1) * ROW_H, :row.shape[1]] = row

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"sheet_{seed}.jpg"
    Image.fromarray(canvas).save(path, quality=88)
    data_vol.commit()

    return path.name, [
        {"shard": s, "key": k, "fg": m.get("fg"), "band": m.get("band"),
         "outside": m.get("outside")} for s, k, m, _ in rows
    ]


@app.function(image=cpu_image, volumes={str(DATA): data_vol}, timeout=900)
def p3m_heads(n: int = 6):
    """Zoomed crops of P3M's human-drawn alphas, to judge real hair detail.

    Crops the top third of the alpha's bounding box -- where the hair is --
    rather than showing whole frames, since hair quality is invisible at
    thumbnail scale.
    """
    import io
    import random
    import tarfile

    import cv2
    import numpy as np
    from PIL import Image

    data_vol.reload()
    tar_path = DATA / "shards" / "p3m" / "train-00000.tar"
    rows = []
    with tarfile.open(tar_path) as tar:
        keys = sorted({m.rsplit(".", 1)[0] for m in tar.getnames() if m.endswith(".jpg")})
        for key in random.Random(1).sample(keys, min(n, len(keys))):
            im = np.asarray(Image.open(io.BytesIO(
                tar.extractfile(f"{key}.jpg").read())).convert("RGB"))
            al = np.asarray(Image.open(io.BytesIO(
                tar.extractfile(f"{key}.png").read())).convert("L")).astype(np.float32) / 255

            ys, xs = np.where(al > 0.5)
            if ys.size == 0:
                continue
            y0, y1 = ys.min(), ys.min() + max(40, int((ys.max() - ys.min()) * 0.33))
            x0, x1 = xs.min(), xs.max()
            sub, sa = im[y0:y1, x0:x1], al[y0:y1, x0:x1]
            if sub.shape[0] < 10 or sub.shape[1] < 10:
                continue

            s = ROW_H / sub.shape[0]
            tw = max(1, int(sub.shape[1] * s))
            sub = cv2.resize(sub, (tw, ROW_H), interpolation=cv2.INTER_CUBIC)
            sa = cv2.resize(sa, (tw, ROW_H), interpolation=cv2.INTER_CUBIC)[..., None]
            comp = (sub * sa + 255 * (1 - sa)).astype(np.uint8)
            rows.append(np.concatenate(
                [sub, np.repeat((sa * 255).astype(np.uint8), 3, 2), comp], axis=1))

    width = max(r.shape[1] for r in rows)
    canvas = np.full((ROW_H * len(rows), width, 3), 20, np.uint8)
    for i, r in enumerate(rows):
        canvas[i * ROW_H:(i + 1) * ROW_H, :r.shape[1]] = r
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(OUT / "p3m_heads.jpg", quality=90)
    data_vol.commit()
    return len(rows)


@app.local_entrypoint()
def p3m():
    print("rows:", p3m_heads.remote(6))
    print("get: modal volume get bg-matting-data qa/p3m_heads.jpg ./p3m_heads.jpg")


@app.local_entrypoint()
def main():
    name, rows = contact_sheet.remote(0)
    print(f"\n{'shard':>6} {'key':>14} {'fg':>7} {'band':>7} {'outside':>8}")
    for r in rows:
        print(f"{r['shard']:>6} {r['key'][-12:]:>14} {r['fg']:>7} "
              f"{r['band']:>7} {r['outside']:>8}")
    print(f"\nsheet: modal volume get bg-matting-data qa/{name} ./{name}")
