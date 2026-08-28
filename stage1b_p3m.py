"""Stage 1b: fetch P3M-10k -- real human-annotated alpha mattes.

This is the highest-value data in the project. A distilled student cannot exceed
its teacher, so pseudo-labels from BiRefNet cap our hair quality. P3M's alphas
were drawn by people, which is the only way past that ceiling. It also supplies
P3M_500_NP, the standard held-out benchmark split.

Source: nobg/P3M-10K on Hugging Face (MIT), stored as parquet with `image` and
`mask` columns. No GPU anywhere in this stage -- the labels already exist, so
there is nothing to infer.

    modal run stage1b_p3m.py::main
"""
import modal

from common import DATA, app, cpu_image, data_vol, is_done, mark_done

REPO = "nobg/P3M-10K"
OUT_TRAIN = DATA / "shards" / "p3m"
OUT_EVAL = DATA / "shards" / "p3m-eval"
MARKERS = DATA / "markers" / "p3m"

N_TRAIN_FILES = 11
EVAL_FILES = ["P3M_500_NP", "P3M_500_P"]
LONG_SIDE = 1024


def _pack(table, tar_path, prefix):
    """Decode a parquet table into a webdataset tar of jpg/png/json triples."""
    import io
    import json
    import tarfile

    import numpy as np
    from PIL import Image

    tmp = tar_path.with_suffix(".part")
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    images = table.column("image").to_pylist()
    masks = table.column("mask").to_pylist()
    kept = 0
    bands = []

    with tarfile.open(tmp, "w") as tar:
        for i, (imrec, mrec) in enumerate(zip(images, masks)):
            try:
                im = Image.open(io.BytesIO(imrec["bytes"])).convert("RGB")
                al = Image.open(io.BytesIO(mrec["bytes"])).convert("L")
            except Exception:
                continue
            if al.size != im.size:
                al = al.resize(im.size, Image.BILINEAR)

            w, h = im.size
            s = LONG_SIDE / max(w, h)
            if s < 1.0:
                tw, th = max(1, round(w * s)), max(1, round(h * s))
                im = im.resize((tw, th), Image.LANCZOS)
                al = al.resize((tw, th), Image.BILINEAR)

            a = np.asarray(al).astype(np.float32) / 255.0
            band = float(((a > 0.05) & (a < 0.95)).mean())
            fg = float((a > 0.5).mean())
            if fg < 0.02 or fg > 0.99:
                continue
            bands.append(band)

            key = f"{prefix}_{i:06d}"
            for ext, img, kw in (("jpg", im, {"format": "JPEG", "quality": 92}),
                                 ("png", al, {"format": "PNG", "optimize": True})):
                buf = io.BytesIO()
                img.save(buf, **kw)
                info = tarfile.TarInfo(f"{key}.{ext}")
                info.size = buf.tell()
                buf.seek(0)
                tar.addfile(info, buf)
            meta = json.dumps({"key": key, "src": "p3m", "fg": round(fg, 4),
                               "band": round(band, 4), "gt": True}).encode()
            info = tarfile.TarInfo(f"{key}.json")
            info.size = len(meta)
            tar.addfile(info, __import__("io").BytesIO(meta))
            kept += 1

    tmp.replace(tar_path)
    mean_band = float(np.mean(bands)) if bands else 0.0
    return kept, mean_band


@app.function(image=cpu_image, volumes={str(DATA): data_vol}, timeout=3600,
              retries=modal.Retries(max_retries=3, backoff_coefficient=2.0))
def fetch_p3m_file(spec: str):
    """Fetch and pack one parquet file. `spec` is 'train:<idx>' or 'eval:<name>'."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    kind, ident = spec.split(":", 1)
    marker = MARKERS / f"{kind}_{ident}.done"
    if is_done(marker):
        return spec, 0, 0.0

    data_vol.reload()
    if kind == "train":
        idx = int(ident)
        remote = f"data/train-{idx:05d}-of-{N_TRAIN_FILES:05d}.parquet"
        out = OUT_TRAIN / f"train-{idx:05d}.tar"
        prefix = f"p3mtr{idx:02d}"
    else:
        remote = f"data/{ident}-00000-of-00001.parquet"
        out = OUT_EVAL / f"{ident}.tar"
        prefix = ident.lower()

    path = hf_hub_download(REPO, remote, repo_type="dataset")
    kept, mean_band = _pack(pq.read_table(path), out, prefix)

    mark_done(marker, {"kept": kept, "band": round(mean_band, 4)}, vol=data_vol)
    return spec, kept, mean_band


@app.local_entrypoint()
def main():
    specs = [f"train:{i}" for i in range(N_TRAIN_FILES)]
    specs += [f"eval:{n}" for n in EVAL_FILES]

    total = 0
    bands = []
    for spec, kept, band in fetch_p3m_file.map(specs, order_outputs=False):
        total += kept
        if kept:
            bands.append(band)
        print(f"  {spec:<22} kept {kept:>5}  band {band:.4f}")

    mean = sum(bands) / len(bands) if bands else 0.0
    print(f"\np3m complete: {total} pairs with real GT alpha")
    print(f"mean soft-transition band: {mean:.4f}")
    print("(BiRefNet_HR-matting pseudo-labels measured 0.0563 in the head region;")
    print(" a higher number here means P3M's human alphas carry more hair detail)")
