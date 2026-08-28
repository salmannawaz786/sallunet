"""Stage 1: collect source human images and build a training manifest.

Three resumable steps:
  1. fetch_annotations  -- COCO 2017 annotations (~241 MB), one marker
  2. build_manifest     -- filter to matting-suitable person images
  3. fetch_images       -- download in shards of 500, one marker per shard

Only the filtered subset is downloaded (~3 GB), not the full 19 GB COCO zip.

    modal run stage1_fetch.py                # all three steps
    modal run stage1_fetch.py::manifest_only  # count images, no download
"""
import pathlib

import modal

from common import DATA, app, cpu_image, data_vol, is_done, mark_done, read_json

ANN_DIR = DATA / "coco" / "annotations"
RAW_DIR = DATA / "raw" / "coco"
MARKERS = DATA / "markers" / "stage1"
MANIFEST = DATA / "manifest_coco.json"

SHARD_SIZE = 500
TARGET_IMAGES = 20_000

# Matting-suitable person images: subject large enough to matter, not a crowd.
MIN_PERSON_FRAC = 0.10
MAX_PERSON_FRAC = 0.95
MAX_PEOPLE = 4
MIN_SIDE = 480


@app.function(image=cpu_image, volumes={str(DATA): data_vol}, timeout=3600)
def fetch_annotations():
    """Download + unzip COCO 2017 annotations. Skipped if already present."""
    import subprocess

    marker = MARKERS / "annotations.done"
    if is_done(marker):
        print("annotations: already done, skipping")
        return

    ANN_DIR.parent.mkdir(parents=True, exist_ok=True)
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = DATA / "coco" / "ann.zip"

    # -c so an interrupted download resumes instead of restarting.
    subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", str(zip_path), url],
                   check=True)
    subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(DATA / "coco")],
                   check=True)
    zip_path.unlink(missing_ok=True)

    mark_done(marker, {"url": url}, vol=data_vol)
    print(f"annotations: ready at {ANN_DIR}")


@app.function(image=cpu_image, volumes={str(DATA): data_vol}, timeout=3600)
def build_manifest():
    """Filter COCO to matting-suitable person images; write manifest.json."""
    import json

    from pycocotools.coco import COCO

    marker = MARKERS / "manifest.done"
    if is_done(marker):
        existing = read_json(MANIFEST, {})
        print(f"manifest: already done, {len(existing.get('images', []))} images")
        return existing

    data_vol.reload()
    coco = COCO(str(ANN_DIR / "instances_train2017.json"))
    person_cat = coco.getCatIds(catNms=["person"])[0]

    kept = []
    for img_id in coco.getImgIds(catIds=[person_cat]):
        info = coco.loadImgs(img_id)[0]
        w, h = info["width"], info["height"]
        if min(w, h) < MIN_SIDE:
            continue

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[person_cat]))
        people = [a for a in anns if not a.get("iscrowd", 0)]
        if not people or len(people) > MAX_PEOPLE:
            continue

        frac = sum(a["area"] for a in people) / float(w * h)
        if not (MIN_PERSON_FRAC <= frac <= MAX_PERSON_FRAC):
            continue

        kept.append({
            "id": img_id,
            "url": info["coco_url"],
            "file": info["file_name"],
            "person_frac": round(frac, 4),
            "n_people": len(people),
        })

    # Largest-subject-first: if we ever truncate, we keep the best images.
    kept.sort(key=lambda r: -r["person_frac"])
    kept = kept[:TARGET_IMAGES]

    manifest = {"source": "coco2017", "count": len(kept), "images": kept}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest))

    mark_done(marker, {"count": len(kept)}, vol=data_vol)
    print(f"manifest: kept {len(kept)} images -> {MANIFEST}")
    return manifest


@app.function(image=cpu_image, volumes={str(DATA): data_vol}, timeout=3600,
              retries=modal.Retries(max_retries=3, backoff_coefficient=2.0))
def fetch_shard(shard_idx: int):
    """Download one shard of images. Idempotent: re-running is a no-op."""
    import concurrent.futures as cf

    import requests

    marker = MARKERS / f"shard_{shard_idx:05d}.done"
    if is_done(marker):
        return shard_idx, 0, 0

    data_vol.reload()
    images = read_json(MANIFEST, {}).get("images", [])
    chunk = images[shard_idx * SHARD_SIZE:(shard_idx + 1) * SHARD_SIZE]
    out_dir = RAW_DIR / f"{shard_idx:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    def grab(rec):
        dest = out_dir / rec["file"]
        if dest.exists() and dest.stat().st_size > 0:
            return True
        try:
            r = session.get(rec["url"], timeout=30)
            r.raise_for_status()
            # Temp + rename: a killed container never leaves a half-written JPEG
            # that a later run would mistake for a complete one.
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return True
        except Exception:
            return False

    with cf.ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(grab, chunk))

    ok, failed = sum(results), len(results) - sum(results)
    # Tolerate a few dead URLs; bail loudly if the shard is mostly broken so the
    # retry policy gets a chance instead of marking a bad shard complete.
    if chunk and ok < 0.8 * len(chunk):
        raise RuntimeError(f"shard {shard_idx}: only {ok}/{len(chunk)} downloaded")

    mark_done(marker, {"ok": ok, "failed": failed}, vol=data_vol)
    return shard_idx, ok, failed


@app.local_entrypoint()
def manifest_only():
    """Steps 1-2 only: report how many images pass the filter, download none."""
    fetch_annotations.remote()
    m = build_manifest.remote()
    imgs = m["images"]
    n_shards = (len(imgs) + SHARD_SIZE - 1) // SHARD_SIZE
    est_gb = len(imgs) * 0.16 / 1024
    print(f"{len(imgs)} images pass the filter -> {n_shards} shards, ~{est_gb:.1f} GB")
    if imgs:
        fr = [r["person_frac"] for r in imgs]
        print(f"person_frac: max {fr[0]:.2f}  median {fr[len(fr)//2]:.2f}  min {fr[-1]:.2f}")


@app.local_entrypoint()
def main():
    fetch_annotations.remote()
    manifest = build_manifest.remote()

    n = len(manifest["images"])
    n_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"fetching {n} images across {n_shards} shards")

    total_ok = total_failed = 0
    for idx, ok, failed in fetch_shard.map(range(n_shards), order_outputs=False):
        total_ok += ok
        total_failed += failed
        print(f"  shard {idx:05d}: +{ok} ok, {failed} failed")

    print(f"\nstage 1 complete: {total_ok} downloaded, {total_failed} failed")
    print("re-running is safe -- finished shards are skipped via markers")
