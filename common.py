"""Shared Modal infrastructure: app, volumes, images, and resume primitives.

Resume model
------------
Every stage of this pipeline is decomposed into units of work that each write a
`.done` marker on completion. A stage restarted after any failure re-scans the
markers and skips finished units, so a crash costs at most one unit -- never the
whole run. Markers live on the same Volume as the data they describe, so the two
can never disagree.

Note: functions are never `serialized=True`. Local Python here is 3.14 and the
images are 3.12; serialized functions require those to match.
"""
import json
import os
import pathlib

import modal

app = modal.App("bg-matting")

# Bulk data: source images, packed webdataset shards, manifests.
data_vol = modal.Volume.from_name("bg-matting-data", create_if_missing=True)
# Training checkpoints, kept separate so frequent commits don't contend with
# the multi-GB data volume.
ckpt_vol = modal.Volume.from_name("bg-matting-ckpt", create_if_missing=True)

DATA = pathlib.Path("/data")
CKPT = pathlib.Path("/ckpt")
VOLUMES = {str(DATA): data_vol, str(CKPT): ckpt_vol}

# CPU-only image for fetching, filtering, and packing.
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "unzip")
    .pip_install("pillow", "tqdm", "requests", "numpy", "pycocotools", "webdataset")
    .pip_install("pyarrow", "huggingface_hub", "opencv-python-headless")
    .env({"HF_HOME": "/data/hf"})
    .add_local_python_source("common")
)

# GPU image: adds torch + the BiRefNet teacher's dependencies.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "unzip", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "pillow", "tqdm", "numpy", "webdataset", "opencv-python-headless",
        "timm", "einops", "kornia", "scikit-image",
        "transformers==4.46.3", "huggingface_hub", "safetensors",
    )
    # Separate layer: added later than the block above, so adding it here
    # does not invalidate the expensive cached torch/transformers layers.
    .pip_install("pycocotools")
    .env({"HF_HOME": "/data/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("common", "model", "losses", "data",
                            "stage3_train")
)


# --------------------------------------------------------------------------
# Resume primitives
# --------------------------------------------------------------------------

def is_done(marker: pathlib.Path) -> bool:
    """True if this unit of work already completed successfully."""
    return marker.exists()


def mark_done(marker: pathlib.Path, meta: dict | None = None, vol=None) -> None:
    """Record a unit as complete, then commit so a later container sees it.

    Written last, after the real output is safely on disk -- a marker must never
    exist for work that did not finish.
    """
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(meta or {"ok": True}))
    if vol is not None:
        vol.commit()


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    """Write via a temp file + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def read_json(path: pathlib.Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default
