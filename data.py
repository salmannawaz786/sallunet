"""Training data: webdataset shards -> augmented 512x512 (image, alpha) pairs.

Two sources are mixed, deliberately unevenly:

  p3m  -- 9,420 pairs, alphas drawn by human annotators. The quality anchor:
          a distilled student cannot exceed its teacher, and these are the only
          labels in the project that beat the teacher.
  coco -- 4,344 pairs pseudo-labelled by BiRefNet_HR-matting. Real faces,
          cluttered real-world backgrounds, arbitrary poses. P3M has none of
          those (its faces are deliberately blurred for privacy).

P3M is sampled more heavily for label quality, COCO enough to carry the domain
variety P3M lacks.

Background replacement uses other COCO images rather than a downloaded
background set. Note this is an approximation: the true foreground colour is
unknown, so compositing reuses the observed pixel, which bleeds a little of the
original background into semi-transparent hair. Applied at moderate probability
it is a net win for background robustness; at p=1.0 it would teach the model
that hair always carries a colour fringe.
"""
import io
import random

import numpy as np
import torch
from PIL import Image

CROP = 512
BG_REPLACE_P = 0.3  # v1 value; 0.5 measurably hurt


def _decode(sample):
    im = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
    al = Image.open(io.BytesIO(sample["png"])).convert("L")
    if al.size != im.size:
        al = al.resize(im.size, Image.BILINEAR)
    return im, al


def _random_resized_crop(im, al, size=CROP, rng=random):
    """Crop biased toward the subject: a crop of pure background teaches nothing."""
    w, h = im.size
    scale = rng.uniform(0.5, 1.0)
    ratio = rng.uniform(0.85, 1.18)
    cw = min(w, max(32, int(round(w * scale * ratio))))
    ch = min(h, max(32, int(round(h * scale))))

    a = np.asarray(al)
    ys, xs = np.where(a > 128)
    if ys.size and rng.random() < 0.8:
        cy, cx = int(rng.choice(ys)), int(rng.choice(xs))
        x0 = int(np.clip(cx - cw // 2, 0, w - cw))
        y0 = int(np.clip(cy - ch // 2, 0, h - ch))
    else:
        x0 = rng.randint(0, w - cw) if w > cw else 0
        y0 = rng.randint(0, h - ch) if h > ch else 0

    im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((size, size), Image.BILINEAR)
    al = al.crop((x0, y0, x0 + cw, y0 + ch)).resize((size, size), Image.BILINEAR)
    return im, al


def _augment(im, al, bg_pool, rng=random):
    """Reverted to the v1 recipe, which measurably outperformed a heavier one.

    A v2 run added rotation, grayscale, sensor noise, wider crops and raised
    background replacement to 0.5. It was 43% WORSE on held-out data at every
    step (band_mad 204 vs 142). Two likely causes: rotation stamps artificial
    black corners into both image and alpha, and background replacement is only
    an approximation -- at high probability it teaches the model that hair
    always carries a colour fringe. Do not re-raise these without an isolated
    experiment showing a gain.
    """
    if rng.random() < 0.5:
        im, al = im.transpose(Image.FLIP_LEFT_RIGHT), al.transpose(Image.FLIP_LEFT_RIGHT)

    img = np.asarray(im).astype(np.float32) / 255.0
    a = np.asarray(al).astype(np.float32)[..., None] / 255.0

    if bg_pool and rng.random() < BG_REPLACE_P:
        bg = bg_pool[rng.randrange(len(bg_pool))]
        img = img * a + bg * (1.0 - a)

    if rng.random() < 0.7:  # brightness / contrast / saturation
        img = img * rng.uniform(0.75, 1.25)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * rng.uniform(0.8, 1.2) + mean
        gray = img.mean(axis=2, keepdims=True)
        img = gray + (img - gray) * rng.uniform(0.7, 1.3)

    # Force float32: numpy silently promotes to float64 on mixed-dtype
    # arithmetic, which reaches the model as a DoubleTensor and fails against
    # bf16 weights. Pinning it here covers every augmentation branch above.
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    a = a.astype(np.float32)

    return (torch.from_numpy(img).permute(2, 0, 1).contiguous(),
            torch.from_numpy(a).permute(2, 0, 1).contiguous())


class ShardDataset(torch.utils.data.IterableDataset):
    """Streams (image, alpha) from webdataset tars, mixing sources by weight."""

    def __init__(self, sources, weights, bg_tars=None, crop=CROP,
                 seed=0, bg_pool_size=64):
        super().__init__()
        self.sources = [list(map(str, s)) for s in sources]
        self.weights = list(weights)
        self.bg_tars = [str(p) for p in (bg_tars or [])]
        self.crop = crop
        self.seed = seed
        self.bg_pool_size = bg_pool_size

    def _load_bg_pool(self, rng):
        """Small in-memory pool of background images, refreshed per worker."""
        import tarfile
        pool = []
        if not self.bg_tars:
            return pool
        for _ in range(8):
            try:
                with tarfile.open(rng.choice(self.bg_tars)) as tar:
                    names = [n for n in tar.getnames() if n.endswith(".jpg")]
                    for name in rng.sample(names, min(8, len(names))):
                        f = tar.extractfile(name)
                        if not f:
                            continue
                        im = Image.open(io.BytesIO(f.read())).convert("RGB")
                        im = im.resize((self.crop, self.crop), Image.BILINEAR)
                        pool.append(np.asarray(im).astype(np.float32) / 255.0)
                        if len(pool) >= self.bg_pool_size:
                            return pool
            except Exception:
                continue
        return pool

    def __iter__(self):
        import tarfile

        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1
        rng = random.Random(self.seed * 9973 + wid)

        bg_pool = self._load_bg_pool(rng)
        # Each worker owns a disjoint slice of every source's shards, so no two
        # workers emit the same sample within an epoch.
        streams = [s[wid::nworkers] or s for s in self.sources]

        while True:
            src = rng.choices(range(len(streams)), weights=self.weights)[0]
            shard = rng.choice(streams[src])
            try:
                with tarfile.open(shard) as tar:
                    members = [m for m in tar.getnames() if m.endswith(".jpg")]
                    rng.shuffle(members)
                    for name in members:
                        key = name[:-4]
                        try:
                            jf = tar.extractfile(f"{key}.jpg")
                            pf = tar.extractfile(f"{key}.png")
                            if not (jf and pf):
                                continue
                            im, al = _decode({"jpg": jf.read(), "png": pf.read()})
                        except Exception:
                            continue
                        im, al = _random_resized_crop(im, al, self.crop, rng)
                        yield _augment(im, al, bg_pool, rng)
            except Exception:
                continue


class EvalDataset(torch.utils.data.Dataset):
    """P3M_500_NP held out for metrics. Resized, never cropped or augmented."""

    def __init__(self, tar_path, size=CROP, limit=None):
        import tarfile
        self.samples = []
        with tarfile.open(tar_path) as tar:
            keys = sorted({n[:-4] for n in tar.getnames() if n.endswith(".jpg")})
            if limit:
                keys = keys[:limit]
            for key in keys:
                jf, pf = tar.extractfile(f"{key}.jpg"), tar.extractfile(f"{key}.png")
                if jf and pf:
                    self.samples.append((jf.read(), pf.read()))
        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        jb, pb = self.samples[i]
        im, al = _decode({"jpg": jb, "png": pb})
        im = im.resize((self.size, self.size), Image.BILINEAR)
        al = al.resize((self.size, self.size), Image.BILINEAR)
        img = np.asarray(im).astype(np.float32) / 255.0
        a = np.asarray(al).astype(np.float32)[..., None] / 255.0
        return (torch.from_numpy(img).permute(2, 0, 1).contiguous(),
                torch.from_numpy(a).permute(2, 0, 1).contiguous())
