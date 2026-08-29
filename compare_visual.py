"""Side-by-side visual comparison: rembg / U^2-Net vs RVM vs SalluNet.

The numbers in the README are aggregate. This produces the picture behind them,
on real photographs rather than the P3M eval split — which matters, because
P3M is the split SalluNet was largely trained on and rembg and RVM never saw.

Every model is run exactly as `baseline_eval.py` runs it, so the comparison is
the same one the table reports:

  * the image is squashed to a 512x512 square, not letterboxed
  * the predicted alpha is resized back to the original frame
  * RVM's four recurrent states start at zero, which is how RVM itself
    handles the first frame of a video

Output per image is a two-row sheet: the cutouts over a checkerboard, and a
zoomed crop of the hair region, chosen automatically as the densest patch of
soft alpha (0.05 < a < 0.95) in the upper half of the frame.

    python compare_visual.py --images IMG_DIR --rvm rvm.onnx --out docs/compare
"""
import argparse
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

SIZE = 512
DEFAULT_IMAGES = Path("testdata")
SALLUNET = Path("dist/sallunet_512.onnx")
# rvm_mobilenetv3_fp32.onnx, from the RobustVideoMatting v1.0.0 release.
RVM = Path("rvm.onnx")

MODELS = ["rembg / U\u00b2-Net", "RVM mobilenetv3", "SalluNet"]


def _square(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)


def _to_full(alpha: np.ndarray, size) -> np.ndarray:
    """Resize a 512x512 alpha in [0,1] back to the original frame."""
    a = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8))
    return np.asarray(a.resize(size, Image.BILINEAR), np.float32) / 255.0


# --------------------------------------------------------------------------
# the three models

def alpha_sallunet(img, sess):
    x = np.asarray(_square(img), np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    return _to_full(out, img.size)


def alpha_rvm(img, sess):
    x = np.asarray(_square(img), np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    zero = np.zeros([1, 1, 1, 1], np.float32)
    names = [o.name for o in sess.get_outputs()]
    feed = {"src": x, "r1i": zero, "r2i": zero, "r3i": zero, "r4i": zero,
            "downsample_ratio": np.array([1.0], np.float32)}
    pha = sess.run(None, feed)[names.index("pha")]
    return _to_full(pha[0, 0], img.size)


def alpha_rembg(img, session):
    from rembg import remove
    buf = io.BytesIO()
    _square(img).save(buf, format="PNG")
    mask = remove(buf.getvalue(), session=session, only_mask=True)
    a = np.asarray(Image.open(io.BytesIO(mask)).convert("L"), np.float32) / 255.0
    return _to_full(a, img.size)


# --------------------------------------------------------------------------
# rendering

def checkerboard(w, h, cell=16, a=(228, 228, 236), b=(202, 202, 214)):
    y, x = np.mgrid[0:h, 0:w]
    m = (((x // cell) + (y // cell)) % 2).astype(bool)
    out = np.empty((h, w, 3), np.uint8)
    out[m] = a
    out[~m] = b
    return out


def composite(img: Image.Image, alpha: np.ndarray) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"), np.float32)
    bg = checkerboard(img.size[0], img.size[1]).astype(np.float32)
    a = alpha[..., None]
    return Image.fromarray((rgb * a + bg * (1 - a)).astype(np.uint8))


def hair_crop_box(alpha: np.ndarray, frac=0.30):
    """Densest window of soft alpha in the upper half — i.e. the hairline.

    Soft pixels are what distinguishes a matte from a segmentation, so this
    puts the crop wherever the models actually disagree.
    """
    h, w = alpha.shape
    band = ((alpha > 0.05) & (alpha < 0.95)).astype(np.float32)
    band[int(h * 0.55):] = 0.0  # hair lives up top; hands and hems do not count

    bh, bw = max(32, int(h * frac)), max(32, int(w * frac))
    # Integral image, so every candidate window is O(1).
    ii = np.pad(band.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.arange(0, h - bh + 1, max(1, bh // 8))
    xs = np.arange(0, w - bw + 1, max(1, bw // 8))
    best, box = -1.0, (0, 0)
    for y in ys:
        for x in xs:
            s = (ii[y + bh, x + bw] - ii[y, x + bw]
                 - ii[y + bh, x] + ii[y, x])
            if s > best:
                best, box = s, (x, y)
    x, y = box
    return (x, y, x + bw, y + bh)


def _font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet(img, alphas, col_w=380, pad=14, label_h=30):
    """Two rows: full cutouts, then the hair crop, one column per model."""
    cols = ["original"] + MODELS
    box = hair_crop_box(alphas["SalluNet"])

    fulls, crops = [], []
    for name in cols:
        pic = img.convert("RGB") if name == "original" \
            else composite(img, alphas[name])
        fulls.append(pic)
        crops.append(pic.crop(box))

    def row(images, width):
        out = []
        for im in images:
            h = round(im.height * width / im.width)
            out.append(im.resize((width, h), Image.LANCZOS))
        return out

    r1 = row(fulls, col_w)
    r2 = row(crops, col_w)
    h1, h2 = r1[0].height, r2[0].height

    W = len(cols) * col_w + (len(cols) + 1) * pad
    H = pad + label_h + h1 + pad + h2 + pad + label_h
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f = _font(17)
    fs = _font(14)

    for i, name in enumerate(cols):
        x = pad + i * (col_w + pad)
        label = name if name != "original" else "input"
        d.text((x, pad + 4), label, fill=(20, 20, 26), font=f)
        canvas.paste(r1[i], (x, pad + label_h))
        canvas.paste(r2[i], (x, pad + label_h + h1 + pad))

    d.text((pad, H - label_h + 2),
           "top: cutout over checkerboard    bottom: hair detail, same crop"
           "    all models run identically at 512\u00b2",
           fill=(110, 110, 125), font=fs)
    return canvas


# --------------------------------------------------------------------------

def band_fraction(alpha):
    """Share of pixels in the soft transition band — a proxy for hair kept."""
    return float(((alpha > 0.05) & (alpha < 0.95)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--out", type=Path, default=Path("docs/compare"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rvm", type=Path, default=RVM)
    ap.add_argument("--sallunet", type=Path, default=SALLUNET)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    s_sallu = ort.InferenceSession(str(args.sallunet), opts,
                                   providers=["CPUExecutionProvider"])
    s_rvm = ort.InferenceSession(str(args.rvm), opts,
                                 providers=["CPUExecutionProvider"])
    from rembg import new_session
    s_rembg = new_session("u2net")

    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".jfif",
                                           ".webp"}
                   and "bg-removed" not in p.name)
    if args.limit:
        paths = paths[:args.limit]

    for p in paths:
        img = Image.open(p).convert("RGB")
        alphas = {
            "rembg / U\u00b2-Net": alpha_rembg(img, s_rembg),
            "RVM mobilenetv3": alpha_rvm(img, s_rvm),
            "SalluNet": alpha_sallunet(img, s_sallu),
        }
        out = args.out / (p.stem + ".jpg")
        sheet(img, alphas).save(out, quality=92)
        stats = "  ".join(f"{k.split()[0]}={band_fraction(v)*100:.2f}%"
                          for k, v in alphas.items())
        print(f"{p.name:55s} soft-band  {stats}")

    print(f"\nwrote {len(paths)} sheets to {args.out}")


if __name__ == "__main__":
    main()
