"""SalluNet demo — human background removal, running on CPU via ONNX Runtime.

Inference deliberately mirrors training: the image is resized to a 512x512
square (not letterboxed) because that is the distribution the model was trained
on. Aspect ratio is restored by resizing the predicted alpha back to the
original dimensions, so the output matches the input frame.
"""
import os

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = os.environ.get("SALLUNET_ONNX", "sallunet_512.onnx")
SIZE = 512

_opts = ort.SessionOptions()
_opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
SESSION = ort.InferenceSession(MODEL_PATH, _opts,
                               providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name


def _box_filter(img: np.ndarray, r: int) -> np.ndarray:
    try:
        import cv2
        return cv2.boxFilter(img, -1, (r, r))
    except Exception:
        h, w = img.shape
        pad = np.pad(img, ((r, r), (r, r)), mode="edge")
        ii = np.pad(pad.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        res = (ii[2 * r + 1:2 * r + 1 + h, 2 * r + 1:2 * r + 1 + w]
               - ii[0:h, 2 * r + 1:2 * r + 1 + w]
               - ii[2 * r + 1:2 * r + 1 + h, 0:w]
               + ii[0:h, 0:w])
        return res / ((2 * r + 1) * (2 * r + 1))


def _guided_filter(guide_rgb: np.ndarray, src_alpha: np.ndarray, r: int = 4, eps: float = 1e-3) -> np.ndarray:
    """Guided filter snapping 512 alpha edges to full-res RGB photo boundaries."""
    gray = (0.299 * guide_rgb[..., 0] + 0.587 * guide_rgb[..., 1] + 0.114 * guide_rgb[..., 2]).astype(np.float32)
    mean_I = _box_filter(gray, r)
    mean_p = _box_filter(src_alpha, r)
    mean_Ip = _box_filter(gray * src_alpha, r)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = _box_filter(gray * gray, r)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)
    q = mean_a * gray + mean_b
    return np.clip(q, 0.0, 1.0)


def _predict_alpha(image: Image.Image) -> np.ndarray:
    """Return a crisp, edge-aligned alpha matte in [0,1] at original resolution."""
    w, h = image.size
    small = image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]  # NCHW

    alpha_512 = SESSION.run(None, {INPUT_NAME: x})[0][0, 0]
    alpha_512 = np.clip(alpha_512, 0.0, 1.0)

    # 1. Bicubic upsample to original dimensions
    a_img = Image.fromarray((alpha_512 * 255).astype(np.uint8))
    alpha_up = np.asarray(a_img.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0

    # 2. Guided Filter alignment with original full-res RGB image
    rgb_norm = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    alpha_guided = _guided_filter(rgb_norm, alpha_up, r=4, eps=1e-3)

    # 3. Clean background noise floor & solid foreground core
    alpha_guided[alpha_guided < 0.015] = 0.0
    alpha_guided[alpha_guided > 0.985] = 1.0

    # 4. Adaptive contrast S-curve for crisp solid edges + soft hair
    gamma = 1.4
    alpha_refined = np.where(
        alpha_guided < 0.5,
        0.5 * np.power(2.0 * alpha_guided, gamma),
        1.0 - 0.5 * np.power(2.0 * (1.0 - alpha_guided), gamma)
    )
    return np.clip(alpha_refined, 0.0, 1.0)


def _hex_to_rgb(value: str):
    value = (value or "#FFFFFF").lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (255, 255, 255)


def run(image, background, colour):
    if image is None:
        return None, None

    image = image.convert("RGB")
    alpha = _predict_alpha(image)
    rgb = np.asarray(image, dtype=np.float32)
    a3 = alpha[..., None]

    if background == "Transparent":
        rgba = np.dstack([rgb, alpha * 255]).astype(np.uint8)
        cutout = Image.fromarray(rgba, mode="RGBA")
    else:
        bg = np.array(_hex_to_rgb(colour), dtype=np.float32)
        cutout = Image.fromarray(
            (rgb * a3 + bg * (1.0 - a3)).astype(np.uint8), mode="RGB")

    matte = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    return cutout, matte


DESCRIPTION = """
# SalluNet

**A 15 MB background-removal model for people.** 4.03M parameters, distilled
from BiRefNet, running here on CPU.

On a held-out benchmark of 500 portraits it produces **45% more accurate hair
edges than rembg** (a 168 MB model) and **18% more accurate than RVM**, the
closest competitor by size.

It only does humans — that narrowness is exactly why it can be this small.
"""

NOTES = """
### Honest limitations

* **People only.** Give it a car or a dog and the output is meaningless.
* **Still images.** No temporal consistency, so it will flicker on video.
* **Benchmarked in-domain.** The eval set and most training data come from the
  same portrait dataset, so real-world photos may be harder than the numbers
  suggest.
* **Hard cases stay hard.** Fine flyaway hair against a busy background, heavy
  backlighting, and motion blur all degrade the matte.

Full benchmark, method, and the complete record of what *didn't* work:
[github.com/salmannawaz786/sallunet](https://github.com/salmannawaz786/sallunet)
"""

with gr.Blocks(title="SalluNet — human background removal") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Photo of a person")
            background = gr.Radio(["Transparent", "Solid colour"],
                                  value="Transparent", label="Background")
            colour = gr.ColorPicker(value="#FFFFFF", label="Colour",
                                    visible=False)
            go = gr.Button("Remove background", variant="primary")
        with gr.Column():
            out_cutout = gr.Image(type="pil", label="Result", format="png")
            out_matte = gr.Image(type="pil", label="Alpha matte")

    background.change(lambda choice: gr.update(visible=choice == "Solid colour"),
                      inputs=background, outputs=colour)
    go.click(run, inputs=[inp, background, colour],
             outputs=[out_cutout, out_matte])
    inp.upload(run, inputs=[inp, background, colour],
               outputs=[out_cutout, out_matte])

    gr.Markdown(NOTES)

if __name__ == "__main__":
    demo.launch()
