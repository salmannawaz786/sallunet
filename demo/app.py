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


def _predict_alpha(image: Image.Image) -> np.ndarray:
    """Return an alpha matte in [0,1] at the image's original resolution."""
    w, h = image.size
    small = image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]  # NCHW

    alpha = SESSION.run(None, {INPUT_NAME: x})[0][0, 0]
    alpha = np.clip(alpha, 0.0, 1.0)

    return np.asarray(
        Image.fromarray((alpha * 255).astype(np.uint8)).resize((w, h),
                                                               Image.BILINEAR),
        dtype=np.float32) / 255.0


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
