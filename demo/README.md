---
title: SalluNet
emoji: ✂️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: 15MB human background removal, better hair than rembg
---

# SalluNet

A 4.03M-parameter (15 MB) background-removal model specialised to people,
distilled from BiRefNet. Runs on CPU.

On 500 held-out portraits it produces 45% more accurate hair edges than
rembg/U²-Net (168 MB) and 18% more accurate than RVM.

Code, full benchmark, and the record of failed experiments:
https://github.com/salmannawaz786/sallunet

**Setup:** place `sallunet_512.onnx` in this Space's root directory
(Files → Add file → Upload). The app reads `SALLUNET_ONNX` if you name it
something else.
