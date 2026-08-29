---
title: SalluNet
emoji: ✂️
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: 15MB human background removal, better hair than rembg
models:
  - salluu3432/sallunet
---

# SalluNet — in-browser demo

A 4.03M-parameter (15 MB) background-removal model specialised to people,
distilled from BiRefNet. Inference runs entirely in the browser through
onnxruntime-web, so no image is ever uploaded.

On 500 held-out portraits it produces 45% more accurate hair edges than
rembg/U²-Net (168 MB) and 18% more accurate than RVM.

* Live: https://huggingface.co/spaces/salluu3432/sallunet-demo
* Weights: https://huggingface.co/salluu3432/sallunet
* Code and full benchmark: https://github.com/salmannawaz786/sallunet

## Why a static Space rather than Gradio

Gradio Spaces need a PRO subscription on free CPU hardware; static Spaces are
free. Moving inference into the browser also removes the server round-trip and
means the demo scales to any number of visitors at zero cost. The Gradio version
is still in [`demo/`](../demo) for anyone who wants a server-side app.

## Threading

Serving `onnxruntime-web`'s threaded wasm build requires `SharedArrayBuffer`,
which requires cross-origin isolation, which requires COOP/COEP headers a static
host cannot set. `coi-serviceworker.js` installs a service worker that re-serves
same-origin responses with those headers.

Thread count is `hardwareConcurrency / 2`, not `hardwareConcurrency`: that
property counts hyperthreads, and oversubscribing measurably hurts. Measured
natively on a 2-core/4-thread laptop:

| threads | latency |
|---|---|
| 1 | 1676 ms |
| 2 | **1159 ms** |
| 4 | 1602 ms |

If the service worker does not install — an insecure context, or an embedding
frame that does not grant `cross-origin-isolated` — the page falls back to a
single thread and still works, just slower.

The comparison figure is inlined as a `data:` URI rather than linked. Hugging
Face serves files from a separate CDN origin, and a `no-cors` `<img>` yields an
opaque response the service worker cannot attach CORP to, so a linked image
would be blocked exactly when isolation succeeds.

## Vendored files not in git

Two binaries are excluded by `.gitignore` and must be fetched before deploying:

```bash
cp ../dist/sallunet_512.onnx .
curl -L -o ort/ort-wasm-simd-threaded.wasm \
  https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.wasm
```

`ort/ort.min.mjs` and `ort/ort-wasm-simd-threaded.mjs` are committed. Everything
the page loads is same-origin — nothing is pulled from a CDN at runtime, which
is what makes `require-corp` safe.

## Running locally

Any static server works, but it must be a secure context for the service worker:

```bash
python -m http.server 5201
```

Then open http://localhost:5201 (localhost counts as secure).
