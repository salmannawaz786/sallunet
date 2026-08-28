"""Pick an encoder/decoder combination that fills the <20 MB budget.

The first student came out at 1.55M params (5.9 MB) -- under a third of the
size budget. For a matting model, unused capacity is unused hair detail, so
this measures real parameter counts for candidate configurations rather than
guessing from published totals (which include a classifier head that
features_only discards).

Runs on CPU: constructing a model and counting parameters needs no GPU.
"""
import modal

from common import app, gpu_image

CANDIDATES = [
    ("mobilenetv4_conv_small.e2400_r224_in1k", 64, 24),
    ("mobilenetv4_conv_small.e2400_r224_in1k", 160, 48),
    ("mobilenetv4_conv_small.e2400_r224_in1k", 224, 64),
    ("mobilenetv3_large_100.ra_in1k", 96, 32),
    ("efficientnet_lite0.ra_in1k", 96, 32),
    ("mobilenetv4_conv_medium.e500_r256_in1k", 64, 24),
    ("efficientvit_b1.r224_in1k", 96, 32),
]


@app.function(image=gpu_image, cpu=4, timeout=1800)
def measure():
    import torch

    import model as M

    rows = []
    for enc, dec_ch, det_ch in CANDIDATES:
        try:
            net = M.MattingStudent(encoder=enc, pretrained=False,
                                   decoder_ch=dec_ch, detail_ch=det_ch)
            n, mb = M.count_params(net)
            with torch.no_grad():
                a, c = net(torch.zeros(1, 3, 512, 512))
            rows.append((enc.split(".")[0], dec_ch, det_ch, n / 1e6, mb,
                         tuple(a.shape[-2:]), "ok"))
        except Exception as e:
            rows.append((enc.split(".")[0], dec_ch, det_ch, 0, 0, None,
                         str(e)[:60]))
    return rows


@app.local_entrypoint()
def main():
    rows = measure.remote()
    print(f"\n{'encoder':<32}{'dec':>5}{'det':>5}{'params(M)':>11}{'MB fp32':>10}  note")
    print("-" * 82)
    for enc, dec, det, n, mb, shape, note in rows:
        flag = "  <-- fits 20MB" if 0 < mb <= 20 else ""
        print(f"{enc:<32}{dec:>5}{det:>5}{n:>11.2f}{mb:>10.1f}  {note}{flag}")
