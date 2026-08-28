"""Student matting network: ~5M params, targeting a <20 MB fp32 ONNX export.

Structure follows the standard matting split, which exists because the two jobs
have opposite requirements:

  * a semantic branch, run on strided encoder features, decides *where* the
    person is -- needs receptive field, tolerates low resolution;
  * a detail branch, run near full resolution but only a few channels deep,
    decides how the *edge* falls -- needs resolution, not receptive field.

Predicting one alpha from strided features alone is what produces the blobby
silhouettes we saw from the teacher comparison: by the time features reach
stride 32 the hair is gone, and no decoder can invent it back.

The detail branch outputs a residual rather than an alpha, so at initialisation
the network reproduces the semantic prediction and learns edges from there.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER = "mobilenetv4_conv_small.e2400_r224_in1k"


def conv_bn(cin, cout, k=3, s=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class FPNDecoder(nn.Module):
    """Light top-down decoder. Lateral 1x1 + add + 3x3 smooth.

    `stop_at` is the finest encoder level the decoder descends to. Profiling
    showed that running this to stride 2 (256x256 at a 512 input) cost 61% of
    total inference time, because a channel at stride 2 costs 16x the FLOPs of
    the same channel at stride 8. It was also redundant: resolving the edge at
    full resolution is the detail branch's job, so the semantic decoder was
    paying 16x to duplicate work done better elsewhere.

    Stopping at stride 4 keeps the channel width -- and the semantic capacity --
    while cutting the decoder's cost roughly 4x.
    """

    def __init__(self, in_chs, ch=64, stop_at=1):
        super().__init__()
        self.stop_at = stop_at
        self.lateral = nn.ModuleList(nn.Conv2d(c, ch, 1) for c in in_chs)
        self.smooth = nn.ModuleList(conv_bn(ch, ch) for _ in in_chs)

    def forward(self, feats):
        out = self.lateral[-1](feats[-1])
        out = self.smooth[-1](out)
        for i in range(len(feats) - 2, self.stop_at - 1, -1):
            lat = self.lateral[i](feats[i])
            out = F.interpolate(out, size=lat.shape[-2:], mode="bilinear",
                                align_corners=False) + lat
            out = self.smooth[i](out)
        return out


class DetailBranch(nn.Module):
    """Shallow, high-resolution refinement.

    Runs at stride 2 rather than stride 1: full resolution roughly quadruples
    this branch's cost for a marginal gain, since the final bilinear upsample
    of a *residual* is far less damaging than upsampling an alpha.
    """

    def __init__(self, ch=24, sem_ch=64):
        super().__init__()
        self.stem = conv_bn(4, ch, k=3, s=2)
        self.body = nn.Sequential(conv_bn(ch, ch), conv_bn(ch, ch))
        self.fuse = conv_bn(ch + sem_ch, ch)
        self.head = nn.Conv2d(ch, 1, 3, 1, 1)
        # Start as a no-op so training begins from the semantic prediction.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image, coarse_logit, sem_feat):
        x = self.stem(torch.cat([image, coarse_logit], dim=1))
        x = self.body(x)
        sem = F.interpolate(sem_feat, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        return self.head(self.fuse(torch.cat([x, sem], dim=1)))


class MattingStudent(nn.Module):
    # Capacity is weighted toward the decoder and detail branch rather than a
    # bigger encoder: locating a person is the easy half of this task, resolving
    # the hair edge is the hard half.
    #
    # decoder_stop=1 keeps the semantic decoder at stride 4. With stop=0 the
    # same model measured 601 ms on 4 CPU threads (3.4x slower than RVM) with
    # 61% of that in the decoder alone; the detail branch already handles fine
    # structure at higher resolution.
    def __init__(self, encoder=ENCODER, pretrained=True, decoder_ch=224,
                 detail_ch=32, decoder_stop=1):
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            encoder, pretrained=pretrained, features_only=True
        )
        chs = self.encoder.feature_info.channels()
        self.decoder = FPNDecoder(chs, decoder_ch, stop_at=decoder_stop)
        self.sem_head = nn.Conv2d(decoder_ch, 1, 3, 1, 1)
        self.detail = DetailBranch(detail_ch, sem_ch=decoder_ch)

    def forward(self, image):
        """image: float tensor in [0,1], NCHW. Returns (alpha, coarse_alpha)."""
        feats = self.encoder(image)
        sem_feat = self.decoder(feats)
        coarse_logit = self.sem_head(sem_feat)

        up = F.interpolate(coarse_logit, size=image.shape[-2:], mode="bilinear",
                           align_corners=False)
        residual = self.detail(image, up, sem_feat)
        residual = F.interpolate(residual, size=image.shape[-2:], mode="bilinear",
                                 align_corners=False)

        alpha = torch.sigmoid(up + residual)
        coarse = torch.sigmoid(up)
        return alpha, coarse


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total, total * 4 / 1024 ** 2  # params, fp32 MB


def arch_from_checkpoint(ck):
    """Recover the architecture a checkpoint was trained with.

    Checkpoints saved before this existed carry no architecture record, so the
    widths are read back out of the tensor shapes. `decoder_stop` cannot be
    recovered that way -- the FPN builds a lateral/smooth pair for every level
    regardless of how far down it actually descends, so the state dict is
    identical either way. It is inferred from detail_ch instead, which is exact
    for this project's runs: the stride-2 decoder (stop=0) only ever shipped
    with detail_ch=64, and the stride-4 rewrite with detail_ch=32.
    """
    if ck.get("arch"):
        return dict(ck["arch"])

    sd = ck.get("model") or {}
    detail_ch = sd["detail.stem.0.weight"].shape[0] if "detail.stem.0.weight" in sd else 64
    decoder_ch = (sd["decoder.lateral.0.weight"].shape[0]
                  if "decoder.lateral.0.weight" in sd else 224)
    return {"decoder_ch": int(decoder_ch), "detail_ch": int(detail_ch),
            "decoder_stop": 0 if detail_ch == 64 else 1}


def build_from_checkpoint(ck):
    """Instantiate the right architecture, then load EMA weights if present."""
    arch = arch_from_checkpoint(ck)
    model = MattingStudent(pretrained=False, **arch).eval()
    sd = model.state_dict()
    if ck.get("ema"):
        shadow, buffers = ck["ema"]["shadow"], ck["ema"]["buffers"]
        model.load_state_dict({
            k: (shadow[k].to(sd[k].dtype) if k in shadow else buffers[k])
            for k in sd})
        which = "ema"
    else:
        model.load_state_dict(ck["model"])
        which = "raw"
    return model, arch, which
