"""Matting losses, built around one problem: the transition band is tiny.

In a typical frame ~95% of pixels are trivially background or trivially body.
A plain L1 loss reaches a very low value by predicting a clean silhouette and
ignoring hair entirely -- which is exactly the blobby output we are trying to
avoid. Every term here exists to stop that:

  band-weighted L1  -- pixels where the ground-truth alpha is fractional count
                       BAND_WEIGHT times more, so hair dominates the gradient
  gradient (Sobel)  -- penalises a smooth blob where the target has structure;
                       an L1-optimal blur has near-zero gradient error only if
                       the target is also blurred
  Laplacian pyramid -- matches alpha across scales, which stabilises the large
                       flat regions that the band weighting deliberately
                       de-emphasises

Metrics (SAD/MAD/Grad) follow the conventions used by the matting literature so
the eventual README table is comparable to published numbers.
"""
import torch
import torch.nn.functional as F

BAND_LO, BAND_HI = 0.05, 0.95
BAND_WEIGHT = 9.0  # transition pixels weigh 1 + 9 = 10x a trivial pixel


def band_mask(gt):
    return ((gt > BAND_LO) & (gt < BAND_HI)).float()


def weighted_l1(pred, gt, band_weight=BAND_WEIGHT):
    """L1 with transition pixels up-weighted.

    The weight is a direct quality trade, not a free win: at band_weight=9 the
    measured result was best-in-class hair (band_mad 136 vs RVM's 166) but
    whole-image accuracy well behind RVM (mad 20.6 vs 11.7). Lowering it moves
    error back from the interior to the edge.
    """
    w = 1.0 + band_weight * band_mask(gt)
    return ((pred - gt).abs() * w).sum() / w.sum().clamp_min(1.0)


def _sobel(x):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy


def gradient_loss(pred, gt):
    px, py = _sobel(pred)
    gx, gy = _sobel(gt)
    return (px - gx).abs().mean() + (py - gy).abs().mean()


def laplacian_loss(pred, gt, levels=4):
    """L1 between Laplacian pyramids, weighted so coarse levels matter less."""
    loss = 0.0
    p, g = pred, gt
    for i in range(levels):
        if min(p.shape[-2:]) < 4:
            break
        pd = F.avg_pool2d(p, 2)
        gd = F.avg_pool2d(g, 2)
        pu = F.interpolate(pd, size=p.shape[-2:], mode="bilinear", align_corners=False)
        gu = F.interpolate(gd, size=g.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + (2 ** -i) * ((p - pu) - (g - gu)).abs().mean()
        p, g = pd, gd
    return loss


def matting_loss(alpha, coarse, gt, w_l1=1.0, w_grad=0.5, w_lap=0.5,
                 w_coarse=0.25, band_weight=BAND_WEIGHT):
    """Total loss. `coarse` is supervised too so the semantic branch stays honest.

    Without the coarse term the detail branch can mask a badly-trained semantic
    branch on easy frames, then fall apart when the subject is unusual.
    """
    l1 = weighted_l1(alpha, gt, band_weight)
    grad = gradient_loss(alpha, gt)
    lap = laplacian_loss(alpha, gt)
    coarse_l1 = F.l1_loss(coarse, gt)
    total = w_l1 * l1 + w_grad * grad + w_lap * lap + w_coarse * coarse_l1
    return total, {"l1": l1.detach(), "grad": grad.detach(),
                   "lap": lap.detach(), "coarse": coarse_l1.detach()}


# ---------------------------------------------------------------------------
# Evaluation metrics (alpha in [0,1]; literature reports SAD/MSE scaled by 1e3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def matting_metrics(pred, gt):
    n = pred.shape[0]
    diff = (pred - gt).abs().view(n, -1)
    sad = diff.sum(1) / 1000.0                 # SAD, conventional 1e-3 scale
    mad = diff.mean(1) * 1e3                   # MAD x1e3
    mse = ((pred - gt) ** 2).view(n, -1).mean(1) * 1e3

    px, py = _sobel(pred)
    gx, gy = _sobel(gt)
    grad = ((px - gx).abs() + (py - gy).abs()).view(n, -1).mean(1) * 1e3

    band = band_mask(gt).view(n, -1)
    band_mad = (diff * band).sum(1) / band.sum(1).clamp_min(1.0) * 1e3

    return {"sad": sad.mean().item(), "mad": mad.mean().item(),
            "mse": mse.mean().item(), "grad": grad.mean().item(),
            "band_mad": band_mad.mean().item()}
