"""Stage 4: fine-tune v1/best.pt at 1024x1024 for hair detail.

The v1 model was trained entirely at 512. Its held-out failures were mostly
edges that are blurrier than the human ground truth, and edge sharpness is
resolution-bound: at 512 a hair strand is under a pixel wide, so no loss
function can recover it. Doubling the training resolution is the direct fix.

This deliberately changes ONE thing at a time -- resolution. Augmentation is
back to the exact v1 recipe (a heavier v2 recipe measured 43% worse), the loss
weights are v1's, and the initialisation is v1's EMA weights. If this run wins,
resolution was the cause; if it loses, it was not.

Metric note: SAD scales with pixel count, so numbers here are NOT comparable to
v1's 512-resolution numbers. Compare via evaluate.py, which runs both models
through an identical pipeline.

    modal run stage4_finetune.py::smoke     # 100 steps, ~$0.10
    python launch_finetune.py               # real run, server-side
"""
import modal

from common import CKPT, DATA, app, ckpt_vol, data_vol, gpu_image

VOLS = {str(DATA): data_vol, str(CKPT): ckpt_vol}

P3M_DIR = DATA / "shards" / "p3m"
COCO_DIR = DATA / "shards" / "BiRefNet_HR-matting"
EVAL_TAR = DATA / "shards" / "p3m-eval" / "P3M_500_NP.tar"

INIT_FROM = CKPT / "v1" / "best.pt"
RUN = "ft1024"
CKPT_DIR = CKPT / RUN
LATEST = CKPT_DIR / "latest.pt"
HISTORY = CKPT_DIR / "history.jsonl"

CROP = 1024
BATCH = 8
LR = 5e-5          # fine-tune, not a fresh run: 6x lower than v1's 3e-4
WEIGHT_DECAY = 0.01
# 250 rather than v1's 100: v1's eval bounced so hard between adjacent steps
# that a three-part "fix" got built on what was partly sampling noise.
EVAL_N = 250
EVAL_EVERY = 250
CKPT_EVERY = 250
LOG_EVERY = 25

W_L1, W_GRAD, W_LAP, W_COARSE = 1.0, 0.5, 0.5, 0.25
MIX_WEIGHTS = [0.7, 0.3]


def _load_init(model):
    """Initialise from v1's EMA weights -- the ones its best.pt was chosen on."""
    import torch

    ck = torch.load(INIT_FROM, map_location="cuda", weights_only=False)
    sd = model.state_dict()
    if ck.get("ema"):
        shadow, buffers = ck["ema"]["shadow"], ck["ema"]["buffers"]
        model.load_state_dict({
            k: (shadow[k].to(sd[k].dtype) if k in shadow else buffers[k]).cuda()
            for k in sd})
        src = "ema"
    else:
        model.load_state_dict(ck["model"])
        src = "raw"
    return ck.get("step"), src


def _run(total_steps, batch=BATCH, resume=True, tag="ft"):
    import json
    import time

    import torch
    from torch.utils.data import DataLoader

    from data import EvalDataset, ShardDataset
    from losses import matting_loss, matting_metrics
    from model import MattingStudent, count_params
    from stage3_train import EMA, _load, _save

    p3m = sorted(P3M_DIR.glob("*.tar"))
    coco = sorted(COCO_DIR.glob("*.tar"))
    if not p3m or not coco:
        raise RuntimeError(f"missing shards: p3m={len(p3m)} coco={len(coco)}")
    if not INIT_FROM.exists():
        raise RuntimeError(f"no init checkpoint at {INIT_FROM}")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    model = MattingStudent().cuda().to(memory_format=torch.channels_last)
    n_params, mb = count_params(model)
    init_step, src = _load_init(model)
    print(f"student: {n_params/1e6:.2f}M params, {mb:.1f} MB fp32", flush=True)
    print(f"init: v1/best.pt step {init_step} ({src} weights)", flush=True)
    print(f"config: run={RUN} crop={CROP} steps={total_steps} batch={batch} "
          f"lr={LR} w_coarse={W_COARSE} eval_n={EVAL_N}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    ema = EMA(model)
    eval_model = MattingStudent(pretrained=False).cuda().to(
        memory_format=torch.channels_last)

    step, best = 0, float("inf")
    if resume and LATEST.exists():
        step, best = _load(LATEST, model, opt, scaler, sched, total_steps, ema)
        print(f"resumed from step {step} (best band_mad {best:.3f})", flush=True)
    if step >= total_steps:
        return {"step": step, "best": best}

    train_ds = ShardDataset([p3m, coco], MIX_WEIGHTS, bg_tars=coco,
                            crop=CROP, seed=step + 1)
    loader = DataLoader(train_ds, batch_size=batch, num_workers=8,
                        pin_memory=True, prefetch_factor=4, persistent_workers=True)
    eval_ds = EvalDataset(EVAL_TAR, size=CROP, limit=EVAL_N)
    eval_loader = DataLoader(eval_ds, batch_size=4, num_workers=2)
    print(f"eval: {len(eval_ds)} images at {CROP}x{CROP}", flush=True)

    t0, seen = time.time(), 0
    model.train()
    for images, alphas in loader:
        if step >= total_steps:
            break
        images = images.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        alphas = alphas.cuda(non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, coarse = model(images)
            loss, parts = matting_loss(pred.float(), coarse.float(), alphas,
                                       w_l1=W_L1, w_grad=W_GRAD,
                                       w_lap=W_LAP, w_coarse=W_COARSE)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        seen += images.shape[0]
        ema.update(model, step)

        if step % LOG_EVERY == 0:
            print(f"step {step:>5}/{total_steps}  loss {loss.item():.4f}  "
                  f"l1 {parts['l1']:.4f}  grad {parts['grad']:.4f}  "
                  f"{seen/(time.time()-t0):.1f} img/s", flush=True)

        if step % EVAL_EVERY == 0 or step == total_steps:
            ema.copy_to(eval_model)
            eval_model.eval()
            agg, nb = {}, 0
            with torch.no_grad():
                for ims, als in eval_loader:
                    ims = ims.cuda().to(memory_format=torch.channels_last)
                    als = als.cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p, _ = eval_model(ims)
                    for k, v in matting_metrics(p.float().clamp(0, 1), als).items():
                        agg[k] = agg.get(k, 0.0) + v
                    nb += 1
            metrics = {k: v / max(nb, 1) for k, v in agg.items()}
            print(f"  EVAL step {step}: " +
                  "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()), flush=True)

            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            with open(HISTORY, "a") as fh:
                fh.write(json.dumps({"step": step, "tag": tag, **metrics}) + "\n")
            if metrics["band_mad"] < best:
                best = metrics["band_mad"]
                _save(CKPT_DIR / "best.pt", model, opt, scaler, sched, step, best,
                      total_steps, ema)

        if step % CKPT_EVERY == 0:
            _save(LATEST, model, opt, scaler, sched, step, best, total_steps, ema)

    _save(LATEST, model, opt, scaler, sched, step, best, total_steps, ema)
    el = time.time() - t0
    return {"step": step, "best": best, "seconds": round(el, 1),
            "img_per_s": round(seen / el, 2) if el else 0}


@app.function(image=gpu_image, gpu="A100-40GB", volumes=VOLS, cpu=16,
              timeout=86400, scaledown_window=2,
              retries=modal.Retries(max_retries=3, backoff_coefficient=2.0))
def finetune(steps: int, batch: int = BATCH, resume: bool = True, tag: str = "ft"):
    return _run(steps, batch, resume, tag)


@app.local_entrypoint()
def smoke(steps: int = 100):
    print(finetune.remote(steps, BATCH, False, "smoke"))
