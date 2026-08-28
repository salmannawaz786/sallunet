"""Stage 3: train the student, with resume-from-anywhere checkpointing.

Serverless containers get preempted and time out. Every checkpoint therefore
carries enough state to make a restart indistinguishable from never having
stopped: weights, optimizer moments, AMP scaler, LR schedule position, step
count, and the Python/torch RNG streams. Saving weights alone would silently
restart the LR schedule and reshuffle the data stream, which looks like
training working while quietly wasting the run.

Checkpoints are written to a temp file and renamed, then committed to the
Volume -- a container killed mid-write leaves the previous checkpoint intact
rather than a truncated file that fails to load.

    modal run stage3_train.py::pilot            # 200 steps, ~$0.30, sanity check
    modal run stage3_train.py::bench            # A100 vs H100 cost per 1k steps
    modal run stage3_train.py::main --steps 60000
"""
import modal

from common import CKPT, DATA, app, ckpt_vol, data_vol, gpu_image

VOLS = {str(DATA): data_vol, str(CKPT): ckpt_vol}

P3M_DIR = DATA / "shards" / "p3m"
COCO_DIR = DATA / "shards" / "BiRefNet_HR-matting"
EVAL_TAR = DATA / "shards" / "p3m-eval" / "P3M_500_NP.tar"

DEFAULT_RUN = "v4"


def run_paths(run):
    d = CKPT / run
    return d, d / "latest.pt", d / "history.jsonl"

CKPT_EVERY = 500
EVAL_EVERY = 500
EVAL_N = 100
LOG_EVERY = 50

BATCH = 16
LR = 3e-4
# v1's value. This run changes ONE thing (the decoder stride level), so every
# other knob matches v1 exactly or the comparison means nothing.
W_COARSE = 0.25
W_L1, W_GRAD, W_LAP = 1.0, 0.5, 0.5
# Transition pixels weigh 1 + BAND_WEIGHT. At 9 (v1) hair was
# best-in-class but whole-image accuracy trailed RVM; this is the
# knob that trades between the two.
BAND_WEIGHT = 9.0
WEIGHT_DECAY = 0.01
WARMUP = 300
# P3M's alphas are human-drawn; COCO's are pseudo-labels that cap out at the
# teacher's quality. Weighted toward P3M, but not so far that the model never
# sees a real face or a cluttered background.
MIX_WEIGHTS = [0.7, 0.3]


class EMA:
    """Exponential moving average of weights.

    The raw weights at the end of a run sit wherever the last few noisy batches
    pushed them; an average over recent history is consistently a little better
    and costs nothing at inference. The decay is ramped in over early steps --
    a fixed 0.999 from step 0 would keep the random initialisation weighted in
    for thousands of steps.
    """

    def __init__(self, model, decay=0.999):
        import torch
        self.decay = decay
        self.shadow, self.buffers = {}, {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone().float()
            else:
                self.buffers[k] = v.detach().clone()

    def update(self, model, step):
        import torch
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
                else:
                    self.buffers[k] = v.detach().clone()

    def copy_to(self, target):
        sd = target.state_dict()
        merged = {k: self.shadow[k].to(sd[k].dtype) if k in self.shadow
                  else self.buffers[k] for k in sd}
        target.load_state_dict(merged)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow, "buffers": self.buffers}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = {k: v.cuda() for k, v in sd["shadow"].items()}
        self.buffers = {k: v.cuda() for k, v in sd["buffers"].items()}


def _save(path, model, opt, scaler, sched, step, best, total_steps,
          ema=None, arch=None):
    """Atomic checkpoint write: temp file, fsync, rename, then commit."""
    import os
    import random

    import numpy as np
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "best": best,
        "total_steps": total_steps,
        "arch": arch,
        "ema": ema.state_dict() if ema is not None else None,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "sched": sched.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
    }, tmp)
    os.replace(tmp, path)
    ckpt_vol.commit()


def _load(path, model, opt, scaler, sched, total_steps, ema=None):
    import random

    import numpy as np
    import torch

    ck = torch.load(path, map_location="cuda", weights_only=False)
    # OneCycleLR state is meaningless across different schedule lengths: loading
    # a 300-step schedule into a 12000-step run silently ruins the LR curve and
    # looks like training working. Fail loudly instead.
    prev = ck.get("total_steps")
    if prev is not None and prev != total_steps:
        raise RuntimeError(
            f"checkpoint at {path} was built for total_steps={prev}, "
            f"but this run wants {total_steps}. Use a new RUN name or delete "
            f"the checkpoint -- resuming would corrupt the LR schedule.")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    sched.load_state_dict(ck["sched"])
    if ema is not None and ck.get("ema"):
        ema.load_state_dict(ck["ema"])
    random.setstate(ck["rng_python"])
    np.random.set_state(ck["rng_numpy"])
    torch.set_rng_state(ck["rng_torch"].cpu())
    try:
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
    except Exception:
        pass  # different GPU count on resume; not worth failing the run over
    return ck["step"], ck.get("best", float("inf"))


def _train(total_steps, batch=BATCH, resume=True, eval_n=EVAL_N, tag="",
           run=DEFAULT_RUN, band_weight=None, detail_ch=64,
           decoder_stop=1, w_coarse=None, decoder_ch=224):
    import json
    import time

    import torch
    from torch.utils.data import DataLoader

    from data import EvalDataset, ShardDataset
    from losses import matting_loss, matting_metrics
    from model import MattingStudent, count_params

    # No volume reload here: the mount is already current at container start,
    # and reload() fails outright if anything holds an open file under /data --
    # HuggingFace's xet cache keeps a log there, since HF_HOME lives on the
    # volume so model weights survive between runs.
    p3m = sorted(P3M_DIR.glob("*.tar"))
    coco = sorted(COCO_DIR.glob("*.tar"))
    if not p3m or not coco:
        raise RuntimeError(f"missing shards: p3m={len(p3m)} coco={len(coco)}")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    band_weight = BAND_WEIGHT if band_weight is None else band_weight
    w_coarse = W_COARSE if w_coarse is None else w_coarse
    CKPT_DIR, LATEST, HISTORY = run_paths(run)

    model = MattingStudent(decoder_ch=decoder_ch, detail_ch=detail_ch,
                           decoder_stop=decoder_stop
                           ).cuda().to(memory_format=torch.channels_last)
    n_params, mb = count_params(model)
    print(f"student: {n_params/1e6:.2f}M params, {mb:.1f} MB fp32", flush=True)
    print(f"config: run={run} steps={total_steps} batch={batch} lr={LR} "
          f"w_coarse={w_coarse} band_w={band_weight} detail_ch={detail_ch} "
          f"decoder_stop={decoder_stop} decoder_ch={decoder_ch} mix={MIX_WEIGHTS} "
          f"p3m_shards={len(p3m)} coco_shards={len(coco)}", flush=True)

    arch = {"decoder_ch": decoder_ch, "detail_ch": detail_ch,
            "decoder_stop": decoder_stop}
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    # Clamped: a short pilot run has fewer steps than WARMUP, which would give
    # pct_start > 1 and refuse to build the schedule.
    pct_start = min(0.3, max(0.02, WARMUP / max(total_steps, 1)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=total_steps,
        pct_start=pct_start, anneal_strategy="cos")

    ema = EMA(model)
    # Must mirror the training model exactly: EMA weights are copied into it.
    eval_model = MattingStudent(pretrained=False, decoder_ch=decoder_ch,
                                detail_ch=detail_ch,
                                decoder_stop=decoder_stop).cuda().to(
        memory_format=torch.channels_last)

    step, best = 0, float("inf")
    if resume and LATEST.exists():
        step, best = _load(LATEST, model, opt, scaler, sched, total_steps, ema)
        print(f"resumed from step {step} (best band_mad {best:.3f})", flush=True)
    else:
        print("starting fresh", flush=True)

    if step >= total_steps:
        print(f"already at {step}/{total_steps}; nothing to do", flush=True)
        return {"step": step, "best": best}

    train_ds = ShardDataset([p3m, coco], MIX_WEIGHTS, bg_tars=coco, seed=step)
    loader = DataLoader(train_ds, batch_size=batch, num_workers=8,
                        pin_memory=True, prefetch_factor=4, persistent_workers=True)
    eval_ds = EvalDataset(EVAL_TAR, limit=eval_n)
    eval_loader = DataLoader(eval_ds, batch_size=8, num_workers=2)
    print(f"eval set: {len(eval_ds)} images from P3M_500_NP", flush=True)

    t0 = time.time()
    seen = 0
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
                                       w_lap=W_LAP, w_coarse=w_coarse,
                                       band_weight=band_weight)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        ema.update(model, step)
        seen += images.shape[0]

        if step % LOG_EVERY == 0:
            ips = seen / (time.time() - t0)
            print(f"step {step:>6}/{total_steps}  loss {loss.item():.4f}  "
                  f"l1 {parts['l1']:.4f}  grad {parts['grad']:.4f}  "
                  f"{ips:.1f} img/s", flush=True)

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
                    m = matting_metrics(p.float().clamp(0, 1), als)
                    for k, v in m.items():
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
                      total_steps, ema, arch)

        if step % CKPT_EVERY == 0:
            _save(LATEST, model, opt, scaler, sched, step, best, total_steps, ema, arch)

    _save(LATEST, model, opt, scaler, sched, step, best, total_steps, ema, arch)
    elapsed = time.time() - t0
    return {"step": step, "best": best, "seconds": round(elapsed, 1),
            "img_per_s": round(seen / elapsed, 2) if elapsed else 0,
            "params_m": round(n_params / 1e6, 2)}


@app.function(image=gpu_image, gpu="A100-40GB", volumes=VOLS, cpu=16,
              timeout=86400, scaledown_window=2,
              retries=modal.Retries(max_retries=3, backoff_coefficient=2.0))
def train_a100(steps: int, batch: int = BATCH, resume: bool = True,
               tag: str = "a100", run: str = DEFAULT_RUN,
               band_weight: float = None, detail_ch: int = 64,
               decoder_stop: int = 1, w_coarse: float = None,
               decoder_ch: int = 224):
    return _train(steps, batch, resume, tag=tag, run=run,
                  band_weight=band_weight, detail_ch=detail_ch,
                  decoder_stop=decoder_stop, w_coarse=w_coarse,
                  decoder_ch=decoder_ch)


@app.function(image=gpu_image, gpu="H100", volumes=VOLS, cpu=16,
              timeout=86400, scaledown_window=2,
              retries=modal.Retries(max_retries=3, backoff_coefficient=2.0))
def train_h100(steps: int, batch: int = BATCH, resume: bool = True, tag: str = "h100"):
    return _train(steps, batch, resume, tag=tag)


@app.local_entrypoint()
def pilot(steps: int = 200):
    """Cheap sanity check: does the loss move and is the model the right size?"""
    print(train_a100.remote(steps, BATCH, False, "pilot"))


@app.local_entrypoint()
def bench(steps: int = 200):
    """Measure real cost per 1k steps on each GPU instead of guessing."""
    rates = {}
    for name, fn, price in (("A100-40GB", train_a100, 0.000583),
                            ("H100", train_h100, 0.001097)):
        r = fn.remote(steps, BATCH, False, f"bench-{name}")
        ips = r["img_per_s"]
        sec_per_1k = (1000 * BATCH) / ips if ips else 0
        rates[name] = (ips, sec_per_1k, sec_per_1k * price)
    print(f"\n{'gpu':<12}{'img/s':>9}{'s/1k steps':>13}{'$/1k steps':>13}")
    for k, (ips, s, cost) in rates.items():
        print(f"{k:<12}{ips:>9.1f}{s:>13.0f}{cost:>13.2f}")


@app.local_entrypoint()
def main(steps: int = 60000, gpu: str = "a100", resume: bool = True):
    fn = train_h100 if gpu.lower() == "h100" else train_a100
    print(fn.remote(steps, BATCH, resume, gpu))
