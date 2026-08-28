"""Launch a named training run on Modal, server-side.

Why spawn rather than `modal run --detach`: a detached run is still cancelled if
its local client dies, which killed two multi-hour runs at steps 2300 and 14600.
`Function.from_name(...).spawn(...)` enqueues the call server-side, so closing
the terminal or losing the network has no effect.

    modal deploy stage3_train.py
    python launch.py v1          # the shipped configuration
    python launch.py --list

Every run below was actually trained; the results are in the README. They are
kept here so the experiment history is reproducible rather than described.
"""
import argparse
import sys

import modal

APP, FUNC = "bg-matting", "train_a100"

# (steps, band_weight, detail_ch, decoder_stop, decoder_ch)
RUNS = {
    "v1": dict(steps=30000, band_weight=9.0, detail_ch=64, decoder_stop=0,
               decoder_ch=224,
               note="shipped. band_mad 136.2"),
    "v1r": dict(steps=30000, band_weight=9.0, detail_ch=64, decoder_stop=0,
                decoder_ch=224,
                note="identical replicate, to measure the noise floor. 139.0"),
    "v3": dict(steps=12000, band_weight=9.0, detail_ch=32, decoder_stop=1,
               decoder_ch=224,
               note="stride-4 decoder + narrow detail branch. 163.3, worse"),
    "v4": dict(steps=12000, band_weight=9.0, detail_ch=64, decoder_stop=1,
               decoder_ch=224,
               note="stride-4 decoder alone. 166.1, worse"),
    "v5": dict(steps=12000, band_weight=4.0, detail_ch=64, decoder_stop=1,
               decoder_ch=224,
               note="v4 + band_weight 4. 169.3, worse"),
    "v6": dict(steps=12000, band_weight=4.0, detail_ch=64, decoder_stop=0,
               decoder_ch=224,
               note="band_weight 4 on a 12k schedule. 164.4, worse"),
    "v7": dict(steps=30000, band_weight=4.0, detail_ch=64, decoder_stop=0,
               decoder_ch=224,
               note="band_weight 4 on v1's schedule. 144.5, worse"),
}

BATCH = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="run name from RUNS")
    ap.add_argument("--list", action="store_true", help="show known runs")
    args = ap.parse_args()

    if args.list or not args.run:
        print(f"{'run':<6}{'steps':>7}{'band_w':>8}{'det':>5}{'stop':>6}  note")
        for name, c in RUNS.items():
            print(f"{name:<6}{c['steps']:>7}{c['band_weight']:>8}"
                  f"{c['detail_ch']:>5}{c['decoder_stop']:>6}  {c['note']}")
        return 0

    if args.run not in RUNS:
        print(f"unknown run {args.run!r}; try --list", file=sys.stderr)
        return 1

    c = RUNS[args.run]
    fn = modal.Function.from_name(APP, FUNC)
    call = fn.spawn(c["steps"], BATCH, True, args.run, args.run,
                    c["band_weight"], c["detail_ch"], c["decoder_stop"],
                    None, c["decoder_ch"])
    print(f"spawned {args.run}: steps={c['steps']} band_w={c['band_weight']} "
          f"detail_ch={c['detail_ch']} decoder_stop={c['decoder_stop']}")
    print(f"  -> {call.object_id}")
    print(f"logs: modal app logs {APP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
