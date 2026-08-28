"""Probe which GPU types this Modal workspace can actually schedule.

Runs a ~5s no-op on each candidate GPU and reports what came back.
Cost is a few cents total; run once before committing to a training plan.

Functions are declared explicitly rather than generated in a loop: `gpu=` must
be a literal at decoration time, and serialized=True is not an option here
(local Python is 3.14, the image is 3.12).
"""
import modal

app = modal.App("bg-matting-probe")
image = modal.Image.debian_slim(python_version="3.12")


def _probe():
    import subprocess
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60,
    )
    return out.stdout.strip() or out.stderr.strip()


@app.function(image=image, gpu="L4", timeout=120)
def probe_L4():
    return _probe()


@app.function(image=image, gpu="A10G", timeout=120)
def probe_A10G():
    return _probe()


@app.function(image=image, gpu="L40S", timeout=120)
def probe_L40S():
    return _probe()


@app.function(image=image, gpu="A100-40GB", timeout=120)
def probe_A100_40GB():
    return _probe()


@app.function(image=image, gpu="A100-80GB", timeout=120)
def probe_A100_80GB():
    return _probe()


@app.function(image=image, gpu="H100", timeout=120)
def probe_H100():
    return _probe()


@app.function(image=image, gpu="H200", timeout=120)
def probe_H200():
    return _probe()


@app.function(image=image, gpu="B200", timeout=120)
def probe_B200():
    return _probe()


PROBES = {
    "L4": probe_L4, "A10G": probe_A10G, "L40S": probe_L40S,
    "A100-40GB": probe_A100_40GB, "A100-80GB": probe_A100_80GB,
    "H100": probe_H100, "H200": probe_H200, "B200": probe_B200,
}


@app.local_entrypoint()
def main():
    for gpu, fn in PROBES.items():
        try:
            print(f"{gpu:<12} OK    {fn.remote()}")
        except Exception as e:
            print(f"{gpu:<12} FAIL  {str(e).replace(chr(10), ' ')[:160]}")
