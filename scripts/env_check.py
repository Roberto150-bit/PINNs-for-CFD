"""Record the hardware and software environment of an experiment session.

Writes environment.json following the schema in notes/week-03.md, extended
with GPU driver/CUDA details and the library versions used by the solvers
and networks.

Usage (from the repository root):
    python scripts/env_check.py [--out experiments/experiment_001_re100/environment.json]
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys


def read_cpu_model():
    # /proc/cpuinfo is available on Linux and WSL2
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def query_nvidia_smi():
    # One row per GPU: name, total memory (MiB), driver version
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    cuda_version = None
    for line in header.splitlines():
        if "CUDA" in line and "Version" in line:
            # e.g. "| NVIDIA-SMI 610.55   KMD Version: 610.78   CUDA UMD Version: 13.3 |"
            cuda_version = line.split("CUDA")[-1].split(":")[-1].strip(" |")
    gpus = []
    for row in out.splitlines():
        name, mem, driver = [s.strip() for s in row.split(",")]
        gpus.append({"name": name, "total_vram_bytes": int(float(mem)) * 1024**2,
                     "driver_version": driver})
    return {"gpus": gpus, "driver_cuda_version": cuda_version}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/experiment_001_re100/environment.json")
    args = parser.parse_args()

    import numpy as np
    import psutil
    import scipy
    import tensorflow as tf
    import keras

    build = tf.sysconfig.get_build_info()
    tf_gpus = tf.config.list_physical_devices("GPU")
    smi = query_nvidia_smi()

    env = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "software": {
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "keras_version": keras.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "psutil_version": psutil.__version__,
            "tf_built_cuda_version": build.get("cuda_version"),
            "tf_built_cudnn_version": build.get("cudnn_version"),
        },
        "system": {
            "operating_system": platform.platform(),
            "kernel": platform.release(),
            "cpu": read_cpu_model(),
            "cpu_count": os.cpu_count(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "total_ram_bytes": psutil.virtual_memory().total,
            "available_ram_bytes": psutil.virtual_memory().available,
        },
        "gpu": {
            "available": len(tf_gpus) > 0,
            "tensorflow_visible_gpus": [d.name for d in tf_gpus],
            "device": smi["gpus"][0]["name"] if smi and smi["gpus"] else "",
            "total_vram_bytes": smi["gpus"][0]["total_vram_bytes"] if smi and smi["gpus"] else None,
            "driver_version": smi["gpus"][0]["driver_version"] if smi and smi["gpus"] else None,
            "driver_cuda_version": smi["driver_cuda_version"] if smi else None,
        },
        "python_executable": sys.executable,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
