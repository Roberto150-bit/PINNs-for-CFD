"""Run the Phase A experiment batch, one timed job at a time.

Order (GPU first): data NN GPU seeds 1-5, PINN GPU seeds 1-5, data NN CPU
seeds 1-2, PINN CPU seeds 1-2 (2,000-step timing runs: CPU step time
0.31 s exceeds the 0.25 s limit), then FD timing repeats and the cost of
deterministic ops. A job is skipped when its run folder already holds a
performance.json, so the batch can be restarted safely.

    python experiments/run_batch.py [--dry-run]
"""
import argparse
import os
import subprocess
import sys
import time

EXP = "experiments/experiment_001_re100"
LOG_DIR = os.path.join(EXP, "logs")
PY = sys.executable


def jobs():
    out = []
    for s in range(1, 6):
        out.append((f"{EXP}/data_driven_nn/dnn_re100_gpu_s{s}",
                    [PY, "models/data_driven_nn/src/train.py", "--device", "gpu", "--seed", str(s)]))
    for s in range(1, 6):
        out.append((f"{EXP}/pinn/pinn_re100_gpu_s{s}",
                    [PY, "models/pinn/src/train.py", "--device", "gpu", "--seed", str(s)]))
    for s in (1, 2):
        out.append((f"{EXP}/data_driven_nn/dnn_re100_cpu_s{s}",
                    [PY, "models/data_driven_nn/src/train.py", "--device", "cpu", "--seed", str(s)]))
    for s in (1, 2):
        out.append((f"{EXP}/pinn/pinn_re100_cpu_s{s}_timing",
                    [PY, "models/pinn/src/train.py", "--device", "cpu", "--seed", str(s),
                     "--steps", "2000", "--tag", "timing"]))
    # FD solve-time repeats (no history logging, so only the solver is timed)
    for n in (32, 64, 128):
        for r in (1, 2, 3):
            out.append((f"{EXP}/finite_difference/fd_re100_n{n}_transient_timing_r{r}",
                        [PY, "models/finite_difference/src/run_solver.py", "--n", str(n),
                         "--mode", "transient", "--history-every", "0", "--tag", f"timing_r{r}"]))
    # Speed cost of deterministic ops: 2,000 steps with and without, GPU, seed 1
    for method, script, rid in (("data_driven_nn", "models/data_driven_nn/src/train.py", "dnn"),
                                ("pinn", "models/pinn/src/train.py", "pinn")):
        for det in (True, False):
            tag = "det" if det else "nondet"
            run_id = f"{rid}_re100_gpu_s1_{tag}2000"
            cmd = [PY, script, "--device", "gpu", "--seed", "1", "--steps", "2000",
                   "--run-id", run_id, "--exp-dir", f"{EXP}/determinism_cost"]
            if not det:
                cmd.append("--no-determinism")
            out.append((f"{EXP}/determinism_cost/{method}/{run_id}", cmd))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TF_CPP_MIN_LOG_LEVEL="2")
    for run_dir, cmd in jobs():
        name = os.path.basename(run_dir)
        if os.path.exists(os.path.join(run_dir, "performance.json")):
            print(f"skip {name} (complete)", flush=True)
            continue
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start {name}: {' '.join(cmd[1:])}", flush=True)
        if args.dry_run:
            continue
        t0 = time.perf_counter()
        with open(os.path.join(LOG_DIR, f"{name}.log"), "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} end   {name}: exit {rc}, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
