"""Run the experiment batch for one Reynolds number, one timed job at a time.

Order (GPU first): data NN GPU seeds 1-5, PINN GPU seeds 1-5, data NN CPU
seeds 1-2, PINN CPU seeds 1-2 (2,000-step timing runs: CPU step time
0.31 s exceeds the 0.25 s limit), then FD timing repeats and, at Re 100
only, the cost of deterministic ops. A job is skipped when its run folder
already holds a performance.json, so the batch can be restarted safely.

    python experiments/run_batch.py [--re 400] [--dry-run]
    python experiments/run_batch.py --sensitivity      (studies S1-S3, Re 100)
    python experiments/run_batch.py --extended         (study S5, 100,000 steps)
    python experiments/run_batch.py --trend fd         (Reynolds trend: FD at Re 10 and 1000)
    python experiments/run_batch.py --trend nn         (Reynolds trend: networks at Re 10 and 1000)
    python experiments/run_batch.py --causal           (study S4: causal weighting, PINN Re 400)
    python experiments/run_batch.py --continuation     (study S6: PINN Re 400 started from Re 100)
    python experiments/run_batch.py --continuation 1000  (S6 extension: Re 1000 started from S6 Re 400)
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from models.common import config as C  # noqa: E402

PY = sys.executable


def jobs(re):
    EXP = C.experiment_dir(re)
    tag = f"re{re:g}"
    R = ["--re", f"{re:g}"]
    out = []
    for s in range(1, 6):
        out.append((f"{EXP}/data_driven_nn/dnn_{tag}_gpu_s{s}",
                    [PY, "models/data_driven_nn/src/train.py", "--device", "gpu", "--seed", str(s)] + R))
    for s in range(1, 6):
        out.append((f"{EXP}/pinn/pinn_{tag}_gpu_s{s}",
                    [PY, "models/pinn/src/train.py", "--device", "gpu", "--seed", str(s)] + R))
    for s in (1, 2):
        out.append((f"{EXP}/data_driven_nn/dnn_{tag}_cpu_s{s}",
                    [PY, "models/data_driven_nn/src/train.py", "--device", "cpu", "--seed", str(s)] + R))
    for s in (1, 2):
        out.append((f"{EXP}/pinn/pinn_{tag}_cpu_s{s}_timing",
                    [PY, "models/pinn/src/train.py", "--device", "cpu", "--seed", str(s),
                     "--steps", "2000", "--tag", "timing"] + R))
    # FD solve-time repeats (no history logging, so only the solver is timed)
    for n in (32, 64, 128):
        for r in (1, 2, 3):
            out.append((f"{EXP}/finite_difference/fd_{tag}_n{n}_transient_timing_r{r}",
                        [PY, "models/finite_difference/src/run_solver.py", "--n", str(n),
                         "--mode", "transient", "--history-every", "0", "--tag", f"timing_r{r}",
                         "--exp-dir", EXP] + R))
    if re != 100.0:
        return out
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


def sensitivity_jobs():
    """S1 data NN train fraction, S2 PINN collocation points, S3 network size.

    Re 100, GPU, seeds 1-3, everything else as in the main runs. Order: the
    shortest studies first.
    """
    EXP = f"{C.experiment_dir(100.0)}/sensitivity"
    dnn = [PY, "models/data_driven_nn/src/train.py", "--device", "gpu"]
    pinn = [PY, "models/pinn/src/train.py", "--device", "gpu"]
    out = []
    for frac, name in ((0.01, "frac1"), (0.10, "frac10")):
        for s in (1, 2, 3):
            rid = f"dnn_re100_gpu_s{s}_{name}"
            out.append((f"{EXP}/data_driven_nn/{rid}", dnn + ["--seed", str(s), "--train-fraction", str(frac),
                                                            "--run-id", rid, "--exp-dir", EXP]))
    for n_col in (5000, 10000):
        for s in (1, 2, 3):
            rid = f"pinn_re100_gpu_s{s}_col{n_col}"
            out.append((f"{EXP}/pinn/{rid}", pinn + ["--seed", str(s), "--n-collocation", str(n_col),
                                                     "--run-id", rid, "--exp-dir", EXP]))
    for layers, width in ((4, 50), (6, 100)):
        for method, cmd, short in (("data_driven_nn", dnn, "dnn"), ("pinn", pinn, "pinn")):
            for s in (1, 2, 3):
                rid = f"{short}_re100_gpu_s{s}_arch{layers}x{width}"
                out.append((f"{EXP}/{method}/{rid}", cmd + ["--seed", str(s), "--hidden-layers", str(layers),
                                                          "--neurons", str(width), "--run-id", rid,
                                                          "--exp-dir", EXP]))
    return out


EXTENDED_STEPS = 100_000


def extended_jobs():
    """S5 training length: does more training remove the remaining errors?

    Same setup as the main runs (5 x 64, Adam 1e-3, same data, collocation
    points and seeds) but 100,000 optimizer steps instead of 20,000. Test-A
    error is logged every 500 steps, so each run also gives the error after
    20,000, 40,000, ... steps. Order: data NN Re 100 (short), then one PINN
    seed at Re 100 and one at Re 400 (the case that failed at 20,000 steps),
    then the remaining PINN seeds.
    """
    scripts = {"data_driven_nn": ("models/data_driven_nn/src/train.py", "dnn"),
               "pinn": ("models/pinn/src/train.py", "pinn")}
    order = [(100.0, "data_driven_nn", s) for s in (1, 2, 3)] + [
        (100.0, "pinn", 1), (400.0, "pinn", 1), (100.0, "pinn", 2), (400.0, "pinn", 2), (100.0, "pinn", 3),
        (400.0, "pinn", 3)]
    out = []
    for re, method, s in order:
        script, short = scripts[method]
        EXP = f"{C.experiment_dir(re)}/extended"
        rid = f"{short}_re{re:g}_gpu_s{s}_steps{EXTENDED_STEPS // 1000}k"
        out.append((f"{EXP}/{method}/{rid}",
                    [PY, script, "--device", "gpu", "--seed", str(s), "--re", f"{re:g}",
                     "--steps", str(EXTENDED_STEPS), "--run-id", rid, "--exp-dir", EXP]))
    return out


TREND_RE = (10.0, 1000.0)


def fd_time_step(re, n):
    """Time step for the FD runs of the Reynolds trend.

    Diffusion is explicit, so forward Euler needs dt <= dx^2 / (4 nu). With
    dt = 0.001 this holds for every grid at Re >= 100 but not at Re 10
    (nu = 0.1): the limit is 6.3e-4 at 64^2 and 1.6e-4 at 128^2.
    """
    if re >= 100.0:
        return 1e-3
    return {32: 1e-3, 64: 2.5e-4, 128: 5e-5}[n]


def trend_jobs(part, res=TREND_RE):
    """Reynolds-number trend: Re 10 and 1000 with the Re 100 protocol.

    part "fd": steady (validation) and transient runs at 32, 64, 128, plus the
    64^2 half-time-step check. The steady stop test compares the per-step
    velocity change with a tolerance, so the tolerance is scaled with dt
    (1e-6 at dt = 0.001) to keep the same criterion per unit time.
    part "nn": data NN and PINN, GPU, seeds 1-3, 20,000 steps. These share the
    GPU with another job, so their timings are flagged as not valid.
    """
    out = []
    for re in res:
        EXP = C.experiment_dir(re)
        tag = f"re{re:g}"
        R = ["--re", f"{re:g}", "--exp-dir", EXP]
        if part == "fd":
            for mode in ("steady", "transient"):
                for n in (32, 64, 128):
                    dt = fd_time_step(re, n)
                    cmd = [PY, "models/finite_difference/src/run_solver.py", "--n", str(n),
                           "--mode", mode, "--dt", f"{dt:g}"] + R
                    if mode == "steady":
                        cmd += ["--tol", f"{1e-6 * dt / 1e-3:g}"]
                    out.append((f"{EXP}/finite_difference/fd_{tag}_n{n}_{mode}", cmd))
            dt = fd_time_step(re, 64) / 2
            out.append((f"{EXP}/finite_difference/fd_{tag}_n64_transient_dthalf",
                        [PY, "models/finite_difference/src/run_solver.py", "--n", "64", "--mode",
                         "transient", "--dt", f"{dt:g}", "--tag", "dthalf"] + R))
        else:
            for method, script, short in (("data_driven_nn", "models/data_driven_nn/src/train.py", "dnn"),
                                          ("pinn", "models/pinn/src/train.py", "pinn")):
                for s in (1, 2, 3):
                    out.append((f"{EXP}/{method}/{short}_{tag}_gpu_s{s}",
                                [PY, script, "--device", "gpu", "--seed", str(s), "--shared-gpu"] + R))
    return out


def causal_jobs():
    """S4: PINN with causal weighting at Re 400 (where the main runs fail),
    seeds 1-3, everything else as in the main runs. Shares the GPU with
    another job, so timings are flagged as not valid."""
    EXP = f"{C.experiment_dir(400.0)}/causal"
    out = []
    for s in (1, 2, 3):
        rid = f"pinn_re400_gpu_s{s}_causal"
        out.append((f"{EXP}/pinn/{rid}",
                    [PY, "models/pinn/src/train.py", "--device", "gpu", "--seed", str(s), "--re", "400",
                     "--causal", "--shared-gpu", "--run-id", rid, "--exp-dir", EXP]))
    return out


def continuation_jobs(target=400.0):
    """S6: PINN at Re 400 started from the final Re 100 PINN of the same seed,
    seeds 1-3; extension: PINN at Re 1000 started from those Re 400 runs,
    seeds 1-2. Everything else as in the main runs. Shares the GPU with
    another job, so timings are flagged as not valid."""
    if target == 1000.0:
        EXP = f"{C.experiment_dir(1000.0)}/continuation"
        out = []
        for s in (1, 2):
            rid = f"pinn_re1000_gpu_s{s}_from_re400"
            init = f"{C.experiment_dir(400.0)}/continuation/pinn/pinn_re400_gpu_s{s}_from_re100/model.keras"
            out.append((f"{EXP}/pinn/{rid}",
                        [PY, "models/pinn/src/train.py", "--device", "gpu", "--seed", str(s), "--re", "1000",
                         "--init-from", init, "--shared-gpu", "--run-id", rid, "--exp-dir", EXP]))
        return out
    EXP = f"{C.experiment_dir(400.0)}/continuation"
    out = []
    for s in (1, 2, 3):
        rid = f"pinn_re400_gpu_s{s}_from_re100"
        init = f"{C.experiment_dir(100.0)}/pinn/pinn_re100_gpu_s{s}/model.keras"
        out.append((f"{EXP}/pinn/{rid}",
                    [PY, "models/pinn/src/train.py", "--device", "gpu", "--seed", str(s), "--re", "400",
                     "--init-from", init, "--shared-gpu", "--run-id", rid, "--exp-dir", EXP]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--trend", choices=["fd", "nn"], default=None)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--continuation", type=float, nargs="?", const=400.0, default=None,
                        help="S6 target Re: 400 (from Re 100, default) or 1000 (from the S6 Re 400 runs)")
    parser.add_argument("--trend-re", type=float, nargs="+", default=list(TREND_RE),
                        help="Reynolds numbers for --trend (default: 10 and 1000)")
    args = parser.parse_args()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TF_CPP_MIN_LOG_LEVEL="2")
    if args.continuation:
        job_list = continuation_jobs(args.continuation)
    elif args.causal:
        job_list = causal_jobs()
    elif args.trend:
        job_list = trend_jobs(args.trend, args.trend_re)
    elif args.extended:
        job_list = extended_jobs()
    elif args.sensitivity:
        job_list = sensitivity_jobs()
    else:
        job_list = jobs(args.re)
    for run_dir, cmd in job_list:
        name = os.path.basename(run_dir)
        if args.extended or args.trend or args.causal or args.continuation is not None:
            LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "logs")
        else:
            LOG_DIR = os.path.join(C.experiment_dir(args.re), "sensitivity" if args.sensitivity else "", "logs")
        os.makedirs(LOG_DIR, exist_ok=True)
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
