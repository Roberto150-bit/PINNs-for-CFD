"""Run the experiment batch for one Reynolds number, one timed job at a time.

Order (GPU first): data NN GPU seeds 1-5, PINN GPU seeds 1-5, data NN CPU
seeds 1-2, PINN CPU seeds 1-2 (2,000-step timing runs: CPU step time
0.31 s exceeds the 0.25 s limit), then FD timing repeats and, at Re 100
only, the cost of deterministic ops. A job is skipped when its run folder
already holds a performance.json, so the batch can be restarted safely.

    python experiments/run_batch.py [--re 400] [--dry-run]
    python experiments/run_batch.py --sensitivity      (studies S1-S3, Re 100)
    python experiments/run_batch.py --extended         (study S5, 100,000 steps)
    python experiments/run_batch.py --hybrid           (hybrid network: data + physics loss, Re 10-1000)
    python experiments/run_batch.py --hybrid-control   (hybrid with data loss only: control, Re 100 and 400)
    python experiments/run_batch.py --day              (2026-11-09: seeds 4-5 at Re 10/1000, controls, 100k runs)
    python experiments/run_batch.py --extended-trend   (S5 extension: PINN Re 1000 and Re 10, 100,000 steps)
    python experiments/run_batch.py --trend fd         (Reynolds trend: FD at Re 10 and 1000)
    python experiments/run_batch.py --trend nn         (Reynolds trend: networks at Re 10 and 1000)
    python experiments/run_batch.py --causal           (study S4: causal weighting, PINN Re 400)
    python experiments/run_batch.py --continuation     (study S6: PINN Re 400 started from Re 100)
    python experiments/run_batch.py --continuation 1000  (S6 extension: Re 1000 started from S6 Re 400)
    python experiments/run_batch.py --continuation 10    (S6 extension: Re 10 started from Re 100)
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


def hybrid_jobs():
    """Hybrid network (FD data loss + physics loss), same conditions as the data NN and
    the PINN: same architecture, optimizer, 20,000 steps, data split, batch and physics
    points. Order: seeds 1-3 at every Re first (Re 100, 400, 10, 1000), then seeds 4-5 at
    Re 100 and 400 (5 seeds, as the main runs), then one 100,000-step run at Re 400 (as S5).
    GPU not shared.
    """
    order = ([(re, s) for re in (100.0, 400.0, 10.0, 1000.0) for s in (1, 2, 3)]
             + [(re, s) for re in (100.0, 400.0) for s in (4, 5)])
    out = []
    for re, s in order:
        EXP = C.experiment_dir(re)
        rid = f"hybrid_re{re:g}_gpu_s{s}"
        out.append((f"{EXP}/hybrid/{rid}",
                    [PY, "models/hybrid/src/train.py", "--device", "gpu", "--seed", str(s), "--re", f"{re:g}"]))
    EXP = f"{C.experiment_dir(400.0)}/extended"
    rid = f"hybrid_re400_gpu_s1_steps{EXTENDED_STEPS // 1000}k"
    out.append((f"{EXP}/hybrid/{rid}",
                [PY, "models/hybrid/src/train.py", "--device", "gpu", "--seed", "1", "--re", "400",
                 "--steps", str(EXTENDED_STEPS), "--run-id", rid, "--exp-dir", EXP]))
    return out


def day_jobs():
    """Day of 2026-11-09 (D24), GPU not shared, in this order:
    1. seeds 4-5 at Re 10 and Re 1000 for all three networks (5 seeds at every Re);
    2. the data-only hybrid control at Re 10 and 1000 (seeds 1-3);
    3. hybrid at Re 400, 100,000 steps, seeds 2-3 (3 seeds with seed 1);
    4. PINN at Re 1000, 100,000 steps, seeds 2-3 (3 seeds with seed 1);
    5. hybrid at Re 1000, 100,000 steps, seed 1."""
    scripts = {"dnn": ("data_driven_nn", "models/data_driven_nn/src/train.py"),
               "pinn": ("pinn", "models/pinn/src/train.py"),
               "hybrid": ("hybrid", "models/hybrid/src/train.py")}
    out = []
    for re in (10.0, 1000.0):
        for short in ("dnn", "pinn", "hybrid"):
            method, script = scripts[short]
            for s in (4, 5):
                out.append((f"{C.experiment_dir(re)}/{method}/{short}_re{re:g}_gpu_s{s}",
                            [PY, script, "--device", "gpu", "--seed", str(s), "--re", f"{re:g}"]))
    out += hybrid_control_jobs((10.0, 1000.0))
    for re, short, seeds in ((400.0, "hybrid", (2, 3)), (1000.0, "pinn", (2, 3)), (1000.0, "hybrid", (1,))):
        method, script = scripts[short]
        EXP = f"{C.experiment_dir(re)}/extended"
        for s in seeds:
            rid = f"{short}_re{re:g}_gpu_s{s}_steps{EXTENDED_STEPS // 1000}k"
            out.append((f"{EXP}/{method}/{rid}",
                        [PY, script, "--device", "gpu", "--seed", str(s), "--re", f"{re:g}",
                         "--steps", str(EXTENDED_STEPS), "--run-id", rid, "--exp-dir", EXP]))
    return out


def hybrid_control_jobs(res=(100.0, 400.0)):
    """Control for the hybrid (D23): hybrid code with the data loss only (no physics terms),
    physical outputs, seeds 1-3 at Re 100 and 400. Hybrid vs this control isolates the physics
    loss; this control vs the data NN isolates the output units. GPU not shared."""
    out = []
    for re in res:
        EXP = f"{C.experiment_dir(re)}/control"
        for s in (1, 2, 3):
            rid = f"hybrid_re{re:g}_gpu_s{s}_dataonly"
            out.append((f"{EXP}/hybrid/{rid}",
                        [PY, "models/hybrid/src/train.py", "--device", "gpu", "--seed", str(s), "--re", f"{re:g}",
                         "--data-only", "--run-id", rid, "--exp-dir", EXP]))
    return out


def extended_trend_jobs():
    """S5 extension across Re: PINN at Re 1000 and Re 10, 100,000 steps, seeds 1-3.

    Same setup as S5. Order alternates Re 1000 and Re 10 per seed, so that if
    the night ends early both have the same number of seeds.
    """
    out = []
    for s in (1, 2, 3):
        for re in (1000.0, 10.0):
            EXP = f"{C.experiment_dir(re)}/extended"
            rid = f"pinn_re{re:g}_gpu_s{s}_steps{EXTENDED_STEPS // 1000}k"
            out.append((f"{EXP}/pinn/{rid}",
                        [PY, "models/pinn/src/train.py", "--device", "gpu", "--seed", str(s), "--re", f"{re:g}",
                         "--steps", str(EXTENDED_STEPS), "--run-id", rid, "--exp-dir", EXP]))
    return out


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
        # Seeds 1-2 ran on a shared GPU; seed 3 was added later (GPU not shared).
        EXP = f"{C.experiment_dir(1000.0)}/continuation"
        out = []
        for s in (1, 2, 3):
            rid = f"pinn_re1000_gpu_s{s}_from_re400"
            init = f"{C.experiment_dir(400.0)}/continuation/pinn/pinn_re400_gpu_s{s}_from_re100/model.keras"
            out.append((f"{EXP}/pinn/{rid}",
                        [PY, "models/pinn/src/train.py", "--device", "gpu", "--seed", str(s), "--re", "1000",
                         "--init-from", init, "--run-id", rid, "--exp-dir", EXP]
                        + (["--shared-gpu"] if s < 3 else [])))
        return out
    if target == 10.0:
        # S6 at a slower flow: PINN at Re 10 started from the final Re 100 PINN of the same seed.
        EXP = f"{C.experiment_dir(10.0)}/continuation"
        out = []
        for s in (1, 2, 3):
            rid = f"pinn_re10_gpu_s{s}_from_re100"
            init = f"{C.experiment_dir(100.0)}/pinn/pinn_re100_gpu_s{s}/model.keras"
            out.append((f"{EXP}/pinn/{rid}",
                        [PY, "models/pinn/src/train.py", "--device", "gpu", "--seed", str(s), "--re", "10",
                         "--init-from", init, "--run-id", rid, "--exp-dir", EXP]))
        return out
    EXP = f"{C.experiment_dir(400.0)}/continuation"
    out = []
    # Seeds 1-3 ran on a shared GPU; seeds 4-5 were added later (GPU not shared) so that
    # S6 at Re 400 has the same 5 seeds as the standard Re 400 runs.
    for s in (1, 2, 3, 4, 5):
        rid = f"pinn_re400_gpu_s{s}_from_re100"
        init = f"{C.experiment_dir(100.0)}/pinn/pinn_re100_gpu_s{s}/model.keras"
        out.append((f"{EXP}/pinn/{rid}",
                    [PY, "models/pinn/src/train.py", "--device", "gpu", "--seed", str(s), "--re", "400",
                     "--init-from", init, "--run-id", rid, "--exp-dir", EXP]
                    + (["--shared-gpu"] if s <= 3 else [])))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--extended-trend", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--hybrid-control", action="store_true")
    parser.add_argument("--day", action="store_true", help="runs of 2026-11-09 (D24)")
    parser.add_argument("--trend",choices=["fd", "nn"], default=None)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--continuation", type=float, nargs="?", const=400.0, default=None,
                        help="S6 target Re: 400 (from Re 100, default) or 1000 (from the S6 Re 400 runs)")
    parser.add_argument("--trend-re", type=float, nargs="+", default=list(TREND_RE),
                        help="Reynolds numbers for --trend (default: 10 and 1000)")
    parser.add_argument("--no-start-after", default=None, metavar="HH:MM",
                        help="start no new job after this clock time (the next one if earlier than now)")
    args = parser.parse_args()
    deadline = None
    if args.no_start_after:
        import datetime as dt
        now = dt.datetime.now()
        hh, mm = map(int, args.no_start_after.split(":"))
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= now:
            deadline += dt.timedelta(days=1)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TF_CPP_MIN_LOG_LEVEL="2")
    if args.continuation:
        job_list = continuation_jobs(args.continuation)
    elif args.causal:
        job_list = causal_jobs()
    elif args.trend:
        job_list = trend_jobs(args.trend, args.trend_re)
    elif args.day:
        job_list = day_jobs()
    elif args.hybrid_control:
        job_list = hybrid_control_jobs()
    elif args.hybrid:
        job_list = hybrid_jobs()
    elif args.extended_trend:
        job_list = extended_trend_jobs()
    elif args.extended:
        job_list = extended_jobs()
    elif args.sensitivity:
        job_list = sensitivity_jobs()
    else:
        job_list = jobs(args.re)
    for run_dir, cmd in job_list:
        name = os.path.basename(run_dir)
        if args.hybrid and "/extended/" not in run_dir:
            LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "logs")
        elif args.day and "/extended/" not in run_dir and "/control/" not in run_dir:
            LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "logs")
        elif args.extended or args.extended_trend or args.hybrid or args.hybrid_control or args.day or args.trend or args.causal or args.continuation is not None:
            LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "logs")
        else:
            LOG_DIR = os.path.join(C.experiment_dir(args.re), "sensitivity" if args.sensitivity else "", "logs")
        os.makedirs(LOG_DIR, exist_ok=True)
        if os.path.exists(os.path.join(run_dir, "performance.json")):
            print(f"skip {name} (complete)", flush=True)
            continue
        if deadline is not None and dt.datetime.now() > deadline:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} stop: no new job after {args.no_start_after}", flush=True)
            break
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
