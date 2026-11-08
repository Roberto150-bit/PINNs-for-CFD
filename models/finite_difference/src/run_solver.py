"""Run the finite-difference lid-driven cavity solver.

Original baseline (21 x 21, Jacobi pressure, backward advection, impulsive lid):
    python models/finite_difference/src/run_solver.py --legacy
    (running with no arguments does the same)

Reference runs written as experiment records:
    python models/finite_difference/src/run_solver.py --n 64 --re 100 --mode transient
    python models/finite_difference/src/run_solver.py --n 128 --re 100 --mode steady

Records are written to <experiment dir>/finite_difference/<run_id>/ with
config.json, solution.npz (x, y, t, u, v, p), performance.json and history.csv.
"""
import argparse
import os
import sys
import time

import numpy as np

import config
from config import VELOCITY_TOLERANCE
from solver import run_solver, divergence_rms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
from models.common import profiling  # noqa: E402


def run_legacy():
    # Original behavior: config defaults, results written next to the solver
    start_time = time.perf_counter()

    x, y, u, v, p, velocity_change, steps_completed = run_solver(
        nx=21, ny=21, dt=0.001, nt=6000, nu=0.01, tolerance=1e-5,
        advection="backward", pressure_solver="jacobi", pressure_iterations=50,
        lid_ramp=False, pressure_bc="lid_dirichlet", time_scheme="pressure_source",
    )

    # Check for invalid numerical values
    if (
        np.isnan(u).any()
        or np.isnan(v).any()
        or np.isnan(p).any()
        or np.isinf(u).any()
        or np.isinf(v).any()
        or np.isinf(p).any()
    ):
        print("Warning: NaN or infinite values detected.")
    else:
        print("Solver completed without NaN or infinite values.")

    # Save numerical results
    np.savez(
        "models/finite_difference/src/solver_results.npz",
        x=x,
        y=y,
        u=u,
        v=v,
        p=p,
    )

    end_time = time.perf_counter()
    runtime = end_time - start_time

    print(f"Final velocity change: {velocity_change:.6e}")
    print(f"Steps completed: {steps_completed}")

    if velocity_change < VELOCITY_TOLERANCE:
        print("Convergence criteria reached.")
    else:
        print("Maximum time steps reached before convergence.")

    print(f"Runtime: {runtime:.4f} seconds")


def run_record(args):
    n = args.n
    nu = 1.0 / args.re                  # Re = U L / ν with U = L = 1
    dt = args.dt
    suffix = f"_{args.tag}" if args.tag else ""
    run_id = args.run_id or f"fd_re{args.re:g}_n{n}_{args.mode}{suffix}"
    run_dir = os.path.join(args.exp_dir, "finite_difference", run_id)
    os.makedirs(run_dir, exist_ok=True)

    if args.mode == "transient":
        nt = int(round(args.t_end / dt))
        snapshot_every = int(round(args.snapshot_interval / dt))
        stop_when_steady = False
        tolerance = 0.0
    else:
        nt = args.max_steps
        snapshot_every = None
        stop_when_steady = True
        tolerance = args.tol

    free_ram = profiling.free_ram_bytes()
    sampler = profiling.MemorySampler().start()
    t_start = time.perf_counter()

    result = run_solver(
        nx=n, ny=n, dt=dt, nt=nt, nu=nu, tolerance=tolerance,
        advection=args.advection, pressure_solver=args.pressure,
        pressure_iterations=args.pressure_iterations, lid_ramp=args.ramp,
        pressure_bc=args.pressure_bc, time_scheme=args.time_scheme, snapshot_every=snapshot_every, stop_when_steady=stop_when_steady,
        history_every=args.history_every,
    )
    solve_seconds = time.perf_counter() - t_start
    peak_rss = sampler.stop()
    x, y, u, v, p, velocity_change, steps, snapshots, history = result

    if snapshots is not None:
        t = np.array(snapshots["t"])
        U, V, P = (np.array(snapshots[k]) for k in ("u", "v", "p"))
    else:
        t = np.array([steps * dt])
        U, V, P = u[None], v[None], p[None]

    finite = bool(np.isfinite(U).all() and np.isfinite(V).all() and np.isfinite(P).all())
    dx = x[1] - x[0]
    final_div = divergence_rms(u, v, dx, dx)

    np.savez(os.path.join(run_dir, "solution.npz"), x=x, y=y, t=t, u=U, v=V, p=P)
    profiling.write_csv(os.path.join(run_dir, "history.csv"), history)

    cfg = {
        "run_id": run_id,
        "numerical_method": "finite_difference",
        "mode": args.mode,
        "reynolds_number": args.re,
        "rho": config.RHO,
        "nu": nu,
        "grid": {"nx": n, "ny": n, "dx": dx},
        "time_step": dt,
        "max_steps": nt,
        "advection_scheme": args.advection,
        "time_scheme": args.time_scheme,
        "pressure_solver": {
            "method": args.pressure,
            "iterations": args.pressure_iterations if args.pressure == "jacobi" else None,
            "boundary_condition": args.pressure_bc,
        },
        "lid": {"velocity": config.LID_VELOCITY, "ramp": args.ramp,
                "ramp_profile": "tanh(k t)" if args.ramp else None,
                "ramp_rate": config.LID_RAMP_RATE if args.ramp else None},
        "convergence": {"enabled": stop_when_steady, "tolerance": tolerance},
        "snapshots": {"interval": args.snapshot_interval if snapshots else None,
                      "count": int(len(t))},
        "initial_conditions": {"u": config.U_INITIAL, "v": config.V_INITIAL,
                               "p": config.P_INITIAL},
        "device": "cpu",
        "seed": None,
        "python_version": profiling.python_version(),
        "tf_version": None,
        "deterministic_ops": None,
        "created": profiling.timestamp(),
    }
    profiling.write_json(os.path.join(run_dir, "config.json"), cfg)

    perf = {
        "timing": {
            "total_runtime_seconds": solve_seconds,
            "solve_time_seconds": solve_seconds,
            "steps": steps,
            "time_per_step_seconds": solve_seconds / steps,
            "training_time_seconds": None,
            "inference_time_seconds": None,
        },
        "convergence": {
            "final_velocity_change": float(velocity_change),
            "converged": bool(stop_when_steady and velocity_change < tolerance),
            "final_divergence_rms": final_div,
            "finite": finite,
        },
        "memory": {"peak_ram_bytes": peak_rss, "peak_vram_bytes": None,
                   "free_ram_at_start_bytes": free_ram},
        "storage": {
            "model_size_bytes": None,
            "output_size_bytes": profiling.file_size(os.path.join(run_dir, "solution.npz")),
        },
    }
    profiling.write_json(os.path.join(run_dir, "performance.json"), perf)
    perf["storage"]["total_run_size_bytes"] = profiling.directory_size(run_dir)
    profiling.write_json(os.path.join(run_dir, "performance.json"), perf)

    print(f"{run_id}: steps={steps} t={steps * dt:.3f} solve={solve_seconds:.1f}s "
          f"dV={velocity_change:.3e} div_rms={final_div:.3e} finite={finite}")
    print(f"written to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy", action="store_true",
                        help="reproduce the original 21 x 21 solver exactly")
    parser.add_argument("--n", type=int, help="grid points per side")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--mode", choices=["steady", "transient"], default="transient")
    parser.add_argument("--advection", choices=["backward", "central"], default="central")
    parser.add_argument("--pressure", choices=["jacobi", "direct"], default="direct")
    parser.add_argument("--time-scheme", choices=["pressure_source", "projection",
                                                     "incremental_projection"],
                        default="incremental_projection")
    parser.add_argument("--pressure-bc", choices=["lid_dirichlet", "neumann"],
                        default="neumann",
                        help="p = 0 on the lid, or zero normal gradient on all walls")
    parser.add_argument("--pressure-iterations", type=int, default=config.PRESSURE_ITERATIONS)
    parser.add_argument("--ramp", action=argparse.BooleanOptionalAction, default=True,
                        help="smooth lid start u_lid = tanh(10 t)")
    parser.add_argument("--t-end", type=float, default=5.0, help="transient end time")
    parser.add_argument("--snapshot-interval", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-6, help="steady tolerance per step")
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--history-every", type=int, default=10)
    parser.add_argument("--exp-dir", default="experiments/experiment_001_re100")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tag", default=None, help="suffix added to the run id")
    args = parser.parse_args()

    if args.legacy or args.n is None:
        run_legacy()
    else:
        run_record(args)


if __name__ == "__main__":
    main()
