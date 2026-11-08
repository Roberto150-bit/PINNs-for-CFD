"""Train the PINN from physics only: PDE residuals, initial and boundary conditions.

    python models/pinn/src/train.py --device gpu --seed 1
    python models/pinn/src/train.py --device cpu --seed 1 --steps 2000 --tag timing

No FD data enters the loss. The FD reference is used only to log test-A
relative L2 every 500 steps (monitoring) and in the final evaluation.
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

import config  # noqa: E402
from models.common import config as C  # noqa: E402
from models.common.network import configure_tensorflow, build_network, count_parameters  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=config.TRAINING_STEPS)
    parser.add_argument("--n-collocation", type=int, default=config.N_COLLOCATION)
    parser.add_argument("--no-determinism", action="store_true")
    parser.add_argument("--tag", default=None, help="suffix added to the run id (e.g. timing)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--exp-dir", default=C.EXPERIMENT_DIR)
    args = parser.parse_args()

    tf = configure_tensorflow(args.device, args.seed, deterministic=not args.no_determinism)
    from models.common import dataset, run_record
    from models.common.residuals import network_fields
    import losses

    nu = C.kinematic_viscosity(args.re)
    pts = losses.sample_points(args.n_collocation, config.N_INITIAL,
                               config.N_BOUNDARY_PER_WALL, config.SAMPLING_SEED)
    col = [tf.constant(a) for a in pts["col"]]
    ic = [tf.constant(a) for a in pts["ic"]]
    bc = [tf.constant(a) for a in pts["bc"]]
    bc_is_lid = tf.constant(pts["bc_is_lid"].astype("float32"))
    samples_per_step = sum(len(a[0]) for a in (pts["col"], pts["ic"], pts["bc"]))

    model = build_network(seed=args.seed)
    fields = network_fields(model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    # Loss weights as variables so a one-time rebalance does not retrace
    lam = {k: tf.Variable(v, dtype=tf.float32, trainable=False) for k, v in
           (("ic", config.LAMBDA_IC), ("bc", config.LAMBDA_BC),
            ("c", config.LAMBDA_CONTINUITY), ("m", config.LAMBDA_MOMENTUM))}

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            l_ic = losses.initial_loss(fields, *ic)
            l_bc = losses.boundary_loss(fields, *bc, bc_is_lid)
            l_c, l_x, l_y = losses.residual_losses(fields, *col, nu)
            total = (lam["ic"] * l_ic + lam["bc"] * l_bc + lam["c"] * l_c
                     + lam["m"] * (l_x + l_y))
        grads = tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return tf.stack([l_ic, l_bc, l_c, l_x, l_y, total])

    data = dataset.load(args.re)
    suffix = f"_{args.tag}" if args.tag else ""
    run_id = args.run_id or config.run_id(args.re, args.device, args.seed) + suffix
    recorder = run_record.RunRecorder(config.METHOD, run_id, args.device, args.seed, args.re,
                                      exp_dir=args.exp_dir)

    def cfg(lambda_adjustment=None):
        return {
            "model": {"hidden_layers": config.HIDDEN_LAYERS, "neurons_per_layer": config.NEURONS_PER_LAYER,
                      "activation": config.ACTIVATION, "initializer": C.KERNEL_INITIALIZER,
                      "output_size": 3, "trainable_parameters": count_parameters(model)},
            "training": {"steps": args.steps, "learning_rate": config.LEARNING_RATE,
                         "optimizer": config.OPTIMIZER, "batch": "full point set every step",
                         "timing_run": args.tag == "timing"},
            "physics": {"equations": ["continuity", "x_momentum", "y_momentum"],
                        "nu": nu, "rho": C.RHO,
                        "collocation_points": args.n_collocation,
                        "initial_points": config.N_INITIAL,
                        "boundary_points_per_wall": config.N_BOUNDARY_PER_WALL,
                        "lid_velocity": "tanh(10 t)", "corners_excluded": True,
                        "time_window": [C.T_MIN, C.T_MAX], "sampling_seed": config.SAMPLING_SEED},
            "loss": {"initial_weight": float(lam["ic"].numpy()), "boundary_weight": float(lam["bc"].numpy()),
                     "continuity_weight": float(lam["c"].numpy()),
                     "x_momentum_weight": float(lam["m"].numpy()), "y_momentum_weight": float(lam["m"].numpy()),
                     "initial_weight_start": config.LAMBDA_IC, "boundary_weight_start": config.LAMBDA_BC,
                     "continuity_weight_start": config.LAMBDA_CONTINUITY,
                     "momentum_weight_start": config.LAMBDA_MOMENTUM,
                     "adjustment": lambda_adjustment},
            "data": {"fd_data_in_loss": False,
                     "monitoring": "test-A relative L2 against FD every 500 steps (not used in training)"},
            "reproducibility": {"random_seed": args.seed},
            "deterministic_ops": not args.no_determinism,
        }

    recorder.write_config(cfg())
    recorder.start()

    names = ["loss_ic", "loss_bc", "loss_continuity", "loss_x_momentum", "loss_y_momentum", "loss_total"]
    t0 = time.perf_counter()
    comps = train_step().numpy()
    first_step = time.perf_counter() - t0
    train_time = first_step
    eval_time = 0.0
    adjustment = None
    print(f"first step (trace + compile) {first_step:.2f}s", flush=True)

    for step in range(1, args.steps + 1):
        if step > 1:
            t0 = time.perf_counter()
            out = train_step()
            if step % C.HISTORY_EVERY == 0 or step == args.steps:
                comps = out.numpy()         # forces completion before stopping the clock
            train_time += time.perf_counter() - t0

        if step % C.HISTORY_EVERY == 0 or step == 1:
            row = {"step": step, **{n: float(c) for n, c in zip(names, comps)},
                   "train_time_seconds": train_time}
            recorder.history.append(row)

        if step == config.LAMBDA_CHECK_STEP and adjustment is None:
            # Weighted components (momentum counted as r_x + r_y)
            comp = {"ic": float(comps[0]), "bc": float(comps[1]), "c": float(comps[2]),
                    "m": float(comps[3] + comps[4])}
            weighted = {k: float(lam[k].numpy()) * v for k, v in comp.items()}
            ratio = float(max(weighted.values()) / max(min(weighted.values()), 1e-30))
            if ratio > config.LAMBDA_IMBALANCE_FACTOR:
                # One-time rebalance: every weighted component set to the median
                med = float(np.median(list(weighted.values())))
                old = {k: float(lam[k].numpy()) for k in lam}
                for k in lam:
                    lam[k].assign(old[k] * med / max(weighted[k], 1e-30))
                adjustment = {"step": step, "imbalance_ratio": ratio, "components": weighted,
                              "old_weights": old, "new_weights": {k: float(lam[k].numpy()) for k in lam}}
            else:
                adjustment = {"step": step, "imbalance_ratio": ratio, "components": weighted,
                              "changed": False}
            recorder.write_config(cfg(adjustment))
            print(f"lambda check at step {step}: ratio {ratio:.1f} -> {adjustment}", flush=True)

        if step % C.EVAL_EVERY == 0 or step == 1:
            te = time.perf_counter()
            metrics = run_record.test_a_metrics(model, data, args.re)
            recorder.eval_history.append({"step": step, "train_time_seconds": train_time,
                                          "wall_seconds": recorder.wall(), **metrics})
            eval_time += time.perf_counter() - te
            print(f"step {step:6d} total {comps[5]:.3e} ic {comps[0]:.2e} bc {comps[1]:.2e} "
                  f"c {comps[2]:.2e} x {comps[3]:.2e} y {comps[4]:.2e} | relL2 u {metrics['rel_l2_u']:.3f} "
                  f"v {metrics['rel_l2_v']:.3f} p {metrics['rel_l2_p']:.3f} | t {train_time:.1f}s", flush=True)
            recorder.flush_history()

    timing = {
        "training_time_seconds": train_time,
        "first_step_seconds": first_step,
        "time_per_step_seconds": (train_time - first_step) / max(1, args.steps - 1),
        "evaluation_during_training_seconds": eval_time,
        "steps": args.steps,
        "samples_per_step": samples_per_step,
        "throughput_samples_per_second": samples_per_step * (args.steps - 1) / (train_time - first_step),
        "timing_run": args.tag == "timing",
    }
    perf = recorder.finish(model, timing)
    print(f"done {run_id}: train {train_time:.1f}s, per step {timing['time_per_step_seconds'] * 1e3:.1f} ms, "
          f"peak RAM {perf['memory']['peak_ram_bytes'] / 2**20:.0f} MiB, "
          f"peak VRAM {(perf['memory']['peak_vram_bytes'] or 0) / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
