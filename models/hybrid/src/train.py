"""Train the hybrid network: FD data loss (as the data NN) + physics loss (as the PINN).

    python models/hybrid/src/train.py --device gpu --seed 1
    python models/hybrid/src/train.py --device gpu --seed 1 --re 400

Every setting is shared with the other two methods (see config.py). The only
difference from the PINN is the added data loss. Against the data NN there are
two differences: the added physics loss, and the output units. The network
outputs u, v, p in physical units (as the PINN), so the physics residuals are
taken directly, whereas the data NN outputs standardized values; the data loss
divides each output error by its train-split standard deviation, which gives
the data NN's loss value for the same prediction.

    --data-only   control run: data loss only (physics weights 0, physics terms
                  not computed), physical outputs. It separates the effect of the
                  physics loss from the effect of the output units.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
# The physics loss and training points are reused unchanged from the PINN. Appended
# last so that `import config` finds this folder's config.py, not the PINN's.
sys.path.append(os.path.join(REPO_ROOT, "models", "pinn", "src"))

import numpy as np  # noqa: E402

import config  # noqa: E402  (models/hybrid/src/config.py: this folder comes first)
from models.common import config as C  # noqa: E402
from models.common.network import configure_tensorflow, build_network, count_parameters  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=config.TRAINING_STEPS)
    parser.add_argument("--no-determinism", action="store_true")
    parser.add_argument("--tag", default=None, help="suffix added to the run id (e.g. timing)")
    parser.add_argument("--shared-gpu", action="store_true",
                        help="another job uses the GPU at the same time: timings are flagged as not valid")
    parser.add_argument("--data-only", action="store_true",
                        help="control: data loss only, physical outputs (no physics terms)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--exp-dir", default=None, help="default: the experiment folder for --re")
    args = parser.parse_args()
    args.exp_dir = args.exp_dir or C.experiment_dir(args.re)

    tf = configure_tensorflow(args.device, args.seed, deterministic=not args.no_determinism)
    from models.common import dataset, run_record
    from models.common.residuals import network_fields
    import losses
    import importlib.util
    spec = importlib.util.spec_from_file_location("pinn_config", os.path.join(REPO_ROOT, "models", "pinn", "src",
                                                                              "config.py"))
    pinn_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pinn_config)

    nu = C.kinematic_viscosity(args.re)

    # ---- physics points: identical to the PINN ----
    pts = losses.sample_points(pinn_config.N_COLLOCATION, pinn_config.N_INITIAL,
                               pinn_config.N_BOUNDARY_PER_WALL, pinn_config.SAMPLING_SEED)
    col = [tf.constant(a) for a in pts["col"]]
    ic = [tf.constant(a) for a in pts["ic"]]
    bc = [tf.constant(a) for a in pts["bc"]]
    bc_is_lid = tf.constant(pts["bc_is_lid"].astype("float32"))

    # ---- data: identical to the data NN ----
    data = dataset.load(args.re)
    X_all = data["inputs"].astype("float32")            # physical (x, y, t); scaled inside the network wrapper
    Y_all = data["targets"]
    idx_train = data["idx_train"]
    y_mean = Y_all[idx_train].mean(axis=0)
    y_std = Y_all[idx_train].std(axis=0)
    X_train = tf.constant(X_all[idx_train])
    Y_train = tf.constant(Y_all[idx_train].astype("float32"))
    inv_std = tf.constant((1.0 / y_std).astype("float32"))
    batch = min(config.BATCH_SIZE, len(idx_train))
    n_train = len(idx_train)
    batches_per_epoch = max(1, n_train // batch)
    rng = np.random.default_rng(args.seed)
    samples_per_step = batch + sum(len(a[0]) for a in (pts["col"], pts["ic"], pts["bc"]))

    model = build_network(config.HIDDEN_LAYERS, config.NEURONS_PER_LAYER, seed=args.seed)
    fields = network_fields(model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    lam = {k: tf.Variable(v, dtype=tf.float32, trainable=False) for k, v in
           (("data", config.LAMBDA_DATA), ("ic", pinn_config.LAMBDA_IC), ("bc", pinn_config.LAMBDA_BC),
            ("c", pinn_config.LAMBDA_CONTINUITY), ("m", pinn_config.LAMBDA_MOMENTUM))}

    def data_loss(xb, yb):
        u, v, p = fields(xb[:, 0:1], xb[:, 1:2], xb[:, 2:3])
        err = (tf.concat([u, v, p], axis=1) - yb) * inv_std
        return tf.reduce_mean(tf.square(err))          # (MSE_u + MSE_v + MSE_p) / 3, standardized

    @tf.function
    def train_step(idx):
        xb = tf.gather(X_train, idx)
        yb = tf.gather(Y_train, idx)
        with tf.GradientTape() as tape:
            l_d = data_loss(xb, yb)
            l_ic = losses.initial_loss(fields, *ic)
            l_bc = losses.boundary_loss(fields, *bc, bc_is_lid)
            l_c, l_x, l_y = losses.residual_losses(fields, *col, nu)
            total = (lam["data"] * l_d + lam["ic"] * l_ic + lam["bc"] * l_bc + lam["c"] * l_c
                     + lam["m"] * (l_x + l_y))
        grads = tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return tf.stack([l_d, l_ic, l_bc, l_c, l_x, l_y, total])

    @tf.function
    def data_only_step(idx):
        xb = tf.gather(X_train, idx)
        yb = tf.gather(Y_train, idx)
        with tf.GradientTape() as tape:
            l_d = data_loss(xb, yb)
        grads = tape.gradient(l_d, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        z = tf.zeros(())
        return tf.stack([l_d, z, z, z, z, z, l_d])

    if args.data_only:
        train_step = data_only_step  # noqa: F811
        for k in ("ic", "bc", "c", "m"):
            lam[k].assign(0.0)

    perm = None

    def batch_indices(step):
        # Same scheme as the data NN: new shuffled order every epoch, last partial batch dropped
        nonlocal perm
        b = step % batches_per_epoch
        if b == 0:
            perm = rng.permutation(n_train)
        return perm[b * batch:(b + 1) * batch]

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
                         "optimizer": config.OPTIMIZER,
                         "batch": f"data: {batch} FD points per step (shuffled per epoch); physics: full point set",
                         "batches_per_epoch": batches_per_epoch, "timing_run": args.tag == "timing"},
            "data": {"fd_data_in_loss": True, "input_variables": ["x", "y", "t"],
                     "target_variables": ["u", "v", "p"], "training_samples": int(n_train),
                     "validation_samples": int(len(data["idx_val"])), "dataset": C.dataset_path(args.re),
                     "input_scaling": "[-1, 1]", "target_mean": y_mean.tolist(), "target_std": y_std.tolist(),
                     "data_loss": "mean of per-output MSE, each output divided by its train-split std",
                     "network_output": "physical u, v, p (no output standardization)"},
            "physics": {"equations": ["continuity", "x_momentum", "y_momentum"],
                        "nu": nu, "rho": C.RHO,
                        "collocation_points": pinn_config.N_COLLOCATION,
                        "initial_points": pinn_config.N_INITIAL,
                        "boundary_points_per_wall": pinn_config.N_BOUNDARY_PER_WALL,
                        "lid_velocity": "tanh(10 t)", "corners_excluded": True,
                        "time_window": [C.T_MIN, C.T_MAX], "sampling_seed": pinn_config.SAMPLING_SEED},
            "loss": {**{f"{k}_weight": float(lam[k].numpy()) for k in lam},
                     "weights_start": {"data": config.LAMBDA_DATA, "ic": pinn_config.LAMBDA_IC,
                                       "bc": pinn_config.LAMBDA_BC, "c": pinn_config.LAMBDA_CONTINUITY,
                                       "m": pinn_config.LAMBDA_MOMENTUM},
                     "adjustment": lambda_adjustment},
            "control_data_only": bool(args.data_only),
            "initial_weights": "random (Glorot uniform)",
            "reproducibility": {"random_seed": args.seed, "split_seed": C.SPLIT_SEED},
            "deterministic_ops": not args.no_determinism,
            "timing_valid": not args.shared_gpu,
            "timing_note": "GPU shared with another job: timing not valid" if args.shared_gpu else None,
        }

    recorder.write_config(cfg())
    recorder.start()

    names = ["loss_data", "loss_ic", "loss_bc", "loss_continuity", "loss_x_momentum", "loss_y_momentum",
             "loss_total"]
    t0 = time.perf_counter()
    comps = train_step(tf.constant(batch_indices(0))).numpy()
    first_step = time.perf_counter() - t0
    train_time = first_step
    eval_time = 0.0
    adjustment = None
    print(f"first step (trace + compile) {first_step:.2f}s", flush=True)

    for step in range(1, args.steps + 1):
        if step > 1:
            t0 = time.perf_counter()
            out = train_step(tf.constant(batch_indices(step - 1)))
            if step % C.HISTORY_EVERY == 0 or step == args.steps:
                comps = out.numpy()         # forces completion before stopping the clock
            train_time += time.perf_counter() - t0

        if step % C.HISTORY_EVERY == 0 or step == 1:
            recorder.history.append({"step": step, **{n: float(c) for n, c in zip(names, comps)},
                                     "train_time_seconds": train_time})

        if step == pinn_config.LAMBDA_CHECK_STEP and adjustment is None and not args.data_only:
            # Same rule as the PINN, over all five weighted components
            comp = {"data": float(comps[0]), "ic": float(comps[1]), "bc": float(comps[2]),
                    "c": float(comps[3]), "m": float(comps[4] + comps[5])}
            weighted = {k: float(lam[k].numpy()) * v for k, v in comp.items()}
            ratio = float(max(weighted.values()) / max(min(weighted.values()), 1e-30))
            if ratio > pinn_config.LAMBDA_IMBALANCE_FACTOR:
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
            print(f"step {step:6d} total {comps[6]:.3e} data {comps[0]:.2e} ic {comps[1]:.2e} bc {comps[2]:.2e} "
                  f"c {comps[3]:.2e} x {comps[4]:.2e} y {comps[5]:.2e} | relL2 u {metrics['rel_l2_u']:.3f} "
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
        "timing_valid": not args.shared_gpu,
    }
    perf = recorder.finish(model, timing)
    print(f"done {run_id}: train {train_time:.1f}s, per step {timing['time_per_step_seconds'] * 1e3:.1f} ms, "
          f"peak RAM {perf['memory']['peak_ram_bytes'] / 2**20:.0f} MiB, "
          f"peak VRAM {(perf['memory']['peak_vram_bytes'] or 0) / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
