"""Train the data-driven NN on FD data only (no physics).

    python models/data_driven_nn/src/train.py --device gpu --seed 1
    python models/data_driven_nn/src/train.py --device cpu --seed 1
    python models/data_driven_nn/src/train.py --overfit 1000 --steps 5000   # sanity test

Inputs (x, y, t) are scaled to [-1, 1]; targets are standardized with the
train-split mean and std; the loss is (MSE_u + MSE_v + MSE_p) / 3 on the
standardized targets. All reported metrics are in physical units.
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
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--overfit", type=int, default=0,
                        help="train full-batch on this many train points; no run record")
    parser.add_argument("--lr-drop-step", type=int, default=0,
                        help="overfit test only: lower the learning rate 10x at this step")
    parser.add_argument("--no-determinism", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--exp-dir", default=None, help="default: the experiment folder for --re")
    args = parser.parse_args()
    args.exp_dir = args.exp_dir or C.experiment_dir(args.re)

    tf = configure_tensorflow(args.device, args.seed, deterministic=not args.no_determinism)
    from models.common import dataset, run_record

    data = dataset.load(args.re)
    X_all = dataset.scale_inputs(data["inputs"]).astype("float32")
    Y_all = data["targets"]
    idx_train = data["idx_train"]
    if args.overfit:
        idx_train = np.random.default_rng(args.seed).choice(idx_train, args.overfit, replace=False)
        args.batch = args.overfit

    # Standardization from the train split only
    y_mean = Y_all[idx_train].mean(axis=0)
    y_std = Y_all[idx_train].std(axis=0)
    Ys_all = ((Y_all - y_mean) / y_std).astype("float32")

    X_train = tf.constant(X_all[idx_train])
    Y_train = tf.constant(Ys_all[idx_train])
    X_val = tf.constant(X_all[data["idx_val"]])
    Y_val = tf.constant(Ys_all[data["idx_val"]])

    model = build_network(seed=args.seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    if args.lr_drop_step and not args.overfit:
        raise SystemExit("--lr-drop-step is only for the overfit test")

    def per_output_mse(xb, yb):
        return tf.reduce_mean(tf.square(model(xb, training=True) - yb), axis=0)

    @tf.function
    def train_step(idx):
        xb = tf.gather(X_train, idx)
        yb = tf.gather(Y_train, idx)
        with tf.GradientTape() as tape:
            mse = per_output_mse(xb, yb)
            loss = tf.reduce_mean(mse)          # (MSE_u + MSE_v + MSE_p) / 3
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mse

    @tf.function
    def val_loss():
        return tf.reduce_mean(per_output_mse(X_val, Y_val))

    # Wrap the model so saved predictions are in physical units
    class Physical(tf.keras.Model):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.mean = tf.constant(y_mean.astype("float32"))
            self.std = tf.constant(y_std.astype("float32"))

        def call(self, z, training=False):
            return self.net(z, training=training) * self.std + self.mean

    physical = Physical(model)

    n_train = len(idx_train)
    rng = np.random.default_rng(args.seed)
    batches_per_epoch = max(1, n_train // args.batch)

    def batch_indices(step):
        # New shuffled order every epoch; the last partial batch is dropped
        nonlocal perm
        b = step % batches_per_epoch
        if b == 0:
            perm = rng.permutation(n_train)
        return perm[b * args.batch:(b + 1) * args.batch]

    perm = None
    recorder = None
    run_id = args.run_id or config.run_id(args.re, args.device, args.seed)
    if not args.overfit:
        recorder = run_record.RunRecorder(config.METHOD, run_id, args.device, args.seed, args.re,
                                          exp_dir=args.exp_dir)
        recorder.write_config({
            "model": {"hidden_layers": config.HIDDEN_LAYERS, "neurons_per_layer": config.NEURONS_PER_LAYER,
                      "activation": config.ACTIVATION, "initializer": C.KERNEL_INITIALIZER,
                      "output_size": 3, "trainable_parameters": count_parameters(model)},
            "training": {"steps": args.steps, "batch_size": args.batch,
                         "learning_rate": config.LEARNING_RATE, "optimizer": config.OPTIMIZER,
                         "loss_function": config.LOSS, "batches_per_epoch": batches_per_epoch},
            "data": {"input_variables": ["x", "y", "t"], "target_variables": ["u", "v", "p"],
                     "training_samples": int(n_train), "validation_samples": int(len(data["idx_val"])),
                     "dataset": C.dataset_path(args.re), "input_scaling": "[-1, 1]",
                     "target_mean": y_mean.tolist(), "target_std": y_std.tolist()},
            "reproducibility": {"random_seed": args.seed, "split_seed": C.SPLIT_SEED},
            "deterministic_ops": not args.no_determinism,
        })
        recorder.start()

    # First step separately: includes tracing / graph compilation
    t0 = time.perf_counter()
    loss, mse = train_step(tf.constant(batch_indices(0)))
    loss = float(loss)
    first_step = time.perf_counter() - t0
    train_time = first_step
    eval_time = 0.0
    history = [] if recorder is None else recorder.history

    for step in range(1, args.steps + 1):
        if args.lr_drop_step and step == args.lr_drop_step:
            optimizer.learning_rate.assign(config.LEARNING_RATE / 10)
        if step > 1:
            t0 = time.perf_counter()
            loss, mse = train_step(tf.constant(batch_indices(step - 1)))
            if step % C.HISTORY_EVERY == 0 or step == args.steps:
                loss = float(loss)          # forces completion before stopping the clock
            train_time += time.perf_counter() - t0

        if step % C.HISTORY_EVERY == 0 or step == 1:
            m = mse.numpy()
            history.append({"step": step, "loss": float(loss), "mse_u_std": float(m[0]),
                            "mse_v_std": float(m[1]), "mse_p_std": float(m[2]),
                            "train_time_seconds": train_time})
        if recorder is not None and (step % C.EVAL_EVERY == 0 or step == 1):
            te = time.perf_counter()
            metrics = run_record.test_a_metrics(physical, data, args.re)
            recorder.eval_history.append({"step": step, "train_time_seconds": train_time,
                                          "wall_seconds": recorder.wall(),
                                          "val_loss": float(val_loss()), **metrics})
            eval_time += time.perf_counter() - te
            print(f"step {step:6d} loss {float(loss):.3e} val {recorder.eval_history[-1]['val_loss']:.3e} "
                  f"relL2 u {metrics['rel_l2_u']:.3e} v {metrics['rel_l2_v']:.3e} p {metrics['rel_l2_p']:.3e} "
                  f"t {train_time:.1f}s", flush=True)
            recorder.flush_history()

    final_loss = float(loss)
    if args.overfit:
        print(f"overfit {args.overfit} points, {args.steps} steps: first loss {history[0]['loss']:.3e}, "
              f"final loss {final_loss:.3e}, train time {train_time:.1f}s")
        for h in history[:: max(1, len(history) // 10)] + [history[-1]]:
            print(f"  step {h['step']:6d} loss {h['loss']:.3e} mse_u {h['mse_u_std']:.2e} "
                  f"mse_v {h['mse_v_std']:.2e} mse_p {h['mse_p_std']:.2e}")
        raise SystemExit(0 if final_loss < 1e-4 else 1)

    timing = {
        "training_time_seconds": train_time,
        "first_step_seconds": first_step,
        "time_per_step_seconds": (train_time - first_step) / max(1, args.steps - 1),
        "evaluation_during_training_seconds": eval_time,
        "steps": args.steps,
        "samples_per_step": args.batch,
        "throughput_samples_per_second": args.batch * (args.steps - 1) / (train_time - first_step),
    }
    perf = recorder.finish(physical, timing, save_model=model)
    print(f"done {run_id}: train {train_time:.1f}s, per step {timing['time_per_step_seconds'] * 1e3:.2f} ms, "
          f"peak RAM {perf['memory']['peak_ram_bytes'] / 2**20:.0f} MiB, "
          f"peak VRAM {(perf['memory']['peak_vram_bytes'] or 0) / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
