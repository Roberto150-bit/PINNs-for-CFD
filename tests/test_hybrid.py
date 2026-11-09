"""Hybrid network: the data loss behaves as specified, and every shared setting
equals the data-driven NN and the PINN, so the only difference is the loss.

    python -m tests.test_hybrid

The data loss and the batch order are defined inside main() of the training
scripts, so they are taken from the source files themselves (ast) and run
here, rather than copied: a change to train.py changes what is tested.
The data-only control (--data-only) is checked to use the data loss alone.
"""
import ast
import importlib.util
import os
import sys
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from models.common.network import configure_tensorflow  # noqa: E402

SRC = {m: os.path.join(REPO_ROOT, "models", m, "src") for m in ("hybrid", "data_driven_nn", "pinn")}


def load_config(method):
    spec = importlib.util.spec_from_file_location(f"{method}_config", os.path.join(SRC[method], "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nested_function(method, name):
    """Source of a function defined inside main() of <method>/src/train.py."""
    path = os.path.join(SRC[method], "train.py")
    source = open(path, encoding="utf-8").read()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return textwrap.dedent(ast.get_source_segment(source, node))
    raise LookupError(f"{name} not found in {path}")


def train_source(method):
    return open(os.path.join(SRC[method], "train.py"), encoding="utf-8").read()


def main():
    tf = configure_tensorflow("cpu", seed=0)
    from models.common import config as C, dataset
    from models.common.network import build_network
    from models.common.residuals import network_fields

    hyb, dnn, pinn = (load_config(m) for m in ("hybrid", "data_driven_nn", "pinn"))
    checks = []

    # ---- 1. Data loss: 0 on exact targets, > 0 otherwise ----
    ns = {"tf": tf}
    exec(nested_function("hybrid", "data_loss"), ns)
    rng = np.random.default_rng(0)
    xb = rng.uniform([0, 0, 0], [1, 1, 5], (512, 3)).astype("float32")

    def exact(x, y, t):
        return tf.sin(np.pi * x) * y * t, x * (1 - y) * tf.cos(t), x + y - t

    yb = tf.concat(exact(*(tf.constant(xb[:, k:k + 1]) for k in range(3))), axis=1)
    std = np.array([0.21, 0.12, 0.09], dtype="float32")
    ns["inv_std"] = tf.constant(1.0 / std)
    ns["fields"] = exact
    l0 = float(ns["data_loss"](tf.constant(xb), yb))
    checks.append(("data loss = 0 when the output equals the targets", l0 == 0.0, f"{l0:.3e}"))

    def shifted(x, y, t):
        u, v, p = exact(x, y, t)
        return u + 0.01, v, p

    ns["fields"] = shifted
    l1 = float(ns["data_loss"](tf.constant(xb), yb))
    expected = (0.01 / std[0]) ** 2 / 3
    checks.append(("data loss > 0 otherwise; u off by 0.01 gives (0.01/std_u)^2 / 3",
                   l1 > 0 and abs(l1 - expected) < 1e-4 * expected, f"{l1:.6e} vs {expected:.6e}"))

    # ---- 2. Same value as the data NN loss for the same predictions ----
    # Data NN: network output is standardized, loss = mean of per-output MSE on
    # standardized targets. Hybrid: output in physical units, error / std.
    data = dataset.load(100.0)
    idx = data["idx_train"][:2048]
    Y = data["targets"]
    y_mean = Y[data["idx_train"]].mean(axis=0)
    y_std = Y[data["idx_train"]].std(axis=0)
    X_phys = data["inputs"][idx].astype("float32")
    net = build_network(seed=1)
    out_std = net(dataset.scale_inputs(X_phys).astype("float32"))              # data NN view
    ns_d = {"tf": tf, "model": lambda z, training=False: net(z)}
    exec(nested_function("data_driven_nn", "per_output_mse"), ns_d)
    l_dnn = float(tf.reduce_mean(ns_d["per_output_mse"](dataset.scale_inputs(X_phys).astype("float32"),
                                                         ((Y[idx] - y_mean) / y_std).astype("float32"))))
    phys = out_std * y_std.astype("float32") + y_mean.astype("float32")          # same prediction, physical
    ns["inv_std"] = tf.constant((1.0 / y_std).astype("float32"))
    ns["fields"] = lambda x, y, t: (phys[:, 0:1], phys[:, 1:2], phys[:, 2:3])
    l_hyb = float(ns["data_loss"](tf.constant(X_phys), tf.constant(Y[idx].astype("float32"))))
    checks.append(("hybrid data loss = data NN loss for the same prediction (Re 100 train points)",
                   abs(l_hyb - l_dnn) < 1e-5 * l_dnn, f"{l_hyb:.6e} vs {l_dnn:.6e}"))

    # ---- 3. Input scaling: physical inputs through network_fields = data NN's scaled inputs ----
    a = tf.concat(network_fields(net)(*(tf.constant(X_phys[:, k:k + 1]) for k in range(3))), axis=1)
    b = net(dataset.scale_inputs(X_phys).astype("float32"))
    d = float(tf.reduce_max(tf.abs(a - b)))
    checks.append(("input scaling identical to the data NN", d < 1e-6, f"max |diff| {d:.1e}"))

    # ---- 4. Shared settings ----
    same = [("steps", hyb.TRAINING_STEPS, dnn.TRAINING_STEPS, pinn.TRAINING_STEPS),
            ("learning rate", hyb.LEARNING_RATE, dnn.LEARNING_RATE, pinn.LEARNING_RATE),
            ("optimizer", hyb.OPTIMIZER, dnn.OPTIMIZER, pinn.OPTIMIZER),
            ("hidden layers", hyb.HIDDEN_LAYERS, dnn.HIDDEN_LAYERS, pinn.HIDDEN_LAYERS),
            ("neurons", hyb.NEURONS_PER_LAYER, dnn.NEURONS_PER_LAYER, pinn.NEURONS_PER_LAYER),
            ("activation", hyb.ACTIVATION, dnn.ACTIVATION, pinn.ACTIVATION),
            ("seeds", hyb.SEEDS, dnn.SEEDS, pinn.SEEDS)]
    for name, h, dd, p in same:
        checks.append((f"{name} equal for all three", h == dd == p, f"hybrid {h}, data NN {dd}, PINN {p}"))
    checks.append(("batch size 4,096 as the data NN", hyb.BATCH_SIZE == dnn.BATCH_SIZE == 4096,
                   f"hybrid {hyb.BATCH_SIZE}, data NN {dnn.BATCH_SIZE}"))
    checks.append(("data split seed as the data NN", hyb.SPLIT_SEED == dnn.SPLIT_SEED == C.SPLIT_SEED,
                   f"hybrid {hyb.SPLIT_SEED}, data NN {dnn.SPLIT_SEED}"))
    w = [pinn.LAMBDA_IC, pinn.LAMBDA_BC, pinn.LAMBDA_CONTINUITY, pinn.LAMBDA_MOMENTUM, hyb.LAMBDA_DATA]
    checks.append(("all loss weights start at 1 (physics weights read from the PINN config)",
                   all(v == 1.0 for v in w), f"{w}"))

    # ---- 5. Source-level identity of what the hybrid reuses ----
    hs, ds, ps = train_source("hybrid"), train_source("data_driven_nn"), train_source("pinn")
    # Physics points: same call, same PINN settings
    checks.append(("physics points from the PINN's sample_points and PINN config",
                   "losses.sample_points(pinn_config.N_COLLOCATION, pinn_config.N_INITIAL,\n"
                   "                               pinn_config.N_BOUNDARY_PER_WALL, pinn_config.SAMPLING_SEED)" in hs
                   and "losses.sample_points(args.n_collocation, config.N_INITIAL," in ps
                   and pinn.N_COLLOCATION == 20000, "N_COLLOCATION default 20,000 in both"))
    # Physics terms: the PINN's own loss functions
    terms = ["losses.initial_loss(fields, *ic)", "losses.boundary_loss(fields, *bc, bc_is_lid)",
             "losses.residual_losses(fields, *col, nu)"]
    checks.append(("physics terms are the PINN's functions", all(t in hs and t in ps for t in terms),
                   "initial_loss, boundary_loss, residual_losses"))
    # Weight check: same step and factor from the PINN config, same median rule
    rule = "lam[k].assign(old[k] * med / max(weighted[k], 1e-30))"
    checks.append(("step-2,000 weight check: same step, factor and median rule as the PINN",
                   rule in hs and rule in ps and "pinn_config.LAMBDA_CHECK_STEP" in hs
                   and "pinn_config.LAMBDA_IMBALANCE_FACTOR" in hs,
                   f"step {pinn.LAMBDA_CHECK_STEP}, factor {pinn.LAMBDA_IMBALANCE_FACTOR:g}"))
    # Same initial weights: same builder, same seed argument
    checks.append(("same network builder and initialization seed",
                   "build_network(config.HIDDEN_LAYERS, config.NEURONS_PER_LAYER, seed=args.seed)" in hs
                   and "build_network(args.hidden_layers, args.neurons, seed=args.seed)" in ds
                   and "build_network(args.hidden_layers, args.neurons, seed=args.seed)" in ps, "seed=args.seed"))
    # Optimizer construction
    adam = "tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)"
    checks.append(("same optimizer construction", all(adam in s for s in (hs, ds, ps)), adam))

    # ---- 6. Batch order: identical index sequence to the data NN for the same seed ----
    def batches(method, seed, n_train, batch, steps):
        src = nested_function(method, "batch_indices").replace("args.batch", "batch")
        src = src.replace("nonlocal perm", "global perm")
        g = {"np": np, "rng": np.random.default_rng(seed), "batches_per_epoch": max(1, n_train // batch),
             "n_train": n_train, "batch": batch, "perm": None}
        exec(src, g)
        return [g["batch_indices"](s) for s in range(steps)]

    n_train = len(data["idx_train"])
    steps = 2 * (n_train // 4096) + 3                              # crosses two epoch boundaries
    bh, bd = batches("hybrid", 1, n_train, 4096, steps), batches("data_driven_nn", 1, n_train, 4096, steps)
    checks.append(("batch order identical to the data NN (seed 1, two epochs)",
                   all(np.array_equal(x, y) for x, y in zip(bh, bd)) and len(bh) == steps,
                   f"{steps} batches of 4,096; {n_train // 4096} per epoch"))
    checks.append(("batch rng seeded by the run seed in both",
                   "rng = np.random.default_rng(args.seed)" in hs and "rng = np.random.default_rng(args.seed)" in ds,
                   "default_rng(args.seed)"))

    # ---- 7. Data-only control (--data-only): data loss alone, same batches, no weight check ----
    ctrl = nested_function("hybrid", "data_only_step")
    checks.append(("control step uses only the data loss",
                   "data_loss(xb, yb)" in ctrl and "losses." not in ctrl
                   and "tape.gradient(l_d, model.trainable_variables)" in ctrl, "data_only_step"))
    checks.append(("control skips the step-2,000 weight check",
                   "and adjustment is None and not args.data_only" in hs, "guarded by not args.data_only"))
    checks.append(("control uses the same batch_indices as the hybrid",
                   "train_step(tf.constant(batch_indices(" in hs and "train_step = data_only_step" in hs,
                   "train_step replaced, batch order unchanged"))

    ok = True
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
