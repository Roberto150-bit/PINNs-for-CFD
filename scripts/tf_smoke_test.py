"""Short TensorFlow training test used to confirm a working device setup.

Fits a small tanh network to a smooth 3-input / 3-output function for 200
Adam steps and checks that the loss falls. Run once per device:

    python scripts/tf_smoke_test.py --device gpu
    python scripts/tf_smoke_test.py --device cpu   # hides the GPU before TF loads
"""
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["gpu", "cpu"], required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.device == "cpu":
        # Must be set before TensorFlow is imported
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(0)
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        # Allocate GPU memory on demand instead of reserving most of it up front
        tf.config.experimental.set_memory_growth(g, True)

    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(4096, 3)).astype("float32")
    y = np.stack([np.sin(np.pi * x[:, 0]) * x[:, 2],
                  np.cos(np.pi * x[:, 1]),
                  x[:, 0] * x[:, 1]], axis=1).astype("float32")

    model = tf.keras.Sequential([tf.keras.Input((3,))]
                                + [tf.keras.layers.Dense(64, activation="tanh") for _ in range(5)]
                                + [tf.keras.layers.Dense(3)])
    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def step(xb, yb):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(xb) - yb))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    xt, yt = tf.constant(x), tf.constant(y)
    losses = []
    t0 = time.perf_counter()
    for _ in range(args.steps):
        losses.append(float(step(xt, yt)))
    wall = time.perf_counter() - t0

    device = model.trainable_variables[0].value.device
    falling = np.mean(losses[-20:]) < 0.5 * np.mean(losses[:20])
    result = {"device_arg": args.device, "tf_visible_gpus": len(gpus),
              "variable_device": device, "steps": args.steps,
              "loss_first": losses[0], "loss_last": losses[-1],
              "loss_falling": bool(falling), "wall_seconds": wall}
    print(json.dumps(result, indent=2))
    sys.exit(0 if falling else 1)


if __name__ == "__main__":
    main()
