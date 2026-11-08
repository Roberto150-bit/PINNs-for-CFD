"""Fully connected network shared by the data-driven NN and the PINN.

(x, y, t) scaled to [-1, 1]  ->  5 hidden layers x 64 tanh  ->  (u, v, p)

Parameter count: 3·64 + 64 = 256 (first layer)
               + 4 · (64·64 + 64) = 16,640 (hidden to hidden)
               + 64·3 + 3 = 195 (output)
               = 17,091 trainable parameters.

TensorFlow must be configured (device, seeds) before this module builds a
model; see `configure_tensorflow`.
"""
import os
import random

import numpy as np

from models.common import config as C


def configure_tensorflow(device, seed, deterministic=True):
    """Select the device, seed every RNG and (optionally) force deterministic ops.

    device: "gpu" or "cpu". For "cpu" the GPU is hidden through
    CUDA_VISIBLE_DEVICES, which only works before TensorFlow is first
    imported, so call this at the very start of a run.
    Returns the imported tensorflow module.
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise RuntimeError("GPU requested but TensorFlow sees no GPU")
    for g in gpus:
        # Allocate GPU memory on demand so the peak reflects actual use
        tf.config.experimental.set_memory_growth(g, True)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
    return tf


def build_network(hidden_layers=C.HIDDEN_LAYERS, neurons=C.NEURONS_PER_LAYER,
                  activation=C.ACTIVATION, seed=None):
    import tensorflow as tf

    def init(k):
        # Glorot (Xavier) uniform; a distinct seed per layer keeps the layers
        # from starting identical while staying reproducible
        return tf.keras.initializers.get(
            {"class_name": C.KERNEL_INITIALIZER,
             "config": {"seed": None if seed is None else seed * 1000 + k}})

    layers = [tf.keras.Input(shape=(C.N_INPUTS,), name="xyt_scaled")]
    for k in range(hidden_layers):
        layers.append(tf.keras.layers.Dense(neurons, activation=activation,
                                            kernel_initializer=init(k), name=f"hidden_{k + 1}"))
    layers.append(tf.keras.layers.Dense(C.N_OUTPUTS, activation=None,
                                        kernel_initializer=init(hidden_layers), name="uvp"))
    return tf.keras.Sequential(layers, name="mlp")


def count_parameters(model):
    return int(sum(np.prod(v.shape) for v in model.trainable_variables))
