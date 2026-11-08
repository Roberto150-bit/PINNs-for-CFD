"""G7: the shared network has 17,091 trainable parameters and outputs (N, 3).

    python -m tests.test_network
"""
from models.common.network import build_network, configure_tensorflow, count_parameters


def main():
    tf = configure_tensorflow("cpu", seed=0)
    model = build_network(seed=0)
    n_params = count_parameters(model)
    out = model(tf.zeros((7, 3)))
    ok_params = n_params == 17091
    ok_shape = tuple(out.shape) == (7, 3)
    print(f"[{'PASS' if ok_params else 'FAIL'}] G7 trainable parameters: {n_params} (expected 17,091)")
    print(f"[{'PASS' if ok_shape else 'FAIL'}] G7 output shape for N=7: {tuple(out.shape)} (expected (7, 3))")
    # Same seed gives identical weights; a different seed does not
    a = build_network(seed=3).get_weights()[0]
    b = build_network(seed=3).get_weights()[0]
    c = build_network(seed=4).get_weights()[0]
    ok_seed = (a == b).all() and not (a == c).all()
    print(f"[{'PASS' if ok_seed else 'FAIL'}] initialization reproducible per seed")
    raise SystemExit(0 if (ok_params and ok_shape and ok_seed) else 1)


if __name__ == "__main__":
    main()
