"""Dataset splits and the reduced-data subsets (study S7).

    python -m tests.test_dataset

For every dataset_re*_n64.npz that exists:
  - train / validation / test A / test B are disjoint and together cover every row;
  - the split indices are the same at every Reynolds number.
For the reduced-data subsets (--train-fraction 0.001 and 0.01):
  - the subset code is taken from the two training scripts themselves (ast), so a
    change to either train.py changes what is tested;
  - both scripts contain the same subset code;
  - the subset does not depend on the run seed (args.seed) and lies inside the train split;
  - its size is 118 rows (0.1%) and 1,176 rows (1%).
"""
import ast
import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from models.common import config as C  # noqa: E402

REYNOLDS = (10.0, 100.0, 400.0, 1000.0)
SPLITS = ("train", "val", "test_a", "test_b")
TRAIN_SCRIPTS = {m: os.path.join(REPO_ROOT, "models", m, "src", "train.py") for m in ("data_driven_nn", "hybrid")}
EXPECTED_SUBSET_SIZE = {0.001: 118, 0.01: 1176}


def available_datasets():
    out = {}
    for re in REYNOLDS:
        path = os.path.join(REPO_ROOT, C.dataset_path(re))
        if os.path.exists(path):
            d = np.load(path)
            out[re] = {k: d[f"idx_{k}"] for k in SPLITS} | {"rows": len(d["inputs"])}
    return out


def subset_block(path):
    # The `if args.train_fraction < 1.0:` block of main(), as source code
    src = open(path).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.If) and "args.train_fraction" in ast.unparse(node.test):
            return ast.unparse(node)
    raise AssertionError(f"no train_fraction block in {path}")


def run_subset(block, idx_train, fraction, seed):
    args = types.SimpleNamespace(train_fraction=fraction, seed=seed)
    scope = {"np": np, "C": C, "args": args, "idx_train": idx_train.copy()}
    exec(block, scope)
    return scope["idx_train"]


def test_splits_disjoint_and_cover_every_row():
    data = available_datasets()
    assert data, "no dataset files found"
    for re, d in data.items():
        joined = np.concatenate([d[k] for k in SPLITS])
        assert len(joined) == d["rows"], (re, len(joined), d["rows"])
        assert len(np.unique(joined)) == d["rows"], f"Re {re}: splits overlap"
        assert joined.min() == 0 and joined.max() == d["rows"] - 1


def test_splits_same_at_every_reynolds_number():
    data = list(available_datasets().values())
    for d in data[1:]:
        for k in SPLITS:
            assert np.array_equal(d[k], data[0][k]), k


def test_subset_code_identical_in_both_methods():
    blocks = {m: subset_block(p) for m, p in TRAIN_SCRIPTS.items()}
    # comments are not in the ast, so only the code is compared
    assert blocks["data_driven_nn"] == blocks["hybrid"], blocks


def test_subsets_same_for_every_seed_and_method():
    data = available_datasets()
    for re, d in data.items():
        for fraction, size in EXPECTED_SUBSET_SIZE.items():
            results = [run_subset(subset_block(p), d["train"], fraction, seed)
                       for p in TRAIN_SCRIPTS.values() for seed in (1, 2, 3, 4, 5)]
            for r in results[1:]:
                assert np.array_equal(r, results[0]), (re, fraction)
            s = results[0]
            assert len(s) == size, (re, fraction, len(s))
            assert len(np.unique(s)) == size
            assert np.isin(s, d["train"]).all()
            assert np.all(np.diff(s) > 0)          # sorted


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
