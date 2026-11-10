# Datasets

One space-time dataset per Reynolds number, built from the finite-difference (FD) transient on the
$64 \times 64$ grid. All networks (data-driven NN, hybrid network) train on these files and every
network is scored on the same test rows. Code: `models/common/dataset.py`.

| File | $Re$ |
|---|---|
| `dataset_re10_n64.npz` / `.json` | 10 |
| `dataset_re100_n64.npz` / `.json` | 100 |
| `dataset_re400_n64.npz` / `.json` | 400 |
| `dataset_re1000_n64.npz` / `.json` | 1000 |

The `.json` next to each file records the source FD run, the sha256 of the source `solution.npz`
and of the dataset, the split seed and fractions, the split sizes and the input scaling.

## How a dataset is built

```
python -m models.common.dataset --re 100 --build    # writes the .npz and .json
python -m models.common.dataset --re 100 --check    # gates G5 (fields) and G6 (splits)
```

- Source: the FD transient `solution.npz` of the run below, with $u, v, p$ on the $64 \times 64$ grid
  ($x, y = 0, 1/63, \dots, 1$) at the 51 snapshots $t = 0.0, 0.1, \dots, 5.0$. The lid moves with
  $u_{lid}(t) = \tanh(10 t)$; at $t = 0$ the interior is at rest.
- At $Re = 1000$ the source is `fd_re1000_n127to64_transient`: the $127 \times 127$ FD solution
  taken at every second point, which are exactly the $64 \times 64$ points (the $64 \times 64$ FD
  solution is under-resolved at this $Re$). At $Re = 10, 100, 400$ the source is the
  $64 \times 64$ FD run itself.
- Rows: each snapshot gives 4,096 rows of inputs $(x, y, t)$ and targets $(u, v, p)$:
  $51 \times 4096 = 208{,}896$ rows. Row order is time-major, then $y$, then $x$:
  `row = k * 4096 + j * 64 + i` for snapshot $k$, grid row $j$ ($y$), column $i$ ($x$).
- Arrays in the `.npz`: `inputs` (208,896 x 3, physical units), `targets` (208,896 x 3),
  `x`, `y` (64), `t` (51), `u`, `v`, `p` (51 x 64 x 64), and `idx_train`, `idx_val`, `idx_test_a`,
  `idx_test_b` (sorted row indices).

| $Re$ | sha256 of the `.npz` | source FD run |
|---|---|---|
| 10 | `28ff651d298098577197095c617d2551cfce86c5b7feda7ce666114770fa9f51` | `fd_re10_n64_transient` |
| 100 | `9f7512c3f2f5a1c83e365586236e8d1db4deef12fa2ef7fe24af6514ba0c2f24` | `fd_re100_n64_transient` |
| 400 | `54b13d3304f1709dd34499bcae9c527e98cb6c4ea75b669328782358ecae8fdc` | `fd_re400_n64_transient` |
| 1000 | `6c61caf3365a4307151083772b6144efd4208a8bd4b0a570057fa9afa873db4d` | `fd_re1000_n127to64_transient` |

Each sha256 matches the value in the file's `.json`, and the source sha256 in the `.json` matches
the source `solution.npz` in the repository (checked for all four files).

## Splits

The split depends only on the grid and the snapshot times, so the same row indices are used at
every $Re$ and by every network.

| Split | Rows | How the rows are chosen |
|---|---:|---|
| train | 117,555 | 70% of the rows of the 41 snapshots not in test B, shuffled with seed 0 |
| validation | 25,190 | the next 15% of the same shuffle |
| test A | 25,191 | the remaining rows of that shuffle (random points at seen times) |
| test B | 40,960 | the 10 whole snapshots $t = 0.5, 1.0, \dots, 5.0$ (unseen times) |
| total | 208,896 | |

The four splits are disjoint and together cover every row (gate G6 and `tests/test_dataset.py`).
No train row has a test-B time. sha256 of the index arrays (as int64 bytes):

| Split | sha256 |
|---|---|
| train | `3722e91cffaafde6afc6eadf99f6e7a2d1e4d21c726dae135373ef5257defabf` |
| validation | `48d9173d4ea244e9558f5161923fa49355b39fd734567d638590e46bf7f4a3a8` |
| test A | `ff8e27923a5b8368dc53f759cc7d9c736023ed008ba4d6d1458389b39e75d904` |
| test B | `8d5aa25618ddfdff8a9cdd1baaa3d9bd4e51ab97c51c693ce5e4bb77ca6669f0` |

## Input scaling and target standardization

- Inputs: the networks receive $\tilde{x} = 2 (x - x_{low}) / (x_{high} - x_{low}) - 1$ with
  $x_{low} = (0, 0, 0)$ and $x_{high} = (1, 1, 5)$ for $(x, y, t)$, so every input lies in $[-1, 1]$.
  Physical derivatives (PINN and hybrid physics loss) are taken through this scaling.
- Targets: in the data loss each output's error is divided by the standard deviation of that output
  over the rows the network trains on (the whole train split, or the reduced subset below); the
  data NN also subtracts their mean. The test sets are never used for scaling.

## Target statistics per split

Mean, standard deviation, minimum and maximum of $u, v, p$ (FD values, no scaling).

**$Re = 10$**

| Split | Rows | u mean | u std | u min | u max | v mean | v std | v min | v max | p mean | p std | p min | p max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 117,555 | 0.0094 | 0.2289 | -0.2081 | 1.0000 | 0.0003 | 0.1323 | -0.3915 | 0.3570 | 0.0021 | 0.7350 | -10.6848 | 11.4895 |
| validation | 25,190 | 0.0085 | 0.2278 | -0.2077 | 1.0000 | -0.0006 | 0.1312 | -0.3915 | 0.3570 | -0.0011 | 0.7066 | -10.6848 | 11.4895 |
| test A | 25,191 | 0.0065 | 0.2243 | -0.2079 | 1.0000 | -0.0006 | 0.1314 | -0.3915 | 0.3570 | 0.0056 | 0.7501 | -10.6788 | 11.4895 |
| test B | 40,960 | 0.0091 | 0.2335 | -0.2078 | 1.0000 | 0.0000 | 0.1362 | -0.3915 | 0.3570 | 0.0022 | 0.7446 | -10.6825 | 11.4895 |
| all rows | 208,896 | 0.0089 | 0.2291 | -0.2081 | 1.0000 | 0.0000 | 0.1328 | -0.3915 | 0.3570 | 0.0021 | 0.7354 | -10.6848 | 11.4895 |

**$Re = 100$**

| Split | Rows | u mean | u std | u min | u max | v mean | v std | v min | v max | p mean | p std | p min | p max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 117,555 | 0.0096 | 0.2131 | -0.2385 | 1.0000 | 0.0004 | 0.1168 | -0.5199 | 0.2710 | 0.0018 | 0.0896 | -0.9704 | 1.6169 |
| validation | 25,190 | 0.0086 | 0.2120 | -0.2381 | 1.0000 | -0.0000 | 0.1158 | -0.5181 | 0.2669 | 0.0015 | 0.0857 | -0.9639 | 1.6166 |
| test A | 25,191 | 0.0068 | 0.2078 | -0.2384 | 1.0000 | -0.0002 | 0.1159 | -0.5193 | 0.2700 | 0.0020 | 0.0916 | -0.9161 | 1.6166 |
| test B | 40,960 | 0.0092 | 0.2189 | -0.2385 | 1.0000 | 0.0002 | 0.1220 | -0.5200 | 0.2711 | 0.0019 | 0.0894 | -0.8680 | 1.6172 |
| all rows | 208,896 | 0.0090 | 0.2135 | -0.2385 | 1.0000 | 0.0002 | 0.1176 | -0.5200 | 0.2711 | 0.0018 | 0.0893 | -0.9704 | 1.6172 |

**$Re = 400$**

| Split | Rows | u mean | u std | u min | u max | v mean | v std | v min | v max | p mean | p std | p min | p max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 117,555 | 0.0100 | 0.1812 | -0.2993 | 1.0000 | 0.0007 | 0.0902 | -0.5927 | 0.1913 | 0.0011 | 0.0381 | -0.3026 | 0.7534 |
| validation | 25,190 | 0.0092 | 0.1804 | -0.2991 | 1.0000 | 0.0006 | 0.0902 | -0.5924 | 0.1900 | 0.0010 | 0.0362 | -0.3026 | 0.7527 |
| test A | 25,191 | 0.0074 | 0.1753 | -0.2983 | 1.0000 | 0.0006 | 0.0898 | -0.5926 | 0.1881 | 0.0011 | 0.0388 | -0.1840 | 0.7527 |
| test B | 40,960 | 0.0097 | 0.1861 | -0.2997 | 1.0000 | 0.0008 | 0.0954 | -0.5928 | 0.1927 | 0.0012 | 0.0383 | -0.1508 | 0.7541 |
| all rows | 208,896 | 0.0095 | 0.1814 | -0.2997 | 1.0000 | 0.0007 | 0.0912 | -0.5928 | 0.1927 | 0.0011 | 0.0380 | -0.3026 | 0.7541 |

**$Re = 1000$**

| Split | Rows | u mean | u std | u min | u max | v mean | v std | v min | v max | p mean | p std | p min | p max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 117,555 | 0.0094 | 0.1621 | -0.3492 | 1.0000 | 0.0006 | 0.0783 | -0.6386 | 0.2323 | -0.0017 | 0.0255 | -0.2236 | 0.7752 |
| validation | 25,190 | 0.0089 | 0.1614 | -0.3456 | 1.0000 | 0.0004 | 0.0789 | -0.6384 | 0.2303 | -0.0017 | 0.0235 | -0.1703 | 0.7219 |
| test A | 25,191 | 0.0070 | 0.1558 | -0.3486 | 1.0000 | 0.0005 | 0.0784 | -0.6384 | 0.2285 | -0.0016 | 0.0275 | -0.1279 | 0.7746 |
| test B | 40,960 | 0.0092 | 0.1662 | -0.3488 | 1.0000 | 0.0006 | 0.0832 | -0.6381 | 0.2329 | -0.0018 | 0.0261 | -0.1062 | 0.7758 |
| all rows | 208,896 | 0.0090 | 0.1621 | -0.3492 | 1.0000 | 0.0006 | 0.0793 | -0.6386 | 0.2329 | -0.0017 | 0.0256 | -0.2236 | 0.7758 |

## Data quality

- No NaN or infinite value in `inputs` or `targets`, and no two rows share the same $(x, y, t)$
  (all four files).
- Gates G5 and G6 (`--check`) pass for all four files: field shapes, zero interior at $t = 0$,
  lid row $u = \tanh(10 t)$ to 1e-12, split sizes, no overlap, cover, no test-B time in train,
  scaled inputs in $[-1, 1]$, row order.
- Pressure has sharp spikes at the two lid corners $(0, 1)$ and $(1, 1)$, where the moving lid meets
  a fixed wall (the FD solution is singular there). The largest $|p|$ of every dataset is at a lid
  corner, and 38-64% of the rows with $|p|$ above 5 standard deviations lie in the
  $3 \times 3$ block of grid points at a lid corner (0.4% of the grid):

| $Re$ | std of p (all rows) | max p: value ($t$, $x$, $y$) | min p: value ($t$, $x$, $y$) | rows with abs(p) > 5 std | of these, in the 3 x 3 block at a lid corner |
|---|---:|---|---|---:|---:|
| 10 | 0.7354 | 11.489 (5.0, 0.984, 0.984) | -10.685 (0.4, 0.000, 0.984) | 1,443 | 845 |
| 100 | 0.0893 | 1.617 (5.0, 0.984, 0.984) | -0.970 (0.2, 0.000, 0.984) | 1,433 | 663 |
| 400 | 0.0380 | 0.754 (5.0, 0.984, 0.984) | -0.303 (0.1, 0.000, 0.984) | 1,178 | 449 |
| 1000 | 0.0256 | 0.776 (5.0, 1.000, 1.000) | -0.224 (0.1, 0.000, 1.000) | 670 | 428 |

  After the per-snapshot mean is removed (as the evaluator does), these two $3 \times 3$ blocks hold
  46-54% of the sum of $p^2$ over all snapshots with $t > 0$ (18 of the 4,096 grid
  points). The relative $L_2$ error in $p$ of every network therefore depends
  strongly on how well it fits a few corner points.
- At $Re = 10, 100, 400$ the FD pressure condition on the walls is a zero normal gradient applied as
  $p_{wall} = p_{neighbour}$, so the wall values of $p$ repeat the adjacent row or column (the
  extremes above are at $x, y = 62/63$ and repeated on the wall). At $Re = 1000$ the wall values come
  from the $127 \times 127$ solution and do not repeat their neighbours; its extremes are on the lid
  corners themselves.

## Reduced training data (studies S1 and S7)

With `--train-fraction f < 1` both `models/data_driven_nn/src/train.py` and
`models/hybrid/src/train.py` train on a subset of the train split, chosen by the same code:

```python
n_keep = int(round(f * len(idx_train)))
idx_train = np.sort(np.random.default_rng(SPLIT_SEED).choice(idx_train, n_keep, replace=False))
```

The generator uses the split seed (0), not the run seed, so the subset is the same for every seed
and for both networks; seeds differ only in initialization and batch order. The train split is the
same at every $Re$, so the subset is too. The mini-batch is the whole subset when the subset has
fewer than 4,096 rows. Validation and test sets are unchanged. `tests/test_dataset.py` takes this
code from both training scripts and checks it.

| Fraction | Rows | Snapshots represented (of 41) | sha256 of the sorted indices (int64) |
|---|---:|---:|---|
| 0.1% (`0.001`) | 118 | 38 | `61ee309385bea8e8d7e0f01b3f7fe6ec62e4d4476d81d831663603ef6846e899` |
| 1% (`0.01`) | 1,176 | 41 | `12e7a7c4885602797e8839db0dac455044242f40fbe2068741973782b632b466` |

The two subsets are separate draws: the 0.1% rows are **not** a subset of the 1% rows (they share
0 rows). The 1% subset is the one used by S1.

Rows of the 0.1% subset (indices into `inputs` / `targets`):

```
450, 908, 1406, 2815, 3766, 4803, 5688, 6947, 8232, 9954
12353, 12719, 13406, 13584, 14201, 15098, 25108, 26948, 29465, 33673
33748, 46527, 47117, 48552, 51109, 51553, 52666, 53659, 54878, 58615
59985, 60421, 65567, 66478, 67522, 68879, 69158, 72405, 75327, 75620
75972, 76691, 77571, 78046, 78431, 79591, 79961, 87352, 87418, 92068
93816, 97161, 97970, 100902, 101084, 102120, 108697, 108726, 109732, 111388
111724, 113563, 114496, 116523, 116889, 117727, 120375, 122372, 127902, 129234
130688, 131432, 133188, 133523, 133729, 137119, 137141, 137283, 137492, 139593
140085, 142695, 142948, 148444, 149230, 149360, 149669, 151008, 151049, 156336
156857, 156986, 160882, 161658, 162445, 163820, 169238, 169668, 172608, 173893
174755, 174881, 175550, 176720, 177255, 177678, 179668, 179930, 182089, 182248
183080, 190003, 193662, 193768, 196293, 199700, 201464, 204230
```

The 1% rows are regenerated by the code above (`f = 0.01`); the sha256 above identifies them.

Target statistics of the subsets:

| $Re$ | Fraction | u mean | u std | u min | u max | v mean | v std | v min | v max | p mean | p std | p min | p max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.1% | -0.0261 | 0.1719 | -0.2035 | 1.0000 | 0.0181 | 0.1249 | -0.3889 | 0.3512 | -0.1272 | 0.7847 | -5.8337 | 1.5704 |
| 10 | 1% | 0.0166 | 0.2369 | -0.2071 | 1.0000 | -0.0035 | 0.1323 | -0.3875 | 0.3570 | -0.0046 | 0.6312 | -10.6775 | 5.7519 |
| 100 | 0.1% | -0.0193 | 0.1586 | -0.2319 | 1.0000 | 0.0209 | 0.1060 | -0.4681 | 0.2552 | -0.0097 | 0.0646 | -0.4032 | 0.2131 |
| 100 | 1% | 0.0148 | 0.2213 | -0.2337 | 1.0000 | -0.0023 | 0.1206 | -0.5095 | 0.2557 | 0.0010 | 0.0764 | -0.8254 | 0.8926 |
| 400 | 0.1% | -0.0128 | 0.1209 | -0.2542 | 1.0000 | 0.0154 | 0.0779 | -0.4043 | 0.1780 | 0.0003 | 0.0158 | -0.0548 | 0.0625 |
| 400 | 1% | 0.0131 | 0.1912 | -0.2970 | 1.0000 | -0.0016 | 0.0974 | -0.5926 | 0.1756 | 0.0006 | 0.0319 | -0.1663 | 0.4516 |
| 1000 | 0.1% | -0.0086 | 0.1048 | -0.2225 | 1.0000 | 0.0113 | 0.0680 | -0.4446 | 0.1886 | -0.0016 | 0.0136 | -0.0647 | 0.0186 |
| 1000 | 1% | 0.0116 | 0.1726 | -0.3482 | 1.0000 | -0.0017 | 0.0864 | -0.6242 | 0.2316 | -0.0023 | 0.0203 | -0.1019 | 0.2376 |
