import time
import numpy as np

from solver import run_solver


def main():
    start_time = time.perf_counter()

    x, y, u, v, p, velocity_change = run_solver()

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
        "solver_results.npz",
        x=x,
        y=y,
        u=u,
        v=v,
        p=p,
    )

    end_time = time.perf_counter()
    runtime = end_time - start_time

    print(f"Final velocity change: {velocity_change:.6e}")
    print(f"Runtime: {runtime:.4f} seconds")


if __name__ == "__main__":
    main()