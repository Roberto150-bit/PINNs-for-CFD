import numpy as np
import matplotlib.pyplot as plt

def load_results(filename="solver_results.npz"):
    data = np.load(filename)

    x = data["x"]
    y = data["y"]
    u = data["u"]
    v = data["v"]
    p = data["p"]

    return x, y, u, v, p


def plot_velocity_field(x, y, u, v):
    # Convert 1D coordinates into a 2D grid for plotting
    X, Y = np.meshgrid(x, y)

    # Plot velocity direction and magnitude as arrows
    plt.figure(figsize=(7, 6))
    plt.quiver(X, Y, u, v)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Lid-Driven Cavity Velocity Field")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig("velocity_field.png", dpi=300)
    plt.show() 


def plot_pressure_field(x, y, p):
    # Convert 1D coordinates into a 2D grid for plotting
    X, Y = np.meshgrid(x, y)

    # Plot pressure as filled contours
    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, p, levels=20)
    plt.colorbar(label="Pressure")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Lid-Driven Cavity Pressure Field")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig("pressure_field.png", dpi=300)
    plt.show()


def main():
    x, y, u, v, p = load_results()
    plot_velocity_field(x, y, u, v)
    plot_pressure_field(x, y, p)


if __name__ == "__main__":
    main()