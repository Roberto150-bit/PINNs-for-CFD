import numpy as np

data = np.load("solver_results.npz")

x = data["x"]
y = data["y"]
u = data["u"]
v = data["v"]

velocity_magnitude = np.sqrt(u**2 + v**2)

# Search central region for the primary vortex
mask_x = (x >= 0.3) & (x <= 0.9)
mask_y = (y >= 0.3) & (y <= 0.9)

central_velocity = velocity_magnitude[np.ix_(mask_y, mask_x)]

j_local, i_local = np.unravel_index(
    np.argmin(central_velocity),
    central_velocity.shape
)

i = np.where(mask_x)[0][i_local]
j = np.where(mask_y)[0][j_local]

vortex_x = x[i]
vortex_y = y[j]

print(f"Estimated vortex center: ({vortex_x:.4f}, {vortex_y:.4f})")
print(f"Velocity magnitude: {velocity_magnitude[i, j]:.6e}")