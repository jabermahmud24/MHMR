
import numpy as np
A = np.array([2.8, 1.8])
pos = A
grid_size = (150, 150)

idx_x = int(pos[0] / 10 * (grid_size[0] - 1))
idx_y = int(pos[1] / 10 * (grid_size[1] - 1))
idx_x = np.clip(idx_x, 0, grid_size[0] - 1)
idx_y = np.clip(idx_y, 0, grid_size[1] - 1)


print(idx_x)
print(idx_y)


