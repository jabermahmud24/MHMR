'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.stats import multivariate_normal

# Environment parameters
grid_size = (50, 50)
# grid_size = (100, 100)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
X, Y = np.meshgrid(x_range, y_range)

# Start and end points
A = np.array([0, 0])   # Starting point A
B = np.array([10, 10]) # Target point B

# Obstacles (circles)
# Obstacle 1
O1_center = np.array([6, 5])
O1_radius = 1.5

# Obstacle 2
O2_center = np.array([9, 4])
O2_radius = 1

# Robot's positional uncertainty (covariance matrix)
sigma = 0.5  # Standard deviation in x and y
Sigma = np.array([[sigma**2, 0],
                  [0, sigma**2]])  # Covariance matrix

# Prelec function parameter gamma values
gamma_values = [0.618]

# Number of samples for Monte Carlo simulation
M = 500

# Create obstacle map
def create_obstacle_map():
    obstacle_map = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            point = np.array([X[i, j], Y[i, j]])
            dist_to_O1 = np.linalg.norm(point - O1_center)
            dist_to_O2 = np.linalg.norm(point - O2_center)
            if dist_to_O1 <= O1_radius or dist_to_O2 <= O2_radius:
                obstacle_map[i, j] = 1
    return obstacle_map

# Compute true collision probability map
def compute_collision_probability_map():
    collision_prob_map = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            mu_t = np.array([X[i, j], Y[i, j]])
            # Monte Carlo simulation
            samples = np.random.multivariate_normal(mu_t, Sigma, M)
            # Collision with obstacles
            dist_to_O1 = np.linalg.norm(samples - O1_center, axis=1)
            collision_O1 = dist_to_O1 <= O1_radius
            dist_to_O2 = np.linalg.norm(samples - O2_center, axis=1)
            collision_O2 = dist_to_O2 <= O2_radius
            total_collisions = np.sum(collision_O1 | collision_O2)
            p_collision = total_collisions / M
            collision_prob_map[i, j] = p_collision
    return collision_prob_map

# Prelec weighting function
def prelec_weight(p, gamma):
    return np.exp(-(-np.log(p + 1e-10))**gamma)

# Cost function incorporating Prelec weighting
def compute_cost_map(collision_prob_map, gamma):
    w_p = prelec_weight(collision_prob_map, gamma)
    # The cost is higher where perceived collision probability is higher
    cost_map =  w_p  # Adjust the scaling factor as needed
    # cost_map = 1 + 10 * w_p  # Adjust the scaling factor as needed
    return cost_map

# Path planning using Dijkstra's algorithm
def dijkstra(cost_map, start_idx, end_idx):
    from heapq import heappush, heappop
    visited = np.full(cost_map.shape, False)
    cost_to_come = np.full(cost_map.shape, np.inf)
    parent = np.full(cost_map.shape + (2,), -1, dtype=int)
    heap = []
    heappush(heap, (0, start_idx))
    cost_to_come[start_idx] = 0
    while heap:
        current_cost, current_idx = heappop(heap)
        if visited[current_idx]:
            continue
        visited[current_idx] = True
        if current_idx == end_idx:
            break
        # Neighboring cells (4-connected)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in neighbors:
            nx, ny = current_idx[0] + dx, current_idx[1] + dy
            if 0 <= nx < cost_map.shape[0] and 0 <= ny < cost_map.shape[1]:
                if not visited[nx, ny]:
                    new_cost = current_cost + cost_map[nx, ny]
                    if new_cost < cost_to_come[nx, ny]:
                        cost_to_come[nx, ny] = new_cost
                        parent[nx, ny] = current_idx
                        heappush(heap, (new_cost, (nx, ny)))
    # Reconstruct path
    path = []
    idx = end_idx
    while np.all(parent[idx] != -1):
        path.append(idx)
        idx = tuple(parent[idx])
    path.append(start_idx)
    path = path[::-1]
    return path

# Main loop for different gamma values
obstacle_map = create_obstacle_map()
collision_prob_map = compute_collision_probability_map()

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Robot Paths for Different $\gamma$ Values')
ax.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
ax.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')

for gamma in gamma_values:
    cost_map = compute_cost_map(collision_prob_map, gamma)
    # Increase cost in obstacles to prevent paths through them
    cost_map += obstacle_map * 1e6  # Large cost to avoid obstacles
    # Convert start and end positions to grid indices
    start_idx = (int(A[0] / 10 * (grid_size[0] - 1)), int(A[1] / 10 * (grid_size[1] - 1)))
    end_idx = (int(B[0] / 10 * (grid_size[0] - 1)), int(B[1] / 10 * (grid_size[1] - 1)))
    path = dijkstra(cost_map, start_idx, end_idx)
    path_coords = np.array([(X[idx], Y[idx]) for idx in path])
    ax.plot(path_coords[:, 0], path_coords[:, 1], label=f'Path ($\gamma$={gamma})')

ax.legend()
plt.show()


# # Visualize the true collision probability map
# plt.figure(figsize=(8, 6))
# plt.imshow(np.flipud(collision_prob_map.T), extent=(0, 10, 0, 10), cmap='hot')
# plt.colorbar(label='True Collision Probability')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('True Collision Probability Map')
# plt.show()

# # Visualize perceived collision probability maps for different gamma values
# for gamma in gamma_values:
#     w_p_map = prelec_weight(collision_prob_map, gamma)
#     plt.figure(figsize=(8, 6))
#     plt.imshow(np.flipud(w_p_map.T), extent=(0, 10, 0, 10), cmap='hot')
#     plt.colorbar(label=f'Perceived Collision Probability (γ={gamma})')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.title(f'Perceived Collision Probability Map (γ={gamma})')
#     plt.show()

'''





'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Environment parameters
grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
X, Y = np.meshgrid(x_range, y_range)

# Start and end points
A = np.array([0, 0])   # Starting point A
B = np.array([8,8]) # Target point B

# Obstacles (circles)
# Obstacle 1
O1_center = np.array([0.5, 5])
O1_radius = 1

# Obstacle 2
O2_center = np.array([3, 5])
O2_radius = 1

# Additional obstacles
# Obstacle 3
O3_center = np.array([6, 5])
O3_radius = 1

# Obstacle 4
O4_center = np.array([9, 5])
O4_radius = 1

# Robot's positional uncertainty (covariance matrix)
sigma = 0.05  # Standard deviation in x and y
Sigma = np.array([[sigma**2, 0],
                  [0, sigma**2]])  # Covariance matrix

# Prelec function parameter gamma values
# gamma_values = [10.0]
gamma_values = [0.61, 1]
# gamma_values = [0.618, 0.75, 0.5]

# Number of samples for Monte Carlo simulation
M = 200

# Create obstacle map
def create_obstacle_map():
    obstacle_map = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            point = np.array([X[i, j], Y[i, j]])
            dist_to_O1 = np.linalg.norm(point - O1_center)
            dist_to_O2 = np.linalg.norm(point - O2_center)
            dist_to_O3 = np.linalg.norm(point - O3_center)
            dist_to_O4 = np.linalg.norm(point - O4_center)
            if (dist_to_O1 <= O1_radius or dist_to_O2 <= O2_radius or
                dist_to_O3 <= O3_radius or dist_to_O4 <= O4_radius):
                obstacle_map[i, j] = 1
    return obstacle_map

# Compute true collision probability map
def compute_collision_probability_map():
    collision_prob_map = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            mu_t = np.array([X[i, j], Y[i, j]])
            # Monte Carlo simulation
            samples = np.random.multivariate_normal(mu_t, Sigma, M)
            # Collision with obstacles
            dist_to_O1 = np.linalg.norm(samples - O1_center, axis=1)
            collision_O1 = dist_to_O1 <= O1_radius
            dist_to_O2 = np.linalg.norm(samples - O2_center, axis=1)
            collision_O2 = dist_to_O2 <= O2_radius
            dist_to_O3 = np.linalg.norm(samples - O3_center, axis=1)
            collision_O3 = dist_to_O3 <= O3_radius
            dist_to_O4 = np.linalg.norm(samples - O4_center, axis=1)
            collision_O4 = dist_to_O4 <= O4_radius
            total_collisions = np.sum(collision_O1 | collision_O2 | collision_O3 | collision_O4)
            p_collision = total_collisions / M
            collision_prob_map[i, j] = p_collision
    return collision_prob_map

# Prelec weighting function
def prelec_weight(p, gamma):
    # return p
    return np.exp(-(-np.log(p + 1e-10))**gamma)

# Cost function incorporating Prelec weighting
def compute_cost_map(collision_prob_map, gamma):
    w_p = prelec_weight(collision_prob_map, gamma)
    cost_map = w_p
    return cost_map

# Path planning using Dijkstra's algorithm
def dijkstra(cost_map, start_idx, end_idx):
    from heapq import heappush, heappop
    visited = np.full(cost_map.shape, False)
    cost_to_come = np.full(cost_map.shape, np.inf)
    parent = np.full(cost_map.shape + (2,), -1, dtype=int)
    heap = []
    heappush(heap, (0, start_idx))
    cost_to_come[start_idx] = 0
    while heap:
        current_cost, current_idx = heappop(heap)
        if visited[current_idx]:
            continue
        visited[current_idx] = True
        if current_idx == end_idx:
            break
        # Neighboring cells (8-connected)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        # neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in neighbors:
            nx, ny = current_idx[0] + dx, current_idx[1] + dy
            if 0 <= nx < cost_map.shape[0] and 0 <= ny < cost_map.shape[1]:
                if not visited[nx, ny]:
                    new_cost = current_cost + cost_map[nx, ny]
                    if new_cost < cost_to_come[nx, ny]:
                        cost_to_come[nx, ny] = new_cost
                        parent[nx, ny] = current_idx
                        heappush(heap, (new_cost, (nx, ny)))
    # Reconstruct path
    path = []
    idx = end_idx
    while np.all(parent[idx] != -1):
        path.append(idx)
        idx = tuple(parent[idx])
    path.append(start_idx)
    path = path[::-1]
    return path

# Main loop for different gamma values
obstacle_map = create_obstacle_map()
collision_prob_map = compute_collision_probability_map()


fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Robot Paths for Different $\gamma$ Values')
ax.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
ax.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')

# Define different line styles and widths for each gamma value
line_styles = ['-', ':', '-.', '--']
line_widths = [4, 5, 3, 4]

for idx, gamma in enumerate(gamma_values):
    cost_map = compute_cost_map(collision_prob_map, gamma)
    # Increase cost in obstacles to prevent paths through them
    cost_map += obstacle_map * 1e6  # Large cost to avoid obstacles
    # Convert start and end positions to grid indices
    start_idx = (int(A[0] / 10 * (grid_size[0] - 1)), int(A[1] / 10 * (grid_size[1] - 1)))
    end_idx = (int(B[0] / 10 * (grid_size[0] - 1)), int(B[1] / 10 * (grid_size[1] - 1)))
    path = dijkstra(cost_map, start_idx, end_idx)
    path_coords = np.array([(X[idx], Y[idx]) for idx in path])
    ax.plot(path_coords[:, 0], path_coords[:, 1], label=f'Path ($\gamma$={gamma})',
            linestyle=line_styles[idx % len(line_styles)], linewidth=line_widths[idx % len(line_widths)])

ax.legend()
plt.show()


# fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_title('Robot Paths for Different $\gamma$ Values')
# ax.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
# ax.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')

# for gamma in gamma_values:
#     cost_map = compute_cost_map(collision_prob_map, gamma)
#     # Increase cost in obstacles to prevent paths through them
#     cost_map += obstacle_map * 1e6  # Large cost to avoid obstacles
#     # Convert start and end positions to grid indices
#     start_idx = (int(A[0] / 10 * (grid_size[0] - 1)), int(A[1] / 10 * (grid_size[1] - 1)))
#     end_idx = (int(B[0] / 10 * (grid_size[0] - 1)), int(B[1] / 10 * (grid_size[1] - 1)))
#     path = dijkstra(cost_map, start_idx, end_idx)
#     path_coords = np.array([(X[idx], Y[idx]) for idx in path])
#     ax.plot(path_coords[:, 0], path_coords[:, 1], label=f'Path ($\gamma$={gamma})')

# ax.legend()
# plt.show()


# Visualize the true collision probability map
plt.figure(figsize=(8, 6))
plt.imshow(np.flipud(collision_prob_map.T), extent=(0, 10, 0, 10), cmap='hot')
plt.colorbar(label='True Collision Probability')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('True Collision Probability Map')
plt.show()

# Visualize perceived collision probability maps for different gamma values
for gamma in gamma_values:
    w_p_map = prelec_weight(collision_prob_map, gamma)
    plt.figure(figsize=(8, 6))
    plt.imshow(np.flipud(w_p_map.T), extent=(0, 10, 0, 10), cmap='hot')
    plt.colorbar(label=f'Perceived Collision Probability (γ={gamma})')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Perceived Collision Probability Map (γ={gamma})')
    plt.show()

    
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Environment parameters
grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
# Use 'ij' indexing for consistent indexing
X, Y = np.meshgrid(x_range, y_range, indexing='ij')

# Start and end points
A = np.array([3.8, 1.8])   # Starting point A
# A = np.array([0, 0])   # Starting point A
B = np.array([9, 0])   # Target point B

# Obstacles (circles)
# Obstacle 1
O1_center = np.array([5, 1])
O1_radius = 1

# Obstacle 2
O2_center = np.array([5, 3.5])
O2_radius = 1

# Additional obstacles
# Obstacle 3
O3_center = np.array([5, 6])
O3_radius = 1

# Obstacle 4
O4_center = np.array([5, 9])
O4_radius = 1

# Robot's positional uncertainty (covariance matrix)
sigma = 0.1  # Standard deviation in x and y
Sigma = np.array([[sigma**2, 0],
                  [0, sigma**2]])  # Covariance matrix

# Prelec function parameter gamma values
gamma_values = [0.4, 1]
# gamma_values = [0.4, 0.61, 1]
# gamma_values = [0.61, 1, 2.5, 5.0]

# Number of samples for Monte Carlo simulation
M = 200

# Create obstacle map
def create_obstacle_map():
    obstacle_map = np.zeros(grid_size)
    for i in range(grid_size[0]):  # i corresponds to x index
        for j in range(grid_size[1]):  # j corresponds to y index
            point = np.array([X[i, j], Y[i, j]])
            dist_to_O1 = np.linalg.norm(point - O1_center)
            dist_to_O2 = np.linalg.norm(point - O2_center)
            dist_to_O3 = np.linalg.norm(point - O3_center)
            dist_to_O4 = np.linalg.norm(point - O4_center)
            if (dist_to_O1 <= O1_radius or dist_to_O2 <= O2_radius or
                dist_to_O3 <= O3_radius or dist_to_O4 <= O4_radius):
                obstacle_map[i, j] = 1
    return obstacle_map

# Compute true collision probability map
def compute_collision_probability_map():
    collision_prob_map = np.zeros(grid_size)
    for i in range(grid_size[0]):  # i corresponds to x index
        for j in range(grid_size[1]):  # j corresponds to y index
            mu_t = np.array([X[i, j], Y[i, j]])
            # Monte Carlo simulation
            samples = np.random.multivariate_normal(mu_t, Sigma, M)
            # Collision with obstacles
            dist_to_O1 = np.linalg.norm(samples - O1_center, axis=1)
            collision_O1 = dist_to_O1 <= O1_radius
            dist_to_O2 = np.linalg.norm(samples - O2_center, axis=1)
            collision_O2 = dist_to_O2 <= O2_radius
            dist_to_O3 = np.linalg.norm(samples - O3_center, axis=1)
            collision_O3 = dist_to_O3 <= O3_radius
            dist_to_O4 = np.linalg.norm(samples - O4_center, axis=1)
            collision_O4 = dist_to_O4 <= O4_radius
            total_collisions = np.sum(collision_O1 | collision_O2 | collision_O3 | collision_O4)
            p_collision = total_collisions / M
            collision_prob_map[i, j] = p_collision
    return collision_prob_map

# Prelec weighting function
def prelec_weight(p, gamma):
    return np.exp(-(-np.log(p + 1e-10))**gamma)

# Cost function incorporating Prelec weighting
def compute_cost_map(collision_prob_map, gamma):
    w_p = prelec_weight(collision_prob_map, gamma)
    cost_map = w_p
    return cost_map

# Path planning using Dijkstra's algorithm
def dijkstra(cost_map, start_idx, end_idx):
    from heapq import heappush, heappop
    visited = np.full(cost_map.shape, False)
    cost_to_come = np.full(cost_map.shape, np.inf)
    parent = np.full(cost_map.shape + (2,), -1, dtype=int)
    heap = []
    heappush(heap, (0, start_idx))
    cost_to_come[start_idx] = 0
    while heap:
        current_cost, current_idx = heappop(heap)
        if visited[current_idx]:
            continue
        visited[current_idx] = True
        if current_idx == end_idx:
            break
        # Neighboring cells (8-connected)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in neighbors:
            nx, ny = current_idx[0] + dx, current_idx[1] + dy
            if 0 <= nx < cost_map.shape[0] and 0 <= ny < cost_map.shape[1]:
                if not visited[nx, ny]:
                    # Diagonal movement cost adjustment
                    if dx != 0 and dy != 0:
                        move_cost = np.sqrt(2) * cost_map[nx, ny]
                    else:
                        move_cost = cost_map[nx, ny]
                    new_cost = current_cost + move_cost
                    if new_cost < cost_to_come[nx, ny]:
                        cost_to_come[nx, ny] = new_cost
                        parent[nx, ny] = current_idx
                        heappush(heap, (new_cost, (nx, ny)))
    # Reconstruct path
    path = []
    idx = end_idx
    while np.all(parent[idx] != -1):
        path.append(idx)
        idx = tuple(parent[idx])
    path.append(start_idx)
    path = path[::-1]
    return path

# Main loop for different gamma values
obstacle_map = create_obstacle_map()
collision_prob_map = compute_collision_probability_map()

fig, ax = plt.subplots(figsize=(8, 8))
# Flip the obstacle map vertically to match the coordinate system
ax.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Robot Paths for Different $\gamma$ Values')
ax.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
ax.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')

# Define different line styles and widths for each gamma value
line_styles = ['-', ':', '-.', '--']
line_widths = [4, 5, 3, 4]

# Convert start and end positions to grid indices
start_idx = (int(A[0] / 10 * (grid_size[0] - 1)), int(A[1] / 10 * (grid_size[1] - 1)))
end_idx = (int(B[0] / 10 * (grid_size[0] - 1)), int(B[1] / 10 * (grid_size[1] - 1)))

for idx, gamma in enumerate(gamma_values):
    cost_map = compute_cost_map(collision_prob_map, gamma)
    # Increase cost in obstacles to prevent paths through them
    cost_map += obstacle_map * 1e6  # Large cost to avoid obstacles
    path = dijkstra(cost_map, start_idx, end_idx)
    # Extract path coordinates
    path_coords = np.array([(X[i, j], Y[i, j]) for i, j in path])
    ax.plot(path_coords[:, 0], path_coords[:, 1], label=f'Path ($\gamma$={gamma})',
            linestyle=line_styles[idx % len(line_styles)], linewidth=line_widths[idx % len(line_widths)])

ax.legend()
plt.show()


# # Visualize the true collision probability map
# plt.figure(figsize=(8, 6))
# plt.imshow(np.flipud(collision_prob_map.T), extent=(0, 10, 0, 10), cmap='hot')
# plt.colorbar(label='True Collision Probability')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('True Collision Probability Map')
# plt.show()

# # Visualize perceived collision probability maps for different gamma values
# for gamma in gamma_values:
#     w_p_map = prelec_weight(collision_prob_map, gamma)
#     plt.figure(figsize=(8, 6))
#     plt.imshow(np.flipud(w_p_map.T), extent=(0, 10, 0, 10), cmap='hot')
#     plt.colorbar(label=f'Perceived Collision Probability (γ={gamma})')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.title(f'Perceived Collision Probability Map (γ={gamma})')
#     plt.show()
