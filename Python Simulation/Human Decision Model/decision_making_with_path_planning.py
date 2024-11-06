import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from heapq import heappush, heappop

# Environment parameters
grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
X, Y = np.meshgrid(x_range, y_range, indexing='ij')

# Start and end points
A = np.array([2.8, 1.8])   # Starting point A
B = np.array([7, 4])       # Target point B

# Obstacles (circles)
obstacles = [
    {'center': np.array([5, 1]),   'radius': 1},
    {'center': np.array([5, 3.5]), 'radius': 1},
    {'center': np.array([5, 6]),   'radius': 1},
    {'center': np.array([5, 9]),   'radius': 1},
]

# Robot's positional uncertainty (covariance matrix)
sigma = 0.1  # Standard deviation in x and y
Sigma = np.array([[sigma**2, 0],
                  [0, sigma**2]])  # Covariance matrix

# Prelec function parameter gamma values
gamma_robot = 1       # Robot's gamma
# gamma_human = 0.618    # Human's gamma
gamma_human = 0.4     # Human's gamma

# Parameters for utility function
phi_k = 0.4         # Human preference parameter
alpha = 0.88         # Value function parameter
beta_prelec = 1      # Prelec function parameter (usually set to 1)
gamma_prelec = gamma_human  # Prelec function curvature for utility calculation
C_risk = -10        # Cost associated with collision

# Prelec probability weighting function for utility
def w(p, beta=beta_prelec, gamma=gamma_prelec):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1 - epsilon)
    return np.exp(-beta * (-np.log(p)) ** gamma)

# Value function
def v(D, alpha=alpha):
    return D ** alpha

# Utility function
def utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk):
    U_own = phi_k * (v(D_own) - w(p_risk_own) * C_risk)
    U_adapt = (1 - phi_k) * (v(D_adapt) - w(p_risk_adapt) * C_risk)
    decision_value = U_own - U_adapt
    return decision_value

# Create obstacle map (vectorized)
def create_obstacle_map():
    obstacle_map = np.zeros(grid_size)
    points = np.stack((X, Y), axis=-1)  # Shape (grid_size[0], grid_size[1], 2)
    for obs in obstacles:
        dist = np.linalg.norm(points - obs['center'], axis=-1)
        obstacle_map[dist <= obs['radius']] = 1
    return obstacle_map

# Compute collision probability map using Gaussian filter
def compute_collision_probability_map(obstacle_map, sigma):
    # Convert sigma from physical units to pixels
    pixel_size_x = x_range[1] - x_range[0]
    pixel_size_y = y_range[1] - y_range[0]
    sigma_pixels = [sigma / pixel_size_x, sigma / pixel_size_y]

    # Apply Gaussian filter to obstacle map
    collision_prob_map = gaussian_filter(obstacle_map, sigma=sigma_pixels, mode='constant')

    # Clip values to [0, 1] range
    collision_prob_map = np.clip(collision_prob_map, 0, 1)
    return collision_prob_map

# Prelec weighting function for path planning
def prelec_weight(p, gamma):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1 - epsilon)
    positive_values = -np.log(p)
    return np.exp(-positive_values ** gamma)

# Cost function incorporating Prelec weighting
def compute_cost_map(collision_prob_map, gamma):
    w_p = prelec_weight(collision_prob_map, gamma)
    cost_map = w_p
    return cost_map

# Path planning using Dijkstra's algorithm
def dijkstra(cost_map, start_idx, end_idx):
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
                    move_cost = np.sqrt(2) * cost_map[nx, ny] if dx != 0 and dy != 0 else cost_map[nx, ny]
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

# Function to convert position to grid index
def pos_to_idx(pos):
    idx_x = int(pos[0] / 10 * (grid_size[0] - 1))
    idx_y = int(pos[1] / 10 * (grid_size[1] - 1))
    idx_x = np.clip(idx_x, 0, grid_size[0] - 1)
    idx_y = np.clip(idx_y, 0, grid_size[1] - 1)
    return (idx_x, idx_y)

# Function to compute remaining distance
def compute_remaining_distance(path_coords, current_idx):
    path_segment = path_coords[current_idx:]
    if len(path_segment) < 2:
        return 0.0
    differences = path_segment[1:] - path_segment[:-1]
    distances = np.linalg.norm(differences, axis=1)
    total_distance = np.sum(distances)
    return total_distance

# Main code
# Precompute obstacle map and collision probability map
obstacle_map = create_obstacle_map()
collision_prob_map = compute_collision_probability_map(obstacle_map, sigma)

# Precompute cost maps
cost_map_robot = compute_cost_map(collision_prob_map, gamma_robot) + obstacle_map * 1e6
cost_map_human = compute_cost_map(collision_prob_map, gamma_human) + obstacle_map * 1e6

# Initialize robot's starting position
robot_position = np.array(A)
robot_positions = [robot_position.copy()]
step = 0  # Initialize step counter
max_steps = 500  # Set a maximum number of steps to prevent infinite loops
fig, ax = plt.subplots(figsize=(8, 8))
plt.ion()  # Turn on interactive mode
cnt = 0

utility_values = []  # Collect utility values during simulation

# Simulation loop
while np.linalg.norm(robot_position - B) > 0.1 and step < max_steps:
    cnt += 1
    # Update start index to robot's current position
    start_idx = pos_to_idx(robot_position)
    end_idx = pos_to_idx(B)

    # Compute paths starting from current robot position
    path_robot = dijkstra(cost_map_robot, start_idx, end_idx)
    path_coords_robot = np.array([(X[i, j], Y[i, j]) for i, j in path_robot])

    path_human = dijkstra(cost_map_human, start_idx, end_idx)
    path_coords_human = np.array([(X[i, j], Y[i, j]) for i, j in path_human])

    # Determine next positions
    next_pos_robot = path_coords_robot[1] if len(path_coords_robot) > 1 else path_coords_robot[0]
    next_pos_human = path_coords_human[1] if len(path_coords_human) > 1 else path_coords_human[0]

    # Decision-making logic
    D_own = compute_remaining_distance(path_coords_robot, 0)
    D_adapt = compute_remaining_distance(path_coords_human, 0)
    idx_next_robot = pos_to_idx(next_pos_robot)
    idx_next_human = pos_to_idx(next_pos_human)
    p_risk_own = collision_prob_map[idx_next_robot]
    p_risk_adapt = collision_prob_map[idx_next_human]
    decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
    utility_values.append(decision_value)

    if cnt>2:
        if (robot_positions[-1] == robot_positions[-2]).all():
            next_pos_robot = path_coords_robot[2]
            next_pos_human = path_coords_human[2]

    robot_position = next_pos_human if decision_value >= 0 else next_pos_robot

    

    # # Append robot's current position for plotting
    robot_positions.append(robot_position.copy())

    print("---------------------")
    print(cnt)
    print(decision_value)

    print("---------------------")

    

    # Clear previous paths and replot updated paths
    ax.clear()
    ax.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Robot Navigation with Human Interaction')
    ax.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
    ax.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')
    ax.plot(path_coords_robot[:, 0], path_coords_robot[:, 1], 'b--', label="Robot's Planned Path")
    ax.plot(path_coords_human[:, 0], path_coords_human[:, 1], 'g--', label="Human's Planned Path")
    ax.plot(np.array(robot_positions)[:, 0], np.array(robot_positions)[:, 1], 'r-', linewidth=2, label='Robot Trajectory')
    ax.legend()
    ax.grid(True)

    # Display the updated plot
    plt.pause(0.1)  # Small pause to update the plot

    step += 1

# Turn off interactive mode
plt.ioff()
plt.show()

# After the simulation loop:
plt.figure(figsize=(10, 6))
plt.plot(range(len(utility_values)), utility_values, 'm-', linewidth=2)
plt.xlabel('Decision Step')
plt.ylabel('Utility Value')
plt.title('Utility Value Over Decision Steps')
plt.grid(True)
plt.axhline(0, color='black', linestyle='--')
plt.show()
