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
A = np.array([0, 0])   # Starting point A
B = np.array([9, 0])   # Target point B

# Obstacles (circles)
# Obstacle 1
O1_center = np.array([5, 1])
O1_radius = 1

# Obstacle 2
O2_center = np.array([5, 3.5])
O2_radius = 1

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
gamma_robot = 1       # Robot's gamma
gamma_human = 0.4     # Human's gamma

# Number of samples for Monte Carlo simulation
M = 200

# Parameters for utility function
phi_k = 0.8          # Human preference parameter
alpha = 0.88         # Value function parameter
beta_prelec = 1      # Prelec function parameter (usually set to 1)
C_risk = -10         # Cost associated with collision

# Prelec probability weighting function for utility
def w(p, beta=beta_prelec, gamma=gamma_human):
    return np.exp(-beta * (-np.log(p + 1e-10)) ** gamma)

# Value function
def v(D, alpha=alpha):
    return D ** alpha

# Utility function
def utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk):
    U_own = phi_k * (v(D_own) - w(p_risk_own) * C_risk)
    U_adapt = (1 - phi_k) * (v(D_adapt) - w(p_risk_adapt) * C_risk)
    decision_value = U_own - U_adapt
    return decision_value

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

# Prelec weighting function for path planning
def prelec_weight(p, gamma):
    return np.exp(-(-np.log(p + 1e-10)) ** gamma)

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

# Main code
obstacle_map = create_obstacle_map()
collision_prob_map = compute_collision_probability_map()

# Convert start and end positions to grid indices
def pos_to_idx(pos):
    idx_x = int(pos[0] / 10 * (grid_size[0] - 1))
    idx_y = int(pos[1] / 10 * (grid_size[1] - 1))
    return (idx_x, idx_y)

start_idx = pos_to_idx(A)
end_idx = pos_to_idx(B)

# Compute paths for robot and human
cost_map_robot = compute_cost_map(collision_prob_map, gamma_robot)
cost_map_robot += obstacle_map * 1e6  # Avoid obstacles
path_robot = dijkstra(cost_map_robot, start_idx, end_idx)
path_coords_robot = np.array([(X[i, j], Y[i, j]) for i, j in path_robot])

cost_map_human = compute_cost_map(collision_prob_map, gamma_human)
cost_map_human += obstacle_map * 1e6  # Avoid obstacles
path_human = dijkstra(cost_map_human, start_idx, end_idx)
path_coords_human = np.array([(X[i, j], Y[i, j]) for i, j in path_human])

# Initialize robot's position
robot_position = A.copy()
robot_positions = [robot_position.copy()]  # To store the trajectory

# Time steps (maximum)
max_steps = 500
step = 0

# Simulation loop
while np.linalg.norm(robot_position - B) > 0.1 and step < max_steps:
    # Get next positions from robot's and human's paths
    # Find the current index in the paths
    idx_robot = np.argmin(np.linalg.norm(path_coords_robot - robot_position, axis=1))
    idx_human = np.argmin(np.linalg.norm(path_coords_human - robot_position, axis=1))
    
    # Next positions
    if idx_robot + 1 < len(path_coords_robot):
        next_pos_robot = path_coords_robot[idx_robot + 1]
    else:
        next_pos_robot = path_coords_robot[-1]
        
    if idx_human + 1 < len(path_coords_human):
        next_pos_human = path_coords_human[idx_human + 1]
    else:
        next_pos_human = path_coords_human[-1]
    
    # Remaining distances
    D_own = np.sum(np.linalg.norm(path_coords_robot[idx_robot:] - np.roll(path_coords_robot[idx_robot:], -1, axis=0)[:-1], axis=1))
    D_adapt = np.sum(np.linalg.norm(path_coords_human[idx_human:] - np.roll(path_coords_human[idx_human:], -1, axis=0)[:-1], axis=1))
    
    # Collision probabilities at next positions
    idx_next_robot = pos_to_idx(next_pos_robot)
    idx_next_human = pos_to_idx(next_pos_human)
    p_risk_own = collision_prob_map[idx_next_robot]
    p_risk_adapt = collision_prob_map[idx_next_human]
    
    # Calculate decision value
    decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
    
    # Decide which path to follow
    if decision_value >= 0:
        # Follow human's path
        robot_position = next_pos_human
    else:
        # Follow robot's own path
        robot_position = next_pos_robot
    
    robot_positions.append(robot_position.copy())
    step += 1

# Convert robot_positions to numpy array for plotting
robot_positions = np.array(robot_positions)

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
# Flip the obstacle map vertically to match the coordinate system
ax.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Robot Navigation with Human Interaction')
ax.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
ax.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')

# Plot robot's and human's planned paths
ax.plot(path_coords_robot[:, 0], path_coords_robot[:, 1], 'b--', label="Robot's Planned Path")
ax.plot(path_coords_human[:, 0], path_coords_human[:, 1], 'g--', label="Human's Planned Path")

# Plot robot's actual trajectory
ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'r-', linewidth=2, label='Robot Trajectory')

ax.legend()
ax.grid(True)
plt.show()

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Environment parameters
# grid_size = (100, 100)
grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
# Use 'ij' indexing for consistent indexing
X, Y = np.meshgrid(x_range, y_range, indexing='ij')

# Start and end points
A = np.array([2.8, 1.8])   # Starting point A
# A = np.array([0, 0])   # Starting point A
B = np.array([9, 0])   # Target point B

# Obstacles (circles)
# Obstacle 1
O1_center = np.array([5, 1])
O1_radius = 1

# Obstacle 2
O2_center = np.array([5, 3.5])
O2_radius = 1

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
gamma_robot = 1       # Robot's gamma
gamma_human = 0.4     # Human's gamma

# Number of samples for Monte Carlo simulation
# M = 200
M = 150

# Parameters for utility function
phi_k = 0.6          # Human preference parameter
alpha = 0.88         # Value function parameter
beta_prelec = 1      # Prelec function parameter (usually set to 1)
gamma_prelec = gamma_human  # Prelec function curvature for utility calculation
C_risk = -10         # Cost associated with collision

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

# Prelec weighting function for path planning
# def prelec_weight(p, gamma):
#     epsilon = 1e-10
#     p = np.clip(p, epsilon, 1 - epsilon)
#     positive_values = -np.log(p)
#     return np.exp(-positive_values ** gamma)


def prelec_weight(p, gamma):
    return np.exp(-(-np.log(p + 1e-10)) ** gamma)


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

# # Main code

# Initialize robot's starting position
robot_position = np.array(A)
robot_positions = [robot_position.copy()]
step = 0  # Initialize step counter
max_steps = 500  # Set a maximum number of steps to prevent infinite loops

'''

# Simulation loop
while np.linalg.norm(robot_position - B) > 0.1 and step < max_steps:

    # Main code
    obstacle_map = create_obstacle_map()
    collision_prob_map = compute_collision_probability_map()

    start_idx = pos_to_idx(A)
    end_idx = pos_to_idx(B)

    # Compute paths for robot and human
    cost_map_robot = compute_cost_map(collision_prob_map, gamma_robot)
    cost_map_robot += obstacle_map * 1e6  # Avoid obstacles
    path_robot = dijkstra(cost_map_robot, start_idx, end_idx)
    path_coords_robot = np.array([(X[i, j], Y[i, j]) for i, j in path_robot])

    cost_map_human = compute_cost_map(collision_prob_map, gamma_human)
    cost_map_human += obstacle_map * 1e6  # Avoid obstacles
    path_human = dijkstra(cost_map_human, start_idx, end_idx)
    path_coords_human = np.array([(X[i, j], Y[i, j]) for i, j in path_human])

    # Get next positions from robot's and human's paths
    # Find the current index in the paths
    idx_robot = np.argmin(np.linalg.norm(path_coords_robot - robot_position, axis=1))
    idx_human = np.argmin(np.linalg.norm(path_coords_human - robot_position, axis=1))
    
    # Next positions
    if idx_robot + 1 < len(path_coords_robot):
        next_pos_robot = path_coords_robot[idx_robot + 1]
    else:
        next_pos_robot = path_coords_robot[-1]
        
    if idx_human + 1 < len(path_coords_human):
        next_pos_human = path_coords_human[idx_human + 1]
    else:
        next_pos_human = path_coords_human[-1]
    
    # Remaining distances
    D_own = compute_remaining_distance(path_coords_robot, idx_robot)
    D_adapt = compute_remaining_distance(path_coords_human, idx_human)
    
    # Collision probabilities at next positions
    idx_next_robot = pos_to_idx(next_pos_robot)
    idx_next_human = pos_to_idx(next_pos_human)
    p_risk_own = collision_prob_map[idx_next_robot]
    p_risk_adapt = collision_prob_map[idx_next_human]
    
    # Calculate decision value
    decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
    print(decision_value)
    
    # Decide which path to follow
    if decision_value >= 0:
        # Follow human's path
        robot_position = next_pos_human
    else:
        # Follow robot's own path
        robot_position = next_pos_robot

    # print(robot_position)
    if step>0:
        robot_positions = robot_positions.tolist()
    
    robot_positions.append(robot_position.copy())
    print(step)
    # Convert robot_positions to numpy array for plotting
    robot_positions = np.array(robot_positions)
    print(robot_positions)
    step += 1


# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
# Flip the obstacle map vertically to match the coordinate system
ax.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Robot Navigation with Human Interaction')
ax.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
ax.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')

# Plot robot's and human's planned paths
ax.plot(path_coords_robot[:, 0], path_coords_robot[:, 1], 'b--', label="Robot's Planned Path")
ax.plot(path_coords_human[:, 0], path_coords_human[:, 1], 'g--', label="Human's Planned Path")

# Plot robot's actual trajectory
ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'r-', linewidth=2, label='Robot Trajectory')

ax.legend()
ax.grid(True)
plt.show()
'''



# '''

# Initialize the figure and axis outside the loop
fig, ax = plt.subplots(figsize=(8, 8))
plt.ion()  # Turn on interactive mode
cnt = 0

# Simulation loop
while np.linalg.norm(robot_position - B) > 0.1 and step < max_steps:
    # Main code for map and path computation
    obstacle_map = create_obstacle_map()
    collision_prob_map = compute_collision_probability_map()
    cnt = cnt+1
    print(cnt)
    
    
    # Update start index to robot's current position
    # start_idx = pos_to_idx(robot_positions[-1])
    start_idx = pos_to_idx(robot_position)
    end_idx = pos_to_idx(B)
    
    # Compute paths starting from current robot position
    cost_map_robot = compute_cost_map(collision_prob_map, gamma_robot) + obstacle_map * 1e6
    path_robot = dijkstra(cost_map_robot, start_idx, end_idx)
    path_coords_robot = np.array([(X[i, j], Y[i, j]) for i, j in path_robot])

    # print(path_coords_robot)
    cost_map_human = compute_cost_map(collision_prob_map, gamma_human) + obstacle_map * 1e6
    path_human = dijkstra(cost_map_human, start_idx, end_idx)
    path_coords_human = np.array([(X[i, j], Y[i, j]) for i, j in path_human])

    print(path_coords_human[0])
    print(path_coords_human[1])
    print(path_coords_human[2])

    # Determine next positions
    if len(path_coords_robot) > 1:
        next_pos_robot = path_coords_robot[1]
    else:
        next_pos_robot = path_coords_robot[0]

    if len(path_coords_human) > 1:
        next_pos_human = path_coords_human[1]
    else:
        next_pos_human = path_coords_human[0]

    # Decision-making logic
    D_own = compute_remaining_distance(path_coords_robot, 0)
    D_adapt = compute_remaining_distance(path_coords_human, 0)
    idx_next_robot = pos_to_idx(next_pos_robot)
    idx_next_human = pos_to_idx(next_pos_human)
    p_risk_own = collision_prob_map[idx_next_robot]
    p_risk_adapt = collision_prob_map[idx_next_human]
    decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)

    if cnt>2:
        if (robot_positions[-1] == robot_positions[-2]).all():
            next_pos_robot = path_coords_robot[2]
            next_pos_human = path_coords_human[2]

    robot_position = next_pos_human if decision_value >= 0 else next_pos_robot

    
    print("---------------------")

    print(robot_position)

    print("---------------------")
    # Append robot's current position for plotting
    robot_positions.append(robot_position.copy())

    print("---------------------")

    print(robot_positions[-1])
    print(robot_positions[-2])
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

# '''





