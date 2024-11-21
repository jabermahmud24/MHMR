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
# A = np.array([0.1, 9.8])   # Starting point A
A = np.array([2.8, 1.8])   # Starting point A
B = np.array([7, 4])       # Target point B

# Obstacles (circles)
# obstacles = [
#     {'center': np.array([5, 1]),   'radius': 1},
#     {'center': np.array([5, 3.5]), 'radius': 1},
#     {'center': np.array([5, 6]),   'radius': 1},
#     {'center': np.array([5, 9]),   'radius': 1},
# ]

obstacles = [
    {'center': np.array([5, 0.5]),   'radius': 1},
    {'center': np.array([5, 3]), 'radius': 1},
    {'center': np.array([5, 5.5]),   'radius': 1},
    {'center': np.array([5, 8]),   'radius': 1},
]


# obstacles = [
#     {'center': np.array([5, 0.5]),   'radius': 1},
#     {'center': np.array([5, 3]), 'radius': 1},
#     {'center': np.array([5, 5.5]),   'radius': 1},
#     {'center': np.array([5, 8]),   'radius': 1},
# ]


# Robot's positional uncertainty (covariance matrix)
# sigma = 0.05  # Standard deviation in x and y
sigma = 0.1  # Standard deviation in x and y
# Sigma = np.array([[sigma**2, 0],
                #   [0, sigma**2]])  # Covariance matrix


Sigma = np.array([[0, 0],
                  [0, 0]])  # Covariance matrix

# Prelec function parameter gamma values
gamma_robot = 1       # Robot's gamma
# gamma_human = 0.618    # Human's gamma
gamma_human = 0.4     # Human's gamma

# Parameters for utility function
phi_k = 1.0      # Human preference parameter
# phi_k = 0.38     # Human preference parameter
alpha = 0.88         # Value function parameter
beta_prelec = 1      # Prelec function parameter (usually set to 1)
gamma_prelec = gamma_human  # Prelec function curvature for utility calculation
C_risk = 1      # Cost associated with collision
# C_risk = -10        # Cost associated with collision

# Prelec probability weighting function for utility
# def w(p, beta=beta_prelec, gamma=gamma_prelec):
#     epsilon = 1e-10
#     p = np.clip(p, epsilon, 1 - epsilon)
#     print("=================")
#     print(p)
#     print("=================")
#     return np.exp(-beta * (-np.log(p)) ** gamma)


def w(p, beta=beta_prelec, gamma=gamma_prelec):
    return np.exp(-beta * (-np.log(p + 1e-10)) ** gamma)


# Value function
def v(D, alpha=alpha):
    return -D ** alpha

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



def compute_travelled_distance(robot_positions, utility_values):
    """
    Calculate the cumulative distance the robot has traveled along its preferred path
    (the robot's own path) when the utility is negative.

    Parameters:
    - robot_positions: List of robot positions at each time step.
    - utility_values: List or array of utility values at each time step.

    Returns:
    - distances: List of cumulative distances at each time step.
    """
    cumulative_distance = 0.0
    distances = [0.0]  # Start with zero distance at the initial position
    in_negative_utility = False
    for i in range(1, len(utility_values)):
        if utility_values[i] < 0:
            if not in_negative_utility:
                in_negative_utility = True
                cumulative_distance = 0.0  # Reset when utility becomes negative
            # Compute distance between current and previous positions
            step_distance = np.linalg.norm(robot_positions[i] - robot_positions[i - 1])
            cumulative_distance += step_distance
        else:
            in_negative_utility = False
            cumulative_distance = 0.0  # Reset when utility becomes positive
        distances.append(cumulative_distance)
    return distances


# def compute_travelled_distance(D_adapt_values, utility_values):
#     """
#     Calculate the traveled distance along the human's path from the starting point where
#     the utility becomes negative until the current time step, if the utility remains negative.
#     Reset the cumulative distance to zero if the utility becomes positive.

#     Parameters:
#     - D_adapt_values: List or array of D_adapt values at each time step.
#     - utility_values: List or array of utility values at each time step.

#     Returns:
#     - distances: List of cumulative distances at each time step.
#     """
#     cumulative_distance = []
#     # cumulative_distance = 0.0
#     distances = []
#     in_negative_utility = False
#     D_adapt_initial = 0.0
#     for i in range(len(utility_values)):
#         if utility_values[i] < 0:
#             if not in_negative_utility:
#                 in_negative_utility = True
#                 D_adapt_initial = D_adapt_values[i]
#             cumulative_distance = np.abs(D_adapt_initial - D_adapt_values[i])
#             # cumulative_distance = D_adapt_initial - D_adapt_values[i]
#         else:
#             in_negative_utility = False
#             cumulative_distance = 0.0
#             D_adapt_initial = 0.0  # Reset initial distance when utility becomes positive
#         # distances.append(cumulative_distance)
#     cumulative_distance += cumulative_distance
#     return cumulative_distance
#     # return distances


# def compute_travelled_distance(path_coords, )

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
steps_to_save = [1, 25, 55, 80]  # Steps to save images

utility_values = []  # Collect utility values during simulation
D_adapt_values = []
phi_k_values = []



# Simulation loop
while np.linalg.norm(robot_position - B) > 0.1 and step < max_steps:
    # Update start index to robot's current position
    start_idx = pos_to_idx(robot_position)
    end_idx = pos_to_idx(B)

    # Compute paths starting from current robot position
    path_robot = dijkstra(cost_map_robot, start_idx, end_idx)
    path_coords_robot = np.array([(X[i, j], Y[i, j]) for i, j in path_robot])

    path_human = dijkstra(cost_map_human, start_idx, end_idx)
    path_coords_human = np.array([(X[i, j], Y[i, j]) for i, j in path_human])

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
    D_adapt_values.append(D_adapt)

    travelled_distances = compute_travelled_distance(robot_positions, utility_values)
    phi_k = phi_k * np.exp(-(travelled_distances[-1]/2))


    print('human behavior:', phi_k)
    print("---------------------")

    phi_k_values.append(phi_k)

    idx_next_robot = pos_to_idx(next_pos_robot)
    idx_next_human = pos_to_idx(next_pos_human)
    p_risk_own = collision_prob_map[idx_next_robot]
    p_risk_adapt = collision_prob_map[idx_next_human]
    decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
    utility_values.append(decision_value)
    print('decision value:', decision_value)

    # Update robot position based on decision
    if decision_value >= 0:
        candidate_position = next_pos_human
    else:
        candidate_position = next_pos_robot

    # Check if the candidate position is the same as the current position
    if np.array_equal(candidate_position, robot_position):
        # Try to move further along the path
        if decision_value >= 0 and len(path_coords_human) > 2:
            candidate_position = path_coords_human[2]
        elif decision_value < 0 and len(path_coords_robot) > 2:
            candidate_position = path_coords_robot[2]
        else:
            # As a fallback, move a small step towards the goal
            direction = (B - robot_position)
            direction = direction / np.linalg.norm(direction)
            candidate_position = robot_position + 0.01 * direction  # Adjust step size as needed



    robot_position = candidate_position

    # Append robot's current position for plotting
    robot_positions.append(robot_position.copy())


# # Simulation loop
# while np.linalg.norm(robot_position - B) > 0.1 and step < max_steps:
#     cnt += 1
    

#     # Update start index to robot's current position
#     start_idx = pos_to_idx(robot_position)
#     end_idx = pos_to_idx(B)







#     # Compute paths starting from current robot position
#     path_robot = dijkstra(cost_map_robot, start_idx, end_idx)
#     path_coords_robot = np.array([(X[i, j], Y[i, j]) for i, j in path_robot])

#     path_human = dijkstra(cost_map_human, start_idx, end_idx)
#     path_coords_human = np.array([(X[i, j], Y[i, j]) for i, j in path_human])

#     # Determine next positions
#     next_pos_robot = path_coords_robot[1] if len(path_coords_robot) > 1 else path_coords_robot[0]
#     next_pos_human = path_coords_human[1] if len(path_coords_human) > 1 else path_coords_human[0]

#     #Current positions

#     current_pos_robot = path_coords_robot[0]
#     current_pos_human = path_coords_human[0]

#     # Decision-making logic
#     D_own = compute_remaining_distance(path_coords_robot, 0)
#     D_adapt = compute_remaining_distance(path_coords_human, 0)
#     D_adapt_values.append(D_adapt)
#     # print(D_adapt_values)

#     travelled_distances = compute_travelled_distance(robot_positions, utility_values)
#     # travelled_distances = compute_travelled_distance(D_adapt_values, utility_values)
#     print("---------------------")
#     # print(travelled_distances[-1])
#     # print("---------------------")

#     phi_k = phi_k * np.exp(-(travelled_distances[-1]/10))


#     print(phi_k)
#     print("---------------------")

#     phi_k_values.append(phi_k)



#     # D_travelled_human = compute_travelled_distance(path)

#     idx_next_robot = pos_to_idx(next_pos_robot)
#     idx_next_human = pos_to_idx(next_pos_human)
#     p_risk_own = collision_prob_map[idx_next_robot]
#     # print("=====================")
#     # print("True risk", p_risk_own)
#     # print("=====================")
#     p_risk_adapt = collision_prob_map[idx_next_human]
#     decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
#     utility_values.append(decision_value)

#     # if cnt > 2:
#     #     if (robot_positions[-1] == robot_positions[-2]).all():
#     #         next_pos_robot = path_coords_robot[2]
#     #         next_pos_human = path_coords_human[2]

#     robot_position = next_pos_human if decision_value >= 0 else next_pos_robot
#     # print(robot_position[1])
#     # print(len(path_coords_human))
#     # print(path_coords_human[3].shape)

#     ### Now robot_position is the robot's next position. Here, using the optimal control law,
#     ### the robot will move from current position to the next position.

#     '''
#     Algorithm
#     '''

    

#     # # Append robot's current position for plotting
#     robot_positions.append(robot_position.copy())

#     # print("---------------------")
#     # print(cnt)
#     # print(decision_value)
#     # print("---------------------")

    

    # Clear previous paths and replot updated paths
    ax.clear()
    ax.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Robot Navigation')
    ax.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
    ax.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')
    ax.plot(path_coords_robot[:, 0], path_coords_robot[:, 1], 'b--', label="Robot's Planned Path")
    ax.plot(path_coords_human[:, 0], path_coords_human[:, 1], 'g--', label="Human's Planned Path")
    ax.plot(np.array(robot_positions)[:, 0], np.array(robot_positions)[:, 1], 'r-', linewidth=2, label='Robot Trajectory')
    ax.legend()
    plt.tight_layout()
    ax.grid(True)

    # Save plot for specific steps
    if step in steps_to_save:
        plt.savefig(f"robot_navigation_step_{step}.png")  # Save the image


    # Display the updated plot
    plt.pause(0.1)  # Small pause to update the plot

    step += 1

# Turn off interactive mode
plt.ioff()
plt.show()




# Convert utility_values to a NumPy array
utility_values = np.array(utility_values)

# After the simulation loop
plt.figure(figsize=(10, 6))
plt.plot(range(len(utility_values)), utility_values, 'm-', linewidth=2, label='Utility Value')

# Fill areas based on utility value conditions
plt.fill_between(
    range(len(utility_values)),
    utility_values,
    0,
    where=(utility_values >= 0),
    color='green',
    alpha=0.3,
    label='Positive Utility'
)
plt.fill_between(
    range(len(utility_values)),
    utility_values,
    0,
    where=(utility_values < 0),
    color='blue',
    alpha=0.3,
    label='Negative Utility'
)

# Add labels, title, and grid
plt.xlabel('Decision Step')
plt.ylabel('Utility Value')
plt.title('Utility Value Over Decision Steps')
plt.axhline(0, color='black', linestyle='--')
plt.grid(True)
plt.legend()
# Save the plot
plt.savefig("utility_value_plot.png")

plt.show()




# Example: Plotting the cumulative traveled distances
plt.figure(figsize=(10, 6))
plt.plot(range(len(travelled_distances)), travelled_distances, 'c-', linewidth=2, label='Cumulative Traveled Distance')
plt.xlabel('Decision Step')
plt.ylabel('Cumulative Traveled Distance')
plt.title('Cumulative Traveled Distance Along Human\'s Path During Negative Utility')
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(range(len(phi_k_values)), phi_k_values, 'c-', linewidth=2, label='Phi values')
plt.xlabel('Decision Step')
plt.ylabel('Phi')
# plt.title('Cumulative Traveled Distance Along Human\'s Path During Negative Utility')
plt.grid(True)
plt.legend()
plt.show()
