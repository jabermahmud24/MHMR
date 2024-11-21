import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from heapq import heappush, heappop
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Environment parameters
grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])

# x_range = np.linspace(2, 7,4, grid_size[0])
# y_range = np.linspace(0,  10, grid_size[1])
X, Y = np.meshgrid(x_range, y_range, indexing='ij')

# Start and end points
A = np.array([0.4, 0.4])   # Starting point A
B = np.array([9.5, 4.0])       # Target point B

# A = np.array([1.8, 1.8])   # Starting point A
# B = np.array([7, 6])       # Target point B


# Obstacles (circles)

obstacles = [
    {'center': np.array([0.5, 5]), 'width': 1, 'height': 10},
    {'center': np.array([9.5, 5]), 'width': 1, 'height': 10},
    
    {'center': np.array([2.0, 8.0]), 'width': 5, 'height': 2},
    {'center': np.array([3.5, 9.0]), 'width': 2, 'height': 2},
    {'center': np.array([5.0, 9.0]), 'width': 4, 'height': 2},
    {'center': np.array([7, 8.0]), 'width': 2, 'height': 2},
    {'center': np.array([8.0, 8.0]), 'width': 5, 'height': 2},

    # {'center': np.array([0.5, 5]), 'width': 1, 'height': 10},
    # {'center': np.array([9.5, 5]), 'width': 1, 'height': 10},
    {'center': np.array([2.0, 4.5]), 'width': 2, 'height': 1.5},
    {'center': np.array([3.5, 4.75]), 'width': 1.5, 'height': 2},
    {'center': np.array([4.5, 5.5]), 'width': 4, 'height': 1.5},
    {'center': np.array([7, 5.25]), 'width': 1.5, 'height': 2},
    {'center': np.array([8.0, 4.5]), 'width': 4, 'height': 1.5},

    {'center': np.array([2.0, 1.5]), 'width': 2.5, 'height': 1.2},
    {'center': np.array([3.5, 1.95]), 'width': 0.9, 'height': 2.1},
    {'center': np.array([5.2, 2.5]), 'width': 4.5, 'height': 1.2},
    {'center': np.array([7, 2.0]), 'width': 0.0, 'height': 0.0},
    {'center': np.array([8.0, 1.5]), 'width': 2.2, 'height': 1.0},


    
    # {'center': np.array([5, 5.5]), 'width': 2, 'height': 1},
    # {'center': np.array([5, 8]), 'width': 2, 'height': 1},
]


# Robot's positional uncertainty (covariance matrix)
# sigma = 0.05  # Standard deviation in x and y
sigma = 0.1  # Standard deviation in x and y
Sigma = np.array([[sigma**2, 0],
                  [0, sigma**2]])  # Covariance matrix

# Prelec function parameter gamma values
gamma_robot = 1       # Robot's gamma
# gamma_human = 0.1    # Human's gamma
# gamma_human = 0.618    # Human's gamma
gamma_human = 0.21     # Human's gamma

# Parameters for utility function
phi_k_init = 1.0    # Human preference parameter
phi_k = phi_k_init    # Human preference parameter
# phi_k = 0.6     # Human preference parameter
alpha = 0.88         # Value function parameter
beta_prelec = 1      # Prelec function parameter (usually set to 1)
gamma_prelec = gamma_human  # Prelec function curvature for utility calculation
C_risk = 10       # Cost associated with collision
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
    # U_own = 1.2 * phi_k * (v(D_own) - w(p_risk_own) * C_risk)
    U_adapt = (1 - phi_k) * (v(D_adapt) - w(p_risk_adapt) * C_risk)
    decision_value = U_own - U_adapt
    return decision_value


# def create_obstacle_map():
#     obstacle_map = np.zeros(grid_size)
#     points = np.stack((X, Y), axis=-1)  # Shape (grid_size[0], grid_size[1], 2)
    
#     for obs in obstacles:
#         # Calculate rectangle boundaries
#         x_min = obs['center'][0] - obs['width'] / 2
#         x_max = obs['center'][0] + obs['width'] / 2
#         y_min = obs['center'][1] - obs['height'] / 2
#         y_max = obs['center'][1] + obs['height'] / 2

#         # Create a mask for points inside the rectangle
#         inside_x = (points[..., 0] >= x_min) & (points[..., 0] <= x_max)
#         inside_y = (points[..., 1] >= y_min) & (points[..., 1] <= y_max)
#         obstacle_map[inside_x & inside_y] = 1

#     return obstacle_map


def create_obstacle_map():
    # Initialize the grid with obstacles (1)
    obstacle_map = np.ones(grid_size)
    points = np.stack((X, Y), axis=-1)  # Shape (grid_size[0], grid_size[1], 2)
    
    for obs in obstacles:
        # Calculate rectangle boundaries
        x_min = obs['center'][0] - obs['width'] / 2
        x_max = obs['center'][0] + obs['width'] / 2
        y_min = obs['center'][1] - obs['height'] / 2
        y_max = obs['center'][1] + obs['height'] / 2

        # Create a mask for points inside the rectangle
        inside_x = (points[..., 0] >= x_min) & (points[..., 0] <= x_max)
        inside_y = (points[..., 1] >= y_min) & (points[..., 1] <= y_max)
        
        # Clear the specified rectangle portion (set to 0)
        obstacle_map[inside_x & inside_y] = 0

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

def compute_cost_map(collision_prob_map, gamma):
    w_p = prelec_weight(collision_prob_map, gamma)
    cost_map = w_p
    return cost_map

# def compute_cost_map(collision_prob_map, gamma):
#     w_p = prelec_weight(collision_prob_map, gamma)
#     min_cost = 1e-3  # Small positive value to prevent zero cost
#     cost_map = np.clip(w_p, min_cost, None)
#     return cost_map



def astar(cost_map, start_idx, end_idx):
    import heapq
    from math import sqrt

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
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in neighbors:
            nx, ny = current_idx[0] + dx, current_idx[1] + dy
            if 0 <= nx < cost_map.shape[0] and 0 <= ny < cost_map.shape[1]:
                if not visited[nx, ny]:
                    heuristic = sqrt((nx - end_idx[0])**2 + (ny - end_idx[1])**2)
                    move_cost = sqrt(dx**2 + dy**2) * cost_map[nx, ny]
                    new_cost = current_cost + move_cost
                    total_cost = new_cost + heuristic
                    if total_cost < cost_to_come[nx, ny]:
                        cost_to_come[nx, ny] = total_cost
                        parent[nx, ny] = current_idx
                        heappush(heap, (total_cost, (nx, ny)))

    # Reconstruct path
    path = []
    idx = end_idx
    while np.all(parent[idx] != -1):
        path.append(idx)
        idx = tuple(parent[idx])
    path.append(start_idx)
    path = path[::-1]
    return path






# Path planning using Dijkstra's algorithm
def dijkstra(cost_map, start_idx, end_idx):
    visited = np.full(cost_map.shape, False)
    cost_to_come = np.full(cost_map.shape, np.inf)
    # current_idx = start_idx
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
                        # parent[nx, ny] = current_idx
                        parent[nx, ny] = np.array(current_idx)
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

# def compute_direction_and_extrapolate(robot_positions):
#     if len(robot_positions) < 2:
#         # If not enough history, return a small random movement
#         return np.array([0.1, 0.0])  # Default small movement
#     # Compute direction of last movement
#     direction = robot_positions[-3] - robot_positions[-4]
#     # Normalize the direction to ensure consistent step size
#     step_size = 0.5  # Define a reasonable step size
#     direction_norm = np.linalg.norm(direction)
#     if direction_norm == 0:
#         # If there's no valid movement, assign a default step size
#         direction = np.array([step_size, 0.0])
#     else:
#         direction = direction / direction_norm * step_size
#     return direction

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
                # cumulative_distance = 0.0  # Reset when utility becomes negative
            # Compute distance between current and previous positions
            step_distance = np.linalg.norm(robot_positions[i] - robot_positions[i - 1])
            cumulative_distance += step_distance
        else:
            in_negative_utility = False
            # cumulative_distance = 0.0  # Reset when utility becomes positive
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
steps_to_save = [1, 50, 80, 110]  # Steps to save images

utility_values = []  # Collect utility values during simulation
D_adapt_values = []
phi_k_values = []
stuck_while_following_robot = 0
run_once = True 
stuck_count = 0
# # Set up FFMpegWriter
# writer = FFMpegWriter(fps=4)  # Set the frame rate of the video
# writer.setup(fig, 'robot_navigation_simulation_test_1_phi_0_5.mp4', dpi=300)  # Output file

while np.linalg.norm(robot_position - B) > 0.1 and step < max_steps:
    cnt += 1
    start_idx = pos_to_idx(robot_position)
    end_idx = pos_to_idx(B)

    # Compute paths using Dijkstra's algorithm
    path_robot = dijkstra(cost_map_robot, start_idx, end_idx)
    path_human = dijkstra(cost_map_human, start_idx, end_idx)

    if not path_robot or not path_human:
        print("Path not found for robot or human.")
        break

    # Convert path indices to coordinates
    path_coords_robot = np.array([(X[i, j], Y[i, j]) for i, j in path_robot])
    path_coords_human = np.array([(X[i, j], Y[i, j]) for i, j in path_human])

    # Determine next positions
    next_pos_robot = path_coords_robot[1] if len(path_coords_robot) > 1 else path_coords_robot[0]
    next_pos_human = path_coords_human[1] if len(path_coords_human) > 1 else path_coords_human[0]

    # Current positions
    current_pos_robot = path_coords_robot[0]
    current_pos_human = path_coords_human[0]

    # Decision-making logic
    D_own = compute_remaining_distance(path_coords_robot, 0)
    D_adapt = compute_remaining_distance(path_coords_human, 0)
    D_adapt_values.append(D_adapt)
    travelled_distances = compute_travelled_distance(robot_positions, utility_values)
    print("---------------------")
    print(f"Travelled Distance: {travelled_distances[-1]}")
    print("---------------------")

    # Collision probabilities
    idx_next_robot = pos_to_idx(next_pos_robot)
    idx_next_human = pos_to_idx(next_pos_human)
    p_risk_own = collision_prob_map[idx_next_robot]
    p_risk_adapt = collision_prob_map[idx_next_human]

    if cnt > 3 and decision_value > 0 and run_once:
        # Example condition to modify phi_k or other parameters
        phi_k -= 0.0  # Adjust as needed
        print("Checked decision value condition.")
        run_once = False  # Ensure this block runs only once

    print(f"Human behavior (phi_k): {phi_k}")

    # Compute utility
    decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
    # if 0 > decision_value > -0.7:
    #     decision_value = 0
    print(f"Decision Value: {decision_value}")
    print(f"Run Once Flag: {run_once}")

    utility_values.append(decision_value)
    print('---------------------')

    # Handle potential stuck condition
    if cnt > 2 and np.array_equal(robot_positions[-1], robot_positions[-2]):
        if len(path_coords_robot) > 2 and len(path_coords_human) > 2:
            next_pos_robot = path_coords_robot[2]
            next_pos_human = path_coords_human[2]
            stuck_count += 1
            print(f"Stuck Count: {stuck_count}")
            if decision_value < 0:
                stuck_while_following_robot += 1
                print("stuck_while_following_robot", stuck_while_following_robot)
            print(f"Robot was stuck. Extrapolated to new position: {robot_position}")
        else:
            print("Cannot extrapolate further; path length insufficient.")

            
    # Decide next position based on utility
    robot_position = next_pos_human if decision_value >= 0 else next_pos_robot
    robot_positions.append(robot_position.copy())

    # # Now compute travelled distances and update phi_k
    # if len(robot_positions) > 1:
    #     dist_travelled = np.linalg.norm(robot_positions[-1] - robot_positions[-2])
    #     if len(travelled_distances) > 0:
    #         total_distance = travelled_distances[-1] + dist_travelled
    # travelled_distances.append(total_distance)

    print("---------------------")
    print(f"Travelled Distance: {travelled_distances[-1]}")
    print("---------------------")

    # Update phi_k based on travelled distance
    phi_k = phi_k_init * np.exp(-(travelled_distances[-1] / 2))
    phi_k_values.append(phi_k)

    # Update plot
    ax.clear()
    ax.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Robot Navigation')
    ax.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
    ax.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')
    ax.plot(path_coords_robot[:, 0], path_coords_robot[:, 1], 'b--', label="Robot's Planned Path")
    ax.plot(path_coords_human[:, 0], path_coords_human[:, 1], 'g--', label="Human's Planned Path")
    ax.plot(np.array(robot_positions)[:-1, 0], np.array(robot_positions)[:-1, 1], 'r-', linewidth=2, label='Robot Trajectory')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.pause(0.01)  # Reduced pause for better performance

    step += 1


# Turn off interactive mode
plt.ioff()
plt.show()


print("Total Number of Stucks:", stuck_count)
print("total stuck_while_following_robot", stuck_while_following_robot)

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
# plt.savefig("utility_value_plot.png")

plt.show()


# # After the simulation loop:
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(utility_values)), utility_values, 'm-', linewidth=2)
# plt.xlabel('Decision Step')
# plt.ylabel('Utility Value')
# plt.title('Utility Value Over Decision Steps')
# plt.grid(True)
# plt.axhline(0, color='black', linestyle='--')
# plt.show()


# fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# # Robot Cost Map
# im1 = axes[0].imshow(cost_map_robot.T, origin='lower', extent=(0, 10, 0, 10), cmap='viridis')
# axes[0].set_title("Robot's Cost Map")
# axes[0].set_xlabel('X-axis')
# axes[0].set_ylabel('Y-axis')
# for obs in obstacles:
#     circle = plt.Circle(obs['center'], obs['radius'], color='black', fill=False)
#     axes[0].add_patch(circle)
# fig.colorbar(im1, ax=axes[0], label='Cost')

# # Human Cost Map
# im2 = axes[1].imshow(cost_map_human.T, origin='lower', extent=(0, 10, 0, 10), cmap='viridis')
# axes[1].set_title("Human's Cost Map")
# axes[1].set_xlabel('X-axis')
# axes[1].set_ylabel('Y-axis')
# for obs in obstacles:
#     circle = plt.Circle(obs['center'], obs['radius'], color='black', fill=False)
#     axes[1].add_patch(circle)
# fig.colorbar(im2, ax=axes[1], label='Cost')

# plt.suptitle('Cost Maps Incorporating Prelec Weighting')
# plt.show()




# # gamma_values = [0.1]
# gamma_values = [0.21, 0.61, 1.0]


# # Visualize perceived collision probability maps for different gamma values
# for gamma in gamma_values:
#     w_p_map = prelec_weight(collision_prob_map, gamma)
#     plt.figure(figsize=(8, 6))
#     plt.imshow(np.flipud(w_p_map.T), extent=(0, 10, 0, 10), cmap='hot')
#     plt.colorbar(label=f'Perceived Collision Probability (γ={gamma})')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.title(f'Perceived Collision Probability Map (γ={gamma})')
#      # Save the figure
#     plt.savefig(f'complex_perceived_collision_probability_gamma_{gamma}.png', dpi=300, bbox_inches='tight')
#     plt.show()



# Example: Plotting the cumulative traveled distances
plt.figure(figsize=(10, 6))
plt.plot(range(len(travelled_distances)), travelled_distances, 'c-', linewidth=2, label='Cumulative Traveled Distance')
plt.xlabel('Decision Step')
plt.ylabel('Cumulative Traveled Distance')
plt.title('Cumulative Traveled Distance Along Human\'s Path During Negative Utility')
plt.grid(True)
plt.legend()
plt.show()


# plt.figure(figsize=(10, 6))
# plt.plot(range(len(phi_k_values)), phi_k_values, 'c-', linewidth=2, label='Phi values')
# plt.xlabel('Decision Step')
# plt.ylabel('Phi')
# # plt.savefig("Human tendency.png")
# # plt.title('Cumulative Traveled Distance Along Human\'s Path During Negative Utility')
# plt.grid(True)
# plt.legend()
# plt.show()


# Set font sizes for each component
title_fontsize = 18
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 12

plt.figure(figsize=(10, 6))
plt.plot(range(len(phi_k_values)-1), phi_k_values[1:], 'c-', linewidth=2, label='Phi values')

# Add labels and title
plt.xlabel('Decision Step', fontsize=label_fontsize)
plt.ylabel('Phi', fontsize=label_fontsize)
# plt.title('Cumulative Traveled Distance Along Human\'s Path During Negative Utility', fontsize=title_fontsize)

# Customize grid, legend, and ticks
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=legend_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# Save and display
# plt.savefig("Human_tendency.png", dpi=300, bbox_inches='tight')
plt.show()
