import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from heapq import heappush, heappop
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
# Environment parameters
grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
X, Y = np.meshgrid(x_range, y_range, indexing='ij')

# # Start and end points
A = np.array([2.8, 1.8])   # Starting point A
B = np.array([7, 4])       # Target point B

# A = np.array([1.8, 1.8])   # Starting point A
# B = np.array([7, 6])       # Target point B


# Obstacles (circles)

obstacles = [
    {'center': np.array([5, 0.5]),   'radius': 1},
    {'center': np.array([5, 3]), 'radius': 1},
    {'center': np.array([5, 5.5]),   'radius': 1},
    {'center': np.array([5, 8]),   'radius': 1},
]


# obstacles = [
#     {'center': np.array([4, 0.5]),   'radius': 1},
#     {'center': np.array([6, 3]), 'radius': 1},
#     {'center': np.array([5, 5.5]),   'radius': 1},
#     {'center': np.array([2, 8]),   'radius': 1},
# ]


# Robot's positional uncertainty (covariance matrix)
# sigma = 0.05  # Standard deviation in x and y
sigma = 0.1  # Standard deviation in x and y
Sigma = np.array([[sigma**2, 0],
                  [0, sigma**2]])  # Covariance matrix

# Prelec function parameter gamma values
gamma_robot = 1       # Robot's gamma
# gamma_human = 0.1    # Human's gamma
gamma_human = 0.4     # Human's gamma

# Parameters for utility function
phi_k = 0.5   # Human preference parameter
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







# Assume all necessary functions and variables (like dijkstra, compute_remaining_distance, etc.) are defined above.

# Parameters (ensure these are defined appropriately)
label_fontsize = 12
title_fontsize = 14
legend_fontsize = 10
tick_fontsize = 10
min_utility, max_utility = -14, 5  # Adjust based on expected utility values
min_phi, max_phi = 0, phi_k+0.05  # Adjust based on phi_k range
plot_steps = 140

# # Initialize variables
# robot_position = np.array(A)  # Starting position
# robot_positions = [robot_position.copy()]
# phi_k = 1.0  # Initial phi value
# phi_k_values = [phi_k]
# utility_values = []
# D_adapt_values = []
# step = 0
# cnt = 0
# max_steps = 1000  # Adjust as needed

# Set up the video writer
metadata = dict(title='Simulation Video', artist='Matplotlib', comment='Robot Navigation and Metrics')
writer = FFMpegWriter(fps=4, metadata=metadata)

# Initialize the figure and GridSpec
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], figure=fig)


# Initialize subplots
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

# Initialize plot elements on ax1 (Robot Navigation)
obstacle_image = ax1.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
start_point_a, = ax1.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
target_point_b, = ax1.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')
path_robot_line, = ax1.plot([], [], 'b--', label="Robot's Planned Path")
path_human_line, = ax1.plot([], [], 'g--', label="Human's Planned Path")
robot_trajectory_line, = ax1.plot([], [], 'r-', linewidth=2, label='Robot Trajectory')
ax1.set_xlabel('X-axis', fontsize=label_fontsize)
ax1.set_ylabel('Y-axis', fontsize=label_fontsize)
ax1.set_title('Robot Navigation', fontsize=title_fontsize)
ax1.legend(fontsize=legend_fontsize)
ax1.grid(True)

# Initialize plot elements on ax2 (Utility Values)
utility_line, = ax2.plot([], [], 'm-', linewidth=2, label='Utility Value')
# Initialize empty lists for fill_between patches
ax2.positive_fill = None
ax2.negative_fill = None
ax2.axhline(0, color='black', linestyle='--')
ax2.set_title('Utility Value Over Decision Steps', fontsize=title_fontsize)
ax2.set_xlabel('Decision Step', fontsize=label_fontsize)
ax2.set_ylabel('Utility Value', fontsize=label_fontsize)
# ax2.set_xlim(0, plot_steps)
# ax2.set_ylim(min_utility, max_utility)
ax2.legend(fontsize=legend_fontsize)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

# Initialize plot elements on ax3 (Phi Values)
phi_line, = ax3.plot([], [], 'c-', linewidth=2, label='Phi values')
ax3.set_xlabel('Decision Step', fontsize=label_fontsize)
ax3.set_ylabel('Phi', fontsize=label_fontsize)
ax3.set_title('Human Preference (Phi)', fontsize=title_fontsize)
ax3.set_xlim(0, plot_steps)
ax3.set_ylim(min_phi, max_phi)
ax3.legend(fontsize=legend_fontsize)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.tick_params(axis='both', which='major', labelsize=tick_fontsize)

# Precompute X and Y if not already done
# X, Y = np.meshgrid(np.linspace(0, 10, obstacle_map.shape[0]), np.linspace(0, 10, obstacle_map.shape[1]))

with writer.saving(fig, "gamma_0.4_simulation_video_test_1_phi_0_5.mp4", dpi=100):
    while np.linalg.norm(robot_position - B) > 0.1 and step < max_steps:
        cnt += 1
        print(cnt)

        # Update start and end indices
        start_idx = pos_to_idx(robot_position)
        end_idx = pos_to_idx(B)

        # Compute paths
        path_robot = dijkstra(cost_map_robot, start_idx, end_idx)
        path_coords_robot = np.array([(X[i, j], Y[i, j]) for i, j in path_robot])

        path_human = dijkstra(cost_map_human, start_idx, end_idx)
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

        phi_k *= np.exp(-(travelled_distances[-1] / 100))
        phi_k_values.append(phi_k)

        p_risk_own = collision_prob_map[pos_to_idx(next_pos_robot)]
        p_risk_adapt = collision_prob_map[pos_to_idx(next_pos_human)]
        decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
        utility_values.append(decision_value)

        # Handle potential robot position stagnation
        if cnt > 2 and np.array_equal(robot_positions[-1], robot_positions[-2]):
            if len(path_coords_robot) > 2 and len(path_coords_human) > 2:
                next_pos_robot = path_coords_robot[2]
                next_pos_human = path_coords_human[2]

        # Decide next position based on utility
        robot_position = next_pos_human if decision_value >= 0 else next_pos_robot
        robot_positions.append(robot_position.copy())

        # --- Update Plots ---

        # Update Robot Navigation Plot (ax1)
        path_robot_line.set_data(path_coords_robot[:, 0], path_coords_robot[:, 1])
        path_human_line.set_data(path_coords_human[:, 0], path_coords_human[:, 1])
        traj = np.array(robot_positions)
        robot_trajectory_line.set_data(traj[:, 0], traj[:, 1])

        # Update Utility Values Plot (ax2)
        utility_line.set_data(range(len(utility_values)), utility_values)

        # Remove previous fill_between patches if they exist
        if ax2.positive_fill is not None:
            ax2.positive_fill.remove()
            ax2.positive_fill = None
        if ax2.negative_fill is not None:
            ax2.negative_fill.remove()
            ax2.negative_fill = None

        # Re-plot fill_between
        utility_array = np.array(utility_values)
        steps = np.arange(len(utility_values))
        ax2.positive_fill = ax2.fill_between(
            steps,
            utility_array,
            0,
            where=(utility_array >= 0),
            color='green',
            alpha=0.3,
            interpolate=True
        )
        ax2.negative_fill = ax2.fill_between(
            steps,
            utility_array,
            0,
            where=(utility_array < 0),
            color='blue',
            alpha=0.3,
            interpolate=True
        )

        # Update Phi Values Plot (ax3)
        phi_line.set_data(range(len(phi_k_values)), phi_k_values)

        # Optionally adjust axes limits if necessary
        # ax1.relim(); ax1.autoscale_view()
        # ax2.relim(); ax2.autoscale_view()
        # ax3.relim(); ax3.autoscale_view()

        # Capture the current frame
        writer.grab_frame()

        # Update simulation step
        step += 1

# Turn off interactive mode and display the plot (optional)
plt.ioff()
plt.show()








# title_fontsize = 18
# label_fontsize = 14
# tick_fontsize = 12
# legend_fontsize = 12

# # Set up the video writer
# metadata = dict(title='Simulation Video', artist='Matplotlib', comment='Robot Navigation and Metrics')
# writer = FFMpegWriter(fps=4, metadata=metadata)

# # Initialize the figure and gridspec
# fig = plt.figure(figsize=(14, 10))
# gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], figure=fig)

# with writer.saving(fig, "simulation_video_2x2.mp4", dpi=100):

#     # Simulation loop
#     while np.linalg.norm(robot_position - B) > 0.1 and step < max_steps:
#         cnt += 1
        

#         # Update start index to robot's current position
#         start_idx = pos_to_idx(robot_position)
#         end_idx = pos_to_idx(B)


#         # Compute paths starting from current robot position
#         path_robot = dijkstra(cost_map_robot, start_idx, end_idx)
#         path_coords_robot = np.array([(X[i, j], Y[i, j]) for i, j in path_robot])

#         path_human = dijkstra(cost_map_human, start_idx, end_idx)
#         path_coords_human = np.array([(X[i, j], Y[i, j]) for i, j in path_human])

#         # Determine next positions
#         next_pos_robot = path_coords_robot[1] if len(path_coords_robot) > 1 else path_coords_robot[0]
#         next_pos_human = path_coords_human[1] if len(path_coords_human) > 1 else path_coords_human[0]

#         #Current positions

#         current_pos_robot = path_coords_robot[0]
#         current_pos_human = path_coords_human[0]

#         # Decision-making logic
#         D_own = compute_remaining_distance(path_coords_robot, 0)
#         D_adapt = compute_remaining_distance(path_coords_human, 0)
#         D_adapt_values.append(D_adapt)
#         # print(D_adapt_values)

#         travelled_distances = compute_travelled_distance(robot_positions, utility_values)
#         # travelled_distances = compute_travelled_distance(D_adapt_values, utility_values)
#         print("---------------------")
#         # print(travelled_distances[-1])
#         # print("---------------------")

#         phi_k = phi_k * np.exp(-(travelled_distances[-1]/100))


#         print('human behavior:', phi_k)


#         phi_k_values.append(phi_k)



#         # D_travelled_human = compute_travelled_distance(path)

#         idx_next_robot = pos_to_idx(next_pos_robot)
#         idx_next_human = pos_to_idx(next_pos_human)
#         p_risk_own = collision_prob_map[idx_next_robot]

#         p_risk_adapt = collision_prob_map[idx_next_human]
#         decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
#         utility_values.append(decision_value)
#         print('decision value:', decision_value)

#         print("---------------------")



#         if cnt > 2:
#             if (robot_positions[-1] == robot_positions[-2]).all():
#                 next_pos_robot = path_coords_robot[2]
#                 next_pos_human = path_coords_human[2]

#         robot_position = next_pos_human if decision_value >= 0 else next_pos_robot

#         # # Append robot's current position for plotting
#         robot_positions.append(robot_position.copy())
        

#         # Robot Navigation Plot (entire first column)
#         ax1 = fig.add_subplot(gs[:, 0])
#         ax1.clear()
#         ax1.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
#         ax1.plot(A[0], A[1], 'go', markersize=8, label='Start Point A')
#         ax1.plot(B[0], B[1], 'bo', markersize=8, label='Target Point B')
#         ax1.plot(path_coords_robot[:, 0], path_coords_robot[:, 1], 'b--', label="Robot's Planned Path")
#         ax1.plot(path_coords_human[:, 0], path_coords_human[:, 1], 'g--', label="Human's Planned Path")
#         ax1.plot(np.array(robot_positions)[:-1, 0], np.array(robot_positions)[:-1, 1], 'r-', linewidth=2, label='Robot Trajectory')
#         ax1.set_xlabel('X-axis', fontsize=label_fontsize)
#         ax1.set_ylabel('Y-axis', fontsize=label_fontsize)
#         ax1.set_title('Robot Navigation', fontsize=title_fontsize)
#         ax1.legend(fontsize=legend_fontsize)
#         ax1.grid(True)

#         # Utility Values Plot (top-right)
#         ax2 = fig.add_subplot(gs[0, 1])
#         ax2.clear()
#         ax2.plot(range(len(utility_values)), utility_values, 'm-', linewidth=2, label='Utility Value')
#         ax2.fill_between(
#             range(len(utility_values)),
#             utility_values,
#             0,
#             where=(np.array(utility_values) >= 0),
#             color='green',
#             alpha=0.3,
#             label='Positive Utility'
#         )
#         ax2.fill_between(
#             range(len(utility_values)),
#             utility_values,
#             0,
#             where=(np.array(utility_values) < 0),
#             color='blue',
#             alpha=0.3,
#             label='Negative Utility'
#         )
#         ax2.axhline(0, color='black', linestyle='--')
#         ax2.set_title('Utility Value Over Decision Steps', fontsize=title_fontsize)
#         ax2.set_xlabel('Decision Step', fontsize=label_fontsize)
#         ax2.set_ylabel('Utility Value', fontsize=label_fontsize)
#         ax2.legend(fontsize=legend_fontsize)
#         ax2.grid(True, linestyle='--', alpha=0.7)
#         ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

#         # Phi Values Plot (bottom-right)
#         ax3 = fig.add_subplot(gs[1, 1])
#         ax3.clear()
#         ax3.plot(range(len(phi_k_values)), phi_k_values, 'c-', linewidth=2, label='Phi values')
#         ax3.set_xlabel('Decision Step', fontsize=label_fontsize)
#         ax3.set_ylabel('Phi', fontsize=label_fontsize)
#         ax3.set_title('Human Tendency (Phi)', fontsize=title_fontsize)
#         ax3.grid(True, linestyle='--', alpha=0.7)
#         ax3.legend(fontsize=legend_fontsize)
#         ax3.tick_params(axis='both', which='major', labelsize=tick_fontsize)

#         # Capture the current frame
#         writer.grab_frame()

#         # Update simulation step
#         step += 1

# # Turn off interactive mode
# plt.ioff()
# plt.show()




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












