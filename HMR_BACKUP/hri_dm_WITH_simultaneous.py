import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv
from scipy.ndimage import gaussian_filter
from heapq import heappush, heappop
import random

# Importing the RobotSerial class
from visual_kinematics.RobotSerial import *

start_time = time.time()

# Robot Control Parameters
lqr_Q = 1000 * np.eye(6)
lqr_R = np.eye(9)
dt = 0.1
show_animation = True

# Decision-Making Parameters
grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
X, Y = np.meshgrid(x_range, y_range, indexing='ij')
A_pos = np.array([2.8, 1.8])  # Starting point A
B_pos = np.array([7, 4])      # Target point B

# Obstacles
obstacles = [
    {'center': np.array([5, 0.5]), 'radius': 1},
    {'center': np.array([5, 3]),   'radius': 1},
    {'center': np.array([5, 5.5]), 'radius': 1},
    {'center': np.array([5, 8]),   'radius': 1},
]

# Robot's positional uncertainty (covariance matrix)
sigma = 0.1  # Standard deviation in x and y
Sigma = np.array([[sigma**2, 0],
                  [0, sigma**2]])  # Covariance matrix

# Prelec function parameter gamma values
gamma_robot = 1.0       # Robot's gamma
gamma_human = 0.1       # Human's gamma

# Parameters for utility function
phi_k_init = 1.0       # Human preference parameter
phi_k = 1.0

alpha = 0.88           # Value function parameter
beta_prelec = 1        # Prelec function parameter (usually set to 1)
gamma_prelec = gamma_human  # Prelec function curvature for utility calculation
C_risk = 10            # Cost associated with collision

# Prelec weighting function for utility
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

def prelec_weight(p, gamma):
    return np.exp(-(-np.log(p + 1e-10)) ** gamma)

def compute_cost_map(collision_prob_map, gamma, obstacle_map):
    w_p = prelec_weight(collision_prob_map, gamma)
    cost_map = w_p + obstacle_map * 1e6  # Add high cost for obstacles
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
                    new_cost = cost_to_come[current_idx] + move_cost
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

def pos_to_idx(pos):
    idx_x = int(pos[0] / 10 * (grid_size[0] - 1))
    idx_y = int(pos[1] / 10 * (grid_size[1] - 1))
    # idx_x = np.clip(idx_x, 0, grid_size[0] - 1)
    # idx_y = np.clip(idx_y, 0, grid_size[1] - 1)
    return (idx_x, idx_y)

def compute_remaining_distance(path_coords, current_idx):
    path_segment = path_coords[current_idx:]
    if len(path_segment) < 2:
        return 0.0
    differences = path_segment[1:] - path_segment[:-1]
    distances = np.linalg.norm(differences, axis=1)
    total_distance = np.sum(distances)
    return total_distance

def compute_travelled_distance(robot_positions, utility_values):
    cumulative_distance = 0.0
    distances = [0.0]  # Start with zero distance at the initial position
    in_negative_utility = False
    for i in range(1, len(utility_values)):
        if utility_values[i] < 0:
            if not in_negative_utility:
                in_negative_utility = True
            step_distance = np.linalg.norm(robot_positions[i] - robot_positions[i - 1])
            cumulative_distance += step_distance
        else:
            in_negative_utility = False
        distances.append(cumulative_distance)
    return distances

# Robot Control Functions and Classes
class State:

    def __init__(self, world_ref_traj, x_base=0, y_base=0.4):
        # DH parameters for the robot
        self.dh_params = np.array([
            [0.72, 0, 0, 0],
            [0.06, 0.117, -0.5 * math.pi, 0],
            [0, 0, 0.5 * math.pi, 1.57],
            [0.219 + 0.133, 0, 0.5 * math.pi, 0],
            [0, 0, -0.5 * math.pi, 0],
            [0.197 + 0.1245, 0, -0.5 * math.pi, 0],
            [0, 0, 0.5 * math.pi, 0],
            [0.1385 + 0.1665, 0, 0, 0]
        ])

        self.x_base = x_base
        self.y_base = y_base
        self.yaw_base = 0

        # print(y_base)

        # Ensure world_ref_traj has at least one column
        if world_ref_traj.shape[1] < 1:
            raise ValueError("world_ref_traj must have at least one column.")

        xyz = np.array([
            [world_ref_traj[0, 0] - self.x_base],
            [world_ref_traj[1, 0] - self.y_base],
            [world_ref_traj[2, 0]]
        ])
        abc = np.array([world_ref_traj[3, 0], world_ref_traj[4, 0], world_ref_traj[5, 0]])
        robot = RobotSerial(self.dh_params)
        end = Frame.from_euler_3(abc, xyz)
        print("CHECK")
        robot.inverse(end)
        self.theta = robot.axis_values.copy()
        self.yaw_base = self.theta[0]

        robot = RobotSerial(self.dh_params)
        f = robot.forward(self.theta)

        self.x_world = self.x_base + f.t_3_1.reshape([3, ])[0]
        self.y_world = self.y_base + f.t_3_1.reshape([3, ])[1]
        self.z_world = f.t_3_1.reshape([3, ])[2]
        self.yaw_world = f.euler_3[2]
        self.pitch_world = f.euler_3[1]
        self.roll_world = f.euler_3[0]

def update(state, ustar, f, dh_params, B, joint_angle_combination):
    state.theta = state.theta + (dt * ustar[1:].reshape(1, -1))
    state.theta = state.theta.astype(float)
    state.yaw_base += ustar[1] * dt
    state.yaw_base = float(state.yaw_base)

    state.x_base += (ustar[0] * dt * math.cos(state.yaw_base))
    state.x_base = float(state.x_base)
    state.y_base += (ustar[0] * dt * math.sin(state.yaw_base))
    state.y_base = float(state.y_base)

    robot = RobotSerial(dh_params)
    f = robot.forward(state.theta)

    state.x_body = f.t_3_1.reshape([3, ])[0]
    state.y_body = f.t_3_1.reshape([3, ])[1]
    state.z_body = f.t_3_1.reshape([3, ])[2]
    state.yaw_body = f.euler_3[2]
    state.pitch_body = f.euler_3[1]
    state.roll_body = f.euler_3[0]

    state.x_world = state.x_base + f.t_3_1.reshape([3, ])[0]
    state.y_world = state.y_base + f.t_3_1.reshape([3, ])[1]
    state.z_world = f.t_3_1.reshape([3, ])[2]
    state.yaw_world = f.euler_3[2]
    state.pitch_world = f.euler_3[1]
    state.roll_world = f.euler_3[0]

    ee_pose = np.array([
        [state.x_world], [state.y_world], [state.z_world],
        [state.yaw_world], [state.pitch_world], [state.roll_world]
    ])

    return state, ee_pose

def get_B(dh_params, state, joint_angle_combination):
    robot = RobotSerial(dh_params)
    theta = joint_angle_combination
    f = robot.forward(theta)

    # Placeholder for Jacobian calculation
    # Since the Jacobian is not provided, we'll use an identity matrix for demonstration
    # jacobian = np.eye(6, 8)

    jacobian = []
    with open('jacobian_matrix1.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            jacobian.append([cell for cell in row])
    jacobian = np.array(jacobian, dtype=float)

    B = np.zeros((6, 9))
    B[:, 1:9] = jacobian * dt
    B[0, 0] = dt * math.cos(state.yaw_base)
    B[1, 0] = dt * math.sin(state.yaw_base)

    return B, f

def solve_dare(A, B, Q, R, ref_traj, time_step):
    X = Q
    plannign_horizon = 8
    eps = 0.01

    small_p = np.array([0, 0, 0, 0, 0, 0])
    for i in range(plannign_horizon):
        Xn = A.T @ X @ A - A.T @ X @ B @ \
            np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        small_p_n = small_p + Xn @ (ref_traj[:, time_step] - ref_traj[:, time_step+1]) \
            - Xn @ B @ np.linalg.inv(R + B.T @ X @ B) @ B.T @ Xn @ ((ref_traj[:, time_step] - ref_traj[:, time_step+1])) \
                - Xn @ B @ np.linalg.inv(R + B.T @ X @ B) @ B.T @ small_p
        if (np.abs(Xn - X)).max() < eps:
            X = Xn
            break
        X = Xn
        small_p = small_p_n

    return X, small_p

def dlqr(A, B, Q, R, ref_traj, time_step):
    X, small_p = solve_dare(A, B, Q, R, ref_traj, time_step)
    K = np.linalg.inv(R + B.T @ X @ B) @ (B.T @ X @ A)
    return K, small_p, X

def lqr_control(state, lqr_Q, lqr_R, ref_traj, time_step, dh_params, joint_angle_combination):
    A = np.eye(6)
    B, f = get_B(dh_params, state, joint_angle_combination)
    K, small_p, X = dlqr(A, B, lqr_Q, lqr_R, ref_traj, time_step)

    state_vector = np.array([
        state.x_world, state.y_world, state.z_world,
        state.yaw_world, state.pitch_world, state.roll_world
    ])
    ref_vector = ref_traj[:, time_step]

    error = state_vector - ref_vector
    ref_traj_difference = ref_traj[:, time_step] - ref_traj[:, time_step+1]

    impact_of_small_p = np.linalg.inv(lqr_R + B.T @ X @ B) @ B.T @ small_p
    ustar = -K @ (error + ref_traj_difference) - impact_of_small_p

    return ustar, B, f

def do_simulation(cx, cy, cz, cyaw, cpitch, croll, world_ref_traj, step, state1, state2):
    # Perform simulation steps for both robots
    # This function no longer handles plotting
    # It returns the updated trajectory data

    # Trajectory data for both robots
    x1, y1, z1 = [state1.x_world], [state1.y_world], [state1.z_world]
    yaw1, pitch1, roll1 = [state1.yaw_world], [state1.pitch_world], [state1.roll_world]
    x2, y2, z2 = [state2.x_world], [state2.y_world], [state2.z_world]
    yaw2, pitch2, roll2 = [state2.yaw_world], [state2.pitch_world], [state2.roll_world]
    i = 0
    # For Robot 1
    theta1 = state1.theta
    joint_angle_combination1 = theta1
    ustar1, B1, f1 = lqr_control(state1, lqr_Q, lqr_R, world_ref_traj, i, state1.dh_params, joint_angle_combination1)
    state1, ee_pose1 = update(state1, ustar1, f1, state1.dh_params, B1, joint_angle_combination1)

    x1.append(state1.x_world)
    y1.append(state1.y_world)
    z1.append(state1.z_world)
    yaw1.append(state1.yaw_world)
    pitch1.append(state1.pitch_world)
    roll1.append(state1.roll_world)

    # For Robot 2
    theta2 = state2.theta
    joint_angle_combination2 = theta2
    ustar2, B2, f2 = lqr_control(state2, lqr_Q, lqr_R, world_ref_traj, i, state2.dh_params, joint_angle_combination2)
    state2, ee_pose2 = update(state2, ustar2, f2, state2.dh_params, B2, joint_angle_combination2)

    x2.append(state2.x_world)
    y2.append(state2.y_world)
    z2.append(state2.z_world)
    yaw2.append(state2.yaw_world)
    pitch2.append(state2.pitch_world)
    roll2.append(state2.roll_world)

    return x1, y1, z1, yaw1, pitch1, roll1, x2, y2, z2, yaw2, pitch2, roll2


def main():
    global phi_k
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.ion()  # Enable interactive mode

    # Precompute obstacle map and collision probability map
    obstacle_map = create_obstacle_map()
    collision_prob_map = compute_collision_probability_map(obstacle_map, sigma)

    # Precompute cost maps
    cost_map_robot = compute_cost_map(collision_prob_map, gamma_robot, obstacle_map)
    cost_map_human = compute_cost_map(collision_prob_map, gamma_human, obstacle_map)

    # Initialize robot's starting position
    robot_position = np.array(A_pos)
    robot_positions = [robot_position.copy()]
    step = 0  # Initialize step counter
    max_steps = 500  # Set a maximum number of steps to prevent infinite loops

    utility_values = []  # Collect utility values during simulation
    phi_k_values = []
    travelled_distances = [0.0]

    # Reference trajectory arrays (initialized empty)
    cx = []
    cy = []
    cz = []
    cyaw = []
    cpitch = []
    croll = []
    x1_values = []
    y1_values = []
    z1_values = []

    cx_values = []
    cy_values = []
    cz_values = []
    stuck_count = 0
    cnt = 0

    # Simulation loop
    while np.linalg.norm(robot_position - B_pos) > 0.1 and step < max_steps:
        cnt += 1

        # Update start and end indices based on current position
        start_idx = pos_to_idx(robot_position)
        end_idx = pos_to_idx(B_pos)

        # Compute paths for both robot and human
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

        # phi_k = phi_k_init * np.exp(- (travelled_distances[-1] / 4))
        phi_k = phi_k * np.exp(- (travelled_distances[-1] / 4))
        phi_k_values.append(phi_k)

        if cnt > 3:
            if decision_value > 0:
                phi_k = phi_k - 0.0
                # print("checked")
                run_once = False  # Ensures loop runs only once
        decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
        utility_values.append(decision_value)

        if cnt > 2:

            if np.allclose(robot_positions[-1], robot_positions[-2], atol=1e-6):
                stuck_count += 1
                print("stuck count:", stuck_count)  
                # Robot hasn't moved; compute heading angle based on previous movement
                movement_vector = robot_positions[-2] - robot_positions[-3] if step >= 2 else robot_positions[-1] - robot_positions[-2]
                if np.linalg.norm(movement_vector) != 0:
                    heading_angle = math.atan2(movement_vector[1], movement_vector[0])
                # else:
                #     # Default heading towards the goal if no movement detected
                #     heading_angle = math.atan2(B_pos[1] - robot_position[1], B_pos[0] - robot_position[0])

                # Take a small step in the current heading direction
                step_size = 0.05  # Adjust the step size as needed
                delta_x = step_size * math.cos(heading_angle)
                delta_y = step_size * math.sin(heading_angle)
                robot_position = robot_position + np.array([delta_x, delta_y])


            # if (robot_positions[-1] == robot_positions[-2]).all():
            #     # Avoid index error by checking path length
            #     if len(path_coords_robot) > 2 and len(path_coords_human) > 2:
            #         next_pos_robot = path_coords_robot[2]
            #         next_pos_human = path_coords_human[2]
            #         stuck_count += 1
            #         print("stuck count:", stuck_count)
            #         print(f"Robot was stuck. Extrapolated to new position: {robot_position}")
            #     else:
            #         print("Robot is stuck and cannot extrapolate further.")



        # Choose path based on utility
        if decision_value >= 0:
            chosen_path = path_coords_human
            robot_position = next_pos_human  # Adapt to human's path
        else:
            chosen_path = path_coords_robot
            robot_position = next_pos_robot  # Follow own path

        robot_positions.append(robot_position.copy())

        # Update reference trajectory
        cx = chosen_path[:, 0]
        cy = chosen_path[:, 1]
        cz = np.random.uniform(0.7, 0.8, len(cy))
        cyaw = np.zeros(len(cz))
        cpitch = np.zeros(len(cz))
        croll = np.zeros(len(cz))

        world_ref_traj = np.array([cx, cy, cz, cyaw, cpitch, croll])

        # Compute cumulative traveled distance
        travelled_distances = compute_travelled_distance(robot_positions, utility_values)

        # Initialize states for both robots on the first step
        if step == 0:
            try:
                state1 = State(world_ref_traj=world_ref_traj, x_base=cx[0]-0.6, y_base=cy[0]+0.0)  # Robot 1
                state2 = State(world_ref_traj=world_ref_traj, x_base=cx[0], y_base=cy[0]-0.6)   # Robot 2
            except ValueError as e:
                print(f"Error initializing State: {e}")
                break  # Exit simulation if State cannot be initialized

        # Perform simulation steps for both robots
        x1, y1, z1, yaw1, pitch1, roll1, x2, y2, z2, yaw2, pitch2, roll2 = do_simulation(
            cx, cy, cz, cyaw, cpitch, croll, world_ref_traj, step, state1, state2)
        
        x1_values.append(x1)
        y1_values.append(y1)
        z1_values.append(z1)
        # print(cx[0])

        cx_values.append(cx[0])
        cy_values.append(cy[0])
        cz_values.append(cz[0])

        robot_position = np.array([x1[-1], y1[-1]])

        # Visualization on two subplots
        if show_animation:
            # Clear both axes
            ax1.cla()
            ax2.cla()

            # Plot on the first subplot (Path Planning)
            ax1.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
            ax1.plot(A_pos[0], A_pos[1], 'go', markersize=8, label='Start Point A')
            ax1.plot(B_pos[0], B_pos[1], 'bo', markersize=8, label='Target Point B')
            ax1.plot(chosen_path[:, 0], chosen_path[:, 1], 'r-', label="Chosen Path")
            ax1.plot(path_coords_robot[:, 0], path_coords_robot[:, 1], 'b--', label="Robot's Planned Path")
            ax1.plot(path_coords_human[:, 0], path_coords_human[:, 1], 'g--', label="Human's Planned Path")
            ax1.plot(np.array(robot_positions)[:-1, 0], np.array(robot_positions)[:-1, 1], 'r-', linewidth=2, label='Robot Trajectory')
            ax1.set_title('Path Planning')
            ax1.set_xlim(0, 10)
            ax1.set_ylim(0, 10)
            ax1.grid(True)
            ax1.legend()

            # Plot on the second subplot (Robot States)
            ax2.plot(cx, cy, "or", label="Reference Trajectory")
            ax2.plot(x1, y1, "ob", label="Robot 1 Trajectory")
            ax2.plot(x2, y2, "og", label="Robot 2 Trajectory")
            ax2.plot(state1.x_base, state1.y_base, 'bs', label="Robot 1 Base")
            ax2.plot(state2.x_base, state2.y_base, 'gs', label="Robot 2 Base")
            ax2.set_title('Robot States Update')
            ax2.set_aspect('equal')
            ax2.set_xlim(0, 10)
            ax2.set_ylim(0, 10)
            ax2.grid(True)
            ax2.legend()

            # Adjust layout and pause for animation effect
            plt.tight_layout()
            plt.pause(0.1)

        # print("--- %s seconds ---" % (time.time() - start_time))

        step += 1

    # After the simulation loop ends
    plt.ioff()
    plt.show()

    # Plot utility values
    plt.figure(figsize=(10, 6))
    utility_values_array = np.array(utility_values)
    plt.plot(range(len(utility_values_array)), utility_values_array, 'm-', linewidth=2, label='Utility Value')

    # Fill areas based on utility value conditions
    plt.fill_between(
        range(len(utility_values_array)),
        utility_values_array,
        0,
        where=(utility_values_array >= 0),
        color='green',
        alpha=0.3,
        label='Positive Utility'
    )
    plt.fill_between(
        range(len(utility_values_array)),
        utility_values_array,
        0,
        where=(utility_values_array < 0),
        color='blue',
        alpha=0.3,
        label='Negative Utility'
    )

    plt.xlabel('Decision Step')
    plt.ylabel('Utility Value')
    plt.title('Utility Value Over Decision Steps')
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot phi_k_values
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(phi_k_values)), phi_k_values, 'c-', linewidth=2, label='Phi values')

    # Add labels and title
    plt.xlabel('Decision Step', fontsize=14)
    plt.ylabel('Phi', fontsize=14)
    # plt.title('Cumulative Traveled Distance Along Human\'s Path During Negative Utility', fontsize=title_fontsize)

    # Customize grid, legend, and ticks
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save and display
    # plt.savefig("Human_tendency.png", dpi=300, bbox_inches='tight')
    plt.show()

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot cy vs cx on the first subplot
    ax3.scatter(x1_values, y1_values, color='blue', label='x1 vs y1')
    ax3.scatter(cx_values, cy_values, color='red', label='cx vs cy')
    ax3.set_xlabel('x1', fontsize=14)
    ax3.set_ylabel('y1', fontsize=14)
    ax3.set_title('Robot Trajectory: cy vs cx', fontsize=16)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=12)

    # Plot cz vs cx on the second subplot
    ax4.scatter(x1_values, z1_values, color='blue', label='z1 vs x1')
    ax4.scatter(cx_values, cz_values, color='red', label='cz vs cy')
    ax4.set_xlabel('x1', fontsize=14)
    ax4.set_ylabel('z1', fontsize=14)
    ax4.set_title('Robot Elevation: cz vs cx', fontsize=16)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(fontsize=12)
    ax4.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()


    # print("--- %s seconds ---" % (time.time() - start_time))



# def main():
#     # Create a figure with two subplots side by side
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#     plt.ion()  # Enable interactive mode

#     # Precompute obstacle map and collision probability map
#     obstacle_map = create_obstacle_map()
#     collision_prob_map = compute_collision_probability_map(obstacle_map, sigma)

#     # Precompute cost maps
#     cost_map_robot = compute_cost_map(collision_prob_map, gamma_robot, obstacle_map)
#     cost_map_human = compute_cost_map(collision_prob_map, gamma_human, obstacle_map)

#     # Initialize robot's starting position
#     robot_position = np.array(A_pos)
#     robot_positions = [robot_position.copy()]
#     step = 0  # Initialize step counter
#     max_steps = 500  # Set a maximum number of steps to prevent infinite loops

#     utility_values = []  # Collect utility values during simulation
#     phi_k_values = []
#     travelled_distances = [0.0]

#     # Reference trajectory arrays (initialized empty)
#     cx = []
#     cy = []
#     cz = []
#     cyaw = []
#     cpitch = []
#     croll = []

#     cnt = 0
#     phi_k = 1.0


#     # Simulation loop
#     while np.linalg.norm(robot_position - B_pos) > 0.1 and step < max_steps:
#         cnt += 1

#         # Update start and end indices based on current position
#         start_idx = pos_to_idx(robot_position)
#         end_idx = pos_to_idx(B_pos)

#         # Compute paths for both robot and human
#         path_robot = dijkstra(cost_map_robot, start_idx, end_idx)
#         path_coords_robot = np.array([(X[i, j], Y[i, j]) for i, j in path_robot])

#         path_human = dijkstra(cost_map_human, start_idx, end_idx)
#         path_coords_human = np.array([(X[i, j], Y[i, j]) for i, j in path_human])

#         # Determine next positions
#         next_pos_robot = path_coords_robot[1] if len(path_coords_robot) > 1 else path_coords_robot[0]
#         next_pos_human = path_coords_human[1] if len(path_coords_human) > 1 else path_coords_human[0]

#         # Decision-making logic
#         D_own = compute_remaining_distance(path_coords_robot, 0)
#         D_adapt = compute_remaining_distance(path_coords_human, 0)
#         idx_next_robot = pos_to_idx(next_pos_robot)
#         idx_next_human = pos_to_idx(next_pos_human)
#         p_risk_own = collision_prob_map[idx_next_robot]
#         p_risk_adapt = collision_prob_map[idx_next_human]

#         phi_k = phi_k * np.exp(- (travelled_distances[-1] / 4))
#         # phi_k = phi_k_init * np.exp(- (travelled_distances[-1] / 4))
#         phi_k_values.append(phi_k)
#         decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
#         utility_values.append(decision_value)

#         if cnt > 2:
#             if (robot_positions[-1] == robot_positions[-2]).all():
#                 # Avoid index error by checking path length
#                 if len(path_coords_robot) > 2 and len(path_coords_human) > 2:
#                     next_pos_robot = path_coords_robot[2]
#                     next_pos_human = path_coords_human[2]
#                     print(f"Robot was stuck. Extrapolated to new position: {robot_position}")
#                 else:
#                     print("Robot is stuck and cannot extrapolate further.")

#         # Choose path based on utility
#         if decision_value >= 0:
#             chosen_path = path_coords_human
#             robot_position = next_pos_human  # Adapt to human's path
#         else:
#             chosen_path = path_coords_robot
#             robot_position = next_pos_robot  # Follow own path

#         robot_positions.append(robot_position.copy())

#         # Update reference trajectory
#         cx = chosen_path[:, 0]
#         cy = chosen_path[:, 1]
#         cz = np.random.uniform(0.7, 0.8, len(cy))
#         cyaw = np.zeros(len(cz))
#         cpitch = np.zeros(len(cz))
#         croll = np.zeros(len(cz))

#         world_ref_traj = np.array([cx, cy, cz, cyaw, cpitch, croll])

#         # Compute cumulative traveled distance
#         travelled_distances = compute_travelled_distance(robot_positions, utility_values)

#         # Initialize states for both robots on the first step
#         if step == 0:
#             try:
#                 state1 = State(world_ref_traj=world_ref_traj, x_base=cx[0]-0.6, y_base=cy[0]+0.0)  # Robot 1
#                 state2 = State(world_ref_traj=world_ref_traj, x_base=cx[0], y_base=cy[0]-0.6)   # Robot 2
#             except ValueError as e:
#                 print(f"Error initializing State: {e}")
#                 break  # Exit simulation if State cannot be initialized

#         # Perform simulation steps for both robots
#         x1, y1, z1, yaw1, pitch1, roll1, x2, y2, z2, yaw2, pitch2, roll2 = do_simulation(
#             cx, cy, cz, cyaw, cpitch, croll, world_ref_traj, step, state1, state2)

#         robot_position = np.array([x1[-1], y1[-1]])

#         # Visualization on two subplots
#         if show_animation:
#             # Clear both axes
#             ax1.cla()
#             ax2.cla()

#             # Plot on the first subplot (Path Planning)
#             ax1.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
#             ax1.plot(A_pos[0], A_pos[1], 'go', markersize=8, label='Start Point A')
#             ax1.plot(B_pos[0], B_pos[1], 'bo', markersize=8, label='Target Point B')
#             ax1.plot(chosen_path[:, 0], chosen_path[:, 1], 'b--', label="Chosen Path")
#             ax1.plot(np.array(robot_positions)[:, 0], np.array(robot_positions)[:, 1], 'r-', linewidth=2, label='Robot Trajectory')
#             ax1.set_title('Path Planning')
#             ax1.set_xlim(0, 10)
#             ax1.set_ylim(0, 10)
#             ax1.grid(True)
#             ax1.legend()

#             # Plot on the second subplot (Robot States)
#             ax2.plot(cx, cy, "or", label="Reference Trajectory")
#             ax2.plot(x1, y1, "ob", label="Robot 1 Trajectory")
#             ax2.plot(x2, y2, "og", label="Robot 2 Trajectory")
#             ax2.plot(state1.x_base, state1.y_base, 'bs', label="Robot 1 Base")
#             ax2.plot(state2.x_base, state2.y_base, 'gs', label="Robot 2 Base")
#             ax2.set_title('Robot States Update')
#             ax2.set_aspect('equal')
#             ax2.set_xlim(0, 10)
#             ax2.set_ylim(0, 10)
#             ax2.grid(True)
#             ax2.legend()

#             # Adjust layout and pause for animation effect
#             plt.tight_layout()
#             plt.pause(0.1)

#         print("--- %s seconds ---" % (time.time() - start_time))

#         step += 1

#     # Finalize the animation
#     plt.ioff()
#     plt.show()

#     # Plot utility values
#     plt.figure(figsize=(10, 6))
#     utility_values_array = np.array(utility_values)
#     plt.plot(range(len(utility_values_array)), utility_values_array, 'm-', linewidth=2, label='Utility Value')

#     # Fill areas based on utility value conditions
#     plt.fill_between(
#         range(len(utility_values_array)),
#         utility_values_array,
#         0,
#         where=(utility_values_array >= 0),
#         color='green',
#         alpha=0.3,
#         label='Positive Utility'
#     )
#     plt.fill_between(
#         range(len(utility_values_array)),
#         utility_values_array,
#         0,
#         where=(utility_values_array < 0),
#         color='blue',
#         alpha=0.3,
#         label='Negative Utility'
#     )

#     plt.xlabel('Decision Step')
#     plt.ylabel('Utility Value')
#     plt.title('Utility Value Over Decision Steps')
#     plt.axhline(0, color='black', linestyle='--')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     print(cz)
#     print(z1)
#     print(z2)

#     # Set font sizes for each component
#     title_fontsize = 18
#     label_fontsize = 14
#     tick_fontsize = 12
#     legend_fontsize = 12

#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(phi_k_values)), phi_k_values, 'c-', linewidth=2, label='Phi values')

#     # Add labels and title
#     plt.xlabel('Decision Step', fontsize=label_fontsize)
#     plt.ylabel('Phi', fontsize=label_fontsize)
#     # plt.title('Cumulative Traveled Distance Along Human\'s Path During Negative Utility', fontsize=title_fontsize)

#     # Customize grid, legend, and ticks
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(fontsize=legend_fontsize)
#     plt.xticks(fontsize=tick_fontsize)
#     plt.yticks(fontsize=tick_fontsize)

#     # Save and display
#     # plt.savefig("Human_tendency.png", dpi=300, bbox_inches='tight')
#     plt.show()

    print(cnt)
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
    main()
