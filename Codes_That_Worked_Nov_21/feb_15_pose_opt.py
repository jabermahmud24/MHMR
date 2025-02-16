import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import heapq
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import math
import time
import csv
from scipy.interpolate import interp1d
import copy

# *** IMPORT THE ROBOT SERIAL MODULE (make sure it is in your PYTHONPATH) ***
from visual_kinematics.RobotSerial import *

# =============================================================================
# 1. ENVIRONMENT SETUP
# =============================================================================

grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
X, Y = np.meshgrid(x_range, y_range, indexing='ij')

# Define start and target positions in world coordinates
A_pos = np.array([1, 1])  # Start (A)
B_pos = np.array([7, 4])  # Target (B)

# Define obstacles: centers and radii
obstacles = [
    {'center': np.array([3, 0.5]), 'radius': 0.99},
    {'center': np.array([5, 3.0]), 'radius': 0.85},
    {'center': np.array([5, 5.5]), 'radius': 0.75},
    {'center': np.array([7, 7]), 'radius': 0.5},
    {'center': np.array([8, 8.5]), 'radius': 0.55},
]

# Define robot size
robot_diameter = 0.5
robot_radius = robot_diameter / 2

# Create grid and mark obstacles
grid = np.zeros(grid_size, dtype=int)
for obs in obstacles:
    distance = np.sqrt((X - obs['center'][0])**2 + (Y - obs['center'][1])**2)
    # Mark grid cells within obstacle radius plus robot radius as occupied
    grid[distance <= (obs['radius'] + robot_radius)] = 1

# -------------------------
#   COORDINATE TRANSFORM
# -------------------------
def world_to_grid(pos):
    """
    Convert world coordinates (x, y) to grid indices (i, j).
    """
    x, y = pos
    i = np.argmin(np.abs(x_range - x))
    j = np.argmin(np.abs(y_range - y))
    return (i, j)

def grid_to_world(idx_pos):
    """
    Convert grid indices (i, j) to world coordinates (x, y).
    """
    i, j = idx_pos
    x = x_range[i]
    y = y_range[j]
    return np.array([x, y])

# --- A* Algorithm (8-connected) ---
class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position  # grid cell (i,j)
        self.parent = parent
        self.g = g  # cost from start
        self.h = h  # heuristic cost to goal
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def astar(grid, start, end):
    open_set = []
    start_h = np.linalg.norm(np.array(start) - np.array(end))
    heapq.heappush(open_set, Node(start, None, 0, start_h))
    closed_set = set()
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,-1), (-1, 1), (1, 1), (1, -1)]
    
    while open_set:
        current_node = heapq.heappop(open_set)
        current_pos = current_node.position
        
        if current_pos == end:
            # Reconstruct path
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        
        closed_set.add(current_pos)
        
        for move in neighbors:
            neighbor = (current_pos[0] + move[0], current_pos[1] + move[1])
            if not (0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]):
                continue
            if grid[neighbor[0], neighbor[1]] == 1:
                continue
            if neighbor in closed_set:
                continue
            
            move_cost = 1
            tentative_g = current_node.g + move_cost
            heuristic = np.linalg.norm(np.array(neighbor) - np.array(end))
            
            in_open_set = False
            for node in open_set:
                if neighbor == node.position and tentative_g >= node.g:
                    in_open_set = True
                    break
            if not in_open_set:
                new_node = Node(neighbor, current_node, tentative_g, heuristic)
                heapq.heappush(open_set, new_node)
    
    return None

def path_length_in_world(path):
    """Compute Euclidean path length given grid path (converted to world coordinates)."""
    if path is None or len(path) < 2:
        return None
    path_world = [grid_to_world(p) for p in path]
    total = 0.0
    for i in range(len(path_world) - 1):
        total += np.linalg.norm(path_world[i + 1] - path_world[i])
    return total

# =============================================================================
# 3. OPENING CALCULATION AND UTILITY FUNCTIONS
# =============================================================================
# Order obstacles (e.g., by y-coordinate)
obstacles_sorted = sorted(obstacles, key=lambda obs: obs['center'][1])

def calculate_openings(obstacles_sorted, robot_diameter, start_pos, target_pos):
    """
    For each pair of consecutive obstacles (sorted by y), compute the gap (opening)
    and associated risk (robot_diameter / gap). Also compute a naive “through‐opening” distance.
    """
    openings = []
    for i in range(len(obstacles_sorted) - 1):
        lower = obstacles_sorted[i]
        upper = obstacles_sorted[i + 1]
        center_distance = np.linalg.norm(lower['center'] - upper['center'])
        opening_width = center_distance - (lower['radius'] + upper['radius'])
        if opening_width <= 0:
            continue
        direction = upper['center'] - lower['center']
        unit_vector = direction / center_distance
        lower_circ = lower['center'] + unit_vector * lower['radius']
        upper_circ = upper['center'] - unit_vector * upper['radius']
        opening_pos = (lower_circ + upper_circ) / 2
        risk = robot_diameter / opening_width
        dist_start = np.linalg.norm(start_pos - opening_pos)
        dist_target = np.linalg.norm(target_pos - opening_pos)
        naive_total_dist = dist_start + dist_target
        openings.append({
            'between': (i + 1, i + 2),
            'width': opening_width,
            'risk': risk,
            'position': opening_pos,
            'distance_to_target': naive_total_dist
        })
    return openings

# --- Utility Functions ---
def utility1(p, d, alpha, gamma):
    # p: risk, d: distance, alpha and gamma can be scalars or arrays.
    c_risk = 3
    p_safe = p
    w_p = np.exp(-(-np.log(p_safe + 1e-10)) ** gamma)
    w_d = d ** alpha
    return w_d + (w_p * c_risk)

def utility2(p, d):
    c_risk = 3
    alpha_true = 1
    gamma_true = 1
    w_p = np.exp(-(-np.log(p + 1e-10)) ** gamma_true)
    w_d = d ** alpha_true
    return w_d + (w_p * c_risk)

def linearized_utility(original_func, p, d, alpha0, gamma0, alpha, gamma):
    epsilon = 1e-5
    f0 = original_func(p, d, alpha0, gamma0)
    f_alpha_plus = original_func(p, d, alpha0 + epsilon, gamma0)
    f_alpha_minus = original_func(p, d, alpha0 - epsilon, gamma0)
    df_dalpha = (f_alpha_plus - f_alpha_minus) / (2 * epsilon)
    f_gamma_plus = original_func(p, d, alpha0, gamma0 + epsilon)
    f_gamma_minus = original_func(p, d, alpha0, gamma0 - epsilon)
    df_dgamma = (f_gamma_plus - f_gamma_minus) / (2 * epsilon)
    return f0 + df_dalpha * (alpha - alpha0) + df_dgamma * (gamma - gamma0)

# =============================================================================
# 4. ROBOT CONTROL FUNCTIONS (LQR-BASED) AND STATE UPDATE
# =============================================================================
def upsample_trajectory(world_ref_traj, target_num_points, kind='cubic'):
    """
    Upsample a reference trajectory (of shape (6, N)) to have target_num_points.
    """
    original_num_points = world_ref_traj.shape[1]
    if target_num_points < original_num_points:
        raise ValueError("Target number of points must exceed the original number.")
    t_original = np.linspace(0, 1, original_num_points)
    t_target = np.linspace(0, 1, target_num_points)
    upsampled_traj = np.zeros((world_ref_traj.shape[0], target_num_points))
    for i in range(world_ref_traj.shape[0]):
        interpolator = interp1d(t_original, world_ref_traj[i, :], kind=kind)
        upsampled_traj[i, :] = interpolator(t_target)
    return upsampled_traj

# Control parameters
dt = 0.1
dummy_Q = 1000 * np.eye(6)
r_matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]
])
scalar = 1
lqr_R = scalar * r_matrix

# --- Define the robot state class ---
class State:
    def __init__(self, world_ref_traj, x_base=0, y_base=0.4):
        # DH parameters for the robot (adjust as needed)
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
        robot.inverse(end)
        self.theta = robot.axis_values.copy()
        self.yaw_base = self.theta[0]
        robot = RobotSerial(self.dh_params)
        f = robot.forward(self.theta)
        self.x_world = self.x_base + f.t_3_1.reshape(3,)[0]
        self.y_world = self.y_base + f.t_3_1.reshape(3,)[1]
        self.z_world = f.t_3_1.reshape(3,)[2]
        self.yaw_world = f.euler_3[2]
        self.pitch_world = f.euler_3[1]
        self.roll_world = f.euler_3[0]

def update(state, ustar, f, dh_params, B):
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
    state.x_world = state.x_base + f.t_3_1.reshape(3,)[0]
    state.y_world = state.y_base + f.t_3_1.reshape(3,)[1]
    state.z_world = f.t_3_1.reshape(3,)[2]
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
    planning_horizon = 8
    eps = 0.01
    small_p = np.zeros(6)
    for i in range(planning_horizon):
        Xn = A.T @ X @ A - A.T @ X @ B @ np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        small_p_n = small_p + Xn @ (ref_traj[:, time_step] - ref_traj[:, time_step + 1]) \
                    - Xn @ B @ np.linalg.inv(R + B.T @ X @ B) @ B.T @ Xn @ (ref_traj[:, time_step] - ref_traj[:, time_step + 1]) \
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
    A_mat = np.eye(6)
    B_mat, f = get_B(dh_params, state, joint_angle_combination)
    K, small_p, X = dlqr(A_mat, B_mat, lqr_Q, lqr_R, ref_traj, time_step)
    state_vector = np.array([
        state.x_world, state.y_world, state.z_world,
        state.yaw_world, state.pitch_world, state.roll_world
    ])
    ref_vector = ref_traj[:, time_step]
    error = state_vector - ref_vector
    ref_diff = ref_traj[:, time_step] - ref_traj[:, time_step + 1]
    impact = np.linalg.inv(lqr_R + B_mat.T @ X @ B_mat) @ B_mat.T @ small_p
    ustar = -K @ (error + ref_diff) - impact
    return ustar, B_mat, f

def calculate_true_cost(ustar, state, world_ref_traj, lqr_Q, lqr_R, angle_change_cost):
    error = np.array([[world_ref_traj[0, 0] - state.x_world],
                      [world_ref_traj[1, 0] - state.y_world],
                      [world_ref_traj[2, 0] - state.z_world],
                      [world_ref_traj[3, 0] - state.yaw_world],
                      [world_ref_traj[4, 0] - state.pitch_world],
                      [world_ref_traj[5, 0] - state.roll_world]])
    true_cost = (error.T @ lqr_Q @ error) + (ustar.T @ lqr_R @ ustar) + angle_change_cost
    # print("true cost: ", true_cost)
    return true_cost


def pose_optimization(candidate_state, lqr_Q, world_ref_traj, time_step_control, step, candidate_index, percentages, offset_count, offset):
    # Create a deep copy so that the original candidate_state is preserved.
    candidate_state_copy = copy.deepcopy(candidate_state)

    # Make an explicit copy of theta so that we have the pre-modification values.
    prev_combinations = candidate_state_copy.theta.copy()
    # print("Previous combinations:", prev_combinations)
    
    # Apply the offset to the copy only.
    if step == 0:
        candidate_state_copy.theta[6] = candidate_state_copy.theta[6] + offset
        candidate_state_copy.theta[7] = candidate_state_copy.theta[7] + offset
    else: 
        candidate_state_copy.theta[0, 6] = candidate_state_copy.theta[0, 6] + offset
        candidate_state_copy.theta[0, 7] = candidate_state_copy.theta[0, 7] + offset

    new_combinations = candidate_state_copy.theta
    # print("New combinations:", new_combinations)

    # Use the copy for control and update.
    ustar, B_mat, fwd = lqr_control(candidate_state_copy,
                                    lqr_Q,
                                    lqr_R,
                                    world_ref_traj,
                                    time_step_control,
                                    candidate_state_copy.dh_params,
                                    candidate_state_copy.theta)

    final_candidate_state, ee_pose = update(candidate_state,
                                              ustar,
                                              fwd,
                                              candidate_state.dh_params,
                                              B_mat)
    kappa = 1

    # Compute the difference (rounded) between new and previous combinations.
    diff = np.round(np.abs(new_combinations - prev_combinations), decimals=6)
    if step == 0:
        angle_change_cost = kappa * (diff.T @ diff)
        # Alternatively, if you want to scale by kappa:
        # angle_change_cost = kappa * (np.abs(new_combinations - prev_combinations).T @ np.abs(new_combinations - prev_combinations))
    else:
        angle_change_cost = kappa * (diff @ diff.T)
        # Alternatively, if you want to scale by kappa:
        # angle_change_cost = kappa * (np.abs(new_combinations - prev_combinations) @ np.abs(new_combinations - prev_combinations).T)
    
    np.set_printoptions(precision=12)
    # print("Absolute differences:", np.abs(new_combinations - prev_combinations))
    # print("Differences:", new_combinations - prev_combinations)
    # print("Angle change cost:", angle_change_cost)
    
    pose_optimized_true_cost = calculate_true_cost(ustar,
                                                   final_candidate_state,
                                                   world_ref_traj,
                                                   lqr_Q,
                                                   lqr_R,
                                                   angle_change_cost)
    
    # weighted_pose_optimized_cost = pose_optimized_true_cost
    weighted_pose_optimized_cost = (1 - (percentages[candidate_index] / 100.0)) * pose_optimized_true_cost

    return weighted_pose_optimized_cost, pose_optimized_true_cost, candidate_state_copy, final_candidate_state


# def pose_optimization(candidate_state, lqr_Q, world_ref_traj, time_step_control, step, candidate_index, percentages, offset_count, offset):
#     # Create a deep copy so that the original candidate_state is preserved.
#     candidate_state_copy = copy.deepcopy(candidate_state)


#     prev_combinations = candidate_state_copy.theta
#     print(prev_combinations)
    
#     # Apply the offset to the copy only.
#     if step == 0:
#         candidate_state_copy.theta[6] = candidate_state_copy.theta[6] + offset
#         candidate_state_copy.theta[7] = candidate_state_copy.theta[7] + offset
#     else: 
#         candidate_state_copy.theta[0, 6] = candidate_state_copy.theta[0, 6] + offset
#         candidate_state_copy.theta[0, 7] = candidate_state_copy.theta[0, 7] + offset

#     new_combinations = candidate_state_copy.theta
#     print(new_combinations)

#     # print(new_combinations.shape)

#     # Use the copy for control and update.
#     ustar, B_mat, fwd = lqr_control(candidate_state_copy,
#                                     lqr_Q,
#                                     lqr_R,
#                                     world_ref_traj,
#                                     time_step_control,
#                                     candidate_state_copy.dh_params,
#                                     candidate_state_copy.theta)

#     final_candidate_state, ee_pose = update(candidate_state,
#                                               ustar,
#                                               fwd,
#                                               candidate_state.dh_params,
#                                               B_mat)
#     kappa = 100
#     if step == 0:
#         angle_change_cost = np.round(np.abs(new_combinations - prev_combinations), decimals=6).T @ np.round(np.abs(new_combinations - prev_combinations), decimals=6)

#         # angle_change_cost = kappa * (np.abs(new_combinations - prev_combinations).T @ np.abs(new_combinations - prev_combinations))

#     else:
#         angle_change_cost = np.round(np.abs(new_combinations - prev_combinations), decimals=6) @ np.round(np.abs(new_combinations - prev_combinations), decimals=6).T

#         # angle_change_cost = kappa * ( np.abs(new_combinations - prev_combinations) @ np.abs(new_combinations - prev_combinations).T)
#     np.set_printoptions(precision=12)
#     print(np.abs(new_combinations - prev_combinations))

#     print(new_combinations - prev_combinations)
#     print(angle_change_cost)
    
#     pose_optimized_true_cost = calculate_true_cost(ustar,
#                                                    final_candidate_state,
#                                                    world_ref_traj,
#                                                    lqr_Q,
#                                                    lqr_R, angle_change_cost)
    
#     weighted_pose_optimized_cost = (1 - (percentages[candidate_index] / 100.0)) * pose_optimized_true_cost
#     # print(weighted_pose_optimized_cost)

#     return weighted_pose_optimized_cost, pose_optimized_true_cost, candidate_state_copy, final_candidate_state




# =============================================================================
# 5. MAIN SIMULATION LOOP
# =============================================================================
def main():
    start_time = time.time()
    
    # Initialize current position and target.
    current_position = A_pos.copy()  # used only for step = 0
    target_position = B_pos.copy()
    
    # For storing trajectory (actual robot position in x-y)
    trajectory = [current_position.copy()]
    
    # For tracking performance in 3D (actual and reference positions)
    tracking_actual = []  # will store [x, y, z] of the robot (end-effector)
    tracking_ref = []     # will store [x, y, z] of the reference point used in control
    
    # For storing cost information from planning iterations.
    cost_history = []
    planning_steps = []
    optimal_costs = []
    second_best_costs = []
    
    # Simulation loop parameters
    max_steps = 100
    step = 0
    
    # Flag and storage to handle "committing" to a chosen opening once the robot
    # is near the midpoint of that opening.
    use_planning = True        # We start with planning turned ON
    chosen_opening_midpoint = None
    chosen_path_world = None
    midpoint_threshold = 0.1   # Distance threshold to stop re-planning
    
    state = None  # will be initialized on the first control step
    current_team_index = None  
    phi_init = 0.4
    phi_dec = phi_init  # initial value
    last_team_change_position = current_position.copy()
    traveled_distance = 0
    cnt = 0


    phi_dec_values = []
    time_steps = []
    
    robot_preferred_indices = []
    human_preferred_indices = []
    chosen_indices = []
    
    delta_robot_list = []
    delta_human_list = []
    
    all_true_utils_history = []     # list of arrays [path_index -> true cost]
    all_nominal_utils_history = []  # list of arrays [path_index -> nominal cost]
    
    without_po_times = 0
    with_po_times = 0
    # offsets= [0.0, -0.03, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    # offsets= [0.0, -0.02, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    offsets= [0.0, -0.02, -0.015, -0.01, -0.005, -0.025, -0.03, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]


    cnt_offsets = [0] * len(offsets)

    

    while np.linalg.norm(current_position - target_position) > 0.3 and step < max_steps:
    
        # 1) Ensure we have a valid real state at the start of the loop.
        if state is None and step == 0:
            dummy_ref_traj = np.zeros((6, 1))
            dummy_ref_traj[0, 0] = current_position[0]
            dummy_ref_traj[1, 0] = current_position[1]
            dummy_ref_traj[2, 0] = 0.7
            dummy_ref_traj[3, 0] = 0
            dummy_ref_traj[4, 0] = 0
            dummy_ref_traj[5, 0] = 0

            state = State(dummy_ref_traj,
                          x_base=current_position[0] - 0.6,
                          y_base=current_position[1] + 0.3)
        
        # 2) Branch based on whether we are still planning or simply following the chosen path.
        if (not use_planning) and (chosen_path_world is not None):
            # FOLLOWING BRANCH: Use the stored chosen_path_world to generate a reference trajectory
            best_path_world = chosen_path_world
            chosen_path_array = np.array(best_path_world)
            # Find the index along the chosen path closest to current position
            distances = np.linalg.norm(chosen_path_array - current_position, axis=1)
            closest_idx = np.argmin(distances)
            trimmed_path = chosen_path_array[closest_idx:]
            if trimmed_path.shape[0] < 2:
                trimmed_path = chosen_path_array
            num_points_traj = trimmed_path.shape[0]
            world_ref_traj = np.zeros((6, num_points_traj))
            world_ref_traj[0, :] = trimmed_path[:, 0]
            world_ref_traj[1, :] = trimmed_path[:, 1]
            world_ref_traj[2, :] = 0.7  # constant z
            world_ref_traj[3, :] = 0
            world_ref_traj[4, :] = 0
            world_ref_traj[5, :] = 0
            ref_point = np.array([world_ref_traj[0, 0],
                                  world_ref_traj[1, 0],
                                  world_ref_traj[2, 0]])
            tracking_ref.append(ref_point)
            
            time_step_control = 0
            lqr_Q = dummy_Q
            ustar, B_mat, fwd = lqr_control(state,
                                            lqr_Q,
                                            lqr_R,
                                            world_ref_traj,
                                            time_step_control,
                                            state.dh_params,
                                            state.theta)
            state, ee_pose = update(state,
                                    ustar,
                                    fwd,
                                    state.dh_params,
                                    B_mat)
        else:
            # --------------------------------------------------------------
            # -- PLANNING + POSE-OPTIMIZATION SECTION
            # --------------------------------------------------------------
            # Compute candidate paths via openings
            start_grid = world_to_grid(current_position)
            end_grid = world_to_grid(target_position)
            openings = calculate_openings(obstacles_sorted,
                                          robot_diameter,
                                          current_position,
                                          target_position)
            
            paths_via_openings = []
            for idx, opening in enumerate(openings, start=1):
                opening_grid = world_to_grid(opening['position'])
                path_A_opening = astar(grid, start_grid, opening_grid)
                path_opening_B = astar(grid, opening_grid, end_grid)
                if path_A_opening is None or path_opening_B is None:
                    continue
                dist_A_open = path_length_in_world(path_A_opening) or 0.0
                dist_open_B = path_length_in_world(path_opening_B) or 0.0
                total_dist = dist_A_open + dist_open_B
                # Combine the two parts (avoid duplicate point)
                full_path = path_A_opening + path_opening_B[1:]
                full_path_world = [grid_to_world(p) for p in full_path]
                paths_via_openings.append({
                    'path_world': full_path_world,
                    'total_distance': total_dist,
                    'risk': opening['risk'],
                    'opening_point': opening['position']
                })
            
            if not paths_via_openings:
                print("No viable candidate paths found. Stopping simulation.")
                break
            
            # Compute utilities for each candidate (nominal and true)
            alpha_min, alpha_max = 0.5, 0.99
            gamma_min, gamma_max = 0.2, 0.99
            num_points = 250
            alpha_vals = np.linspace(alpha_min, alpha_max, num_points)
            gamma_vals = np.linspace(gamma_min, gamma_max, num_points)
            Alpha, Gamma = np.meshgrid(alpha_vals, gamma_vals)
            
            final_true_utils = []
            final_nominal_utils = []
            utilities_linearized = []

            alpha0 = (alpha_min + alpha_max) / 2
            gamma0 = (gamma_min + gamma_max) / 2

            for candidate in paths_via_openings:
                p = candidate['risk']
                d = candidate['total_distance']
                utility_nominal = utility1(p, d, alpha0, gamma0)
                true_utility = utility2(p, d)
                final_true_utils.append(true_utility)
                final_nominal_utils.append(utility_nominal)
                
                Utility_lin = linearized_utility(
                    utility1, p, d, alpha0, gamma0, Alpha, Gamma
                )
                utilities_linearized.append(Utility_lin)
            
            # "Percentages" logic: decide how many points in (alpha, gamma) favor each candidate.
            utilities_stack = np.array(utilities_linearized)
            min_indices = np.argmin(utilities_stack, axis=0)
            total_points = min_indices.size
            counts = np.bincount(min_indices.flatten(), minlength=len(paths_via_openings))
            percentages = counts / total_points * 100
            
            if current_team_index is None:
                current_team_index = np.argmin(final_true_utils)
            
            base_true = final_true_utils[current_team_index]
            base_nominal = final_nominal_utils[current_team_index]
            N = len(paths_via_openings)
            costs = np.zeros(N)
            for k in range(N):
                costs[k] = (percentages[k] * (
                    + (1 - phi_dec) * (base_true - final_true_utils[k])
                    - phi_dec * (base_nominal - final_nominal_utils[k])
                ))
            
            if np.min(costs) < 0:
                optimal_index = np.argmin(costs)
            else:
                optimal_index = current_team_index
            
            print("based on O*, the optimal index: ", {optimal_index+1})
            print("Percentages of choosing an opening: ", percentages)

            
            robot_preferred = np.argmin(final_true_utils)
            human_preferred = np.argmin(final_nominal_utils)
            
            robot_preferred_indices.append(robot_preferred)
            human_preferred_indices.append(human_preferred)
            chosen_indices.append(optimal_index)
            
            ut_chosen = final_true_utils[optimal_index]
            ut_robot_pref = final_true_utils[robot_preferred]
            delta_robot = ut_chosen - ut_robot_pref
            
            un_chosen = final_nominal_utils[optimal_index]
            un_human_pref = final_nominal_utils[human_preferred]
            delta_human = un_chosen - un_human_pref
            
            delta_robot_list.append(delta_robot)
            delta_human_list.append(delta_human)
            
            all_true_utils_history.append(final_true_utils)
            all_nominal_utils_history.append(final_nominal_utils)
            
            # Update counters and team index
            if optimal_index == 0: pass  # increment counters if desired
            current_team_index = optimal_index
            if human_preferred != optimal_index:
                cnt += 1
                traveled_distance += np.linalg.norm(current_position - last_team_change_position)
                phi_dec = phi_init * np.exp(traveled_distance / 0.6)
                last_team_change_position = current_position.copy()
            phi_dec_values.append(phi_dec)
            time_steps.append(step)
            
            # Save cost info for plotting/logging
            if use_planning:
                cost_dict = {
                    'step': step,
                    'costs': costs,
                    'optimal_index': optimal_index,
                    'final_true_utils': final_true_utils,
                    'final_nominal_utils': final_nominal_utils,
                    'percentages': percentages
                }
                cost_history.append(cost_dict)
                sorted_costs = np.sort(costs)
                optimal_cost = sorted_costs[0]
                second_best_cost = sorted_costs[1] if len(sorted_costs) > 1 else np.nan
                planning_steps.append(step)
                optimal_costs.append(optimal_cost)
                second_best_costs.append(second_best_cost)
            
            # Get the candidate path chosen by the high-level planning
            best_candidate_overall = paths_via_openings[optimal_index]
            best_path_world = best_candidate_overall['path_world']
            chosen_opening_midpoint = best_candidate_overall['opening_point']
            chosen_path_world = np.array(best_path_world)
            
            # -------------------------------
            # Evaluate candidates with pose optimization across offsets
            # -------------------------------

            offset_count = 0
            time_step_control = 0
            lqr_Q = dummy_Q


            # Dictionaries to accumulate total cost per offset and track best candidate info
            offset_accumulated_costs = {offset: 0.0 for offset in offsets}
            offset_best_info = {offset: {'cost': float('inf'), 'candidate_index': None, 
                                         'initial_state': None, 'final_state': None} 
                                for offset in offsets}
            # print(state.theta)

            # Outer loop: iterate over each offset
            for offset in offsets:
                for candidate_index in range(len(paths_via_openings)):
                    candidate_dict = paths_via_openings[candidate_index]
                    candidate_path_world = candidate_dict['path_world']
                    candidate_path_array = np.array(candidate_path_world)
                    
                    # Optionally trim the candidate path based on current_position
                    distances = np.linalg.norm(candidate_path_array - current_position, axis=1)
                    closest_idx = np.argmin(distances)
                    # Here we use the full candidate path; you could trim if desired.
                    trimmed_path = candidate_path_array
                    
                    # If the path is too short, try a direct path.
                    if trimmed_path.shape[0] < 2:
                        direct_path = astar(grid,
                                            world_to_grid(current_position),
                                            world_to_grid(target_position))
                        if direct_path is not None and len(direct_path) > 1:
                            trimmed_path = np.array([grid_to_world(p) for p in direct_path])
                        else:
                            trimmed_path = candidate_path_array
                    
                    num_points_traj = trimmed_path.shape[0]
                    world_ref_traj = np.zeros((6, num_points_traj))
                    world_ref_traj[0, :] = trimmed_path[:, 0]
                    world_ref_traj[1, :] = trimmed_path[:, 1]
                    world_ref_traj[2, :] = 0.7
                    world_ref_traj[3, :] = 0
                    world_ref_traj[4, :] = 0
                    world_ref_traj[5, :] = 0
                    
                    # (Optional) Save reference point for tracking if needed.
                    ref_point = np.array([world_ref_traj[0, 0],
                                          world_ref_traj[1, 0],
                                          world_ref_traj[2, 0]])
                    tracking_ref.append(ref_point)
                    
                    # ---- POSE-OPTIMIZED Approach for this candidate with the current offset ----
                    candidate_state_opt = copy.deepcopy(state)
                    weighted_pose_optimized_cost, true_cost_opt, initial_candidate_state_opt, final_candidate_state_opt = pose_optimization(
                        candidate_state_opt,
                        lqr_Q,
                        world_ref_traj,
                        time_step_control,
                        step,           # current step
                        candidate_index,
                        percentages,
                        offset_count,
                        offset
                    )
                    offset_count += 1
                    # print("weighted_pose_optimized_cost: ", weighted_pose_optimized_cost)
                    
                    # Accumulate the cost for the current offset
                    offset_accumulated_costs[offset] += weighted_pose_optimized_cost
                    
                    # Update best candidate info for this offset if lower cost found.
                    if weighted_pose_optimized_cost < offset_best_info[offset]['cost']:
                        offset_best_info[offset]['cost'] = weighted_pose_optimized_cost
                        offset_best_info[offset]['candidate_index'] = candidate_index
                        offset_best_info[offset]['initial_state'] = initial_candidate_state_opt
                        offset_best_info[offset]['final_state'] = final_candidate_state_opt

                print("================================")   

            # for off in offsets:
            #     print(f"Total cost for offset {off}: {offset_accumulated_costs[off]}")
            
            # Select the offset with the lowest total cost.
            best_offset = min(offset_accumulated_costs, key=offset_accumulated_costs.get)

            if best_offset in offsets:
                cnt_offsets[offsets.index(best_offset)] += 1

            best_total_cost = offset_accumulated_costs[best_offset]
            # print(f"Best offset: {best_offset} with total cost: {best_total_cost}")

            
            # Retrieve the best candidate information for the chosen offset.
            best_candidate_index = offset_best_info[best_offset]['candidate_index']
            best_overall_state = offset_best_info[best_offset]['final_state']
            # print(f"Best overall state (theta): {best_overall_state.theta}")
            state = copy.deepcopy(best_overall_state)

            # ---- Retrieve the Best Path from paths_via_openings ----
            if best_candidate_index is not None:
                best_candidate = paths_via_openings[best_candidate_index]
                best_path_array = np.array(best_candidate['path_world'])
                chosen_opening_midpoint = best_candidate['opening_point']
                chosen_path_world = best_path_array
            else:
                print("No best candidate was found.")

            # --------------------------------------------------------------
            # -- END OF PLANNING + POSE-OPTIMIZATION SECTION
            # --------------------------------------------------------------

        # 3) Update current_position from the state.
        current_position = np.array([state.x_world, state.y_world])
        trajectory.append(current_position.copy())
        tracking_actual.append(np.array([state.x_world, state.y_world, state.z_world]))
        
        print(f"Step {step}: current_position = {current_position}, "
              f"lowest cost path index = {best_candidate_index + 1 if (use_planning or chosen_path_world is None) else 'N/A (following)'}")
        step += 1
        
        # 4) Check if we reached the opening midpoint => disable planning.
        if chosen_opening_midpoint is not None:
            dist_to_midpoint = np.linalg.norm(current_position - chosen_opening_midpoint)
            if dist_to_midpoint < midpoint_threshold:
                use_planning = False
        
        # 5) Visualization
        plt.figure(figsize=(8, 8))
        plt.plot(target_position[0], target_position[1],
                 'bx', markersize=10, label='Target')
        
        for obs_idx, obs in enumerate(obstacles):
            circle = Circle(obs['center'], obs['radius'], color='red', alpha=1.0)
            plt.gca().add_patch(circle)
            plt.text(obs['center'][0], obs['center'][1],
                     f'Obs{obs_idx}', color='white',
                     ha='center', va='center',
                     fontsize=12, fontweight='bold')
        
        # Plot candidate paths if in planning mode
        if use_planning and 'paths_via_openings' in locals():
            for candidate in paths_via_openings:
                cpath = np.array(candidate['path_world'])
                plt.plot(cpath[:, 0], cpath[:, 1], 'c--', linewidth=1)
        
        if chosen_path_world is not None:
            best_path_array = np.array(chosen_path_world)
            plt.plot(best_path_array[:, 0],
                     best_path_array[:, 1],
                     'b-', linewidth=2, label='Lowest-Cost Path')
        
        traj_array = np.array(trajectory)
        plt.plot(traj_array[:, 0], traj_array[:, 1],
                 'r.-', markersize=4, label='Robot Trajectory')
        
        plt.plot(current_position[0], current_position[1],
                 'go', markersize=10, label='Current Position')
        
        plt.legend()
        plt.title(f"Step {step}")
        plt.pause(0.001)
        plt.close()
    
    print("Simulation complete. Total steps:", step)

    # Assuming offsets is a list and cnt_offsets is updated as per previous logic
    for i, count in enumerate(cnt_offsets, start=1):
        print(f"cnt_offset_{i}: {count}")


    # print("Number of times they chose Opening 1: ", ind_1)
    # print("Number of times they chose Opening 2: ", ind_2)
    # print("Number of times they chose Opening 3: ", ind_3)
    # print("Number of times they chose Opening 4: ", ind_4)
    # print("Number of times they chose Opening 5: ", ind_5)
    # print("Number of times they chose Opening 6: ", ind_6)
    # print("Number of times they chose Opening 7: ", ind_7)
    print("cnt: ", cnt)
    print("with_po_times: ", with_po_times)
    print("without_po_times: ", without_po_times)
    # -------------------------------
    # FINAL PLOTS AFTER SIMULATION
    # -------------------------------
    trajectory = np.array(trajectory)

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot obstacles
    for obs in obstacles:
        circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.4, edgecolor='k')
        ax.add_patch(circle)

    # Plot the robot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', label='Robot Trajectory')

    # Labels and grid
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Trajectory in X-Y Plane with Obstacles')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 10)  # Adjust as needed
    ax.set_ylim(0, 10)  # Adjust as needed
    ax.set_aspect('equal')  # Keep scale equal for proper obstacle shapes

    # Show the plot
    plt.show()
    
    
    # Ensure data is in NumPy array format
    tracking_actual = np.array(tracking_actual)
    tracking_ref = np.array(tracking_ref)

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot obstacles
    for obs in obstacles:
        circle = Circle(obs['center'], obs['radius'], facecolor='red', alpha=0.4, edgecolor='k')
        ax.add_patch(circle)

    # Plot actual vs. reference tracking
    ax.plot(tracking_actual[:, 0], tracking_actual[:, 1], 'r.-', label='Actual Path')
    ax.plot(tracking_ref[:, 0], tracking_ref[:, 1], 'b.-', label='Reference Path')

    # Labels and grid
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Tracking Performance in X-Y Plane with Obstacles')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 10)  # Adjust based on your environment
    ax.set_ylim(0, 10)  # Adjust based on your environment
    ax.set_aspect('equal')  # Keep scale equal for proper obstacle visualization

    # Show the plot
    plt.show()

    
    # x-z plane tracking
    plt.figure()
    plt.plot(tracking_actual[:, 0], tracking_actual[:, 2], 'r.-', label='Actual')
    plt.plot(tracking_ref[:, 0], tracking_ref[:, 2], 'b.-', label='Reference')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Tracking Performance in X-Z Plane')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(time_steps, phi_dec_values, 'bo-')
    plt.xlabel('Time Step')
    plt.ylabel('$\phi$')
    plt.title('Human preference paramter')
    plt.grid(True)
    # plt.legend()
    plt.show()


    print("Step:", step)
    print("   Path   TrueCost   NominalCost   Percentage   DeltaRobot   DeltaHuman")
    for i, path in enumerate(paths_via_openings):
        line = f"   {i+1:4d}   {final_true_utils[i]:9.3f}   {final_nominal_utils[i]:11.3f}   {percentages[i]:9.2f}%"
        # You could also compute deltaRobot / deltaHuman as in item #2
        print(line)
    print(f"Chosen path index: {optimal_index+1}")
    print(f"Robot preferred index: {robot_preferred+1}")
    print(f"Human preferred index: {human_preferred+1}")
    print("---------------------------------------------------------------------")
    

    print("Elapsed time:", time.time() - start_time)

if __name__ == '__main__':
    main()

