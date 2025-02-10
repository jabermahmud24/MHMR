import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import heapq
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import math
import time
import csv
from scipy.interpolate import interp1d

# *** IMPORT THE ROBOT SERIAL MODULE (make sure it is in your PYTHONPATH) ***
from visual_kinematics.RobotSerial import *

# =============================================================================
# 1. ENVIRONMENT SETUP
# =============================================================================
grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
X, Y = np.meshgrid(x_range, y_range, indexing='ij')

# Starting (A) and target (B) positions in world coordinates.
A_pos = np.array([2.0, 2.0])
B_pos = np.array([9, 6])

# Define obstacles (each with a center and radius)
obstacles = [
    {'center': np.array([3, 0.5]), 'radius': 0.2},
    # {'center': np.array([5, 3]),   'radius': 0.65},
    {'center': np.array([5, 5.5]), 'radius': 0.2},
    {'center': np.array([8, 8.5]),   'radius': 0.2},
    
    # # Additional 10 Obstacles
    # {'center': np.array([1, 7]),   'radius': 0.6},
    # {'center': np.array([2, 4.5]), 'radius': 0.7},
    # {'center': np.array([4, 7]),   'radius': 0.7},
    # {'center': np.array([6, 2]),   'radius': 0.65},
    # {'center': np.array([8, 5]),   'radius': 0.55},
    # {'center': np.array([3, 3]), 'radius': 0.5},


    # {'center': np.array([3, 0.5]), 'radius': 0.45},
    {'center': np.array([5, 3]),   'radius': 0.2},
    {'center': np.array([5, 5.5]), 'radius': 0.2},
    # {'center': np.array([1, 7]),   'radius': 0.6},
    {'center': np.array([2, 4.5]), 'radius': 0.3},
    {'center': np.array([4, 7]),   'radius': 0.2},
    {'center': np.array([6, 2]),   'radius': 0.2},
    {'center': np.array([8, 5]),   'radius': 0.2},
    {'center': np.array([3, 3]), 'radius': 0.3},
    # Additional obstacles can be added here.
]

# Define robot size
robot_diameter = 0.5
robot_radius = robot_diameter / 2

# Create grid and mark obstacles (cells within obstacle radius plus robot radius are occupied)
grid = np.zeros(grid_size, dtype=int)
for obs in obstacles:
    distance = np.sqrt((X - obs['center'][0])**2 + (Y - obs['center'][1])**2)
    grid[distance <= (obs['radius'] + robot_radius)] = 1

# =============================================================================
# 2. COORDINATE TRANSFORMATIONS AND A* PATHFINDING
# =============================================================================
def world_to_grid(pos):
    """Convert world (x,y) to grid indices."""
    x, y = pos
    i = np.argmin(np.abs(x_range - x))
    j = np.argmin(np.abs(y_range - y))
    return (i, j)

def grid_to_world(idx_pos):
    """Convert grid indices (i,j) to world (x,y) coordinates."""
    i, j = idx_pos
    x = x_range[i]
    y = y_range[j]
    return np.array([x, y])

# --- A* Algorithm (4-connected) ---
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
    p_safe = p
    w_p = np.exp(-(-np.log(p_safe + 1e-10)) ** gamma)
    w_d = d ** alpha
    return w_d + w_p  # here, c_risk is assumed to be 1

def utility2(p, d):
    alpha_true = 1
    gamma_true = 1
    w_p = np.exp(-(-np.log(p + 1e-10)) ** gamma_true)
    w_d = d ** alpha_true
    # Print statement for debugging; you may remove it if desired.
    print("Distance d:", d)
    return w_d + w_p

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
lqr_Q = 1000 * np.eye(6)
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
    planning_horizon = 1
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

# =============================================================================
# 5. MAIN SIMULATION LOOP: RE-PLANNING + CONTROL UPDATE
# =============================================================================
def main():
    start_time = time.time()
    
    # Initialize current position and target.
    # Issue 1 fix: We only use A_pos for the very FIRST step.
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
    max_steps = 1000
    step = 0
    
    # --------------------------------------------------------------------------
    # A flag and storage to handle "committing" to a chosen opening once the robot
    # is near the midpoint of that opening.
    # --------------------------------------------------------------------------
    use_planning = True        # We start with planning turned ON
    chosen_opening_midpoint = None
    chosen_path_world = None
    midpoint_threshold = 0.5   # Distance threshold to stop re-planning
    
    state = None  # will be initialized on the first control step
    current_team_index = None  
    phi_init = 0.1
    phi_dec = phi_init
    cnt = 0
    ind_1 = 0
    ind_2 = 0
    ind_3 = 0
    ind_4 = 0
    ind_5 = 0
    ind_6 = 0
    ind_7 = 0
    

    
    while np.linalg.norm(current_position - target_position) > 0.3 and step < max_steps:
        
        # If we have committed to an opening and are already near its midpoint,
        # skip the entire path planning step. Just follow the previously chosen path.
        if (not use_planning) and (chosen_path_world is not None):
            best_path_world = chosen_path_world
        else:
            # -------------------------------
            # PLANNING: Compute candidate paths via openings
            # -------------------------------
            start_grid = world_to_grid(current_position)
            end_grid = world_to_grid(target_position)
            openings = calculate_openings(obstacles_sorted, robot_diameter, current_position, target_position)
            
            # List candidate paths using A*.
            paths_via_openings = []
            for idx, opening in enumerate(openings, start=1):
                opening_grid = world_to_grid(opening['position'])
                path_A_opening = astar(grid, start_grid, opening_grid)
                path_opening_B = astar(grid, opening_grid, end_grid)
                if path_A_opening is None or path_opening_B is None:
                    continue
                dist_A_open = path_length_in_world(path_A_opening)
                if dist_A_open is None:
                    dist_A_open = 0.0
                dist_open_B = path_length_in_world(path_opening_B)
                if dist_open_B is None:
                    dist_open_B = 0.0
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
            
            # -------------------------------
            # Utility calculations over a grid of (α, γ) parameters.
            # -------------------------------
            alpha_min, alpha_max = 0.5, 0.8
            gamma_min, gamma_max = 0.5, 0.75
            num_points = 50
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
                Utility_lin = linearized_utility(utility1, p, d, alpha0, gamma0, Alpha, Gamma)
                utilities_linearized.append(Utility_lin)
            
            # Compute “percentages” based on point-wise minimization of the linearized utility.
            utilities_stack = np.array(utilities_linearized)
            min_indices = np.argmin(utilities_stack, axis=0)
            human_preferred = np.argmin(final_nominal_utils, axis=0)
            print(human_preferred)
            robot_preferred = np.argmin(final_true_utils, axis = 0)
            print(robot_preferred)
            total_points = min_indices.size
            counts = np.bincount(min_indices.flatten(), minlength=len(utilities_linearized))
            percentages = counts / total_points * 100
            print(percentages)

            # -------------------------------
            # Final decision cost function.
            # -------------------------------
            N = len(paths_via_openings)
            # Use the current team opening as the baseline.
            # If none has been set yet, initialize with the candidate having the lowest nominal utility.
            if current_team_index is None:
                current_team_index = np.argmin(final_true_utils)
            base_true = final_true_utils[current_team_index]
            base_nominal = final_nominal_utils[current_team_index]
            costs = np.zeros(N)
            for k in range(N):
                costs[k] = percentages[k] * (
                    - (1 - phi_dec) * (base_true - final_true_utils[k])
                    + phi_dec * (base_nominal - final_nominal_utils[k])
                )
            optimal_index = np.argmin(costs)
            print(current_team_index)
            print(optimal_index)

            if optimal_index == 0:
                ind_1 += 1
                print("ind_1: ", ind_1)

            if optimal_index == 1:
                ind_2 += 1
                print("ind_2: ", ind_2)

            if optimal_index == 2:
                ind_3 += 1
                print("ind_3: ", ind_3)

            if optimal_index == 3:
                ind_4 += 1
                print("ind_4: ", ind_4)

            if optimal_index == 4:
                ind_5 += 1
                print("ind_5: ", ind_5)

            if optimal_index == 5:
                ind_6 += 1
                print("ind_6: ", ind_6)

            if optimal_index == 6:
                ind_7 += 1
                print("ind_7: ", ind_7)

            if current_team_index != optimal_index:
                phi_dec = phi_init * np.exp( (cnt*0.4 / 10))
                cnt += 1
                print("cnt: ",cnt)
                print("phi_dec:", phi_dec)
            # Update the team's current opening index for subsequent iterations.
            current_team_index = optimal_index



            best_candidate = paths_via_openings[optimal_index]
            best_path_world = best_candidate['path_world']

            
            # # Final decision cost function.
            # phi_dec = 1.0
            # N = len(paths_via_openings)
            # costs = np.zeros(N)
            # for i in range(N):
            #     cost = 0
            #     for k in range(N):
            #         cost += percentages[k] * ( - (1 - phi_dec) * (final_true_utils[i] - final_true_utils[k]) +
            #                                    phi_dec * (final_nominal_utils[i] - final_nominal_utils[k]) )
            #     costs[i] = cost
            # optimal_index = np.argmin(costs)
            # best_candidate = paths_via_openings[optimal_index]
            # best_path_world = best_candidate['path_world']
            
            # Store the midpoint of this chosen opening
            chosen_opening_midpoint = best_candidate['opening_point']
            chosen_path_world = best_path_world  # We may skip re-planning once near midpoint
            
            # Save cost information for later visualization.
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
                if len(costs) > 1:
                    sorted_costs = np.sort(costs)
                    optimal_cost = sorted_costs[0]
                    second_best_cost = sorted_costs[1]
                else:
                    optimal_cost = costs[0]
                    second_best_cost = np.nan
                planning_steps.append(step)
                optimal_costs.append(optimal_cost)
                second_best_costs.append(second_best_cost)
        
        # -------------------------------------------------------
        # Trim the candidate path so that only the portion ahead of the robot is used.
        # -------------------------------------------------------
        best_path_array = np.array(best_path_world)
        distances = np.linalg.norm(best_path_array - current_position, axis=1)
        closest_idx = np.argmin(distances)
        trimmed_path = best_path_array[closest_idx:]
        # If the trimmed path is too short, try direct path from current to target.
        if trimmed_path.shape[0] < 2:
            direct_path = astar(grid, world_to_grid(current_position), world_to_grid(target_position))
            if direct_path is not None and len(direct_path) > 1:
                trimmed_path = np.array([grid_to_world(p) for p in direct_path])
            else:
                trimmed_path = best_path_array
        
        # -------------------------------------------------------
        # Convert the (trimmed) candidate path into a reference trajectory for control.
        # -------------------------------------------------------
        num_points_traj = trimmed_path.shape[0]
        world_ref_traj = np.zeros((6, num_points_traj))
        world_ref_traj[0, :] = trimmed_path[:, 0]
        world_ref_traj[1, :] = trimmed_path[:, 1]
        world_ref_traj[2, :] = 0.7   # Constant z (end-effector height)
        world_ref_traj[3, :] = 0     # Yaw
        world_ref_traj[4, :] = 0     # Pitch
        world_ref_traj[5, :] = 0     # Roll
        
        # Save the reference point (first point of the trajectory) for tracking plots.
        ref_point = np.array([world_ref_traj[0, 0], world_ref_traj[1, 0], world_ref_traj[2, 0]])
        tracking_ref.append(ref_point)
        
        # -------------------------------
        # CONTROL: Use LQR-based control for one step.
        # -------------------------------
        if step == 0:
            # Initialize the Robot State with the reference for the first time
            state = State(world_ref_traj,
                          x_base=world_ref_traj[0, 0] - 0.6,
                          y_base=world_ref_traj[1, 0] + 0.3)
        
        time_step_control = 0
        ustar, B_mat, fwd = lqr_control(state, lqr_Q, lqr_R, world_ref_traj, time_step_control, state.dh_params, state.theta)
        state, ee_pose = update(state, ustar, fwd, state.dh_params, B_mat, state.theta)
        
        # Update current position from the end-effector state
        current_position = np.array([state.x_world, state.y_world])
        trajectory.append(current_position.copy())
        
        # Save the actual end-effector position for tracking plots.
        tracking_actual.append(np.array([state.x_world, state.y_world, state.z_world]))
        
        # Print status
        print(f"Step {step}: current_position = {current_position}, cost index = {optimal_index + 1 if 'optimal_index' in locals() else 'N/A'}")
        step += 1
        
        # ------------------------------------------------------------------
        # Check if we have reached near the midpoint of the chosen opening.
        # If so, stop any further A* planning and keep using the chosen path.
        # ------------------------------------------------------------------
        if chosen_opening_midpoint is not None:
            dist_to_midpoint = np.linalg.norm(current_position - chosen_opening_midpoint)
            if dist_to_midpoint < midpoint_threshold:
                use_planning = False
        
        # -------------------------------
        # Visualization (optional during simulation)
        # -------------------------------
        plt.figure(figsize=(8, 8))
        plt.imshow(np.flipud(grid.T), extent=(0, 10, 0, 10), cmap='gray_r')
        for obs in obstacles:
            circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.plot(target_position[0], target_position[1], 'bx', markersize=10, label='Target')
        
        if use_planning and 'paths_via_openings' in locals():
            # Show all candidate paths
            for candidate in paths_via_openings:
                candidate_path_array = np.array(candidate['path_world'])
                plt.plot(candidate_path_array[:, 0], candidate_path_array[:, 1],
                         'c--', linewidth=1)
        
        # Show final best path
        plt.plot(best_path_array[:, 0], best_path_array[:, 1], 'b-', linewidth=2, label='Best Path')
        
        # Show actual robot trajectory so far
        traj_array = np.array(trajectory)
        plt.plot(traj_array[:, 0], traj_array[:, 1], 'r.-', markersize=4, label='Robot Trajectory')
        
        plt.plot(current_position[0], current_position[1], 'go', markersize=10, label='Current Position')
        plt.legend()
        plt.title(f"Step {step}")
        plt.pause(0.1)
        plt.close()
    
    print("Simulation complete. Total steps:", step)
    
    print("Number of times they chose Opening 1: ",ind_1)
    print("Number of times they chose Opening 2: ",ind_2)
    print("Number of times they chose Opening 3: ",ind_3)
    print("Number of times they chose Opening 4: ",ind_4)
    print("Number of times they chose Opening 5: ",ind_5)
    print("Number of times they chose Opening 6: ",ind_6)
    print("Number of times they chose Opening 7: ",ind_7)
    # -------------------------------
    # FINAL PLOTS AFTER SIMULATION
    # -------------------------------
    trajectory = np.array(trajectory)
    
    # (a) Plot the overall robot trajectory in the x-y plane.
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', label='Robot Trajectory (x-y)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robot Trajectory in X-Y Plane')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # (b) Plot the tracking performance: Actual vs. Reference in x-y and x-z planes.
    tracking_actual = np.array(tracking_actual)
    tracking_ref = np.array(tracking_ref)
    
    # x-y plane tracking
    plt.figure()
    plt.plot(tracking_actual[:, 0], tracking_actual[:, 1], 'r.-', label='Actual')
    plt.plot(tracking_ref[:, 0], tracking_ref[:, 1], 'b.-', label='Reference')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Tracking Performance in X-Y Plane')
    plt.legend()
    plt.grid(True)
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
    
    # (c) Plot candidate cost evolution over planning iterations.
    if planning_steps:
        plt.figure()
        plt.plot(planning_steps, optimal_costs, 'go-', label='Optimal Candidate Cost')
        plt.plot(planning_steps, second_best_costs, 'ro-', label='Second Best Candidate Cost')
        plt.xlabel('Simulation Step (Planning Iteration)')
        plt.ylabel('Cost')
        plt.title('Candidate Cost Evolution Over Planning Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # (d) For the final planning iteration, display a bar chart of candidate cost details.
    if cost_history:
        final_record = cost_history[-1]
        final_costs = final_record['costs']
        final_percentages = final_record['percentages']
        final_true_utils = final_record['final_true_utils']
        final_nominal_utils = final_record['final_nominal_utils']
        indices = np.arange(len(final_costs))
        width = 0.2
        plt.figure(figsize=(10, 6))
        plt.bar(indices - 1.5*width, final_costs, width, label='Candidate Cost')
        plt.bar(indices - 0.5*width, final_percentages, width, label='Candidate Percentage')
        plt.bar(indices + 0.5*width, final_true_utils, width, label='Final True Utility')
        plt.bar(indices + 1.5*width, final_nominal_utils, width, label='Final Nominal Utility')
        plt.xlabel('Candidate Index')
        plt.ylabel('Value')
        plt.title('Final Planning Iteration: Cost and Utility Details')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    print("Elapsed time:", time.time() - start_time)

if __name__ == '__main__':
    main()


