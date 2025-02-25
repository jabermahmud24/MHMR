import numpy as np
import matplotlib.pyplot as plt
import heapq
import math
import time
import csv
from scipy.interpolate import interp1d
import copy

# *** IMPORT THE ROBOT SERIAL MODULE (make sure it is in your PYTHONPATH) ***
from visual_kinematics.RobotSerial import *

# ----------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP: Rectangular Obstacles & Occupancy Grid
# ----------------------------------------------------------------------------

grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
X, Y = np.meshgrid(x_range, y_range, indexing='ij')

A_pos = np.array([4.2, 0.5])  # Start
B_pos = np.array([5.0, 8.0])  # Target

obstacles = [
    {'center': np.array([4, 3.0]),   'width': 4,  'height': 3.95},
    {'center': np.array([2, 8.0]),   'width': 3,  'height': 4},
    {'center': np.array([6.95, 7]),  'width': 3,  'height': 3},
    # Dummy obstacle to avoid index errors
    {'center': np.array([10.01, 6.5]), 'width': 0,'height': 0},
]

robot_diameter = 0.5
robot_radius   = robot_diameter / 2

# Inflated occupancy grid
grid = np.zeros(grid_size, dtype=int)
for obs in obstacles:
    center = obs['center']
    half_w = obs['width']/2  + robot_radius
    half_h = obs['height']/2 + robot_radius
    xmin, xmax = center[0] - half_w, center[0] + half_w
    ymin, ymax = center[1] - half_h, center[1] + half_h
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x = x_range[i]
            y = y_range[j]
            if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                grid[i, j] = 1

# ----------------------------------------------------------------------------
# 2. COORDINATE TRANSFORMS & A* PATHFINDING
# ----------------------------------------------------------------------------

def world_to_grid(pos):
    i = np.argmin(np.abs(x_range - pos[0]))
    j = np.argmin(np.abs(y_range - pos[1]))
    return (i, j)

def grid_to_world(idx):
    return np.array([x_range[idx[0]], y_range[idx[1]]])

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent   = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def astar(grid, start, end):
    open_set   = []
    start_h    = np.linalg.norm(np.array(start) - np.array(end))
    heapq.heappush(open_set, Node(start, None, 0, start_h))
    closed_set = set()
    neighbors  = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while open_set:
        current_node = heapq.heappop(open_set)
        if current_node.position == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        closed_set.add(current_node.position)
        for move in neighbors:
            nx = current_node.position[0] + move[0]
            ny = current_node.position[1] + move[1]
            neighbor = (nx, ny)
            if not(0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]):
                continue
            if grid[nx, ny] == 1:
                continue
            if neighbor in closed_set:
                continue
            step_cost = math.hypot(move[0], move[1])
            tentative_g = current_node.g + step_cost
            h = np.linalg.norm(np.array(neighbor) - np.array(end))
            skip = False
            for node in open_set:
                if node.position == neighbor and tentative_g >= node.g:
                    skip = True
                    break
            if skip:
                continue
            heapq.heappush(open_set, Node(neighbor, current_node, tentative_g, h))
    return None

def path_length_in_world(path):
    if not path or len(path) < 2:
        return 0
    path_world = [grid_to_world(p) for p in path]
    return sum(np.linalg.norm(path_world[i+1] - path_world[i])
               for i in range(len(path_world)-1))

# ----------------------------------------------------------------------------
# 3. OPENING / WAYPOINT CALCULATION
# ----------------------------------------------------------------------------

def rect_edge_distance(obstacle, direction):
    half_w = obstacle['width']/2
    half_h = obstacle['height']/2
    tol = 1e-8
    t1 = half_w / abs(direction[0]) if abs(direction[0])>tol else float('inf')
    t2 = half_h / abs(direction[1]) if abs(direction[1])>tol else float('inf')
    return min(t1, t2)

def compute_opening(obstacle_a, obstacle_b, robot_diameter, A_pos, B_pos):
    center_a = obstacle_a['center']
    center_b = obstacle_b['center']
    dist_ab  = np.linalg.norm(center_b - center_a)
    if dist_ab == 0:
        return None
    direction  = (center_b - center_a) / dist_ab
    edge_dist_a= rect_edge_distance(obstacle_a, direction)
    edge_dist_b= rect_edge_distance(obstacle_b, -direction)
    gap_width  = dist_ab - (edge_dist_a + edge_dist_b)
    if gap_width <= 0:
        return None
    edge_pt_a  = center_a + direction * edge_dist_a
    edge_pt_b  = center_b - direction * edge_dist_b
    opening_pos= (edge_pt_a + edge_pt_b)/2
    # Example: define risk as ratio of robot diameter to gap (minus diameter, or etc.)
    # Adjust as desired:
    risk       = robot_diameter / gap_width
    naive_dist = np.linalg.norm(A_pos - opening_pos) + np.linalg.norm(opening_pos - B_pos)
    return {
        'width': gap_width,
        'risk': risk,
        'opening_position': opening_pos,
        'distance_to_target': naive_dist
    }

obstacles_sorted = sorted(obstacles, key=lambda obs: obs['center'][1])
pairs_to_consider = [(1, 3), (2, 3), (1, 4), (3, 4)]

candidate_nodes = {"A": {'pos': A_pos, 'type': 'start'},
                   "B": {'pos': B_pos, 'type': 'target'}}

for pair in pairs_to_consider:
    idx_a, idx_b = pair
    obs_a = obstacles_sorted[idx_a - 1]
    obs_b = obstacles_sorted[idx_b - 1]
    opening = compute_opening(obs_a, obs_b, robot_diameter, A_pos, B_pos)
    label   = f"O{idx_a}_{idx_b}"
    if opening is not None:
        candidate_nodes[label] = {
            'pos':  opening['opening_position'],
            'type': 'opening',
            'risk': opening['risk']
        }
        print(f"Opening {label}: width={opening['width']:.2f}, risk={opening['risk']:.2f}")
    else:
        print(f"No valid opening for pair {pair}")

# ----------------------------------------------------------------------------
# 4. BUILD GRAPH BETWEEN NODES USING A*
# ----------------------------------------------------------------------------

graph = {node: {} for node in candidate_nodes}
for node_i, info_i in candidate_nodes.items():
    for node_j, info_j in candidate_nodes.items():
        if node_i == node_j:
            continue
        start_idx = world_to_grid(info_i['pos'])
        goal_idx  = world_to_grid(info_j['pos'])
        path_grid = astar(grid, start_idx, goal_idx)
        if path_grid is not None:
            w = path_length_in_world(path_grid)
            path_world = [grid_to_world(p) for p in path_grid]
            graph[node_i][node_j] = {
                'weight': w,
                'path':   path_world
            }

# ----------------------------------------------------------------------------
# 5. MANUAL PATHS
# ----------------------------------------------------------------------------

manual_paths = [
    ["O1_4", "O1_3", "O2_3"],
    ["O1_4", "O3_4"],
    ["O1_3", "O3_4"],
    ["O2_3"]
]

options = []
for manual_seq in manual_paths:
    full_seq = ["A"] + manual_seq + ["B"]
    merged_path = []
    total_cost  = 0
    valid = True
    for i in range(len(full_seq)-1):
        from_lbl = full_seq[i]
        to_lbl   = full_seq[i+1]
        if to_lbl not in graph[from_lbl]:
            print(f"Path segment {from_lbl}->{to_lbl} not found; skip {manual_seq}")
            valid = False
            break
        seg_info = graph[from_lbl][to_lbl]
        seg_path = seg_info['path']
        seg_cost = seg_info['weight']
        total_cost += seg_cost
        if i>0:
            seg_path = seg_path[1:]  # avoid duplicates
        merged_path.extend(seg_path)
    if valid:
        # We can define path "risk" in various ways: sum, max, average, ...
        # Let's do average of any opening's risk in this sequence
        opening_labels = [lbl for lbl in full_seq if lbl.startswith("O")]
        if opening_labels:
            risk_vals = [candidate_nodes[olbl]['risk'] for olbl in opening_labels]
            path_risk = np.mean(risk_vals)
        else:
            path_risk = 0
        options.append({
            'sequence': full_seq,
            'path':     merged_path,
            'total_cost': total_cost,
            'risk':     path_risk
        })
        print(f"Option {full_seq}: distance={total_cost:.2f}, risk={path_risk:.3f}")


# ----------------------------------------------------------------------------
# 6. UTILITY FUNCTIONS (robot-human preference)
# ----------------------------------------------------------------------------

c_risk = 7.0
weighted_distance = 1.0

def utility1(p, d, alpha, gamma):
    """
    'Human' nominal utility
    """
    p_safe = p
    w_p = np.exp(-(-np.log(p_safe + 1e-10)) ** gamma)
    w_d = weighted_distance * (d ** alpha)
    return w_d + (w_p * c_risk)

def utility2(p, d):
    """
    'Robot' true utility
    """
    alpha_true  = 1
    gamma_true  = 1
    w_p = np.exp(-(-np.log(p + 1e-10)) ** gamma_true)
    w_d = d ** alpha_true
    return w_d + (w_p * c_risk)

def linearized_utility(original_func, p, d, alpha0, gamma0, Alpha, Gamma):
    """
    Evaluate the linearization of original_func around (alpha0, gamma0)
    for arrays of (Alpha, Gamma).
    """
    epsilon = 1e-5
    f0      = original_func(p, d, alpha0, gamma0)
    # partial derivative wrt alpha
    f_a_plus  = original_func(p, d, alpha0 + epsilon, gamma0)
    f_a_minus = original_func(p, d, alpha0 - epsilon, gamma0)
    df_da     = (f_a_plus - f_a_minus)/(2*epsilon)
    # partial derivative wrt gamma
    f_g_plus  = original_func(p, d, alpha0, gamma0 + epsilon)
    f_g_minus = original_func(p, d, alpha0, gamma0 - epsilon)
    df_dg     = (f_g_plus - f_g_minus)/(2*epsilon)

    # linearization f(Alpha, Gamma) ~ f0 + df_da*(Alpha - alpha0) + df_dg*(Gamma - gamma0)
    return f0 + df_da*(Alpha - alpha0) + df_dg*(Gamma - gamma0)


# ----------------------------------------------------------------------------
# 7. LQR CONTROL & POSE OPTIMIZATION
# ----------------------------------------------------------------------------

dt      = 0.1
dummy_Q = 1000*np.eye(6)
r_matrix= np.eye(9)
scalar  = 1
lqr_R   = scalar*r_matrix

class State:
    def __init__(self, world_ref_traj, x_base=0, y_base=0.3):
        # Example DH parameters
        self.dh_params = np.array([
            [0.72,          0,              0,             0 ],
            [0.06,          0.117,          -0.5*math.pi,   0 ],
            [0,             0,               0.5*math.pi,    1.57 ],
            [0.219+0.133,   0,               0.5*math.pi,    0 ],
            [0,             0,              -0.5*math.pi,    0 ],
            [0.197+0.1245,  0,              -0.5*math.pi,    0 ],
            [0,             0,               0.5*math.pi,    0 ],
            [0.1385+0.1665, 0,               0,              0 ]
        ])
        self.x_base   = x_base
        self.y_base   = y_base
        self.yaw_base = 0

        if world_ref_traj.shape[1]<1:
            raise ValueError("Ref traj must have at least 1 column.")

        xyz = np.array([
            [world_ref_traj[0,0] - self.x_base],
            [world_ref_traj[1,0] - self.y_base],
            [world_ref_traj[2,0]]
        ])
        abc = np.array([world_ref_traj[3,0],
                        world_ref_traj[4,0],
                        world_ref_traj[5,0]])
        robot = RobotSerial(self.dh_params)
        end   = Frame.from_euler_3(abc, xyz)
        robot.inverse(end)
        self.theta = robot.axis_values.copy()
        self.yaw_base = self.theta[0]

        f = robot.forward(self.theta)
        self.x_world     = self.x_base + f.t_3_1[0,0]
        self.y_world     = self.y_base + f.t_3_1[1,0]
        self.z_world     = f.t_3_1[2,0]
        self.yaw_world   = f.euler_3[2]
        self.pitch_world = f.euler_3[1]
        self.roll_world  = f.euler_3[0]

def get_B(dh_params, state, joint_angle_combination):
    robot = RobotSerial(dh_params)
    f     = robot.forward(joint_angle_combination)
    # Load a precomputed Jacobian from CSV (example):
    jacobian = []
    with open('jacobian_matrix1.csv','r') as csvf:
        reader = csv.reader(csvf)
        for row in reader:
            jacobian.append([float(cell) for cell in row])
    jacobian = np.array(jacobian, dtype=float)

    B = np.zeros((6,9))
    B[:,1:9] = jacobian*dt
    B[0,0]   = dt*math.cos(state.yaw_base)
    B[1,0]   = dt*math.sin(state.yaw_base)
    return B,f

def solve_dare(A, B, Q, R, ref_traj, time_step):
    X = Q
    planning_horizon = 8
    eps = 0.01
    small_p = np.zeros(6)
    for i in range(planning_horizon):
        Xn = A.T@X@A - A.T@X@B@np.linalg.inv(R + B.T@X@B)@B.T@X@A + Q
        small_p_n = small_p + Xn@(ref_traj[:, time_step] - ref_traj[:, time_step+1]) \
                     - Xn@B@np.linalg.inv(R+B.T@X@B)@B.T@Xn@(ref_traj[:, time_step]-ref_traj[:, time_step+1]) \
                     - Xn@B@np.linalg.inv(R+B.T@X@B)@B.T@small_p
        if (np.abs(Xn - X)).max()<eps:
            X = Xn
            break
        X = Xn
        small_p = small_p_n
    return X, small_p

def dlqr(A, B, Q, R, ref_traj, time_step):
    X, small_p = solve_dare(A, B, Q, R, ref_traj, time_step)
    K = np.linalg.inv(R + B.T@X@B)@(B.T@X@A)
    return K, small_p, X

def lqr_control(state, lqr_Q, lqr_R, ref_traj, time_step,
                dh_params, joint_angle_combination):
    A_mat = np.eye(6)
    B_mat, fwd = get_B(dh_params, state, joint_angle_combination)
    K, small_p, X = dlqr(A_mat, B_mat, lqr_Q, lqr_R, ref_traj, time_step)
    st_vec = np.array([ state.x_world,     state.y_world,     state.z_world,
                        state.yaw_world,   state.pitch_world, state.roll_world ])
    ref_vec= ref_traj[:, time_step]
    err    = st_vec - ref_vec
    ref_diff= ref_traj[:, time_step] - ref_traj[:, time_step+1]
    impact = np.linalg.inv(lqr_R + B_mat.T@X@B_mat)@(B_mat.T@small_p)
    ustar  = -K@(err + ref_diff) - impact
    return ustar, B_mat, fwd

def update(state, ustar, f, dh_params, B):
    # The first control dimension is base linear velocity,
    # second dimension is base yaw rate, next are joint velocities, etc.
    state.theta = state.theta + dt*ustar[1:].reshape(1,-1)
    state.theta = state.theta.astype(float)

    state.yaw_base += ustar[1]*dt
    state.x_base   += ustar[0]*dt*math.cos(state.yaw_base)
    state.y_base   += ustar[0]*dt*math.sin(state.yaw_base)

    robot = RobotSerial(dh_params)
    f = robot.forward(state.theta)
    state.x_world     = state.x_base + f.t_3_1[0,0]
    state.y_world     = state.y_base + f.t_3_1[1,0]
    state.z_world     = f.t_3_1[2,0]
    state.yaw_world   = f.euler_3[2]
    state.pitch_world = f.euler_3[1]
    state.roll_world  = f.euler_3[0]

    ee_pose = np.array([[state.x_world],
                        [state.y_world],
                        [state.z_world],
                        [state.yaw_world],
                        [state.pitch_world],
                        [state.roll_world]])
    return state, ee_pose

def calculate_true_cost(ustar, state, ref_traj, lqr_Q, lqr_R, angle_change_cost):
    err = np.array([
        [ref_traj[0,0] - state.x_world],
        [ref_traj[1,0] - state.y_world],
        [ref_traj[2,0] - state.z_world],
        [ref_traj[3,0] - state.yaw_world],
        [ref_traj[4,0] - state.pitch_world],
        [ref_traj[5,0] - state.roll_world]
    ])
    cost = (err.T @ lqr_Q @ err) + (ustar.T @ lqr_R @ ustar) + angle_change_cost
    return cost

def pose_optimization(candidate_state, lqr_Q, lqr_R, world_ref_traj,
                      time_step_control, step, candidate_index,
                      percentages, offset):
    """
    Try offset on a copy of candidate_state. We run the LQR control
    on that offset to get a 'pose_optimized_true_cost'.
    Return both the weighted cost (for picking best offset) and
    the 'true' cost if we finalize it.
    """
    cand_copy = copy.deepcopy(candidate_state)
    prev_theta= cand_copy.theta.copy()
    # Example: offset the last 2 joints if step=0, else offset them with indexing
    if step==0:
        cand_copy.theta[6] += offset
        cand_copy.theta[7] += offset
    else:
        cand_copy.theta[0,6] += offset
        cand_copy.theta[0,7] += offset

    # LQR step on cand_copy
    ustar, B_mat, fwd = lqr_control(cand_copy, lqr_Q, lqr_R,
                                    world_ref_traj, time_step_control,
                                    cand_copy.dh_params, cand_copy.theta)
    # Now do a copy update to see final pose
    test_state = copy.deepcopy(candidate_state)  # original state
    final_state, ee_pose = update(test_state, ustar, fwd, test_state.dh_params, B_mat)

    # angle-change cost
    diff = np.abs(cand_copy.theta - prev_theta)
    kappa= 1.0
    if diff.ndim==2:
        diff_val = (diff @ diff.T)
    else:
        diff_val = (diff.T @ diff)
    angle_change_cost = kappa*diff_val

    pose_optimized_true_cost = float( calculate_true_cost(ustar, final_state,
                                            world_ref_traj, lqr_Q, lqr_R,
                                            angle_change_cost) )
    # Weighted by how likely humans pick that candidate
    # percentages[candidate_index] is the fraction of param space that picks this path
    # We'll do (1 - p) factor, for example:
    weight = (1 - (percentages[candidate_index]/100.0))
    weighted_cost = weight*pose_optimized_true_cost

    return weighted_cost, pose_optimized_true_cost, cand_copy, final_state


# ----------------------------------------------------------------------------
# 8. CHECK IF ROBOT HAS REACHED ANY OPENING
# ----------------------------------------------------------------------------

def has_reached_an_opening(current_position, candidate_nodes, candidate_sequence, threshold):
    for lbl in candidate_sequence:
        if lbl.startswith("O"):
            opening_pos = candidate_nodes[lbl]['pos']
            if np.linalg.norm(current_position - opening_pos)<threshold:
                return True
    return False


# ----------------------------------------------------------------------------
# 9. MAIN SIMULATION FUNCTION
# ----------------------------------------------------------------------------

def run_simulation(offsets_override=None):
    """
    Run the entire simulation with either single or multiple offset values.
    """
    if offsets_override is None:
        offsets = [0.0]  # default single offset
    else:
        offsets = offsets_override

    current_position = A_pos.copy()
    target_position  = B_pos.copy()

    # We'll make a local copy of 'options' so we don't mutate the global.
    # However, for big changes you might re-build them each iteration.
    local_options = copy.deepcopy(options)

    trajectory       = [current_position.copy()]
    tracking_actual  = []
    max_steps        = 160
    step             = 0
    use_planning     = True
    chosen_sequence  = None
    chosen_path_world= None
    midpoint_threshold= 0.1

    state = None
    current_team_index     = None
    phi_init = 0.2
    phi_dec  = phi_init
    last_team_change_pos = current_position.copy()
    traveled_distance   = 0
    cnt = 0

    phi_dec_values   = []
    time_steps       = []
    robot_pref_inds  = []
    human_pref_inds  = []
    chosen_indices   = []
    final_true_costs = []
    final_nom_costs  = []
    accumulated_cost = 0.0

    while np.linalg.norm(current_position - target_position)>0.3 and step<max_steps:
        # Check if we already reached some opening in chosen_sequence => stop re-planning
        if chosen_sequence is not None:
            if has_reached_an_opening(current_position, candidate_nodes, chosen_sequence, midpoint_threshold):
                use_planning = False

        if state is None and step==0:
            # Initialize robot state
            dummy_ref_traj = np.zeros((6,1))
            dummy_ref_traj[0,0] = current_position[0]
            dummy_ref_traj[1,0] = current_position[1]
            dummy_ref_traj[2,0] = 0.7
            # yaw/pitch/roll=0
            robot_init = State(dummy_ref_traj,
                               x_base=current_position[0]-0.6,
                               y_base=current_position[1]+0.3)
            state = robot_init

        if (not use_planning) and (chosen_path_world is not None):
            # Follow the chosen path (no re-planning)
            cpath_arr = np.array(chosen_path_world)
            dist_arr  = np.linalg.norm(cpath_arr - current_position, axis=1)
            closest   = np.argmin(dist_arr)
            trimmed   = cpath_arr[closest:]
            if trimmed.shape[0]<2:
                trimmed = cpath_arr
            # Build a short ref traj
            npts  = trimmed.shape[0]
            wref  = np.zeros((6,npts))
            wref[0,:] = trimmed[:,0]
            wref[1,:] = trimmed[:,1]
            wref[2,:] = 0.7  # constant
            # LQR step
            time_step_ctrl= 0
            ustar, Bmat, fwd = lqr_control(state, dummy_Q, lqr_R,
                                           wref, time_step_ctrl,
                                           state.dh_params, state.theta)
            state, ee_pose = update(state, ustar, fwd, state.dh_params, Bmat)
        else:
            # Re-planning with updated A*
            # For each path option, we rebuild path from current position to the full sequence.
            updated_candidates = []
            start_idx = world_to_grid(current_position)
            for candidate in local_options:
                seq    = candidate['sequence']
                merged = []
                total_c= 0.0
                valid  = True
                for i in range(len(seq)-1):
                    from_lbl= seq[i]
                    to_lbl  = seq[i+1]
                    if i==0:
                        # from current pos
                        sidx = start_idx
                    else:
                        sidx = world_to_grid(candidate_nodes[from_lbl]['pos'])
                    gidx = world_to_grid(candidate_nodes[to_lbl]['pos'])
                    path_grid = astar(grid, sidx, gidx)
                    if path_grid is None:
                        valid=False
                        break
                    seg_cost= path_length_in_world(path_grid)
                    total_c += seg_cost
                    seg_w   = [grid_to_world(p) for p in path_grid]
                    if i>0:
                        seg_w=seg_w[1:]
                    merged.extend(seg_w)
                if valid:
                    updated_candidates.append({
                        'sequence': seq,
                        'path':     merged,
                        'total_cost': total_c,
                        'risk': candidate.get('risk', 0)
                    })
            if not updated_candidates:
                print("No viable path found; stopping.")
                break

            # Evaluate each candidate's nominal vs. true utility
            alpha0, gamma0 = 0.88, 0.4
            alpha_vals = np.linspace(0.4, 1.0, 300)
            gamma_vals = np.linspace(0.2, 1.4, 300)
            Agrid, Ggrid = np.meshgrid(alpha_vals, gamma_vals)
            nom_utils = []
            true_utils= []
            lin_utils = []
            for cand in updated_candidates:
                p = cand['risk']
                d = cand['total_cost']
                unom = utility1(p, d, alpha0, gamma0)
                utru = utility2(p, d)
                nom_utils.append(unom)
                true_utils.append(utru)
                # linearization
                lu = linearized_utility(utility1, p, d, alpha0, gamma0, Agrid, Ggrid)
                lin_utils.append(lu)
            # build big array
            lu_stack = np.array(lin_utils)
            min_inds = np.argmin(lu_stack, axis=0)
            total_pts= min_inds.size
            counts   = np.bincount(min_inds.flatten(), minlength=len(updated_candidates))
            percentages = (counts/total_pts)*100

            # "Team selection" approach
            if current_team_index is None:
                current_team_index = np.argmin(true_utils)  # default to robot preference

            base_t = true_utils[current_team_index]
            base_n = nom_utils[current_team_index]
            N = len(updated_candidates)
            combined_cost = np.zeros(N)
            for k in range(N):
                # same logic as your code: 
                combined_cost[k] = (percentages[k] * (
                    + (1 - phi_dec)*(-base_t + true_utils[k])
                    - phi_dec*(base_n - nom_utils[k])
                ))
            if np.min(combined_cost)<0:
                best_idx = np.argmin(combined_cost)
            else:
                best_idx = current_team_index

            # record
            robot_idx = np.argmin(true_utils)
            human_idx = np.argmin(nom_utils)
            robot_pref_inds.append(robot_idx)
            human_pref_inds.append(human_idx)
            chosen_indices.append(best_idx)
            final_true_costs.append(true_utils[best_idx])
            final_nom_costs.append(nom_utils[best_idx])

            if human_idx!=best_idx:
                cnt += 1
                traveled_distance += np.linalg.norm(current_position - last_team_change_pos)
                # example exponent
                eta  = 0.5
                phi_dec = phi_init*np.exp(traveled_distance/eta)
                last_team_change_pos = current_position.copy()

            phi_dec_values.append(phi_dec)
            time_steps.append(step)
            current_team_index = best_idx

            # The chosen path
            chosen_candidate = updated_candidates[best_idx]
            chosen_sequence  = chosen_candidate['sequence']
            chosen_path_world= chosen_candidate['path']
            # Now we do "pose optimization" for each offset in `offsets`
            offset_best_cost    = float('inf')
            offset_best         = None
            offset_best_state   = None
            offset_best_true    = float('inf')

            # Build a short ref traj from chosen path
            cpath_arr = np.array(chosen_path_world)
            if cpath_arr.shape[0]<2:
                # fallback if it's just a single point
                pass
            else:
                dist_arr = np.linalg.norm(cpath_arr - current_position, axis=1)
                closest  = np.argmin(dist_arr)
                trimmed  = cpath_arr[closest:]
                if trimmed.shape[0]<2:
                    trimmed = cpath_arr
            npts = trimmed.shape[0]
            wref = np.zeros((6,npts))
            wref[0,:] = trimmed[:,0]
            wref[1,:] = trimmed[:,1]
            wref[2,:] = 0.7
            # The 'time_step_control' we pass to lqr_control
            time_step_ctrl=0

            # pick out this candidate's index among updated_candidates
            # we want the same index used in 'percentages'
            # best_idx is that index
            for off in offsets:
                # pose_optimization returns weighted cost + true cost
                wcost, tcost, cand_copy, final_st = pose_optimization(
                    candidate_state=state,
                    lqr_Q=dummy_Q,
                    lqr_R=lqr_R,
                    world_ref_traj=wref,
                    time_step_control=time_step_ctrl,
                    step=step,
                    candidate_index=best_idx,
                    percentages=percentages,
                    offset=off
                )
                if wcost<offset_best_cost:
                    offset_best_cost  = wcost
                    offset_best_true  = tcost
                    offset_best       = off
                    offset_best_state = copy.deepcopy(final_st)

            # Once we pick the best offset, we commit to it:
            state = offset_best_state
            accumulated_cost += offset_best_true

        current_position = np.array([state.x_world, state.y_world])
        trajectory.append(current_position.copy())
        tracking_actual.append(np.array([state.x_world, state.y_world, state.z_world]))

        step += 1

    # Wrap-up
    print(f"\nSimulation done with offsets={offsets}. Steps={step}. Final pos={current_position}")
    print(f"Accumulated Cost (summing chosen pose costs) ~ {accumulated_cost:.3f}")
    return {
        'offsets': offsets,
        'steps':   step,
        'final_pos': current_position,
        'accumulated_cost': accumulated_cost,
        'phi_dec_values': phi_dec_values,
        'robot_pref_indices': np.array(robot_pref_inds),
        'human_pref_indices': np.array(human_pref_inds),
        'chosen_indices': np.array(chosen_indices),
        'final_true_costs': np.array(final_true_costs),
        'final_nominal_costs':np.array(final_nom_costs)
    }

# ----------------------------------------------------------------------------
# 10. RUN BOTH Single-Offset and Multi-Offset Cases and Compare
# ----------------------------------------------------------------------------
if __name__=='__main__':
    start_all = time.time()
    print("=========== RUN 1: Single offset [0.0] ===========")
    results_single = run_simulation(offsets_override=[0.0])
    print("=========== RUN 2: Multiple offsets [0.0, 0.02, 0.03] ===========")
    results_multi  = run_simulation(offsets_override=[0.0, 0.02, 0.03])

    print("\n===== COMPARISON =====")
    print(f"Single offset final steps: {results_single['steps']}, final pos={results_single['final_pos']}")
    print(f"Multi offset final steps:  {results_multi['steps']},  final pos={results_multi['final_pos']}")
    print(f"Single offset accumulated cost: {results_single['accumulated_cost']:.3f}")
    print(f"Multi offset  accumulated cost: {results_multi['accumulated_cost']:.3f}")

    print("\nElapsed total time:", time.time()-start_all)
