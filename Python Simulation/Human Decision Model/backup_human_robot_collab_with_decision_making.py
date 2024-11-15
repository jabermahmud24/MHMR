import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from heapq import heappush, heappop
from scipy.linalg import solve_discrete_are
import copy
import math
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from visual_kinematics.RobotSerial import *
from visual_kinematics.examples.inverse import *
import subprocess
import time
import numpy as np

lqr_Q = 1000*np.eye(6)
lqr_R = 0.1*np.eye(9)
dt = 0.1


# Robot Model

# dh_params = np.array([[0.72,0,0,0],
#                           [0, 0, -0.5*pi, 0],
#                           [0, 0, 0.5*pi, 0],
#                           [0, 0.219,  0.5 * pi, 0],
#                           [0, 0, -0.5 * pi, 0],
#                           [0, 0.197, 0.5 * pi, 0],
#                           [0, 0, -0.5 * pi,0],
#                           [0,.1385, 0, 0]])


dh_params = np.array([[0.72,0,0,0],
                          [0.06,0.117, -0.5*pi, 0],
                          [0, 0, 0.5*pi, 1.57],
                          [0.219+0.133, 0,  0.5 * pi, 0],
                          [0, 0, -0.5 * pi, 0],
                          [0.197+0.1245, 0, -0.5 * pi, 0],
                          [0, 0, +0.5 * pi,0],
                          [0.1385+0.1665,0, 0, 0]])



# Environment parameters
grid_size = (150, 150)
x_range = np.linspace(0, 10, grid_size[0])
y_range = np.linspace(0, 10, grid_size[1])
X, Y = np.meshgrid(x_range, y_range, indexing='ij')

# Start and end points
# A = np.array([0.9, 0.1])   # Starting point A
A = np.array([2.5, 1.4])   # Starting point A
B = np.array([7, 4])       # Target point B

# Obstacles (circles)
obstacles = [
    {'center': np.array([5, 0.5]),   'radius': 1},
    {'center': np.array([5, 3]), 'radius': 1},
    {'center': np.array([5, 5.5]),   'radius': 1},
    {'center': np.array([5, 8]),   'radius': 1},
]

# Robot's positional uncertainty (covariance matrix)
sigma = 0.1  # Standard deviation in x and y
# Sigma = np.array([[sigma**2, 0],
#                   [0, sigma**2]])  # Covariance matrix

# Prelec function parameter gamma values
gamma_robot = 1       # Robot's gamma
gamma_human = 0.4     # Human's gamma

# Parameters for utility function
phi_k = 1.0      # Human preference parameter
alpha = 0.88         # Value function parameter
beta_prelec = 1      # Prelec function parameter (usually set to 1)
gamma_prelec = gamma_human  # Prelec function curvature for utility calculation
C_risk = 10        # Cost associated with collision

# Prelec probability weighting function for utility
# def w(p, beta=beta_prelec, gamma=gamma_prelec):
#     epsilon = 1e-10
#     p = np.clip(p, epsilon, 1 - epsilon)
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

# """


class State:

    def __init__(self,world_ref_traj):   

        # dh_params = np.array([[0.72,0,0,0],
        #                   [0, 0, -0.5*pi, 0],
        #                   [0, 0, 0.5*pi, 0],
        #                   [0, 0.219,  0.5 * pi, 0],
        #                   [0, 0, -0.5 * pi, 0],
        #                   [0, 0.197, 0.5 * pi, 0],
        #                   [0, 0, -0.5 * pi,0],
        #                   [0,.1385, 0, 0]])


        dh_params = np.array([[0.72,0,0,0],
                          [0.06,0.117, -0.5*pi, 0],
                          [0, 0, 0.5*pi, 1.57],
                          [0.219+0.133, 0,  0.5 * pi, 0],
                          [0, 0, -0.5 * pi, 0],
                          [0.197+0.1245, 0, -0.5 * pi, 0],
                          [0, 0, +0.5 * pi,0],
                          [0.1385+0.1665,0, 0, 0]])
        

    
        self.x_base = 2.5
        self.y_base = 1.9
        # self.x_base = 0.1
        # self.y_base = 0.6
        self.yaw_base = 0
        xyz = np.array([[desired_position[0]-self.x_base], [desired_position[1]-self.y_base], [world_ref_traj[0,2]]])
        abc= np.array([world_ref_traj[0, 3], world_ref_traj[0, 4], world_ref_traj[0, 5]])

     


        robot = RobotSerial(dh_params)
        end = Frame.from_euler_3(abc, xyz)
        robot.inverse(end)
        self.theta = robot.axis_values.copy()


        self.yaw_base = self.theta[0]
       
     
        robot = RobotSerial(dh_params)
        f = robot.forward(self.theta)

        # print(f)

        x_world = self.x_base + f.t_3_1.reshape([3, ])[0]
        
        self.x_world = x_world
        # self.x_world = self.x_base + f.t_3_1.reshape([3, ])[0]*math.cos(self.yaw_base) - f.t_3_1.reshape([3, ])[1]*math.sin(self.yaw_base)
        self.y_world = self.y_base +f.t_3_1.reshape([3, ])[1]
        # self.y_world = self.y_base + f.t_3_1.reshape([3, ])[0]*math.sin(self.yaw_base) + f.t_3_1.reshape([3, ])[1]*math.cos(self.yaw_base)
        self.z_world = f.t_3_1.reshape([3, ])[2]
        self.yaw_world = f.euler_3[2]
        self.pitch_world = f.euler_3[1]
        self.roll_world = f.euler_3[0]
        


        self.x_body = f.t_3_1.reshape([3, ])[0]
        self.y_body = f.t_3_1.reshape([3, ])[1]
        self.z_body = f.t_3_1.reshape([3, ])[2]
        self.yaw_body = f.euler_3[2]
        self.pitch_body = f.euler_3[1]
        self.roll_body = f.euler_3[0]
   


def lqr_speed_steering_control(state, lqr_Q, lqr_R, world_ref_traj, n_sum, dh_params, i, std_dev, world_ref_traj_without_noise_test, D, joint_angle_combination, sigma):
    
    A = np.eye(6)

    B, f =  get_B(dh_params, state, joint_angle_combination)



    ustar, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t = dlqr(A, B, lqr_Q, lqr_R, world_ref_traj, n_sum, i, state, std_dev, world_ref_traj_without_noise_test, D, sigma)
    # ustar, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t = dlqr(A, B, lqr_Q, lqr_R, world_ref_traj, n_sum, i, state, std_dev, world_ref_traj_without_noise_test, D, sigma)

    return ustar, f
    # return ustar, B, f, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t, ustar_cost, error_cost


def get_B (dh_params, state, joint_angle_combination):

    robot = RobotSerial(dh_params)
    theta = joint_angle_combination
    f = robot.forward(theta)

    jacobian = []
    with open('jacobian_matrix1.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            jacobian.append([cell for cell in row])
    jacobian = np.array(jacobian, dtype= float) 

    # print(jacobian)
    
    B = np.zeros((6,9))
    B[:, 1:9] = jacobian * dt

    B[0,0] = dt * math.cos(state.yaw_base)
    B[1,0] = dt * math.sin(state.yaw_base)

    
    return B, f


def dlqr(A, B, Q, R, world_ref_traj, n_sum, i, state, std_dev,  world_ref_traj_without_noise, D, sigma):

    P, p, c, rt_c, rt_p, c_t, P_t, p_t = solve_dare(A, B, Q, R, world_ref_traj, n_sum, i, std_dev, state, world_ref_traj_without_noise, D, sigma)
    # In your main script or before calling lqr_speed_steering_control
    
    # In your lqr_speed_steering_control function, before calling dlqr
    M = la.inv(R + (B.T @ P @ B)) @ B.T
    # print("==========")
    # print(np.array([state.x_world, state.y_world, state.z_world, state.yaw_world, state.pitch_world, state.roll_world]).reshape(-1,1))
    # print(world_ref_traj[i].reshape(-1,1))
    # print("==========")


    state_error_world = np.array([state.x_world, state.y_world, state.z_world, state.yaw_world, state.pitch_world, state.roll_world]).reshape(-1,1) - world_ref_traj[i].reshape(-1,1) 
    # state_error_world = np.array([state.x_world, state.y_world, state.z_world, state.yaw_world, state.pitch_world, state.roll_world]).reshape(-1,1) - (world_ref_traj[:,i+1].reshape(-1,1) )

    ustar = - M @ (P @ (state_error_world + (rt_c.reshape(-1,1) -  rt_p.reshape(-1,1))) + p )

    # print("==========")
    # print(ustar)
    # print("==========")
    ustar[1] = ustar[1] % (2 * math.pi)
    if ustar[1] > math.pi:
        ustar[1] -= 2 * math.pi
    elif ustar[1] < -math.pi:
        ustar[1] += 2 * math.pi
   
    return ustar, P, p , c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t


def solve_dare(A, B, Q, R, world_ref_traj, n_sum,i, std_dev, state, world_ref_traj_without_noise, D, sigma):

    P = Q
    P_next = Q

    p = np.array([[0], [0],[0],[0],[0],[0]])
    p_next = np.array([[0], [0],[0],[0],[0],[0]])
    
    c = 0
    c_next = 0
    
    horizon = 8

    noise_expectation = sigma @ (D.T @ P_next @ D)
    trace_noise_expectation = np.trace(noise_expectation)
        
    for j in range(horizon-1,-1,-1): 

        M = la.inv(R + (B.T @ P_next @ B)) @ B.T
        P_plus = P_next.copy()
        P_next = P_next - P_next @ B @ M @ P_next + Q

        # print(world_ref_traj_without_noise[i+j][2])


        '''
             

        world_xyz_in_horizon_1 =([world_ref_traj_without_noise[i][0]], 
                                 [world_ref_traj_without_noise[i][1]], 
                                 [world_ref_traj_without_noise[i][2]] )  

        world_xyz_in_horizon_2 = ([world_ref_traj_without_noise[i+j][0]],
                                  [world_ref_traj_without_noise[i+j][1]],
                                  [world_ref_traj_without_noise[i+j][2]] ) 
        
        world_xyz_in_horizon_1_orientation = np.array([
            [world_ref_traj_without_noise[i][3]],
            [world_ref_traj_without_noise[i][4]],
            [world_ref_traj_without_noise[i][5]]
        ])

        world_xyz_in_horizon_1 = np.vstack((world_xyz_in_horizon_1, world_xyz_in_horizon_1_orientation))
        
        
        world_xyz_in_horizon_2_orientation = np.array([
            [world_ref_traj_without_noise[i+j][3]],
            [world_ref_traj_without_noise[i+j][4]],
            [world_ref_traj_without_noise[i+j][5]]
        ])

        world_xyz_in_horizon_2 = np.vstack((world_xyz_in_horizon_2, world_xyz_in_horizon_2_orientation))

        '''
        #BETTER

        world_xyz_in_horizon_1 =([world_ref_traj_without_noise[i+j][0]], 
                                 [world_ref_traj_without_noise[i+j][1]], 
                                 [world_ref_traj_without_noise[i+j][2]] )  

        world_xyz_in_horizon_2 = ([world_ref_traj_without_noise[i+j+1][0]],
                                  [world_ref_traj_without_noise[i+j+1][1]],
                                  [world_ref_traj_without_noise[i+j+1][2]] ) 
        
        world_xyz_in_horizon_1_orientation = np.array([
            [world_ref_traj_without_noise[i+j][3]],
            [world_ref_traj_without_noise[i+j][4]],
            [world_ref_traj_without_noise[i+j][5]]
        ])

        world_xyz_in_horizon_1 = np.vstack((world_xyz_in_horizon_1, world_xyz_in_horizon_1_orientation))
        
        
        world_xyz_in_horizon_2_orientation = np.array([
            [world_ref_traj_without_noise[i+j+1][3]],
            [world_ref_traj_without_noise[i+j+1][4]],
            [world_ref_traj_without_noise[i+j+1][5]]
        ])

        world_xyz_in_horizon_2 = np.vstack((world_xyz_in_horizon_2, world_xyz_in_horizon_2_orientation))


        
        p_plus = p_next.copy()
        p_next = p_next  + P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) 
        p_next = p_next - P_next @ B @ M @ P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) 
        p_next = p_next  - P_next @ B @ M @ p_next
            
        noise_expectation = sigma @ (D.T @ P_plus @ D)
        trace_noise_expectation = np.trace(noise_expectation)
    
        c_next = c_next + (world_xyz_in_horizon_1 - world_xyz_in_horizon_2).T @ P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2)
        
        c_next  = c_next + trace_noise_expectation
        
        c_next = c_next - (P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + p_next).T @ B @ M @ (P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + p_next)
        c_next = c_next + 2 * (world_xyz_in_horizon_1 - world_xyz_in_horizon_2).T @ p_next
    
    return P_plus, p_plus, c_next, world_xyz_in_horizon_1, world_xyz_in_horizon_2, c_next, P_next, p_next

def update(state, ustar, f, dh_params, B, joint_angle_combination):  
    state.theta = state.theta + (dt * ustar[1:].reshape(1,-1))
    state.theta = state.theta.astype(float)
    # print(state.theta)
    # print("================================================")
    state.yaw_base += ustar[1] * dt
    state.yaw_base = float(state.yaw_base)
    
    state.yaw_base = state.yaw_base % (2 * math.pi)
    if state.yaw_base > math.pi:
        state.yaw_base -= 2 * math.pi
    elif state.yaw_base < -math.pi:
        state.yaw_base += 2 * math.pi
    
    # print(ustar.shape)
    # print(ustar)
    
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
    
    # state.x_world = state.x_base + f.t_3_1.reshape([3, ])[0]*math.cos(state.yaw_base) - f.t_3_1.reshape([3, ])[1]*math.sin(state.yaw_base)
    state.x_world = state.x_base + f.t_3_1.reshape([3, ])[0]
    # state.y_world = state.y_base + f.t_3_1.reshape([3, ])[0]*math.sin(state.yaw_base) + f.t_3_1.reshape([3, ])[1]*math.cos(state.yaw_base)
    state.y_world = state.y_base + f.t_3_1.reshape([3, ])[1]
    state.z_world = f.t_3_1.reshape([3, ])[2]
    state.yaw_world =  f.euler_3[2]
    state.pitch_world =  f.euler_3[1]
    state.roll_world =  f.euler_3[0]

    ee_pose = np.array([[state.x_world], [state.y_world], [state.z_world], [state.yaw_world], [state.pitch_world], [state.roll_world]]) 
    
    return state, ee_pose

# """

# Main code
# Precompute obstacle map and collision probability map
obstacle_map = create_obstacle_map()
collision_prob_map = compute_collision_probability_map(obstacle_map, sigma)

# Precompute cost maps
cost_map_robot = compute_cost_map(collision_prob_map, gamma_robot) + obstacle_map * 1e6
cost_map_human = compute_cost_map(collision_prob_map, gamma_human) + obstacle_map * 1e6

# Initialize robot's starting position and state
robot_position = np.array(A)
robot_positions = [robot_position.copy()]

step = 0  # Initialize step counter
max_steps = 500  # Set a maximum number of steps to prevent infinite loops
fig, ax = plt.subplots(figsize=(8, 8))
plt.ion()  # Turn on interactive mode
cnt = 0
steps_to_save = [10, 30, 40, 50]  # Steps to save images

utility_values = []  # Collect utility values during simulation

D_adapt_values = []
phi_k_values = []


# Simulation loop
while np.linalg.norm(robot_position - B) > 0.1 and step < max_steps:
    cnt += 1
    print(cnt)
    

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

    #Current positions

    current_pos_robot = path_coords_robot[0]
    current_pos_human = path_coords_human[0]

    # Decision-making logic
    D_own = compute_remaining_distance(path_coords_robot, 0)
    D_adapt = compute_remaining_distance(path_coords_human, 0)
    D_adapt_values.append(D_adapt)
    # print(D_adapt_values)

    travelled_distances = compute_travelled_distance(robot_positions, utility_values)
    # travelled_distances = compute_travelled_distance(D_adapt_values, utility_values)
    print("---------------------")
    # print(travelled_distances[-1])
    # print("---------------------")

    phi_k = phi_k * np.exp(-(travelled_distances[-1]/100))


    # print('human behavior:', phi_k)


    phi_k_values.append(phi_k)



    # D_travelled_human = compute_travelled_distance(path)

    idx_next_robot = pos_to_idx(next_pos_robot)
    idx_next_human = pos_to_idx(next_pos_human)
    p_risk_own = collision_prob_map[idx_next_robot]
    # print("=====================")
    # print("True risk", p_risk_own)
    # print("=====================")
    p_risk_adapt = collision_prob_map[idx_next_human]
    decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk)
    utility_values.append(decision_value)
    # print('decision value:', decision_value)

    # print("---------------------")



    if cnt > 2:
        if (robot_positions[-1] == robot_positions[-2]).all():
            next_pos_robot = path_coords_robot[2]
            next_pos_human = path_coords_human[2]

    desired_position = next_pos_human if decision_value >= 0 else next_pos_robot
    world_ref_traj_without_noise_test = path_coords_human if decision_value >= 0 else path_coords_robot
    print(world_ref_traj_without_noise_test.shape)




    # Initial number of points
    num_points = len(world_ref_traj_without_noise_test)

    # Container for storing the reduced trajectories
    large_trajectories = []


    # Separate x and y coordinates
    cx, cy = world_ref_traj_without_noise_test[:, 0], world_ref_traj_without_noise_test[:, 1]
    
    # Create interpolation functions
    cx1 = np.linspace(0, 1, len(cx))
    cy1 = np.linspace(0, 1, len(cy))
    f_cx = interp1d(cx1, cx, kind='linear')
    f_cy = interp1d(cy1, cy, kind='linear')
    
    # Define the new number of points and the new linspace
    new_linspace = np.linspace(0, 1, num_points - 1)
    
    # Interpolate to reduce points
    new_cx = f_cx(new_linspace)
    new_cy = f_cy(new_linspace)
    
    # Combine the new trajectory
    new_traj = np.column_stack((new_cx, new_cy))
    large_trajectories.append(new_traj)
    
    # Update trajectory and point count for the next step
    world_ref_traj_without_noise_test = new_traj
    print(world_ref_traj_without_noise_test.shape)







    # print(world_ref_traj_without_noise_test)
    # Define the values you want to embed
    values_to_embed = np.array([0.8, 0.0, 0, 0])

    # Create a new array by repeating `values_to_embed` for each row
    # The number of rows is the same as in the original array
    embedding = np.tile(values_to_embed, (world_ref_traj_without_noise_test.shape[0], 1))

    # Concatenate the embedding to the original array along the columns (axis=1)
    world_ref_traj_without_noise_test = np.hstack((world_ref_traj_without_noise_test, embedding))

    # def lqr_speed_steering_control(state, lqr_Q, lqr_R, world_ref_traj, n_sum, dh_params, i, std_dev, world_ref_traj_without_noise_test, D, joint_angle_combination, sigma):

    world_ref_traj = world_ref_traj_without_noise_test
    n_sum = 0
    std_dev = 0
    # D = 0
    sigma = np.array([[0.015*std_dev, 0, 0], 
        [0, 0.025*std_dev, 0],
        [0, 0, 0.015*std_dev]])
    D = np.array([[0, 0, 0], 
        [0, 0, 0],
        [0, 0, 0],
        [0,0,0],
        [0,0,0],
        [0,0,0]])
    if step == 0:
        state = State(world_ref_traj=world_ref_traj)
        theta = state.theta
        joint_angle_combination = theta



    # for i in range(len(path_coords_human)+1):
    i = 0
        

    ustar, f = lqr_speed_steering_control(state, lqr_Q, lqr_R, world_ref_traj, n_sum, dh_params, i, std_dev, world_ref_traj_without_noise_test, D, joint_angle_combination, sigma)
    state, ee_pose = update(state, ustar, f, dh_params, B,joint_angle_combination)    


    robot_position[0] = ee_pose[0]
    robot_position[1] = ee_pose[1]
    num_points -= 1


    ### Algorithm: Update robot's state using LQR control ###

    '''
    ALGORITHM

    Get ee_pose

    Get robot_position which are the updated x_world and y_world.
    '''


    robot_positions.append(robot_position.copy())

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
    ax.grid(True)

    # # Save plot for specific steps
    # if step in steps_to_save:
    #     plt.savefig(f"robot_navigation_step_{step}.png")  # Save the image

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
# # Save the plot
# plt.savefig("utility_value_plot.png")

plt.show()
