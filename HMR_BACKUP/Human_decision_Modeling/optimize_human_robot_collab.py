import os
import sys
import math
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.linalg import solve_discrete_are, inv
from scipy.interpolate import interp1d
from heapq import heappush, heappop

import pybullet as p
import pybullet_data

from get_inverse_kinematics import generate_new_joint_angles
from get_jacobian_function import compute_jacobian
from get_forward_kinematics import compute_end_effector_info


# Constants and Parameters
LQR_Q = 1000 * np.eye(6)
LQR_R = np.eye(9)
DT = 0.1

GRID_SIZE = (150, 150)
X_RANGE = np.linspace(0, 10, GRID_SIZE[0])
Y_RANGE = np.linspace(0, 10, GRID_SIZE[1])
X, Y = np.meshgrid(X_RANGE, Y_RANGE, indexing='ij')

START_POINT = np.array([2.8, 1.8])
TARGET_POINT = np.array([7, 4])

OBSTACLES = [
    {'center': np.array([5, 0.5]), 'radius': 1},
    {'center': np.array([5, 3]), 'radius': 1},
    {'center': np.array([5, 5.5]), 'radius': 1},
    {'center': np.array([5, 8]), 'radius': 1},
]

SIGMA = 0.1  # Positional uncertainty

# Prelec Function Parameters
GAMMA_ROBOT = 1
GAMMA_HUMAN = 0.4

# Utility Function Parameters
PHI_K_INITIAL = 1.0
ALPHA = 0.88
BETA_PRELEC = 1
C_RISK = 10

# Steps to Save Images
STEPS_TO_SAVE = [10, 30, 40, 50]

# Initialize Collision Probability Sigma Matrix
SIGMA_MATRIX = np.array([
    [0.015 * 0, 0, 0],
    [0, 0.025 * 0, 0],
    [0, 0, 0.015 * 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])


def prelec_weight(p, beta=BETA_PRELEC, gamma=GAMMA_HUMAN):
    """Prelec weighting function for probability p."""
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1 - epsilon)
    return np.exp(-beta * (-np.log(p)) ** gamma)


def value_function(D, alpha=ALPHA):
    """Value function v(D)."""
    return -D ** alpha


def utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_risk):
    """Utility function for decision-making."""
    U_own = phi_k * (value_function(D_own) - prelec_weight(p_risk_own) * C_RISK)
    U_adapt = (1 - phi_k) * (value_function(D_adapt) - prelec_weight(p_risk_adapt) * C_RISK)
    return U_own - U_adapt


def create_obstacle_map():
    """Create a binary obstacle map based on defined obstacles."""
    obstacle_map = np.zeros(GRID_SIZE)
    points = np.stack((X, Y), axis=-1)  # Shape: (grid_x, grid_y, 2)
    for obs in OBSTACLES:
        dist = np.linalg.norm(points - obs['center'], axis=-1)
        obstacle_map[dist <= obs['radius']] = 1
    return obstacle_map


def compute_collision_probability_map(obstacle_map, sigma):
    """Compute collision probability map using Gaussian filter."""
    pixel_size_x = X_RANGE[1] - X_RANGE[0]
    pixel_size_y = Y_RANGE[1] - Y_RANGE[0]
    sigma_pixels = [sigma / pixel_size_x, sigma / pixel_size_y]
    collision_prob_map = gaussian_filter(obstacle_map, sigma=sigma_pixels, mode='constant')
    return np.clip(collision_prob_map, 0, 1)


def compute_cost_map(collision_prob_map, gamma):
    """Compute cost map incorporating Prelec weighting."""
    w_p = prelec_weight(collision_prob_map, gamma=gamma)
    return w_p


def pos_to_idx(pos):
    """Convert continuous position to grid index."""
    idx_x = int(np.clip(pos[0] / 10 * (GRID_SIZE[0] - 1), 0, GRID_SIZE[0] - 1))
    idx_y = int(np.clip(pos[1] / 10 * (GRID_SIZE[1] - 1), 0, GRID_SIZE[1] - 1))
    return (idx_x, idx_y)


def compute_remaining_distance(path_coords, current_idx):
    """Compute remaining distance from current index to the end of the path."""
    path_segment = path_coords[current_idx:]
    if len(path_segment) < 2:
        return 0.0
    differences = path_segment[1:] - path_segment[:-1]
    return np.sum(np.linalg.norm(differences, axis=1))


def compute_travelled_distance(robot_positions, utility_values):
    """
    Calculate the cumulative distance the robot has traveled along its path when utility is negative.
    """
    cumulative_distance = 0.0
    distances = [0.0]
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


def dijkstra(cost_map, start_idx, end_idx):
    """Path planning using Dijkstra's algorithm."""
    visited = np.full(GRID_SIZE, False, dtype=bool)
    cost_to_come = np.full(GRID_SIZE, np.inf)
    parent = -np.ones(GRID_SIZE + (2,), dtype=int)
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
            if 0 <= nx < GRID_SIZE[0] and 0 <= ny < GRID_SIZE[1]:
                if not visited[nx, ny]:
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
    return path[::-1]


class State:
    """Class to represent the robot's state."""

    def __init__(self, world_ref_traj):
        self.x_base = 2.4
        self.y_base = 1.8
        self.yaw_base = 0.0

        desired_position = world_ref_traj[0, :2]
        example_end_effector_pose = [
            desired_position[0] - self.x_base,
            desired_position[1] - self.y_base,
            world_ref_traj[0, 2],
            world_ref_traj[0, 3],
            world_ref_traj[0, 4],
            world_ref_traj[0, 5]
        ]

        example_initial_theta = [0] * 7
        new_joint_angles = generate_new_joint_angles(
            end_effector_pose=example_end_effector_pose,
            initial_theta=example_initial_theta,
            model_path='Pose Optimization/nadam_cvae_model_with_pose.pth',
            urdf_path='fetch.urdf',
            n_samples=1
        )

        self.theta = new_joint_angles.flatten()
        print('Inverse Kinematics Theta:', self.theta)

        self.yaw_base = self.theta[0]

        ee_pos, ee_euler = compute_end_effector_info(self.theta)
        print("End Effector Position:", ee_pos[0])

        self.x_world = self.x_base + ee_pos[0]
        self.y_world = self.y_base + ee_pos[1]
        self.z_world = ee_pos[2]
        self.roll_world, self.pitch_world, self.yaw_world = ee_euler

        # Body frame is identical to world frame initially
        self.x_body = self.x_world
        self.y_body = self.y_world
        self.z_body = self.z_world
        self.roll_body = self.roll_world
        self.pitch_body = self.pitch_world
        self.yaw_body = self.yaw_world


def get_B_matrix(state, joint_angle_combination, dt=DT):
    """Compute the B matrix based on the current state and joint angles."""
    jacobian = compute_jacobian(joint_angle_combination)
    print("Jacobian:", jacobian)

    B = np.zeros((6, 9))
    B[:, 2:9] = jacobian * dt

    cos_yaw = math.cos(state.yaw_base)
    sin_yaw = math.sin(state.yaw_base)

    B[0, 0] = dt * cos_yaw
    B[1, 0] = dt * sin_yaw
    B[0, 1] = -dt * sin_yaw
    B[1, 1] = dt * cos_yaw

    return B


def dlqr(A, B, Q, R, P_next, state_error):
    """Solve the discrete-time LQR controller."""
    M = inv(R + B.T @ P_next @ B) @ B.T
    u_star = -M @ P_next @ state_error
    return u_star


def solve_dare(A, B, Q, R):
    """Solve the Discrete-time Algebraic Riccati Equation (DARE)."""
    return solve_discrete_are(A, B, Q, R)


def update_state(state, u_star, B, dt=DT):
    """Update the robot's state based on the control input."""
    # Update joint angles
    state.theta += (dt * u_star[2:]).flatten()
    state.theta = state.theta.astype(float)

    # Update yaw base
    state.yaw_base += u_star[1] * dt
    state.yaw_base = (state.yaw_base + np.pi) % (2 * np.pi) - np.pi  # Normalize between [-π, π]

    # Update base position
    state.x_base += u_star[0] * dt * math.cos(state.yaw_base) - u_star[1] * dt * math.sin(state.yaw_base)
    state.y_base += u_star[0] * dt * math.sin(state.yaw_base) + u_star[1] * dt * math.cos(state.yaw_base)

    # Compute end-effector position and orientation
    ee_pos, ee_euler = compute_end_effector_info(state.theta)
    state.x_world = state.x_base + ee_pos[0]
    state.y_world = state.y_base + ee_pos[1]
    state.z_world = ee_pos[2]
    state.roll_world, state.pitch_world, state.yaw_world = ee_euler

    # Update body frame (assuming identical to world frame)
    state.x_body = state.x_world
    state.y_body = state.y_world
    state.z_body = state.z_world
    state.roll_body = state.roll_world
    state.pitch_body = state.pitch_world
    state.yaw_body = state.yaw_world

    # Create ee_pose as a 1D array
    ee_pose = np.array([
        state.x_world,
        state.y_world,
        state.z_world,
        state.yaw_world,
        state.pitch_world,
        state.roll_world
    ])

    return state, ee_pose


def main():
    # Precompute obstacle and collision probability maps
    obstacle_map = create_obstacle_map()
    collision_prob_map = compute_collision_probability_map(obstacle_map, SIGMA)

    # Compute cost maps for robot and human
    cost_map_robot = compute_cost_map(collision_prob_map, GAMMA_ROBOT) + obstacle_map * 1e6
    cost_map_human = compute_cost_map(collision_prob_map, GAMMA_HUMAN) + obstacle_map * 1e6

    # Initialize robot's position and state
    robot_position = START_POINT.copy()
    robot_positions = [robot_position.copy()]

    # Initialize simulation parameters
    step = 0
    max_steps = 500
    phi_k = PHI_K_INITIAL

    utility_values = []
    D_adapt_values = []
    phi_k_values = []

    # Setup plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()  # Interactive mode

    state = None  # Will be initialized in the loop

    while np.linalg.norm(robot_position - TARGET_POINT) > 0.1 and step < max_steps:
        # Update grid indices
        start_idx = pos_to_idx(robot_position)
        end_idx = pos_to_idx(TARGET_POINT)

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

        # Compute travelled distances based on utility
        travelled_distances = compute_travelled_distance(robot_positions, utility_values)
        phi_k = PHI_K_INITIAL * np.exp(-(travelled_distances[-1] / 100))
        phi_k_values.append(phi_k)
        print('Human behavior phi_k:', phi_k)

        # Update indices for next step positions
        idx_next_robot = pos_to_idx(next_pos_robot)
        idx_next_human = pos_to_idx(next_pos_human)
        p_risk_own = collision_prob_map[idx_next_robot]
        p_risk_adapt = collision_prob_map[idx_next_human]

        # Compute utility
        decision_value = utility(phi_k, D_own, D_adapt, p_risk_own, p_risk_adapt, C_RISK)
        utility_values.append(decision_value)
        print('Decision value:', decision_value)

        # Adjust next positions based on previous steps
        if step > 2 and np.array_equal(robot_positions[-1], robot_positions[-2]):
            if len(path_coords_robot) > 2 and len(path_coords_human) > 2:
                next_pos_robot = path_coords_robot[2]
                next_pos_human = path_coords_human[2]

        # Choose desired position based on utility
        desired_position = next_pos_human if decision_value >= 0 else next_pos_robot
        world_ref_traj = path_coords_human if decision_value >= 0 else path_coords_robot

        # Embed additional values into the trajectory if needed
        embedding = np.tile(np.array([0.5, 0.2, 0, 0]), (world_ref_traj.shape[0], 1))
        world_ref_traj = np.hstack((world_ref_traj, embedding))

        # Initialize state
        if step == 0:
            state = State(world_ref_traj=world_ref_traj)
            joint_angle_combination = state.theta.copy()

        # Compute B matrix
        B = get_B_matrix(state, joint_angle_combination)

        # Define A matrix for LQR (identity)
        A = np.eye(6)

        # Compute state error
        state_error = np.array([
            state.x_world,
            state.y_world,
            state.z_world,
            state.yaw_world,
            state.pitch_world,
            state.roll_world
        ]).reshape(-1, 1) - world_ref_traj[0, :6].reshape(-1, 1)

        # Solve DARE to get P
        P = solve_dare(A, B, LQR_Q, LQR_R)

        # Compute LQR control input
        u_star = dlqr(A, B, LQR_Q, LQR_R, P, state_error)
        print("Control Input u_star:", u_star)

        # Update state with control input
        state, ee_pose = update_state(state, u_star, B)

        # Update robot's world position with scalar values
        robot_position[0] = ee_pose[0]  # x_world
        robot_position[1] = ee_pose[1]  # y_world
        robot_positions.append(robot_position.copy())

        # Plotting
        ax.clear()
        ax.imshow(np.flipud(obstacle_map.T), extent=(0, 10, 0, 10), cmap='gray_r', alpha=0.5)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Robot Navigation')
        ax.plot(START_POINT[0], START_POINT[1], 'go', markersize=8, label='Start Point A')
        ax.plot(TARGET_POINT[0], TARGET_POINT[1], 'bo', markersize=8, label='Target Point B')
        ax.plot(path_coords_robot[:, 0], path_coords_robot[:, 1], 'b--', label="Robot's Planned Path")
        ax.plot(path_coords_human[:, 0], path_coords_human[:, 1], 'g--', label="Human's Planned Path")
        ax.plot(np.array(robot_positions)[:, 0], np.array(robot_positions)[:, 1], 'r-', linewidth=2, label='Robot Trajectory')
        ax.legend()
        ax.grid(True)

        # Save plots for specific steps
        if step in STEPS_TO_SAVE:
            plt.savefig(f"robot_navigation_step_{step}.png")

        plt.pause(0.1)
        step += 1

    plt.ioff()
    plt.show()

    # Plot Utility Values
    utility_values = np.array(utility_values)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(utility_values)), utility_values, 'm-', linewidth=2, label='Utility Value')
    plt.fill_between(range(len(utility_values)), utility_values, 0, where=(utility_values >= 0),
                     color='green', alpha=0.3, label='Positive Utility')
    plt.fill_between(range(len(utility_values)), utility_values, 0, where=(utility_values < 0),
                     color='blue', alpha=0.3, label='Negative Utility')
    plt.xlabel('Decision Step')
    plt.ylabel('Utility Value')
    plt.title('Utility Value Over Decision Steps')
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True)
    plt.legend()
    plt.savefig("utility_value_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
