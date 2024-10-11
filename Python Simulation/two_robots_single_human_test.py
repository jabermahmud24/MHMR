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

start_time = time.time()

try:
    import cubic_spline_planner
except ImportError:
    raise

lqr_Q = 1000 * np.eye(6)
r_matrix = np.array([
    [30, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 30, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]
])
# Define the scalar
scalar = 1

# Multiply the matrix by the scalar
lqr_R = scalar * r_matrix
# lqr_R = 1 * np.eye(9)
dt = 0.1
show_animation = True

class State:

    def __init__(self, world_ref_traj, x_base=0, y_base=0.4):
        # DH parameters for the robot
        dh_params = np.array([
            [0.72, 0, 0, 0],
            [0.06, 0.117, -0.5 * pi, 0],
            [0, 0, 0.5 * pi, 1.57],
            [0.219 + 0.133, 0, 0.5 * pi, 0],
            [0, 0, -0.5 * pi, 0],
            [0.197 + 0.1245, 0, -0.5 * pi, 0],
            [0, 0, 0.5 * pi, 0],
            [0.1385 + 0.1665, 0, 0, 0]
        ])

        self.dh_params = dh_params

        self.x_base = x_base
        self.y_base = y_base
        self.yaw_base = 0
        xyz = np.array([
            [world_ref_traj[0, 0] - self.x_base],
            [world_ref_traj[1, 0] - self.y_base],
            [world_ref_traj[2, 0]]
        ])
        abc = np.array([world_ref_traj[3, 0], world_ref_traj[4, 0], world_ref_traj[5, 0]])
        robot = RobotSerial(dh_params)
        end = Frame.from_euler_3(abc, xyz)
        robot.inverse(end)
        self.theta = robot.axis_values.copy()
        self.yaw_base = self.theta[0]

        robot = RobotSerial(dh_params)
        f = robot.forward(self.theta)

        self.x_world = self.x_base + f.t_3_1.reshape([3, ])[0]
        self.y_world = self.y_base + f.t_3_1.reshape([3, ])[1]
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

def solve_dare(A, B, Q, R, world_ref_traj, n_sum, i, std_dev, state, world_ref_traj_without_noise, D, sigma):
    P = Q
    P_next = Q

    p_next = np.zeros((6, 1))
    c_next = 0

    horizon = 8

    for j in range(horizon - 1, -1, -1):
        M = la.inv(R + (B.T @ P_next @ B)) @ B.T
        P_plus = P_next.copy()
        P_next = P_next - P_next @ B @ M @ P_next + Q

        world_xyz_in_horizon_1 = world_ref_traj_without_noise[:, i + j + 1].reshape(-1, 1)
        world_xyz_in_horizon_2 = world_ref_traj_without_noise[:, i + j + 2].reshape(-1, 1)

        p_plus = p_next.copy()
        p_next = p_next + P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2)
        p_next = p_next - P_next @ B @ M @ P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + P_next @ (
                world_xyz_in_horizon_1 - world_xyz_in_horizon_2)
        p_next = p_next - P_next @ B @ M @ p_next

        noise_expectation = sigma @ (D.T @ P_plus @ D)
        trace_noise_expectation = np.trace(noise_expectation)
        c_next = c_next + (world_xyz_in_horizon_1 - world_xyz_in_horizon_2).T @ P_next @ (
                    world_xyz_in_horizon_1 - world_xyz_in_horizon_2)
        c_next = c_next + trace_noise_expectation
        c_next = c_next - (P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + p_next).T @ B @ M @ (
                    P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + p_next)
        c_next = c_next + 2 * (world_xyz_in_horizon_1 - world_xyz_in_horizon_2).T @ p_next

    return P_plus, p_plus, c_next, world_xyz_in_horizon_1, world_xyz_in_horizon_2, c_next, P_next, p_next

def dlqr(A, B, Q, R, world_ref_traj, n_sum, i, state, std_dev, world_ref_traj_without_noise, D, sigma):
    P, p, c, rt_c, rt_p, c_t, P_t, p_t = solve_dare(A, B, Q, R, world_ref_traj, n_sum, i, std_dev, state,
                                                     world_ref_traj_without_noise, D, sigma)
    M = la.inv(R + (B.T @ P @ B)) @ B.T

    state_error_world = np.array([
        state.x_world, state.y_world, state.z_world,
        state.yaw_world, state.pitch_world, state.roll_world
    ]).reshape(-1, 1) - world_ref_traj[:, i + 1].reshape(-1, 1)

    ustar = - M @ (P @ (state_error_world + (rt_c.reshape(-1, 1) - rt_p.reshape(-1, 1))) + p)

    return ustar, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t

def lqr_speed_steering_control(state, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, dh_params, i, std_dev,
                               world_ref_traj_without_noise, D, joint_angle_combination, sigma):
    A = np.eye(6)
    B, f = get_B(dh_params, state, joint_angle_combination)
    ustar, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t = dlqr(A, B, lqr_Q, lqr_R, world_ref_traj, n_sum,
                                                                            i, state, std_dev, world_ref_traj_without_noise,
                                                                            D, sigma)
    ustar_cost = ustar.T @ lqr_R @ ustar
    error_cost = state_error_world.T @ lqr_Q @ state_error_world

    return ustar, B, f, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t, ustar_cost, error_cost

def check_cost(state, lqr_Q, B, P, p, theta, joint_angle_combination, state_error_world, c, rt_c, rt_p, M, c_t, P_t, p_t):
    angle_change_cost = np.sum(np.abs(state.theta - joint_angle_combination))
    cost = state_error_world.T @ P_t @ state_error_world + 2 * state_error_world.T @ p_t + c_t
    total_cost = cost + angle_change_cost
    return total_cost, angle_change_cost, cost, state_error_world

def do_simulation(cx, cy, cz, cyaw, cpitch, croll, world_ref_traj, noise, D, std_dev, world_ref_traj_without_noise, sigma):

    n_sum = np.array([[0], [0], [0]], dtype=float)
    time_ind = np.arange(0.0, len(cx), 1).astype(int).reshape(1, -1)

    # Initialize states for both robots with different base positions
    state1 = State(world_ref_traj=world_ref_traj, x_base=0.4, y_base=0.0)  # Robot 1
    state2 = State(world_ref_traj=world_ref_traj, x_base=0.4, y_base=-0.05)   # Robot 2

    # Trajectory data for both robots
    x1, y1, z1 = [state1.x_world], [state1.y_world], [state1.z_world]
    yaw1, pitch1, roll1 = [state1.yaw_world], [state1.pitch_world], [state1.roll_world]
    x2, y2, z2 = [state2.x_world], [state2.y_world], [state2.z_world]
    yaw2, pitch2, roll2 = [state2.yaw_world], [state2.pitch_world], [state2.roll_world]

    ustar1 = [0] * 9
    ustar2 = [0] * 9

    best_costs1 = []
    angle_change_costs1 = []
    prev_costs1 = []
    values_for_offset_zero1 = []
    state_error_world_dataset1 = []
    ustar_dataset1 = []

    best_costs2 = []
    angle_change_costs2 = []
    prev_costs2 = []
    values_for_offset_zero2 = []
    state_error_world_dataset2 = []
    ustar_dataset2 = []

    for i in range(len(cx) + 1):

        if i == len(cx) - 10:
            break
        if i >= 1:
            n_sum = n_sum + noise[:, i - 1].reshape(-1, 1)

        # Update world_ref_traj with accumulated noise
        world_ref_traj[:, i][0] = np.array([[(world_ref_traj[:, i][0] + n_sum[0])]])
        world_ref_traj[:, i][1] = np.array([[(world_ref_traj[:, i][1] + n_sum[1])]])
        world_ref_traj[:, i][2] = np.array([[(world_ref_traj[:, i][2] + n_sum[2])]])

        CT = 1
        PO = 0  # Pose Optimization
        if CT == 1:
            # For Robot 1
            theta1 = state1.theta
            joint_angle_combination1 = theta1
            ustar1, B1, f1, P1, p1, c1, state_error_world1, c_t1, rt_c1, rt_p1, M1, P_t1, p_t1, ustar_cost1, error_cost1 = lqr_speed_steering_control(
                state1, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, state1.dh_params, i, std_dev,
                world_ref_traj_without_noise, D, joint_angle_combination1, sigma)
            total_cost1, angle_change_cost1, cost1, state_error_world1 = check_cost(
                state1, lqr_Q, B1, P1, p1, theta1, joint_angle_combination1, state_error_world1, c1, rt_c1, rt_p1,
                M1, c_t1, P_t1, p_t1)
            best_costs1.append(total_cost1)
            angle_change_costs1.append(angle_change_cost1)
            prev_costs1.append(cost1)
            state_error_world_dataset1.append(error_cost1)
            ustar_dataset1.append(ustar_cost1)

            # For Robot 2
            theta2 = state2.theta
            joint_angle_combination2 = theta2
            ustar2, B2, f2, P2, p2, c2, state_error_world2, c_t2, rt_c2, rt_p2, M2, P_t2, p_t2, ustar_cost2, error_cost2 = lqr_speed_steering_control(
                state2, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, state2.dh_params, i, std_dev,
                world_ref_traj_without_noise, D, joint_angle_combination2, sigma)
            total_cost2, angle_change_cost2, cost2, state_error_world2 = check_cost(
                state2, lqr_Q, B2, P2, p2, theta2, joint_angle_combination2, state_error_world2, c2, rt_c2, rt_p2,
                M2, c_t2, P_t2, p_t2)
            best_costs2.append(total_cost2)
            angle_change_costs2.append(angle_change_cost2)
            prev_costs2.append(cost2)
            state_error_world_dataset2.append(error_cost2)
            ustar_dataset2.append(ustar_cost2)

        if PO == 1:
            # Pose Optimization for Robot 1
            costs_and_combinations1 = []
            theta1 = state1.theta
            joint_angle_combination1 = theta1
            robot1 = RobotSerial(state1.dh_params)

            ustar1, B1, f1, P1, p1, c1, state_error_world1, c_t1, rt_c1, rt_p1, M1, P_t1, p_t1, ustar_cost1, error_cost1 = lqr_speed_steering_control(
                state1, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, state1.dh_params, i, std_dev,
                world_ref_traj_without_noise, D, joint_angle_combination1, sigma)
            total_cost1, angle_change_cost1, cost1, state_error_world1 = check_cost(
                state1, lqr_Q, B1, P1, p1, theta1, joint_angle_combination1, state_error_world1, c1, rt_c1, rt_p1,
                M1, c_t1, P_t1, p_t1)
            costs_and_combinations1.append(
                (total_cost1, joint_angle_combination1, ustar1, B1, angle_change_cost1, cost1, state_error_world1))
            values_for_offset_zero1.append(total_cost1)

            # Pose Optimization for Robot 2
            costs_and_combinations2 = []
            theta2 = state2.theta
            joint_angle_combination2 = theta2
            robot2 = RobotSerial(state2.dh_params)

            ustar2, B2, f2, P2, p2, c2, state_error_world2, c_t2, rt_c2, rt_p2, M2, P_t2, p_t2, ustar_cost2, error_cost2 = lqr_speed_steering_control(
                state2, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, state2.dh_params, i, std_dev,
                world_ref_traj_without_noise, D, joint_angle_combination2, sigma)
            total_cost2, angle_change_cost2, cost2, state_error_world2 = check_cost(
                state2, lqr_Q, B2, P2, p2, theta2, joint_angle_combination2, state_error_world2, c2, rt_c2, rt_p2,
                M2, c_t2, P_t2, p_t2)
            costs_and_combinations2.append(
                (total_cost2, joint_angle_combination2, ustar2, B2, angle_change_cost2, cost2, state_error_world2))
            values_for_offset_zero2.append(total_cost2)

            # Optimization offsets (adjust as needed)
            offsets = [-0.02, -0.015, -0.01, -0.005, -0.025, -0.03, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
            if i % 5 == 0:
                # For Robot 1
                if len(joint_angle_combination1) == 8:
                    for offset in offsets:
                        joint_angle_combination1[6] += offset
                        joint_angle_combination1[7] += offset
                        joint_angle_combination1[5] += offset
                        ustar1, B1, f1, P1, p1, c1, state_error_world1, c_t1, rt_c1, rt_p1, M1, P_t1, p_t1, ustar_cost1, error_cost1 = lqr_speed_steering_control(
                            state1, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, state1.dh_params, i, std_dev,
                            world_ref_traj_without_noise, D, joint_angle_combination1, sigma)
                        total_cost1, angle_change_cost1, cost1, state_error_world1 = check_cost(
                            state1, lqr_Q, B1, P1, p1, theta1, joint_angle_combination1, state_error_world1, c1,
                            rt_c1, rt_p1, M1, c_t1, P_t1, p_t1)
                        costs_and_combinations1.append(
                            (total_cost1, joint_angle_combination1.copy(), ustar1, B1, angle_change_cost1, cost1,
                             state_error_world1))

                # For Robot 2
                if len(joint_angle_combination2) == 8:
                    for offset in offsets:
                        joint_angle_combination2[6] += offset
                        joint_angle_combination2[7] += offset
                        joint_angle_combination2[5] += offset
                        ustar2, B2, f2, P2, p2, c2, state_error_world2, c_t2, rt_c2, rt_p2, M2, P_t2, p_t2, ustar_cost2, error_cost2 = lqr_speed_steering_control(
                            state2, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, state2.dh_params, i, std_dev,
                            world_ref_traj_without_noise, D, joint_angle_combination2, sigma)
                        total_cost2, angle_change_cost2, cost2, state_error_world2 = check_cost(
                            state2, lqr_Q, B2, P2, p2, theta2, joint_angle_combination2, state_error_world2, c2,
                            rt_c2, rt_p2, M2, c_t2, P_t2, p_t2)
                        costs_and_combinations2.append(
                            (total_cost2, joint_angle_combination2.copy(), ustar2, B2, angle_change_cost2, cost2,
                             state_error_world2))

                # Select best cost and corresponding parameters for Robot 1
                best_cost1, joint_angle_combination1, ustar1, B1, angle_change_cost1, prev_cost1, state_error_world1 = min(
                    costs_and_combinations1, key=lambda x: x[0])
                best_costs1.append(best_cost1)
                angle_change_costs1.append(angle_change_cost1)
                prev_costs1.append(prev_cost1)
                state_error_world_dataset1.append(state_error_world1)
                ustar_dataset1.append(ustar_cost1)

                # Select best cost and corresponding parameters for Robot 2
                best_cost2, joint_angle_combination2, ustar2, B2, angle_change_cost2, prev_cost2, state_error_world2 = min(
                    costs_and_combinations2, key=lambda x: x[0])
                best_costs2.append(best_cost2)
                angle_change_costs2.append(angle_change_cost2)
                prev_costs2.append(prev_cost2)
                state_error_world_dataset2.append(state_error_world2)
                ustar_dataset2.append(ustar_cost2)

        # Update states
        state1, ee_pose1 = update(state1, ustar1, f1, state1.dh_params, B1, joint_angle_combination1)
        state2, ee_pose2 = update(state2, ustar2, f2, state2.dh_params, B2, joint_angle_combination2)

        x1.append(state1.x_world)
        y1.append(state1.y_world)
        z1.append(state1.z_world)
        yaw1.append(state1.yaw_world)
        pitch1.append(state1.pitch_world)
        roll1.append(state1.roll_world)

        x2.append(state2.x_world)
        y2.append(state2.y_world)
        z2.append(state2.z_world)
        yaw2.append(state2.yaw_world)
        pitch2.append(state2.pitch_world)
        roll2.append(state2.roll_world)

        if i % 1 == 0 and show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "or", label="Reference Trajectory with Noise")
            plt.plot(x1, y1, "ob", label="Robot 1 Trajectory")
            plt.plot(x2, y2, "og", label="Robot 2 Trajectory")
            plt.plot(state1.x_base, state1.y_base, 'bs', label="Robot 1 Base")
            plt.plot(state2.x_base, state2.y_base, 'gs', label="Robot 2 Base")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.0001)
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.show()

    return x1, y1, z1, yaw1, pitch1, roll1, x2, y2, z2, yaw2, pitch2, roll2, PO, CT, best_costs1, angle_change_costs1, prev_costs1, values_for_offset_zero1, state_error_world_dataset1, ustar_dataset1, best_costs2, angle_change_costs2, prev_costs2, values_for_offset_zero2, state_error_world_dataset2, ustar_dataset2

def main():
    # ax = [0, 0.25, 0.5, 0.4, 0.25, 0.45, 0.65, 0.85, 0.75, 0.65, 0.75, 1.75,
    #       2.5, 3.25, 3.5, 3.75, 4, 4.25]
    # ay = [0, 0, 0.05, 0.35, 0.45, 0.35, 0, 0, 0, 0, -0.07, -0.15, -0.35,
    #       -0.45, -0.35, -0.15, 0, 0.1]
    # az = [0.9, 0.94, 0.98, 1.02, 1.06, 1.1, 1.14, 1.14, 1.12, 1.09, 1.1,
    #       1.07, 1.03, 0.99, 0.95, 0.96, 1.02, 1.03]

    ax = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25]
    ay = [0, 0.3, 0, -0.3, 0, 0.5, 0 ,-0.5 ,0, 0.7, 0, -0.7, 0, 0.2, 0, -0.2, -0.15, -0.1]
    # ay = [0, 0, 0, 0, 0.35, 0.7, 1 ,1 ,1, 1, 1, 1, 1, 1, 1, 0.7, 0.35, 0]

    # az = [0.9,0.94,0.98,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.06,1.04,1.03,1.01,0.99, 0.98, 0.99, 1.01]
    # az = [0.9,0.94,0.98,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.06,1.04,1.03,1.01,0.99, 0.98, 0.99, 1.01]
    az = [0.9,0.94,0.98,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95, 0.96, 1.02,1.03]
    


    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)
    _, cz, _, _, _ = cubic_spline_planner.calc_spline_course(
        ax, az, ds=0.1)

    cx1 = np.linspace(0, 1, len(cx))
    cx2 = np.linspace(0, 1, 509)
    f = interp1d(cx1, cx)
    cx = f(cx2)

    cy1 = np.linspace(0, 1, len(cy))
    cy2 = np.linspace(0, 1, 509)
    f = interp1d(cy1, cy)
    cy = f(cy2)

    cyaw1 = np.linspace(0, 1, len(cyaw))
    cyaw2 = np.linspace(0, 1, 509)
    f = interp1d(cyaw1, cyaw)
    cyaw = f(cyaw2)

    cz1 = np.linspace(0, 1, len(cz))
    cz2 = np.linspace(0, 1, 509)
    f = interp1d(cz1, cz)
    cz = f(cz2)

    cpitch = np.zeros(len(cx))
    croll = np.zeros(len(cx))
    world_ref_traj_without_noise = np.array([cx, cy, cz, cyaw, cpitch, croll])

    csv_filename = 'traj3_without_noise_reference_trajectory_data.csv'

    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['cx', 'cy', 'cz', 'cyaw', 'cpitch', 'croll'])

        for i in range(len(cx)):
            csv_writer.writerow([cx[i], cy[i], cz[i], cyaw[i], cpitch[i], croll[i]])

    cx = np.array(cx)
    mean = 0

    std_dev = 0.4

    np.random.seed(42)
    noise1 = np.random.normal(mean, std_dev, size=cx.shape)
    noise2 = np.random.normal(mean, std_dev, size=cx.shape)
    noise3 = np.random.normal(mean, std_dev, size=cx.shape)
    sigma = np.array([[0.015 * std_dev, 0, 0],
                      [0, 0.025 * std_dev, 0],
                      [0, 0, 0.015 * std_dev]])
    D = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])
    cx = cx + (noise1 * 0.015)
    cy = cy + (noise2 * 0.025)
    cz = cz + (noise3 * 0.015)
    noise = np.array([(noise1 * 0.015), (noise2 * 0.025), (noise3 * 0.015)])
    cpitch = np.zeros(len(cx))
    croll = np.zeros(len(cx))
    world_ref_traj = np.array([cx, cy, cz, cyaw, cpitch, croll])

    csv_filename = 'traj3_reference_trajectory_data.csv'

    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['cx', 'cy', 'cz', 'cyaw', 'cpitch', 'croll'])

        for i in range(len(cx)):
            csv_writer.writerow([cx[i], cy[i], cz[i], cyaw[i], cpitch[i], croll[i]])

    x1, y1, z1, yaw1, pitch1, roll1, x2, y2, z2, yaw2, pitch2, roll2, PO, CT, best_costs1, angle_change_costs1, prev_costs1, values_for_offset_zero1, state_error_world_dataset1, ustar_dataset1, best_costs2, angle_change_costs2, prev_costs2, values_for_offset_zero2, state_error_world_dataset2, ustar_dataset2 = do_simulation(
        cx, cy, cz, cyaw, cpitch, croll, world_ref_traj, noise, D, std_dev, world_ref_traj_without_noise, sigma)

    if PO == 1:
        # Save Robot 1 trajectory
        csv_filename = 'Robot1_PPO_500traj3_0.04_pose_optimized_tracking_trajectory_data.csv'
        with open(csv_filename, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['x', 'y', 'z', 'yaw', 'pitch', 'roll'])
            for i in range(len(x1)):
                csv_writer.writerow([x1[i], y1[i], z1[i], yaw1[i], pitch1[i], roll1[i]])

        # Save Robot 2 trajectory
        csv_filename = 'Robot2_PPO_500traj3_0.04_pose_optimized_tracking_trajectory_data.csv'
        with open(csv_filename, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['x', 'y', 'z', 'yaw', 'pitch', 'roll'])
            for i in range(len(x2)):
                csv_writer.writerow([x2[i], y2[i], z2[i], yaw2[i], pitch2[i], roll2[i]])

    if CT == 1:
        # Save Robot 1 trajectory
        csv_filename = 'Robot1_1000traj3_0.04_coupled_tracking_trajectory_data.csv'
        with open(csv_filename, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['x', 'y', 'z', 'yaw', 'pitch', 'roll'])
            for i in range(len(x1)):
                csv_writer.writerow([x1[i], y1[i], z1[i], yaw1[i], pitch1[i], roll1[i]])

        # Save Robot 2 trajectory
        csv_filename = 'Robot2_1000traj3_0.04_coupled_tracking_trajectory_data.csv'
        with open(csv_filename, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['x', 'y', 'z', 'yaw', 'pitch', 'roll'])
            for i in range(len(x2)):
                csv_writer.writerow([x2[i], y2[i], z2[i], yaw2[i], pitch2[i], roll2[i]])

    # Data saving for Robot 1
    data_to_write1 = []
    data_to_write1.append(["Header for opt_total_best_costs"])
    data_to_write1.extend([[total_cost] for total_cost in best_costs1])

    data_to_write1.append([])
    data_to_write1.append(["Header for opt_angle_change_costs"])
    data_to_write1.extend([[angle_change_cost] for angle_change_cost in angle_change_costs1])

    data_to_write1.append([])
    data_to_write1.append(["Header for opt_PREV_best_costs"])
    data_to_write1.extend([[cost] for cost in prev_costs1])

    data_to_write1.append([])
    data_to_write1.append(["Header for cost_offset_zero"])
    data_to_write1.extend([[total_cost_without] for total_cost_without in values_for_offset_zero1])

    data_to_write1.append([])
    data_to_write1.append(["Header for ustar"])
    data_to_write1.extend([[ustarcost] for ustarcost in ustar_dataset1])

    data_to_write1.append([])
    data_to_write1.append(["Header for error"])
    data_to_write1.extend([[state_error_world] for state_error_world in state_error_world_dataset1])

    # Data saving for Robot 2
    data_to_write2 = []
    data_to_write2.append(["Header for opt_total_best_costs"])
    data_to_write2.extend([[total_cost] for total_cost in best_costs2])

    data_to_write2.append([])
    data_to_write2.append(["Header for opt_angle_change_costs"])
    data_to_write2.extend([[angle_change_cost] for angle_change_cost in angle_change_costs2])

    data_to_write2.append([])
    data_to_write2.append(["Header for opt_PREV_best_costs"])
    data_to_write2.extend([[cost] for cost in prev_costs2])

    data_to_write2.append([])
    data_to_write2.append(["Header for cost_offset_zero"])
    data_to_write2.extend([[total_cost_without] for total_cost_without in values_for_offset_zero2])

    data_to_write2.append([])
    data_to_write2.append(["Header for ustar"])
    data_to_write2.extend([[ustarcost] for ustarcost in ustar_dataset2])

    data_to_write2.append([])
    data_to_write2.append(["Header for error"])
    data_to_write2.extend([[state_error_world] for state_error_world in state_error_world_dataset2])

    # Write data to CSV for Robot 1
    if CT == 1:
        with open('Robot1_1000traj3_0.04_coupled_candidate_cost.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_to_write1)
    if PO == 1:
        with open('Robot1_PPO_500traj3_0.04_PO_candidate_cost.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_to_write1)

    # Write data to CSV for Robot 2
    if CT == 1:
        with open('Robot2_1000traj3_0.04_coupled_candidate_cost.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_to_write2)
    if PO == 1:
        with open('Robot2_PPO_500traj3_0.04_PO_candidate_cost.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_to_write2)

if __name__ == '__main__':
    main()
