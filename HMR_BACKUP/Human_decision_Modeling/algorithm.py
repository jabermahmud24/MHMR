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
from get_inverse_kinematics import generate_new_joint_angles

# from get_end_effector_body_frame import transform_pose_world_to_body
import numpy as np
import os

import pybullet as p
import pybullet_data

from get_jacobian_function import compute_jacobian

from get_forward_kinematics import compute_end_effector_info


class State:

    def __init__(self,world_ref_traj):   

    
        self.x_base = 2.8
        self.y_base = 2.2
        self.yaw_base = 0

        example_end_effector_pose = [desired_position[0]-self.x_base, desired_position[1]-self.y_base, world_ref_traj[0,2],
                                     world_ref_traj[0, 3], world_ref_traj[0, 4], world_ref_traj[0, 5]]
        
        example_initial_theta = [0, 0, 0, 0, 0, 0, 0]

        n_samples = 1

        new_joint_angles = generate_new_joint_angles(
        end_effector_pose=example_end_effector_pose,
        initial_theta=example_initial_theta,
        model_path='Pose Optimization/nadam_cvae_model_with_pose.pth',
        urdf_path='fetch.urdf',
        n_samples=n_samples)

        self.theta = new_joint_angles

        self.theta = np.insert(self.theta, 0, 0, axis=1)

        print(example_end_effector_pose)
        print(self.theta)

        self.theta = self.theta.flatten()


        self.yaw_base = self.theta[0]

        ee_pos, ee_euler, ee_pos_body, ee_euler_body = compute_end_effector_info(self.theta)



        self.x_world = self.x_base + ee_pos_body[0]
        self.y_world = self.y_base + ee_pos_body[1]
        self.z_world = ee_pos_body[2]
        self.roll_world = ee_euler_body[0]
        self.pitch_world = ee_euler_body[1]
        self.yaw_world = ee_euler_body[2]

        self.x_body = ee_pos_body[0]
        self.y_body = ee_pos_body[1]
        self.z_body = ee_pos_body[2]
        self.yaw_body = ee_euler_body[2]
        self.pitch_body = ee_euler_body[1]
        self.roll_body = ee_euler_body[0]


def lqr_speed_steering_control(state, lqr_Q, lqr_R, world_ref_traj, n_sum, i, std_dev, world_ref_traj_without_noise_test, D, joint_angle_combination, sigma):
    
    A = np.eye(6)

    B=  get_B(state, joint_angle_combination)



    ustar, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t = dlqr(A, B, lqr_Q, lqr_R, world_ref_traj, n_sum, i, state, std_dev, world_ref_traj_without_noise_test, D, sigma)

    return ustar



def get_B(state, joint_angle_combination):

    jacobian = compute_jacobian(joint_angle_combination)

    B = np.zeros((6, 9))

    B[:, 1:9] = jacobian * dt

    B[0, 0] = dt * math.cos(state.yaw_base)
    B[1, 0] = dt * math.sin(state.yaw_base)



    return B

def dlqr(A, B, Q, R, world_ref_traj, n_sum, i, state, std_dev,  world_ref_traj_without_noise, D, sigma):

    P, p, c, rt_c, rt_p, c_t, P_t, p_t = solve_dare(A, B, Q, R, world_ref_traj, n_sum, i, std_dev, state, world_ref_traj_without_noise, D, sigma)
    
    M = la.inv(R + (B.T @ P @ B)) @ B.T


    state_error_world = np.array([state.x_world, state.y_world, state.z_world, state.yaw_world, state.pitch_world, state.roll_world]).reshape(-1,1) - world_ref_traj[i].reshape(-1,1) 

    ustar = - M @ (P @ (state_error_world + (rt_c.reshape(-1,1) -  rt_p.reshape(-1,1))) + p )

    ustar[1] = ustar[1] % (2 * math.pi)

    return ustar, P, p , c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t


def solve_dare(A, B, Q, R, world_ref_traj, n_sum,i, std_dev, state, world_ref_traj_without_noise, D, sigma):

    P = Q
    P_next = Q

    p = np.array([[0], [0],[0],[0],[0],[0]])
    p_next = np.array([[0], [0],[0],[0],[0],[0]])
    
    c = 0
    c_next = 0
    
    horizon = 1

    noise_expectation = sigma @ (D.T @ P_next @ D)
    trace_noise_expectation = np.trace(noise_expectation)
        
    for j in range(horizon-1,-1,-1): 

        M = la.inv(R + (B.T @ P_next @ B)) @ B.T
        P_plus = P_next.copy()
        P_next = P_next - P_next @ B @ M @ P_next + Q


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





def update(state, ustar, B, joint_angle_combination):  
    state.theta = state.theta + (dt * ustar[1:].reshape(1,-1))
    state.theta = state.theta.astype(float)
    state.yaw_base += ustar[1] * dt
    state.yaw_base = float(state.yaw_base)
    
    state.yaw_base = state.yaw_base % (2 * math.pi)
   
    state.x_base += (ustar[0] * dt * math.cos(state.yaw_base))
    state.x_base = float(state.x_base)
    state.y_base += (ustar[0] * dt * math.sin(state.yaw_base)) 
    state.y_base = float(state.y_base)


    state.theta = state.theta.flatten()
    ee_pos, ee_euler, ee_pos_body, ee_euler_body = compute_end_effector_info(state.theta)


    state.x_body = ee_pos_body[0]
    state.y_body = ee_pos_body[1]
    state.z_body = ee_pos_body[2]
    state.yaw_body = ee_euler_body[2]
    state.pitch_body = ee_euler_body[1]
    state.roll_body = ee_euler_body[0]
    
    state.x_world = state.x_base + ee_pos_body[0]
    state.y_world = state.y_base + ee_pos_body[1]

    
    state.z_world = ee_pos_body[2]
    state.roll_world = ee_euler_body[0]
    state.pitch_world = ee_euler_body[1]
    state.yaw_world = ee_euler_body[2]


    ee_pose = np.array([[state.x_world], [state.y_world], [state.z_world], [state.yaw_world], [state.pitch_world], [state.roll_world]]) 

    
    return state, ee_pose


lqr_Q = 1000*np.eye(6)
lqr_R = 1*np.eye(9)
dt = 0.1



world_ref_traj_without_noise_test = path_coords_human if decision_value >= 0 else path_coords_robot

values_to_embed = np.array([0.5, 0.0, 0, 0])


embedding = np.tile(values_to_embed, (world_ref_traj_without_noise_test.shape[0], 1))


world_ref_traj_without_noise_test = np.hstack((world_ref_traj_without_noise_test, embedding))


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



i = 0
    

ustar= lqr_speed_steering_control(state, lqr_Q, lqr_R, world_ref_traj, n_sum, i, std_dev, world_ref_traj_without_noise_test, D, joint_angle_combination, sigma)
state, ee_pose = update(state, ustar, B,joint_angle_combination)    


