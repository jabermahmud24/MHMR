
import matplotlib.pyplot as plt
import numpy as np
from visual_kinematics.RobotSerial import *
from visual_kinematics.examples.inverse import *



lqr_Q = 1000*np.eye(6)
lqr_R = 1*np.eye(9)
dt = 0.1


class State:

    def __init__(self,world_ref_traj ):   

        dh_params = np.array([[0.72,0,0,0],
                          [0.06,0.117, -0.5*pi, 0],
                          [0, 0, 0.5*pi, 1.57],
                          [0.219+0.133, 0,  0.5 * pi, 0],
                          [0, 0, -0.5 * pi, 0],
                          [0.197+0.1245, 0, -0.5 * pi, 0],
                          [0, 0, +0.5 * pi,0],
                          [0.1385+0.1665,0, 0, 0]])
        
    
        self.x_base = 0
        self.y_base = 0.4
        self.yaw_base = 0
        xyz = np.array([[desired_position[0]-self.x_base], [desired_position[1]-self.y_base], [world_ref_traj[2,0]]])
        abc= np.array([world_ref_traj[3,0], world_ref_traj[4,0], world_ref_traj[5,0]])
        robot = RobotSerial(dh_params)
        end = Frame.from_euler_3(abc, xyz)
        robot.inverse(end)
        self.theta = robot.axis_values.copy()
        self.yaw_base = self.theta[0]
       
     
        robot = RobotSerial(dh_params)
        f = robot.forward(self.theta)
        
        self.x_world = self.x_base + f.t_3_1.reshape([3, ])[0]
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
    ustar_cost = ustar.T @ lqr_R @ ustar
    error_cost = state_error_world.T @ lqr_Q @ state_error_world


    return ustar, B, f, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t, ustar_cost, error_cost


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
    
    B = np.zeros((6,9))
    B[:, 1:9] = jacobian * dt

    B[0,0] = dt * math.cos(state.yaw_base)
    B[1,0] = dt * math.sin(state.yaw_base)
    
    return B, f


def dlqr(A, B, Q, R, world_ref_traj, n_sum, i, std_dev, state, world_ref_traj_without_noise, D, sigma):
    P, p, c, rt_c, rt_p, c_t, P_t, p_t = solve_dare(A, B, Q, R, world_ref_traj, n_sum, i, std_dev, state, world_ref_traj_without_noise, D, sigma)
    M = la.inv(R + (B.T @ P @ B)) @ B.T
    state_error_world = np.array([state.x_world, state.y_world, state.z_world, state.yaw_world, state.pitch_world, state.roll_world]).reshape(-1,1) - (world_ref_traj[:,i+1].reshape(-1,1) )

    ustar = - M @ (P @ (state_error_world + (rt_c.reshape(-1,1) -  rt_p.reshape(-1,1))) + p )
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

        

        world_xyz_in_horizon_1 =([world_ref_traj_without_noise[:,i+j+1][0]], 
                                 [world_ref_traj_without_noise[:,i+j+1][1]], 
                                 [world_ref_traj_without_noise[:,i+j+1][2]] )  

        world_xyz_in_horizon_2 = ([world_ref_traj_without_noise[:,i+j+2][0]],
                                  [world_ref_traj_without_noise[:,i+j+2][1]],
                                  [world_ref_traj_without_noise[:,i+j+2][2]] ) 
        
        world_xyz_in_horizon_1_orientation = np.array([
            [world_ref_traj_without_noise[:,i+j+1][3]],
            [world_ref_traj_without_noise[:,i+j+1][4]],
            [world_ref_traj_without_noise[:,i+j+1][5]]
        ])

        world_xyz_in_horizon_1 = np.vstack((world_xyz_in_horizon_1, world_xyz_in_horizon_1_orientation))
        
        
        world_xyz_in_horizon_2_orientation = np.array([
            [world_ref_traj_without_noise[:,i+j+2][3]],
            [world_ref_traj_without_noise[:,i+j+2][4]],
            [world_ref_traj_without_noise[:,i+j+2][5]]
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

