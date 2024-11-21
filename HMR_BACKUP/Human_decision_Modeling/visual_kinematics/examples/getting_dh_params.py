import numpy as np
from scipy.optimize import minimize

def dh_transformation(a, alpha, d, theta):
    """Compute the DH transformation matrix."""
    T = np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])
    return T

def forward_kinematics(dh_params, joint_angles):
    """Compute the end-effector pose using DH parameters and joint angles."""
    n_joints = len(joint_angles)
    T = np.eye(4)
    for i in range(n_joints):
        a, alpha, d, _ = dh_params[i]
        theta = joint_angles[i] + dh_params[i][3]  # theta_i = joint_angle_i + theta_offset_i
        T_i = dh_transformation(a, alpha, d, theta)
        T = np.dot(T, T_i)
    return T

def objective_function(dh_params_flat, joint_angle_sets, pose_targets):
    """Objective function to minimize."""
    n_joints = len(joint_angle_sets[0])
    dh_params = np.array(dh_params_flat).reshape((n_joints, 4))
    error = 0.0
    for joint_angles, pose_target in zip(joint_angle_sets, pose_targets):
        T = forward_kinematics(dh_params, joint_angles)
        pose_diff = T - pose_target
        error += np.linalg.norm(pose_diff)
    return error

# # Sample data: Replace these with your actual joint angles and poses
joint_angle_sets = [
    [0.24773788452148438, 0.7336263656616211, -2.193209171295166, 1.1121363639831543, 0.2826355993747711, 0.03719902038574219, 1.1144371032714844],
    [-0.059058189392089844, 1.2858595848083496, 2.899223804473877, 1.8691558837890625, 2.756563663482666, 0.6396698951721191, -1.1581555604934692],  # End-effector pose corresponding to joint angles in joint_angle_sets[0]
    # np.eye(4),  # End-effector pose corresponding to joint angles in joint_angle_sets[1]
    # Add more pose targets as needed
]
pose_targets = [
    np.array([[ 0.81917536,  0.34333701,  0.45942511,  0.94089733],
              [-0.5622932,   0.32288401,  0.76129645, -0.23098457],
              [ 0.11304022, -0.8819669,   0.45755468,  0.78676971],
              [ 0.0,          0.0,          0.0,          1.0        ]]),
            np.array([[ 0.99839401, -0.01116632,  0.05554014,  0.81579053],
 [-0.05523419,  0.02605905,  0.99813331,  0.02855432],
 [-0.0125928,  -0.99959804,  0.02540043,  0.81990476],
 [ 0.0,          0.0,          0.0,        1.0        ]]),  # 8 joint angles for pose 2
    # Add more joint angle sets as needed
]

# Initial guess for DH parameters: [a, alpha, d, theta_offset] for each joint
n_joints = len(joint_angle_sets[0])
initial_dh_params = np.zeros((n_joints, 4))

# Flatten the initial DH parameters for optimization
initial_dh_params_flat = initial_dh_params.flatten()

# Set bounds for DH parameters if necessary
bounds = [(-1, 1)] * len(initial_dh_params_flat)  # Adjust bounds based on your robot's dimensions

# Run the optimization
result = minimize(
    objective_function,
    initial_dh_params_flat,
    args=(joint_angle_sets, pose_targets),
    bounds=bounds,
    method='L-BFGS-B'
)

# Retrieve the optimized DH parameters
optimized_dh_params = result.x.reshape((n_joints, 4))

print("Optimized DH Parameters:")
for i, params in enumerate(optimized_dh_params):
    a, alpha, d, theta_offset = params
    print(f"Joint {i+1}: a={a}, alpha={alpha}, d={d}, theta_offset={theta_offset}")






# import numpy as np
# from scipy.optimize import minimize

# def dh_transformation(a, alpha, d, theta):
#     """Compute the DH transformation matrix."""
#     # Ensure inputs are scalar floats
#     a = float(a)
#     alpha = float(alpha)
#     d = float(d)
#     theta = float(theta)

#     T = np.array([
#         [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
#         [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
#         [0,              np.sin(alpha),                np.cos(alpha),               d],
#         [0,              0,                            0,                           1]
#     ])
#     return T

# def forward_kinematics(dh_params, joint_angles):
#     """Compute the end-effector pose using DH parameters and joint angles."""
#     n_joints = len(joint_angles)
#     T = np.eye(4)
#     for i in range(n_joints):
#         a, alpha, d, theta_offset = dh_params[i]
#         theta = joint_angles[i] + theta_offset  # theta_i = joint_angle_i + theta_offset_i
#         T_i = dh_transformation(a, alpha, d, theta)
#         T = np.dot(T, T_i)
#     return T

# def objective_function(dh_params_flat, joint_angle_sets, pose_targets, fixed_dh_params_first_row):
#     """Objective function to minimize."""
#     n_joints = len(joint_angle_sets[0])
#     dh_params = np.zeros((n_joints, 4))
#     dh_params[0] = fixed_dh_params_first_row
#     dh_params[1:] = dh_params_flat.reshape((n_joints - 1, 4))
#     error = 0.0
#     for joint_angles, pose_target in zip(joint_angle_sets, pose_targets):
#         T = forward_kinematics(dh_params, joint_angles)
#         pose_diff = T - pose_target
#         error += np.linalg.norm(pose_diff)
#     return error

# # Sample data: Replace these with your actual joint angles and poses
# pose_targets = [
#     np.array([[ 0.81917536,  0.34333701,  0.45942511,  0.94089733],
#               [-0.5622932,   0.32288401,  0.76129645, -0.23098457],
#               [ 0.11304022, -0.8819669,   0.45755468,  0.78676971],
#               [ 0.0,          0.0,          0.0,          1.0        ]]),
#             np.array([[ 0.99839401, -0.01116632,  0.05554014,  0.81579053],
#  [-0.05523419,  0.02605905,  0.99813331,  0.02855432],
#  [-0.0125928,  -0.99959804,  0.02540043,  0.81990476],
#  [ 0.0,          0.0,          0.0,        1.0        ]]),  # 8 joint angles for pose 2
#     # Add more joint angle sets as needed
# ]

# joint_angle_sets = [
#     [0, 0.24773788452148438, 0.7336263656616211, -2.193209171295166, 1.1121363639831543, 0.2826355993747711, 0.03719902038574219, 1.1144371032714844],
#     [0, -0.059058189392089844, 1.2858595848083496, 2.899223804473877, 1.8691558837890625, 2.756563663482666, 0.6396698951721191, -1.1581555604934692],  # End-effector pose corresponding to joint angles in joint_angle_sets[0]
#     # np.eye(4),  # End-effector pose corresponding to joint angles in joint_angle_sets[1]
#     # Add more pose targets as needed
# ]

# n_joints = len(joint_angle_sets[0])

# # Define the first row of DH parameters (fixed)
# fixed_dh_params_first_row = [0, 0, 0.72, 0]  # Replace with your actual values

# # Initial guess for the last 7 rows of DH parameters: [a, alpha, d, theta_offset] for each joint
# initial_dh_params_last7 = np.zeros((n_joints - 1, 4))

# # Flatten the DH parameters to be optimized
# initial_dh_params_flat = initial_dh_params_last7.flatten()

# # Set bounds for DH parameters if necessary
# bounds = [(-1, 1)] * len(initial_dh_params_flat)  # Adjust bounds based on your robot's dimensions

# # Run the optimization
# result = minimize(
#     objective_function,
#     initial_dh_params_flat,
#     args=(joint_angle_sets, pose_targets, fixed_dh_params_first_row),
#     bounds=bounds,
#     method='L-BFGS-B'
# )

# # Retrieve the optimized DH parameters
# optimized_dh_params_last7 = result.x.reshape((n_joints - 1, 4))

# # Combine the fixed first row with the optimized last 7 rows
# optimized_dh_params = np.vstack((fixed_dh_params_first_row, optimized_dh_params_last7))

# print("Optimized DH Parameters:")
# print(f"Joint 1 (Fixed): a={fixed_dh_params_first_row[0]}, alpha={fixed_dh_params_first_row[1]}, "
#       f"d={fixed_dh_params_first_row[2]}, theta_offset={fixed_dh_params_first_row[3]}")

# for i, params in enumerate(optimized_dh_params_last7, start=2):
#     a, alpha, d, theta_offset = params
#     print(f"Joint {i}: a={a}, alpha={alpha}, d={d}, theta_offset={theta_offset}")






# import numpy as np
# from scipy.optimize import minimize

# def dh_transformation(a, alpha, d, theta):
#     """Compute the DH transformation matrix."""
#     # Ensure inputs are scalar floats
#     a = float(a)
#     alpha = float(alpha)
#     d = float(d)
#     theta = float(theta)

#     T = np.array([
#         [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
#         [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
#         [0.0,            np.sin(alpha),                np.cos(alpha),               d],
#         [0.0,            0.0,                          0.0,                         1.0]
#     ])
#     return T

# def forward_kinematics(dh_params, joint_angles):
#     """Compute the end-effector pose using DH parameters and joint angles."""
#     n_joints = len(joint_angles)
#     T = np.eye(4)
#     for i in range(n_joints):
#         a, alpha, d, theta_offset = dh_params[i]
#         theta = joint_angles[i] + theta_offset  # theta_i = joint_angle_i + theta_offset_i
#         T_i = dh_transformation(a, alpha, d, theta)
#         T = np.dot(T, T_i)
#     return T

# def objective_function(dh_params_flat, joint_angle_sets, pose_targets, fixed_dh_params_first_row):
#     """Objective function to minimize."""
#     n_joints = len(joint_angle_sets[0])
#     dh_params = np.zeros((n_joints, 4))
#     dh_params[0] = fixed_dh_params_first_row
#     dh_params[1:] = dh_params_flat.reshape((n_joints - 1, 4))
#     error = 0.0
#     for joint_angles, pose_target in zip(joint_angle_sets, pose_targets):
#         T = forward_kinematics(dh_params, joint_angles)
#         pose_diff = T - pose_target
#         error += np.linalg.norm(pose_diff)
#     return error

# # Sample data: Replace these with your actual joint angles and poses
# pose_targets = [
#     np.array([[ 0.81917536,  0.34333701,  0.45942511,  0.94089733],
#               [-0.5622932,   0.32288401,  0.76129645, -0.23098457],
#               [ 0.11304022, -0.8819669,   0.45755468,  0.78676971],
#               [ 0.0,          0.0,          0.0,          1.0        ]]),  # 8 joint angles for pose 2
#     # Add more joint angle sets as needed
# ]

# joint_angle_sets = [
#     [0, 0.24773788452148438, 0.7336263656616211, -2.193209171295166, 1.1121363639831543, 0.2826355993747711, 0.03719902038574219, 1.1144371032714844],  # End-effector pose corresponding to joint angles in joint_angle_sets[0]
#     # np.eye(4),  # End-effector pose corresponding to joint angles in joint_angle_sets[1]
#     # Add more pose targets as needed
# ]

# n_joints = len(joint_angle_sets[0])

# # Define the fixed first row of DH parameters with actual scalar values
# # Replace with your robot's actual DH parameters for the first joint
# fixed_dh_params_first_row = [0.0, -np.pi/2, 0.1, 0.0]  # Example values

# # Initial guess for the last 7 rows of DH parameters
# initial_dh_params_last7 = np.zeros((n_joints - 1, 4))  # Shape: (7, 4)

# # Flatten the DH parameters to be optimized
# initial_dh_params_flat = initial_dh_params_last7.flatten()

# # Set bounds for DH parameters if necessary
# bounds = [(-1.0, 1.0)] * len(initial_dh_params_flat)  # Adjust bounds based on your robot's dimensions

# # Run the optimization
# result = minimize(
#     objective_function,
#     initial_dh_params_flat,
#     args=(joint_angle_sets, pose_targets, fixed_dh_params_first_row),
#     bounds=bounds,
#     method='L-BFGS-B'
# )

# # Retrieve the optimized DH parameters
# optimized_dh_params_last7 = result.x.reshape((n_joints - 1, 4))

# # Combine the fixed first row with the optimized last 7 rows
# optimized_dh_params = np.vstack((fixed_dh_params_first_row, optimized_dh_params_last7))

# print("Optimized DH Parameters:")
# print(f"Joint 1 (Fixed): a={fixed_dh_params_first_row[0]}, alpha={fixed_dh_params_first_row[1]}, "
#       f"d={fixed_dh_params_first_row[2]}, theta_offset={fixed_dh_params_first_row[3]}")

# for i, params in enumerate(optimized_dh_params_last7, start=2):
#     a, alpha, d, theta_offset = params
#     print(f"Joint {i}: a={a}, alpha={alpha}, d={d}, theta_offset={theta_offset}")
