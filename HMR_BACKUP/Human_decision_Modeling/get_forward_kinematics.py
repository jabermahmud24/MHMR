# test_jacobian.py

import os
import pybullet as p
import pybullet_data
import numpy as np
import math
import time

from get_end_effector_body_frame import transform_pose_world_to_body

def compute_end_effector_info(joint_angle_combination):
    """
    Compute the end-effector pose, orientation, and Jacobian for the Fetch robot given joint angles.

    Parameters:
    - joint_angle_combination (list or array): Joint angles for the robot's revolute joints in radians.
    - urdf_path (str, optional): Path to the fetch.urdf file. If None, assumes it's in the current directory.
    - end_effector_name (str, optional): Name of the end-effector link as defined in the URDF.
    - use_gui (bool, optional): If True, connects to PyBullet with a GUI for visualization.

    Returns:
    - end_effector_pos (tuple): (x, y, z) position of the end-effector.
    - end_effector_orient (tuple): Orientation of the end-effector as a quaternion (x, y, z, w).
    - end_effector_euler (tuple): Orientation of the end-effector as Euler angles (roll, pitch, yaw).
    - jacobian (numpy.ndarray): 6xN Jacobian matrix, where N is the number of revolute joints.
    """
    # -------------------------------
    # 1. Initialize PyBullet
    # -------------------------------
    
    # Connect to PyBullet


    physicsClient = p.connect(p.DIRECT)
    
    # Set additional search paths (e.g., for PyBullet data)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set gravity (optional, not needed for kinematic calculations)
    p.setGravity(0, 0, -9.81)
    
    # -------------------------------
    # 2. Load the Fetch Robot URDF
    # -------------------------------
    
    # Define the path to the fetch.urdf file

    current_directory = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_directory, "fetch_with_base.urdf")
    # urdf_path = os.path.join(current_directory, "fetch.urdf")

    # Load the robot into the simulation
    robot_id = p.loadURDF(urdf_path, useFixedBase=False)
    # robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    end_effector_name="gripper_link"
    
    # -------------------------------
    # 3. Retrieve Joint Information
    # -------------------------------
    
    num_joints = p.getNumJoints(robot_id)
    # print(f"Number of joints in the robot: {num_joints}")
    
    # List to store joint indices for revolute (rotational) joints
    joint_indices = []
    
    # Dictionary to map joint indices to joint names
    joint_names = {}
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        joint_name = joint_info[1].decode('utf-8')
        joint_names[i] = joint_name
        
        # Check if the joint is revolute (type 0)
        if joint_type == p.JOINT_REVOLUTE:
            joint_indices.append(i)
            # print(f"Joint {i}: {joint_name} is revolute.")
        # else:
            # print(f"Joint {i}: {joint_name} is not revolute and will be ignored.")
    
    # -------------------------------
    # 4. Identify the End-Effector Link
    # -------------------------------
    
    end_effector_index = -1  # Initialize with invalid index
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        link_name = joint_info[12].decode('utf-8')  # Link name is at index 12
        if end_effector_name in link_name:
            end_effector_index = i
            # print(f"End-effector found at joint index: {end_effector_index}, Link name: {link_name}")
            break
    
    if end_effector_index == -1:
        # print(f"End-effector link '{end_effector_name}' not found. Please check the URDF.")
        p.disconnect()
        return None, None, None, None
    
    # -------------------------------
    # 5. Define Joint Angles
    # -------------------------------
    # print(len(joint_angle_combination))
    # print(len(joint_indices))
    
    # Ensure that the number of joint angles matches the number of revolute joints
    if len(joint_angle_combination) != len(joint_indices):
        # print("Error: Number of joint angles provided does not match the number of revolute joints.")
        p.disconnect()
        return None, None, None, None
    
    # -------------------------------
    # 6. Set Joint Angles
    # -------------------------------
    
    # Iterate through each revolute joint and set its angle
    for idx, joint_idx in enumerate(joint_indices):
        angle = joint_angle_combination[idx]
        p.resetJointState(robot_id, joint_idx, angle)
        # print(f"Set Joint {joint_idx} ({joint_names[joint_idx]}) to angle {angle} radians.")
    
    # -------------------------------
    # 7. Compute End-Effector Pose
    # -------------------------------
    
    # Retrieve the state of the end-effector link
    link_state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)
    
    # Extract position and orientation
    end_effector_pos = link_state[4]  # World position of the end-effector
    
    ######### X compensation
    # Convert tuple to list
    end_effector_pos = list(end_effector_pos)
    
    # Modify the x-coordinate (first element)
    end_effector_pos[0] = end_effector_pos[0] - 0.0  # Adjust as needed
    # end_effector_pos[0] = end_effector_pos[0] - 0.08  # Example adjustment
    
    # Convert back to tuple
    end_effector_pos = tuple(end_effector_pos)
    
    ###### Z compensation 
    # print(f"Original Z position: {end_effector_pos[2]}")
    
    # Convert tuple to list
    end_effector_pos = list(end_effector_pos)
    
    # Modify the z-coordinate (third element)
    end_effector_pos[2] = end_effector_pos[2] + 0.0  # Adjust as needed
    # end_effector_pos[2] = end_effector_pos[2] + 0.4  # Example adjustment
    
    # Convert back to tuple
    end_effector_pos = tuple(end_effector_pos)
    
    end_effector_orient = link_state[5]  # World orientation (quaternion) of the end-effector
    
    # Convert quaternion to Euler angles for easier interpretation (optional)
    end_effector_euler = p.getEulerFromQuaternion(end_effector_orient)
    

    base_rotation_angle = joint_angle_combination[0]

    end_effector_pos_body_frame, end_effector_orient_body_frame = transform_pose_world_to_body(end_effector_pos, end_effector_orient, base_rotation_angle)

    end_effector_orient_body_frame_euler = p.getEulerFromQuaternion(end_effector_orient_body_frame)

    # -------------------------------
    # 8. Compute Jacobian Matrix
    # -------------------------------
    
    # # Get joint positions, velocities, and accelerations for all revolute joints
    # joint_positions = []
    # joint_velocities = []
    # joint_accelerations = []
    
    # for joint_idx in joint_indices:
    #     joint_state = p.getJointState(robot_id, joint_idx)
    #     joint_positions.append(joint_state[0])
    #     joint_velocities.append(joint_state[1])  # Use actual velocities if available
    #     joint_accelerations.append(0.0)  # Assuming zero accelerations
    


    
    # -------------------------------
    # 9. Display the Results (Optional)
    # -------------------------------
     
    # print("\n--- End-Effector Pose ---")
    # print(f"Position (x, y, z): {end_effector_pos}")
    # print(f"Orientation (quaternion [x, y, z, w]): {end_effector_orient}")
    # print(f"Orientation (Euler angles [roll, pitch, yaw] in radians): {end_effector_euler}")

    # Optionally, compute the transformation matrix
    transformation_matrix = p.getMatrixFromQuaternion(end_effector_orient)
    transformation_matrix = np.array(transformation_matrix).reshape(3, 3)
    # print(f"\nRotation Matrix:\n{transformation_matrix}")
    
    # -------------------------------
    # 10. Clean Up
    # -------------------------------
    
    # Allow some time for visualization if GUI is used

    # Disconnect from PyBullet
    p.disconnect()
    
    return end_effector_pos, end_effector_euler, end_effector_pos_body_frame, end_effector_orient_body_frame_euler