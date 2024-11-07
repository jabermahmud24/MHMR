import os
import pybullet as p
import pybullet_data
import numpy as np
import time

def main():
    # -------------------------------
    # 1. Initialize PyBullet
    # -------------------------------
    
    # Connect to PyBullet in DIRECT mode (no GUI). Use GUI for visualization.
    physicsClient = p.connect(p.DIRECT)  # Use p.GUI for graphical mode
    
    # Optionally, set additional search paths (e.g., for PyBullet data)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set gravity (optional, not needed for kinematic calculations)
    p.setGravity(0, 0, -9.81)
    
    # -------------------------------
    # 2. Load the Fetch Robot URDF
    # -------------------------------
    
    # Define the path to the fetch.urdf file
    # Ensure that 'fetch.urdf' is in the current directory or provide the full path
    current_directory = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_directory, "fetch.urdf")
    # urdf_path = os.path.join(current_directory, "fetch.urdf")
    
    if not os.path.exists(urdf_path):
        print(f"URDF file not found at: {urdf_path}")
        p.disconnect()
        return
    
    # Load the robot into the simulation
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    
    # -------------------------------
    # 3. Retrieve Joint Information
    # -------------------------------
    
    num_joints = p.getNumJoints(robot_id)
    print(f"Number of joints in the robot: {num_joints}")
    
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
            print(f"Joint {i}: {joint_name} is revolute.")
        else:
            print(f"Joint {i}: {joint_name} is not revolute and will be ignored.")
    
    # -------------------------------
    # 4. Identify the End-Effector Link
    # -------------------------------
    
    # For Fetch robot, the end-effector is typically the "gripper" link.
    # You may need to adjust this based on your URDF.
    # Here, we search for the link named "gripper_link" or similar.
    
    # end_effector_name = "wrist_roll_link"  # Change as per your URDF
    end_effector_name = "gripper_link"  # Change as per your URDF
    # end_effector_name = "r_gripper_finger_link"  # Change as per your URDF
    end_effector_index = -1  # Initialize with invalid index
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        link_name = joint_info[12].decode('utf-8')  # Link name is at index 12
        if end_effector_name in link_name:
            end_effector_index = i
            print(f"End-effector found at joint index: {end_effector_index}, Link name: {link_name}")
            break
    
    if end_effector_index == -1:
        print(f"End-effector link '{end_effector_name}' not found. Please check the URDF.")
        p.disconnect()
        return
    
    # -------------------------------
    # 5. Define Joint Angles
    # -------------------------------
    
    # Define a specific set of joint angles (in radians)
    # The number of angles should match the number of revolute joints
    # Example for a 7-DOF Fetch robot:
    # joint_angles = [0.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5]
    
    # For demonstration, define a sample joint angle configuration
    # Replace these values with your desired joint angles
    
    # sample_joint_angles = [0.2, -0.18, -0.5, -0.68, -0.24, 0.95, 2.1]
    sample_joint_angles = [0.1593387333878335, 0.3884387680913637, 0.5827158571440167, -0.9631914645613588, 0.7758054813829567, 0.4165144917126984, 0.5581246666552617]
    # sample_joint_angles  = [-0.55, 0.5, -1.0, -1.60, -0.4, 0.65, 2.9]
    
    # Ensure that the number of joint angles matches the number of revolute joints
    if len(sample_joint_angles) != len(joint_indices):
        print("Error: Number of joint angles provided does not match the number of revolute joints.")
        p.disconnect()
        return
    
    # -------------------------------
    # 6. Set Joint Angles
    # -------------------------------
    
    # Iterate through each revolute joint and set its angle
    for idx, joint_idx in enumerate(joint_indices):
        angle = sample_joint_angles[idx]
        p.resetJointState(robot_id, joint_idx, angle)
        print(f"Set Joint {joint_idx} ({joint_names[joint_idx]}) to angle {angle} radians.")
    
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

    # Modify the z-coordinate (third element)
    end_effector_pos[0] = end_effector_pos[0] - 0.0
    # end_effector_pos[0] = end_effector_pos[0] - 0.08

    # If you need it back as a tuple, you can convert it back to a tuple
    end_effector_pos = tuple(end_effector_pos)



    ###### Z compensation 


    print(end_effector_pos[2])

    # Convert tuple to list
    end_effector_pos = list(end_effector_pos)

    # Modify the z-coordinate (third element)
    end_effector_pos[2] = end_effector_pos[2] + 0.0
    # end_effector_pos[2] = end_effector_pos[2] + 0.4

    # If you need it back as a tuple, you can convert it back to a tuple
    end_effector_pos = tuple(end_effector_pos)


    end_effector_orient = link_state[5]  # World orientation (quaternion) of the end-effector
    
    # Convert quaternion to Euler angles for easier interpretation (optional)
    end_effector_euler = p.getEulerFromQuaternion(end_effector_orient)
    
    # -------------------------------
    # 8. Display the Results
    # -------------------------------
    
    print("\n--- End-Effector Pose ---")
    print(f"Position (x, y, z): {end_effector_pos}")
    print(f"Orientation (quaternion [x, y, z, w]): {end_effector_orient}")
    print(f"Orientation (Euler angles [roll, pitch, yaw] in radians): {end_effector_euler}")
    
    # Optionally, compute the transformation matrix
    transformation_matrix = p.getMatrixFromQuaternion(end_effector_orient)
    transformation_matrix = np.array(transformation_matrix).reshape(3, 3)
    print(f"Rotation Matrix:\n{transformation_matrix}")
    
    # -------------------------------
    # 9. Clean Up
    # -------------------------------
    
    # Disconnect from PyBullet
    p.disconnect()

if __name__ == "__main__":
    main()
