import os
import pybullet as p
import pybullet_data
import numpy as np
import itertools
import csv
import time

def main():
    # -------------------------------
    # 1. Initialize PyBullet
    # -------------------------------

    # Connect to PyBullet in DIRECT mode (no GUI). Use p.GUI for visualization.
    physicsClient = p.connect(p.DIRECT)  # Use p.GUI for graphical mode

    # Optionally, set additional search paths (e.g., for PyBullet data)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity (optional, not needed for kinematic calculations)
    p.setGravity(0, 0, -9.81)

    # -------------------------------
    # 2. Load the Fetch Robot URDF
    # -------------------------------

    # Define the path to the fetch.urdf file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_directory, "fetch.urdf")

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

    # Dictionary to map joint names to joint indices
    joint_name_to_index = {}

    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        joint_name = joint_info[1].decode('utf-8')
        joint_name_to_index[joint_name] = i
        print(f"Joint {i}: {joint_name}")

    # -------------------------------
    # 4. Identify Arm Joints
    # -------------------------------

    # Joint names for the Fetch robot's arm
    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ]

    # Get joint indices for arm joints
    arm_joint_indices = []
    for joint_name in arm_joint_names:
        if joint_name in joint_name_to_index:
            arm_joint_indices.append(joint_name_to_index[joint_name])
        else:
            print(f"Joint {joint_name} not found in the robot.")
            p.disconnect()
            return

    print("Arm joint indices:", arm_joint_indices)

    # -------------------------------
    # 5. Identify the End-Effector Link
    # -------------------------------

    # For Fetch robot, the end-effector is typically the "gripper_link"
    end_effector_name = "gripper_link"  # Adjust as per your URDF
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
    # 6. Define Joint Angle Ranges
    # -------------------------------

    # Define the increment (gap) for each joint
    increment = 0.2  # Adjust as needed

    # Define joint angle ranges
    angle_shoulder_pan = np.arange(-0.57, 0.57 + increment, increment)
    angle_shoulder_lift = np.arange(-0.5, 0.7 + increment, increment)
    angle_upperarm_roll = np.arange(-1.0, 1.0 + increment, increment)
    angle_elbow_flex = np.arange(-1.1, 1.1 + increment, increment)
    angle_forearm_roll = np.arange(-1.0, 1.0 + increment, increment)
    angle_wrist_flex = np.arange(-0.6, 0.6 + increment, increment)
    angle_wrist_roll = np.arange(-1.1, 1.1 + increment, increment)

    # Generate all combinations
    all_combinations = list(itertools.product(
        angle_shoulder_pan,
        angle_shoulder_lift,
        angle_upperarm_roll,
        angle_elbow_flex,
        angle_forearm_roll,
        angle_wrist_flex,
        angle_wrist_roll
    ))

    total_combinations = len(all_combinations)
    print("Total number of combinations:", total_combinations)

    # -------------------------------
    # 7. Compute Forward Kinematics and Collect Data
    # -------------------------------

    data = []
    start_time = time.time()

    for i, angle_combination in enumerate(all_combinations):
        print("Iteration:", i + 1, "/", total_combinations)
        print("Angle combination:", angle_combination)

        # Set joint angles
        if len(angle_combination) != len(arm_joint_indices):
            print("Error: Number of joint angles provided does not match the number of arm joints.")
            continue

        for idx, joint_idx in enumerate(arm_joint_indices):
            angle = angle_combination[idx]
            p.resetJointState(robot_id, joint_idx, angle)

        # Retrieve the state of the end-effector link
        link_state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)

        # Extract position and orientation
        end_effector_pos = link_state[4]  # World position of the end-effector
        end_effector_orient = link_state[5]  # World orientation (quaternion) of the end-effector

        # Apply compensation if needed
        # Adjust these offsets based on your setup
        end_effector_pos = list(end_effector_pos)
        end_effector_pos[0] -= 0.08  # X compensation
        end_effector_pos[2] += 0.4   # Z compensation
        end_effector_pos = tuple(end_effector_pos)

        # Convert quaternion to Euler angles
        end_effector_euler = p.getEulerFromQuaternion(end_effector_orient)
        roll, pitch, yaw = end_effector_euler

        # Check if within tolerance
        within_tolerance = abs(pitch) <= 0.2 and abs(yaw) <= 0.2

        # Collect data
        position_x, position_y, position_z = end_effector_pos
        data_row = list(angle_combination) + [position_x, position_y, position_z, roll, pitch, yaw, within_tolerance]
        data.append(data_row)

    end_time = time.time()
    duration = end_time - start_time
    print("Total time: {} seconds".format(duration))

    # -------------------------------
    # 8. Save Data to CSV File
    # -------------------------------

    header = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
        "position_x",
        "position_y",
        "position_z",
        "roll",
        "pitch",
        "yaw",
        "within_tolerance"
    ]

    with open('robot_IK_data.csv', 'w', newline='') as csvfile:
        data_writer = csv.writer(csvfile)
        data_writer.writerow(header)
        data_writer.writerows(data)

    print("Data saved to robot_IK_data.csv")

    # -------------------------------
    # 9. Clean Up
    # -------------------------------

    # Disconnect from PyBullet
    p.disconnect()

if __name__ == "__main__":
    main()
