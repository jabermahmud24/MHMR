import os
import pybullet as p
import pybullet_data
import numpy as np

def main():
    # -------------------------------
    # 1. Initialize PyBullet
    # -------------------------------

    # Connect to PyBullet in DIRECT mode (no GUI)
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

    # List of desired joint names for the 7-DOF arm
    desired_joint_names = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'upperarm_roll_joint',
        'elbow_flex_joint',
        'forearm_roll_joint',
        'wrist_flex_joint',
        'wrist_roll_joint'
    ]

    # Names of gripper prismatic joints (included in DoF)
    gripper_prismatic_joints = ['l_gripper_finger_joint', 'r_gripper_finger_joint']

    # Lists to store joint indices and names
    joint_indices = []  # Indices of desired joints
    joint_names = []    # Names of desired joints
    dof_indices = []    # Indices of all movable joints

    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]

        # Check if the joint is movable (not fixed)
        if joint_type != p.JOINT_FIXED:
            dof_indices.append(i)

            # Collect the indices and names of the desired joints
            if joint_name in desired_joint_names:
                joint_indices.append(i)
                joint_names.append(joint_name)
                print(f"Joint {i}: {joint_name} is included.")
            else:
                print(f"Joint {i}: {joint_name} is included as a movable joint.")

    # Debugging: Print total DoF
    print(f"Total number of movable joints (DoF): {len(dof_indices)}")

    # -------------------------------
    # 4. Identify the End-Effector Link
    # -------------------------------

    # For Fetch robot, the end-effector is typically the "gripper_link"
    end_effector_name = "gripper_link"
    end_effector_index = -1  # Initialize with invalid index

    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        link_name = joint_info[12].decode('utf-8')  # Link name is at index 12
        if end_effector_name == link_name:
            end_effector_index = i
            print(f"End-effector found at joint index: {end_effector_index}, Link name: {link_name}")
            break

    if end_effector_index == -1:
        print(f"End-effector link '{end_effector_name}' not found. Please check the URDF.")
        p.disconnect()
        return

    # -------------------------------
    # 5. Define and Set Joint Angles
    # -------------------------------

    # Define a specific set of joint angles (in radians)
    sample_joint_angles = [
        0.1593387333878335,
        0.3884387680913637,
        0.5827158571440167,
        -0.9631914645613588,
        0.7758054813829567,
        0.4165144917126984,
        0.5581246666552617
    ]

    # Ensure that the number of joint angles matches the number of desired joints
    if len(sample_joint_angles) != len(joint_indices):
        print("Error: Number of joint angles provided does not match the number of desired joints.")
        p.disconnect()
        return

    # Set joint angles for all movable joints
    for i in dof_indices:
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')

        if joint_name in desired_joint_names:
            # Set to the provided sample angle
            idx = joint_names.index(joint_name)
            angle = sample_joint_angles[idx]
            p.resetJointState(robot_id, i, angle)
            print(f"Set Joint {i} ({joint_name}) to angle {angle} radians.")
        else:
            # Set other movable joints (e.g., gripper prismatic joints) to zero
            p.resetJointState(robot_id, i, 0.0)
            print(f"Set Joint {i} ({joint_name}) to angle 0.0 radians.")

    # -------------------------------
    # 6. Compute End-Effector Pose and Jacobian
    # -------------------------------

    # Retrieve the state of the end-effector link
    link_state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)

    # Extract position and orientation
    end_effector_pos = link_state[4]  # World position of the end-effector
    end_effector_orient = link_state[5]  # World orientation (quaternion) of the end-effector

    # Convert quaternion to Euler angles for easier interpretation (optional)
    end_effector_euler = p.getEulerFromQuaternion(end_effector_orient)

    # -------------------------------
    # 7. Compute Jacobian Matrix
    # -------------------------------

    # Get joint positions, velocities, and accelerations for all movable joints
    joint_positions = []
    joint_velocities = []
    joint_accelerations = []

    for i in dof_indices:
        joint_state = p.getJointState(robot_id, i)
        joint_positions.append(joint_state[0])
        joint_velocities.append(joint_state[1])  # Use actual velocities if available
        joint_accelerations.append(0.0)  # Assuming zero accelerations

    # Debugging: Verify the size of the lists
    print(f"Number of joint positions: {len(joint_positions)}")
    print(f"Number of joint velocities: {len(joint_velocities)}")
    print(f"Number of joint accelerations: {len(joint_accelerations)}")

    # Define local position where we want the Jacobian (e.g., [0, 0, 0] in the end-effector frame)
    localPosition = [0, 0, 0]

    # Calculate Jacobian
    jacobian_linear, jacobian_angular = p.calculateJacobian(
        robot_id,
        end_effector_index,
        localPosition,
        joint_positions,
        joint_velocities,
        joint_accelerations
    )

    # Convert Jacobian matrices to numpy arrays for easier manipulation
    jacobian_linear = np.array(jacobian_linear)
    jacobian_angular = np.array(jacobian_angular)

    # Map from joint_indices to indices in dof_indices
    desired_dof_indices = [dof_indices.index(j) for j in joint_indices]

    # Extract Jacobian columns corresponding to desired joints
    jacobian_linear_desired = jacobian_linear[:, desired_dof_indices]
    jacobian_angular_desired = jacobian_angular[:, desired_dof_indices]

    # -------------------------------
    # 8. Display the Results
    # -------------------------------

    print("\n--- End-Effector Pose ---")
    print(f"Position (x, y, z): {end_effector_pos}")
    print(f"Orientation (quaternion [x, y, z, w]): {end_effector_orient}")
    print(f"Orientation (Euler angles [roll, pitch, yaw] in radians): {end_effector_euler}")

    print("\n--- Jacobian Matrices for Desired Joints ---")
    print("Linear Jacobian (Desired):")
    print(jacobian_linear_desired)
    print("Angular Jacobian (Desired):")
    print(jacobian_angular_desired)

    # -------------------------------
    # 9. Clean Up
    # -------------------------------

    # Disconnect from PyBullet
    p.disconnect()

if __name__ == "__main__":
    main()
