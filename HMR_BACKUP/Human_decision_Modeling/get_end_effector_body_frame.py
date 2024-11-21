# transform_pose.py

import numpy as np
from scipy.spatial.transform import Rotation as R

def transform_pose_world_to_body(position_world, orientation_world, base_rotation_angle):
    """
    Transforms the end-effector pose from the world frame to the body frame.

    Parameters:
    ----------
    position_world : array-like
        The position of the end-effector in the world frame as [x, y, z].
    orientation_world : array-like
        The orientation of the end-effector in the world frame as a quaternion [x, y, z, w].
    base_rotation_angle : float
        The rotation angle of the base around the Z-axis in degrees.

    Returns:
    -------
    position_body : numpy.ndarray
        The position of the end-effector in the body frame as [x, y, z].
    orientation_body : numpy.ndarray
        The orientation of the end-effector in the body frame as a quaternion [x, y, z, w].

    Example:
    -------
    >>> position_world = [0.01382, 1.03227, 0.88155]
    >>> orientation_world = [0.59428, 0.49081, 0.53987, 0.33834]
    >>> base_rotation_angle = 90
    >>> pos_body, ori_body = transform_pose_world_to_body(position_world, orientation_world, base_rotation_angle)
    >>> print("Position in Body Frame:", pos_body)
    >>> print("Orientation in Body Frame:", ori_body)
    """
    # Convert inputs to NumPy arrays
    position_world = np.array(position_world)
    orientation_world = np.array(orientation_world)

    # Validate input dimensions
    if position_world.shape != (3,):
        raise ValueError("position_world must be a 3-element array-like object.")
    if orientation_world.shape != (4,):
        raise ValueError("orientation_world must be a 4-element array-like object representing a quaternion [x, y, z, w].")

    # Define the body frame pose in the world frame
    # No linear movement
    P_b_world = np.array([0.0, 0.0, 0.0])

    # Create rotation object for base rotation about Z-axis
    rot = R.from_euler('z', base_rotation_angle, degrees=False)
    Q_b_world = rot.as_quat()  # Quaternion in [x, y, z, w] format

    # Convert Quaternions to Rotation Matrices
    R_e_world = R.from_quat(orientation_world).as_matrix()
    R_b_world = R.from_quat(Q_b_world).as_matrix()

    # Compute Inverse Transformation (World to Body)
    R_world_body = R_b_world.T
    P_world_body = -R_world_body @ P_b_world  # Since P_b_world is [0,0,0], this remains [0,0,0]

    # Transform Position
    P_e_body = R_world_body @ position_world + P_world_body  # Simplifies to R_world_body @ position_world

    # Transform Orientation
    R_e_body = R_world_body @ R_e_world
    Q_e_body = R.from_matrix(R_e_body).as_quat()

    return P_e_body, Q_e_body
