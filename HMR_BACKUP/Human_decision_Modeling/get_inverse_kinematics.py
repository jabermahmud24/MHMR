import torch
import numpy as np
import torch.nn as nn
import os
import pybullet as p
import pybullet_data

current_directory = os.path.dirname(os.path.abspath(__file__))
# urdf_path = os.path.join(current_directory, "fetch_with_base.urdf")
urdf_path = os.path.join(current_directory, "fetch.urdf")

def generate_new_joint_angles(
    end_effector_pose,
    initial_theta,
    model_path='nadam_cvae_model_with_pose.pth',
    urdf_path='fetch.urdf',
    n_samples=1
):
    """
    Generate new joint angles given a desired end-effector pose and initial joint configuration.

    Parameters:
    - end_effector_pose (list or array): Desired end-effector pose [x, y, z, roll, pitch, yaw].
    - initial_theta (list or array): Initial joint angles [7 values].
    - model_path (str): Path to the trained CVAE model weights.
    - urdf_path (str): Path to the robot URDF file.
    - n_samples (int): Number of joint angle samples to generate.

    Returns:
    - new_joint_angles (np.ndarray): Generated joint angles of shape (n_samples, 7).
    - mse_list (list): List of Mean Squared Errors for each generated sample.
    """
    
    # Define the CVAE model
    class CVAE(nn.Module):
        def __init__(self, latent_dim=10):
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim
            
            # Encoder layers
            self.encoder_fc1 = nn.Linear(20, 128)
            self.encoder_fc2 = nn.Linear(128, 64)
            self.encoder_mu = nn.Linear(64, latent_dim)
            self.encoder_logvar = nn.Linear(64, latent_dim)
            
            # Decoder layers
            self.decoder_fc1 = nn.Linear(latent_dim + 13, 64)
            self.decoder_fc2 = nn.Linear(64, 128)
            self.decoder_fc3 = nn.Linear(128, 7)
            
        def encode(self, theta, x, theta0):
            encoder_input = torch.cat([theta, x, theta0], dim=1)
            h = torch.relu(self.encoder_fc1(encoder_input))
            h = torch.relu(self.encoder_fc2(h))
            z_mean = self.encoder_mu(h)
            z_logvar = self.encoder_logvar(h)
            return z_mean, z_logvar
        
        def reparameterize(self, z_mean, z_logvar):
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            z = z_mean + eps * std
            return z
        
        def decode(self, z, x, theta0):
            decoder_input = torch.cat([z, x, theta0], dim=1)
            h = torch.relu(self.decoder_fc1(decoder_input))
            h = torch.relu(self.decoder_fc2(h))
            theta_hat = self.decoder_fc3(h)
            return theta_hat
        
        def forward(self, theta, x, theta0):
            z_mean, z_logvar = self.encode(theta, x, theta0)
            z = self.reparameterize(z_mean, z_logvar)
            theta_hat = self.decode(z, x, theta0)
            return theta_hat, z_mean, z_logvar

    # Define the PyBullet Forward Kinematics class
    class PyBulletFK:
        def __init__(self, urdf_path):
            # Initialize PyBullet in DIRECT mode
            self.physicsClient = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            # Load the robot URDF
            self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
            # Get joint indices for revolute joints
            num_joints = p.getNumJoints(self.robot_id)
            self.joint_indices = []
            self.joint_names = {}
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                joint_type = joint_info[2]
                joint_name = joint_info[1].decode('utf-8')
                self.joint_names[i] = joint_name
                if joint_type == p.JOINT_REVOLUTE:
                    self.joint_indices.append(i)
            # Find end-effector index
            self.end_effector_name = "gripper_link"  # Adjust as per your URDF
            self.end_effector_index = -1
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                link_name = joint_info[12].decode('utf-8')  # Link name is at index 12
                if self.end_effector_name in link_name:
                    self.end_effector_index = i
                    break
            if self.end_effector_index == -1:
                raise ValueError(f"End-effector link '{self.end_effector_name}' not found.")
        
        def compute_fk(self, theta_i):
            # Set joint angles
            theta_i = theta_i.detach().cpu().numpy()  # Convert to numpy
            for idx, joint_idx in enumerate(self.joint_indices):
                angle = theta_i[idx]
                p.resetJointState(self.robot_id, joint_idx, angle)
            # Compute end-effector pose
            link_state = p.getLinkState(self.robot_id, self.end_effector_index, computeForwardKinematics=True)
            end_effector_pos = link_state[4]  # World position
            end_effector_orient = link_state[5]  # World orientation (quaternion)
            # Convert to torch tensor
            end_effector_pos = torch.tensor(end_effector_pos, dtype=torch.float32)
            end_effector_orient = torch.tensor(end_effector_orient, dtype=torch.float32)
            # Convert quaternion to Euler angles
            end_effector_euler = torch.tensor(p.getEulerFromQuaternion(end_effector_orient.numpy()), dtype=torch.float32)
            # Return position and orientation as a single tensor
            x_hat_i = torch.cat([end_effector_pos, end_effector_euler])
            return x_hat_i
        
        def disconnect(self):
            p.disconnect(self.physicsClient)

    # Initialize the Forward Kinematics module
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found at: {urdf_path}")
    
    pybullet_fk = PyBulletFK(urdf_path)
    
    # Define the forward_kinematics function
    def forward_kinematics(theta):
        x_hat_list = []
        for theta_i in theta:
            x_hat_i = pybullet_fk.compute_fk(theta_i)
            x_hat_list.append(x_hat_i)
        x_hat = torch.stack(x_hat_list)
        return x_hat

    # Initialize the CVAE model and load the trained weights
    model = CVAE(latent_dim=10)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Define the test_model function to generate new joint angles
    def test_model(end_effector_pose, initial_theta, n_samples=1):
        # Convert inputs to tensors
        end_effector_pose = torch.tensor(end_effector_pose, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 6)
        initial_theta = torch.tensor(initial_theta, dtype=torch.float32).unsqueeze(0)          # Shape: (1, 7)
        
        # Create placeholder for the new joint angle samples
        new_theta_samples = []
        
        for _ in range(n_samples):
            # Sample random latent vector
            z_sample = torch.randn(1, model.latent_dim)
            
            # Generate new joint angles using the decoder
            with torch.no_grad():  # Disable gradient calculation for inference
                theta_hat = model.decode(z_sample, end_effector_pose, initial_theta)
                new_theta_samples.append(theta_hat.squeeze(0).numpy())
        
        return np.array(new_theta_samples)

    # Generate new joint angles
    new_joint_angles = test_model(end_effector_pose, initial_theta, n_samples=n_samples)
    
    # Precompute desired pose tensor for MSE calculation
    desired_pose_tensor = torch.tensor(end_effector_pose, dtype=torch.float32).unsqueeze(0)
    
    # Verify the end-effector pose for each generated joint angle configuration
    mse_list = []
    for i, joint_angles in enumerate(new_joint_angles):
        joint_angles_tensor = torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0)
        predicted_pose = forward_kinematics(torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0))
        # Compute MSE between predicted and desired pose
        mse = torch.mean((predicted_pose - desired_pose_tensor) ** 2).item()
        mse_list.append(mse)
        # Optionally, you can print or log the results
        # print(f"Sample {i+1} Joint Angles: {joint_angles}")
        # print(f"Predicted End-Effector Pose: {predicted_pose.squeeze(0).numpy()}")
        # print(f"Desired End-Effector Pose: {end_effector_pose}")
        # print(f"MSE between predicted and desired pose: {mse:.6f}")
        # print("-------------")
    
    # Disconnect PyBullet
    pybullet_fk.disconnect()
    
    return new_joint_angles

# # Example usage:
# if __name__ == "__main__":
#     example_end_effector_pose = [0.9875610303878785, 0.20118485391139984, 1.1050693035125732, 
#                                  1.4669270961586376, -0.04649062848867051, -0.09337069994631944]
#     example_initial_theta = [0.2, -0.18, -0.5, -0.68, -0.24, 0.95, 2.1]
#     n_samples = 1
    
#     new_joint_angles, mse_list = generate_new_joint_angles(
#         end_effector_pose=example_end_effector_pose,
#         initial_theta=example_initial_theta,
#         model_path='Pose Optimization/nadam_cvae_model_with_pose.pth',
#         urdf_path='fetch.urdf',
#         n_samples=n_samples
#     )
    
#     # Optionally, process the returned joint angles and MSEs as needed
#     # For example, selecting the joint angles with the lowest MSE
#     best_index = np.argmin(mse_list)
#     best_joint_angles = new_joint_angles[best_index]
#     print(f"Best Joint Angles (Sample {best_index+1}): {best_joint_angles}")
#     print(f"Lowest MSE: {mse_list[best_index]:.6f}")
