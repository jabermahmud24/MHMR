import torch
import numpy as np
import pandas as pd
import torch.nn as nn  # Import the 'nn' module for building neural networks
import os
# Load the trained model
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

# # Initialize the model and load the saved weights
# model = CVAE(latent_dim=10)
# model.load_state_dict(torch.load('cvae_model.pth'))
# model.eval()  # Set the model to evaluation mode

# # Test function to generate new joint angles given end-effector pose and initial configuration
# def test_model(end_effector_pose, initial_theta, n_samples=1):
#     # Convert inputs to tensors
#     end_effector_pose = torch.tensor(end_effector_pose, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 6)
#     initial_theta = torch.tensor(initial_theta, dtype=torch.float32).unsqueeze(0)          # Shape: (1, 7)
    
#     # Create placeholder for the new joint angle samples
#     new_theta_samples = []
    
#     for _ in range(n_samples):
#         # Sample random latent vector
#         z_sample = torch.randn(1, model.latent_dim)
        
#         # Generate new joint angles using the decoder
#         with torch.no_grad():  # Disable gradient calculation for inference
#             new_theta = model.decode(z_sample, end_effector_pose, initial_theta)
#             new_theta_samples.append(new_theta.squeeze(0).numpy())
    
#     return np.array(new_theta_samples)

# # Example test case:
# # Provide an example end-effector pose (6 values) and initial joint configuration (7 values)
# # example_end_effector_pose = [0.821209, 0.291622, 0.218917, 0.466656, 0.884660, 0.758261]  # Example pose (x, y, z, roll, pitch, yaw)
# # example_initial_theta = [0.03012805,  0.4444292,   0.8140529,   0.3515997 , -0.19826019 , 0.36017722,
# #  -0.60355014]  # Example initial joint angles



# example_end_effector_pose = [1.072140, 0.032626, 0.474656, 0.108692, 0.399095, 0.021535]  # Example pose (x, y, z, roll, pitch, yaw)
# example_initial_theta = [0.0, 0.2, 0.1, 0.2, 0.0 ,0.0 ,0.0]  # Example initial joint angles

# # Test the model with this example data
# n_samples = 5  # Generate 5 different joint angle configurations
# new_joint_angles = test_model(example_end_effector_pose, example_initial_theta, n_samples=n_samples)

# # Print the generated joint angles
# for i, joint_angles in enumerate(new_joint_angles):
#     print(f"Sample {i+1}: {joint_angles}")



# Implement the forward kinematics function using PyBullet
import pybullet as p
import pybullet_data

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

        ######### X compensation

        # Convert tuple to list
        end_effector_pos = list(end_effector_pos)

        # Modify the z-coordinate (third element)
        end_effector_pos[0] = end_effector_pos[0] 

        # If you need it back as a tuple, you can convert it back to a tuple
        end_effector_pos = tuple(end_effector_pos)

        # Convert tuple to list
        end_effector_pos = list(end_effector_pos)

        # Modify the z-coordinate (third element)
        end_effector_pos[2] = end_effector_pos[2]

        # If you need it back as a tuple, you can convert it back to a tuple
        end_effector_pos = tuple(end_effector_pos)

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

# Initialize the FK module
current_directory = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_directory, "fetch.urdf")
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


# Load the trained model
model = CVAE(latent_dim=10)
model.load_state_dict(torch.load('Pose Optimization/nadam_cvae_model_with_pose.pth'))
model.eval()  # Set the model to evaluation mode

def test_model(end_effector_pose, initial_theta, n_samples=1):
    end_effector_pose = torch.tensor(end_effector_pose, dtype=torch.float32).unsqueeze(0)
    initial_theta = torch.tensor(initial_theta, dtype=torch.float32).unsqueeze(0)
    
    new_theta_samples = []
    
    for _ in range(n_samples):
        z_sample = torch.randn(1, model.latent_dim)
        with torch.no_grad():
            theta_hat = model.decode(z_sample, end_effector_pose, initial_theta)
            new_theta_samples.append(theta_hat.squeeze(0).numpy())
    return np.array(new_theta_samples)

# Example test case
example_end_effector_pose = [0.9875610303878785, 0.20118485391139984, 1.1050693035125732, 1.4669270961586376, -0.04649062848867051, -0.09337069994631944]
example_initial_theta = [0.2, -0.18, -0.5, -0.68, -0.24, 0.95, 2.1]

n_samples = 10
new_joint_angles = test_model(example_end_effector_pose, example_initial_theta, n_samples=n_samples)

# Precompute desired pose tensor for MSE calculation
desired_pose_tensor = torch.tensor(example_end_effector_pose, dtype=torch.float32).unsqueeze(0)


# Verify the end-effector pose for each generated joint angle configuration
for i, joint_angles in enumerate(new_joint_angles):
    joint_angles_tensor = torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0)
    predicted_pose = forward_kinematics(torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0))
    # Compute MSE between predicted and desired pose
    mse = torch.mean((predicted_pose - desired_pose_tensor) ** 2).item()
    
    # Print the results
    print(f"Sample {i+1} Joint Angles: {joint_angles}")
    print(f"Predicted End-Effector Pose: {predicted_pose.squeeze(0).numpy()}")
    print(f"Desired End-Effector Pose: {example_end_effector_pose}")
    print(f"MSE between predicted and desired pose: {mse:.6f}")
    print("-------------")
