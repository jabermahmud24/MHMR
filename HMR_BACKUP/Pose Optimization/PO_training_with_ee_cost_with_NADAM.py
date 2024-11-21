import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# from torch.optim import Nadam  # Changed import to Nadam
from torch.utils.data import Dataset, DataLoader
import os
import pybullet as p
import pybullet_data
# from torch_optimizer import Nadam
from torch.optim import NAdam
import matplotlib.pyplot as plt


# Step 1: Load and preprocess the data
data = pd.read_csv('Pose Optimization/100sample_robot_IK_data.csv')
# data = pd.read_csv('0.5_robot_IK_data.csv')
theta = data.iloc[:, :7].values  # Joint angles
x = data.iloc[:, 7:13].values    # End-effector poses

# Generate initial configurations (theta0)
np.random.seed(42)
theta0 = theta + np.random.normal(0, 0.0, theta.shape)

# Convert data to tensors
theta = torch.tensor(theta, dtype=torch.float32)
x = torch.tensor(x, dtype=torch.float32)
theta0 = torch.tensor(theta0, dtype=torch.float32)

# Step 2: Create the dataset and data loader
class RobotDataset(Dataset):
    def __init__(self, theta, x, theta0):
        self.theta = theta
        self.x = x
        self.theta0 = theta0
        
    def __len__(self):
        return len(self.theta)
    
    def __getitem__(self, idx):
        return self.theta[idx], x[idx], theta0[idx]

dataset = RobotDataset(theta, x, theta0)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 3: Define the CVAE model
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

        # Modify the x-coordinate (first element)
        end_effector_pos[0] = end_effector_pos[0] 

        # Modify the z-coordinate (third element)
        end_effector_pos[2] = end_effector_pos[2] 

        # Convert back to tuple
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

# Step 4: Define the loss function
def loss_function(theta, theta_hat, z_mean, z_logvar, theta0, x, pose_weight, kl_weight, init_weight):
    recon_loss = nn.MSELoss()(theta_hat, theta)
    kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    init_loss = nn.MSELoss()(theta_hat, theta0)
    x_hat = forward_kinematics(theta_hat)

    Q = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(x.device)  # Example weights
    # Q = torch.tensor([0.5, 0.5, 0.5, 0.5, 50.0, 50.0], dtype=torch.float32).to(x.device)  # Example weights
    pose_error = (x_hat - x) ** 2  # Element-wise squared error
    weighted_pose_error = Q * pose_error  # Apply weights to each axis
    
    pose_loss = torch.mean(weighted_pose_error)  # Mean loss across the batch
    
    # pose_loss = nn.MSELoss()(x_hat, x)
    total_loss = recon_loss + kl_weight * kl_loss + init_weight * init_loss + pose_weight * pose_loss
    return total_loss, recon_loss, kl_loss, init_loss, pose_loss

# Step 5: Train the model
model = CVAE(latent_dim=10)
optimizer = NAdam(model.parameters(), lr=0.0001)  # Changed optimizer to Nadam
# optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 500
# num_epochs = 50
kl_weight = 1.0
init_weight = 1.0
pose_weight = 1.0

losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (theta_batch, x_batch, theta0_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        theta_hat, z_mean, z_logvar = model(theta_batch, x_batch, theta0_batch)
        
        # Compute the total loss with the end-effector pose penalty
        loss, recon_loss, kl_loss, init_loss, pose_loss = loss_function(
            theta_batch, theta_hat, z_mean, z_logvar, theta0_batch, x_batch,
            pose_weight=pose_weight, kl_weight=kl_weight, init_weight=init_weight)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(dataloader)
    losses.append(avg_loss)  # Store the average loss for this epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
    
# Disconnect PyBullet
pybullet_fk.disconnect()

# Step 6: Save the trained model
torch.save(model.state_dict(), 'nadam_cvae_model_with_pose.pth')
print("Model saved as 'nadam_cvae_model_with_pose.pth'")


# Step 7: Plot the training loss
plt.plot(range(1, num_epochs+1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()
