import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from scipy.stats import multivariate_normal

# Parameters
# Start and end points
A = np.array([0, 0])   # Starting point A
B = np.array([10, 10]) # Target point B

# Obstacles (defined as circles for simplicity)
# Obstacle 1
O1_center = np.array([4, 5])
O1_radius = 0.5

# Obstacle 2
O2_center = np.array([8, 6])
O2_radius = 1

# Robot's positional uncertainty (covariance matrix)
sigma = 0.5  # Standard deviation in x and y
Sigma = np.array([[sigma**2, 0],
                  [0, sigma**2]])  # Covariance matrix

# Number of time steps
N = 50

# Number of samples for Monte Carlo simulation
M = 1000

# Time steps
t_values = np.linspace(0, 1, N)

# Initialize arrays to store results
mu_t_values = []            # Expected positions
collision_probabilities = []  # Collision probabilities

# Visualization setup
fig, ax = plt.subplots(figsize=(8, 8))

# Draw obstacles
obstacle1 = Circle(O1_center, O1_radius, color='red', alpha=0.5)
obstacle2 = Circle(O2_center, O2_radius, color='red', alpha=0.5)
ax.add_patch(obstacle1)
ax.add_patch(obstacle2)

# Draw start and end points
ax.plot(A[0], A[1], 'go', markersize=10, label='Start Point A')
ax.plot(B[0], B[1], 'bo', markersize=10, label='Target Point B')

# Theory:
# The robot's position at time t is modeled as a random variable x_t ~ N(mu_t, Sigma)
# where mu_t is the expected position at time t, and Sigma is the covariance matrix representing uncertainty.

for t in t_values:
    # Expected position at time t (linear interpolation from A to B)
    mu_t = A + t * (B - A)
    mu_t_values.append(mu_t)
    
    # Monte Carlo simulation to estimate collision probability
    # Sample M positions from the Gaussian distribution
    samples = np.random.multivariate_normal(mu_t, Sigma, M)
    
    # Check for collisions with obstacles
    # For obstacle 1
    dist_to_O1 = np.linalg.norm(samples - O1_center, axis=1)
    collision_O1 = dist_to_O1 <= O1_radius
    
    # For obstacle 2
    dist_to_O2 = np.linalg.norm(samples - O2_center, axis=1)
    collision_O2 = dist_to_O2 <= O2_radius
    
    # Total collisions
    total_collisions = np.sum(collision_O1 | collision_O2)
    
    # Collision probability
    collision_probability = total_collisions / M
    collision_probabilities.append(collision_probability)
    
    # Visualization of robot's expected position
    ax.plot(mu_t[0], mu_t[1], 'k.', alpha=0.5)
    
    # Optionally, plot uncertainty ellipse (commented out to reduce clutter)
    # eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    # angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
    # width, height = 2 * np.sqrt(eigenvalues)
    # ellipse = Ellipse(mu_t, width, height, angle, edgecolor='blue', facecolor='none', alpha=0.2)
    # ax.add_patch(ellipse)

# Plot settings
ax.set_xlim(-1, 12)
ax.set_ylim(-1, 12)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Robot Path with Collision Probabilities')
ax.legend()
ax.grid(True)
plt.show()

# Plot collision probabilities over time
plt.figure(figsize=(8, 4))
plt.plot(t_values, collision_probabilities, 'r-')
plt.xlabel('Time (normalized)')
plt.ylabel('Collision Probability')
plt.title('Collision Probability Over Time')
plt.grid(True)
plt.show()
