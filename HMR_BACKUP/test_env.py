import numpy as np
import matplotlib.pyplot as plt

# Define grid size and initialize grid points
grid_size = (100, 100)  # 100x100 grid
X, Y = np.meshgrid(np.linspace(0, 10, grid_size[0]), np.linspace(0, 10, grid_size[1]))

# Obstacles definition
obstacles = [
    {'center': np.array([0.5, 5]), 'width': 1, 'height': 10},
    {'center': np.array([9.5, 5]), 'width': 1, 'height': 10},
    {'center': np.array([2.0, 8.0]), 'width': 5, 'height': 2},
    {'center': np.array([3.5, 9.0]), 'width': 2, 'height': 2},
    {'center': np.array([5.0, 9.0]), 'width': 4, 'height': 2},
    {'center': np.array([7, 8.0]), 'width': 2, 'height': 2},
    {'center': np.array([8.0, 8.0]), 'width': 5, 'height': 2},
    {'center': np.array([2.0, 4.5]), 'width': 2, 'height': 1.5},
    {'center': np.array([3.5, 4.75]), 'width': 1.5, 'height': 2},
    {'center': np.array([4.5, 5.5]), 'width': 4, 'height': 1.5},
    {'center': np.array([7, 5.25]), 'width': 1.5, 'height': 2},
    {'center': np.array([8.0, 4.5]), 'width': 4, 'height': 1.5},
    {'center': np.array([2.0, 1.5]), 'width': 2.5, 'height': 1.2},
    {'center': np.array([3.5, 1.95]), 'width': 0.9, 'height': 2.1},
    {'center': np.array([5.2, 2.5]), 'width': 4.5, 'height': 1.2},
    {'center': np.array([7, 2.0]), 'width': 0.0, 'height': 0.0},
    {'center': np.array([8.0, 1.5]), 'width': 2.2, 'height': 1.0},
]

# Function to create the obstacle map
def create_obstacle_map():
    obstacle_map = np.ones(grid_size)
    points = np.stack((X, Y), axis=-1)  # Shape (grid_size[0], grid_size[1], 2)
    
    for obs in obstacles:
        x_min = obs['center'][0] - obs['width'] / 2
        x_max = obs['center'][0] + obs['width'] / 2
        y_min = obs['center'][1] - obs['height'] / 2
        y_max = obs['center'][1] + obs['height'] / 2

        inside_x = (points[..., 0] >= x_min) & (points[..., 0] <= x_max)
        inside_y = (points[..., 1] >= y_min) & (points[..., 1] <= y_max)
        
        obstacle_map[inside_x & inside_y] = 0

    return obstacle_map

# Generate obstacle map
obstacle_map = create_obstacle_map()

# Visualize the obstacle map
plt.figure(figsize=(10, 10))
plt.imshow(obstacle_map, origin='lower', extent=(0, 10, 0, 10), cmap='Greys')
# plt.colorbar(label='Obstacle Map Value (1 = Free, 0 = Obstacle)')
plt.title("Obstacle Map Visualization")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
plt.savefig('obstacle_map.png', dpi=300, bbox_inches='tight')  # Adjust file name and options as needed
plt.show()
