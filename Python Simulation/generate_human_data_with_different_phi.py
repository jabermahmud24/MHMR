

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cubic_spline_planner  # Ensure this module is available in your working directory

def generate_data(phi_k, mu, sigma, s_choice):
    """
    Generate synthetic data for the human-robot collaboration model.

    Parameters:
    - phi_k: Human preference parameter (scalar between 0 and 1)
    - mu: Mean of the noise term epsilon
    - sigma: Standard deviation of the noise term epsilon
    - s_choice: State based on human's own preference (T x 3 array)

    Returns:
    - s_star: True state of human k at each time step (T x 3 array)
    - s_dummy: Human's dummy state designed by robot (T x 3 array)
    """
    T = s_choice.shape[0]
    # Initialize arrays
    s_dummy = np.zeros((T, 3))
    s_star = np.zeros((T, 3))

    # Ensure s_dummy starts and ends at the same points as s_choice
    s_dummy[0] = s_choice[0]
    s_dummy[-1] = s_choice[-1]

    # Generate a smooth trajectory for s_dummy between the start and end points
    # We'll create random waypoints and use cubic spline interpolation
    num_waypoints = 5  # Number of waypoints excluding start and end
    waypoints_t = np.linspace(0, T-1, num_waypoints + 2)
    waypoints_s = np.zeros((num_waypoints + 2, 3))
    waypoints_s[0] = s_dummy[0]
    waypoints_s[-1] = s_dummy[-1]
    for i in range(1, num_waypoints + 1):
        waypoints_s[i] = s_choice[int(waypoints_t[i])] + np.random.normal(0, sigma, 3)
        # Ensure z is within [0.8, 1.2]
        waypoints_s[i, 2] = np.clip(waypoints_s[i, 2], 0.8, 1.2)

    # Interpolate s_dummy using cubic splines for smoothness
    s_dummy_interp = np.zeros((T, 3))
    for dim in range(3):
        cs = interp1d(waypoints_t, waypoints_s[:, dim], kind='cubic')
        s_dummy_interp[:, dim] = cs(np.arange(T))
    s_dummy = s_dummy_interp

    # Ensure z is within [0.8, 1.2]
    s_dummy[:, 2] = np.clip(s_dummy[:, 2], 0.8, 1.2)

    # Generate s_star based on the model and ensure it starts and ends at s_choice
    s_star[0] = s_choice[0]
    s_star[-1] = s_choice[-1]

    for t in range(1, T-1):
        # Simulate epsilon
        epsilon = np.random.normal(mu, sigma, 3)

        # Compute s_star based on the model
        s_star[t] = phi_k * s_choice[t] + (1 - phi_k) * s_dummy[t] + epsilon
        # Ensure z is within [0.8, 1.2]
        s_star[t, 2] = np.clip(s_star[t, 2], 0.8, 1.2)

    return s_star, s_dummy

def estimate_phi(s_star, s_choice, s_dummy):
    """
    Estimate the human preference parameter phi_k using Least Squares.

    Parameters:
    - s_star: True state of human k at each time step (T x 3 array)
    - s_choice: State based on human's own preference (T x 3 array)
    - s_dummy: Human's dummy state designed by robot (T x 3 array)

    Returns:
    - phi_hat: Estimated phi_k
    """
    # Compute y and x
    y = s_star - s_dummy
    x = s_choice - s_dummy

    # Flatten the arrays for all dimensions and time steps
    y_flat = y.flatten()
    x_flat = x.flatten()

    # Center the data
    x_mean = np.mean(x_flat)
    y_mean = np.mean(y_flat)
    x_centered = x_flat - x_mean
    y_centered = y_flat - y_mean

    # Estimate phi_k using Least Squares
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sum(x_centered ** 2)
    phi_hat = numerator / denominator

    return phi_hat

def main():
    # Simulation parameters
    true_phi_k = 0.95  # Example true phi_k value (between 0 and 1)
    mu = 0.0           # Mean of epsilon
    sigma = 0.0       # Standard deviation of epsilon

    # Trajectory data for s_choice
    ax = [0, 0.25, 0.5, 0.4, 0.25, 0.45, 0.65, 0.85, 0.75, 0.65, 0.75, 1.75, 
          2.5, 3.25, 3.5, 3.75, 4, 4.25]
    ay = [0, 0, 0.05, 0.35, 0.45, 0.35, 0, 0, 0, 0, -0.07, -0.15, -0.35, -0.45, 
          -0.35, -0.15, 0, 0.1]
    az = [0.9, 0.94, 0.98, 1.02, 1.06, 1.1, 1.14, 1.14, 1.12, 1.09, 1.1, 1.07, 
          1.03, 0.99, 0.95, 0.96, 1.02, 1.03]

    # Generate cubic spline for x and y
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)

    # Generate cubic spline for z
    # Note: Assuming cubic_spline_planner can handle separate z trajectories
    # If not, you might need to modify the spline planner accordingly
    _, cz, _, _, _ = cubic_spline_planner.calc_spline_course(ax, az, ds=0.1)

    # Interpolation to have consistent number of points
    num_points = 509  # Desired number of points
    # Interpolate x
    cx_original = np.linspace(0, 1, len(cx))
    cx_target = np.linspace(0, 1, num_points)
    fx = interp1d(cx_original, cx, kind='cubic')
    cx_interp = fx(cx_target)

    # Interpolate y
    cy_original = np.linspace(0, 1, len(cy))
    fy = interp1d(cy_original, cy, kind='cubic')
    cy_interp = fy(cx_target)

    # Interpolate z
    cz_original = np.linspace(0, 1, len(cz))
    fz = interp1d(cz_original, cz, kind='cubic')
    cz_interp = fz(cx_target)

    # Ensure z is within [0.8, 1.2] after interpolation
    cz_interp = np.clip(cz_interp, 0.8, 1.2)

    # Assemble s_choice
    s_choice = np.vstack((cx_interp, cy_interp, cz_interp)).T  # Shape (T, 3)

    # Generate synthetic data
    s_star, s_dummy = generate_data(true_phi_k, mu, sigma, s_choice)

    # Estimate phi_k
    phi_hat = estimate_phi(s_star, s_choice, s_dummy)

    print(f"True phi_k: {true_phi_k}")
    print(f"Estimated phi_k: {phi_hat:.4f}")

    # Plotting the results for x, y, z
    T = s_choice.shape[0]
    time_steps = np.arange(T)

    # Plot XY dimension
    plt.figure(figsize=(12, 4))
    plt.plot(s_star[:, 0], s_star[:, 1], label='s_star z', linewidth=2)
    plt.plot(s_choice[:, 0], s_choice[:, 1], label='s_choice z', linestyle='--')
    plt.plot(s_dummy[:, 0], s_dummy[:, 1], label='s_dummy z', linestyle=':')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.title('States Over Time (Z Dimension)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot XZ dimension
    plt.figure(figsize=(12, 4))
    plt.plot(s_star[:, 0], s_star[:, 2], label='s_star z', linewidth=2)
    plt.plot(s_choice[:, 0], s_choice[:, 2], label='s_choice z', linestyle='--')
    plt.plot(s_dummy[:, 0], s_dummy[:, 2], label='s_dummy z', linestyle=':')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Z')
    # plt.title('States Over Time (Z Dimension)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # # Plot X dimension
    # plt.figure(figsize=(12, 4))
    # plt.plot(time_steps, s_star[:, 0], label='s_star x', linewidth=2)
    # plt.plot(time_steps, s_choice[:, 0], label='s_choice x', linestyle='--')
    # plt.plot(time_steps, s_dummy[:, 0], label='s_dummy x', linestyle=':')
    # plt.legend()
    # plt.xlabel('Time Step')
    # plt.ylabel('X Value')
    # plt.title('States Over Time (X Dimension)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # Plot Y dimension
    # plt.figure(figsize=(12, 4))
    # plt.plot(time_steps, s_star[:, 1], label='s_star y', linewidth=2)
    # plt.plot(time_steps, s_choice[:, 1], label='s_choice y', linestyle='--')
    # plt.plot(time_steps, s_dummy[:, 1], label='s_dummy y', linestyle=':')
    # plt.legend()
    # plt.xlabel('Time Step')
    # plt.ylabel('Y Value')
    # plt.title('States Over Time (Y Dimension)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # Plot Z dimension
    # plt.figure(figsize=(12, 4))
    # plt.plot(time_steps, s_star[:, 2], label='s_star z', linewidth=2)
    # plt.plot(time_steps, s_choice[:, 2], label='s_choice z', linestyle='--')
    # plt.plot(time_steps, s_dummy[:, 2], label='s_dummy z', linestyle=':')
    # plt.legend()
    # plt.xlabel('Time Step')
    # plt.ylabel('Z Value')
    # plt.title('States Over Time (Z Dimension)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
