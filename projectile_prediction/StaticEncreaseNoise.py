import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from matplotlib.gridspec import GridSpec

from predictor_new import predict_future  # Using the new predictor

# -------------------------
# Trajectory Generators and addition functions (same as before)
# -------------------------
def generate_straight_trajectory(t, start_pos=(0, 0), velocity=(200, 0)):
    x = start_pos[0] + velocity[0] * t
    y = start_pos[1] + velocity[1] * t
    return (x, y)

def generate_circle_trajectory(t, center=(200, 200), radius=100, angular_speed=3.0, phase=0):
    angle = angular_speed * t + phase
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    return (x, y)

def generate_sine_trajectory(t, start_pos=(0, 0), x_velocity=200, amplitude=50, frequency=2, phase=0):
    x = start_pos[0] + x_velocity * t
    y = start_pos[1] + amplitude * np.sin(2 * np.pi * frequency * x / 300 + phase)
    return (x, y)

def add_noise(positions, noise_level=2.0):
    noisy_positions = []
    for pos in positions:
        noise_x = np.random.normal(0, noise_level)
        noise_y = np.random.normal(0, noise_level)
        noisy_positions.append((pos[0] + noise_x, pos[1] + noise_y))
    return noisy_positions

trajectory_generators = [
    ("Straight", lambda t: generate_straight_trajectory(t, start_pos=(0, 0), velocity=(200, 0))),
    ("Circle", lambda t: generate_circle_trajectory(t, center=(200, 200), radius=100, angular_speed=3.0)),
    ("Sine", lambda t: generate_sine_trajectory(t, start_pos=(0, 0), x_velocity=200, amplitude=50, frequency=2))
]

# -------------------------
# Simulation Loop Settings
# -------------------------
dt = 0.05       
n_steps = 20    
future_steps = 20  

noise_level = 0.0
iteration = 0

plt.ion()   # Enable interactive mode.
fig = plt.figure(figsize=(10, 16))
gs = GridSpec(len(trajectory_generators), 1, figure=fig, height_ratios=[3]*len(trajectory_generators))

while True:
    iteration += 1
    mse_list = []
    all_errors = {}
    fig.clf()  # Clear the figure to update the plots in the same window.
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for i, (name, trajectory_func) in enumerate(trajectory_generators):
        ax = fig.add_subplot(gs[i])
        
        total_steps = n_steps + future_steps
        all_times = np.arange(0, total_steps * dt, dt)
        all_positions = [trajectory_func(t) for t in all_times]
        
        history_times = all_times[:n_steps]
        history_pos_clean = all_positions[:n_steps]
        actual_future_times = all_times[n_steps:total_steps]
        actual_future_pos = all_positions[n_steps:total_steps]
        
        if name == "Straight":
            noise_level_straight = noise_level / 1
            history_pos_noisy = add_noise(history_pos_clean, noise_level_straight)
            ax.set_ylim(-10, 10)
        else:
            history_pos_noisy = add_noise(history_pos_clean, noise_level)
            
        time_position_history = [[t, pos[0], pos[1]] for t, pos in zip(history_times, history_pos_noisy)]
        predicted_future = predict_future(time_position_history,
                                          future_times=actual_future_times,
                                          num_points=future_steps)
        
        history_np_clean = np.array(history_pos_clean)
        history_np_noisy = np.array(history_pos_noisy)
        actual_future_np  = np.array(actual_future_pos)
        predicted_future_np = np.array(predicted_future)
        
        errors = [euclidean(pred, actual) for pred, actual in zip(predicted_future_np, actual_future_np)]
        mse = np.mean(np.square(errors))
        mse_list.append(mse)
        rmse = np.sqrt(mse)
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        all_errors[name] = {
            "mean_error": mean_error,
            "rmse": rmse,
            "max_error": max_error,
            "errors_over_time": errors
        }
        
        ax.plot(actual_future_np[:, 0], actual_future_np[:, 1], 'y-', label="Actual Future")
        ax.plot(predicted_future_np[:, 0], predicted_future_np[:, 1], linestyle=(0, (5, 5)), color='m', label="Predicted Future")
        ax.plot(history_np_clean[:, 0], history_np_clean[:, 1], 'co', alpha=0.3, label="True Historical Path")
        ax.plot(history_np_noisy[:, 0], history_np_noisy[:, 1], 'bo', label="Noisy Input Points")
        
        ax.set_title(f"{name} Movement - RMSE: {rmse:.2f}, Noise σ: {noise_level:.1f} px")
        ax.set_xlabel("x position (pixels)")
        ax.set_ylabel("y position (pixels)")
        ax.legend()
        ax.grid(True)
        if name == "Circle":
            ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Small pause to ensure the figure updates
    
    avg_mse = np.mean(mse_list)
    print(f"Iteration {iteration}: Noise level = {noise_level:.1f} px, Average MSE = {avg_mse:.2f} px²")
    

    noise_level += 0.5
    
    input("Press Enter to run the next iteration...")

print("Detailed Error Metrics:")
print("-" * 50)
for name, metrics in all_errors.items():
    print(f"Movement Type: {name}")
    print(f"  Mean Error: {metrics['mean_error']:.2f} pixels")
    print(f"  RMSE: {metrics['rmse']:.2f} pixels")
    print(f"  Maximum Error: {metrics['max_error']:.2f} pixels")
    print("-" * 50)