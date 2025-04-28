import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from matplotlib.gridspec import GridSpec

from predictor_new import predict_future  # Using the new predictor

# Define manual trajectory functions for each movement type
def generate_straight_trajectory(t, start_pos=(0, 0), velocity=(30, 0)):
    """Generate position at time t for straight-line movement"""
    x = start_pos[0] + velocity[0] * t
    y = start_pos[1] + velocity[1] * t
    return (x, y)

def generate_circle_trajectory(t, center=(200, 200), radius=100, angular_speed=1.0, phase=0):
    """Generate position at time t for circular movement"""
    angle = angular_speed * t + phase
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    return (x, y)

def generate_sine_trajectory(t, start_pos=(0, 0), x_velocity=30, amplitude=50, frequency=2, phase=0):
    """Generate position at time t for sinusoidal movement"""
    x = start_pos[0] + x_velocity * t
    y = start_pos[1] + amplitude * np.sin(2 * np.pi * frequency * x / 300 + phase)
    return (x, y)

def add_noise(positions, noise_level=2.0):
    """Add Gaussian noise to a list of positions"""
    noisy_positions = []
    for pos in positions:
        noise_x = np.random.normal(0, noise_level)
        noise_y = np.random.normal(0, noise_level)
        noisy_positions.append((pos[0] + noise_x, pos[1] + noise_y))
    return noisy_positions

# List of segment names and their corresponding generator functions
trajectory_generators = [
    ("Straight", lambda t: generate_straight_trajectory(t, start_pos=(0, 0), velocity=(200, 0))),
    ("Circle", lambda t: generate_circle_trajectory(t, center=(200, 200), radius=100, angular_speed=3.0)),
    ("Sine", lambda t: generate_sine_trajectory(t, start_pos=(0, 0), x_velocity=200, amplitude=50, frequency=2))
]

# Simulation parameters
dt = 0.05                 # time step in seconds
n_steps = 20              # number of simulation steps for history
future_steps = 20        # number of future steps to predict and validate
noise_level = 1.0         # standard deviation of the Gaussian noise (in pixels)

# Create figure with proper size for all plots
fig = plt.figure(figsize=(10, 16))
num_plots = len(trajectory_generators) + 1  # +1 for the error plot

# Create GridSpec for better control over subplot layout
gs = GridSpec(num_plots, 1, figure=fig, height_ratios=[3]*len(trajectory_generators) + [0])

# Store error metrics for all trajectory types
all_errors = {}
actual_future_times = None  # Will be set in the loop

# Set random seed for reproducible results
np.random.seed(42)

# Loop over each trajectory type, generate positions, predict future positions, and plot
for i, (name, trajectory_func) in enumerate(trajectory_generators):
    print("Name:", name)
    # Create subplot using GridSpec
    ax = fig.add_subplot(gs[i])
    
    # Generate complete trajectory (history + actual future)
    total_steps = n_steps + future_steps
    all_times = np.arange(0, total_steps * dt, dt)
    all_positions = [trajectory_func(t) for t in all_times]
    
    # Split into history and actual future
    history_times = all_times[:n_steps]
    history_pos_clean = all_positions[:n_steps]
    actual_future_times = all_times[n_steps:total_steps]
    actual_future_pos = all_positions[n_steps:total_steps]
    
    # Add noise to history positions (but keep true future positions clean)
    if name == "Straight":
        noise_level_straight = noise_level / 10
        history_pos_noisy = add_noise(history_pos_clean, noise_level_straight)
        ax.set_ylim(-10, 10)
    else:
        history_pos_noisy = add_noise(history_pos_clean, noise_level)
    
    # Create time-position history in format [time, x, y] with noise
    time_position_history = []
    for t, pos in zip(history_times, history_pos_noisy):
        time_position_history.append([t, pos[0], pos[1]])
    
    # Use predictor to compute future positions
    predicted_future = predict_future(
        time_position_history, 
        future_times=actual_future_times,
        num_points=future_steps
    )
    
    # Convert to numpy arrays for easier manipulation
    history_np_clean = np.array(history_pos_clean)
    history_np_noisy = np.array(history_pos_noisy)
    actual_future_np = np.array(actual_future_pos)
    predicted_future_np = np.array(predicted_future)
    
    # Calculate error metrics
    errors = []
    for pred, actual in zip(predicted_future_np, actual_future_np):
        error = euclidean(pred, actual)
        errors.append(error)
    
    mse = np.mean(np.square(errors))
    rmse = np.sqrt(mse)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    # Store error metrics
    all_errors[name] = {
        "mean_error": mean_error,
        "rmse": rmse,
        "max_error": max_error,
        "errors_over_time": errors
    }
    
    # Plot actual future as a solid green line
    ax.plot(actual_future_np[:, 0], actual_future_np[:, 1], 'y-', label="Actual Future")
    
    # Plot predicted future as a dashed magenta line with slightly less spacing between dashes
    ax.plot(predicted_future_np[:, 0], predicted_future_np[:, 1], linestyle=(0, (5, 5)), color='m', label="Predicted Future")
    
    # Plot true history points (faded)
    ax.plot(history_np_clean[:, 0], history_np_clean[:, 1], 'co', alpha=0.3, label="True Historical Path")
    
    # Plot noisy history points (these are what the predictor sees)
    ax.plot(history_np_noisy[:, 0], history_np_noisy[:, 1], 'bo', label="Noisy Input Points")
    
    # Add error metrics and noise level to the title
    ax.set_title(f"{name} Movement - RMSE: {rmse:.2f}, Noise σ: {noise_level:.1f} px")
    ax.set_xlabel("x position (pixels)")
    ax.set_ylabel("y position (pixels)")
    ax.legend()
    ax.grid(True)
    
    # Set aspect ratio to 1:1 for the circle plot
    if name == "Circle":
        ax.set_aspect('equal', adjustable='datalim')

# End of trajectory plotting figure
plt.tight_layout()
# plt.show()

# Create a new figure for the error plot (separate window)
error_fig, error_ax = plt.subplots(figsize=(8, 6))
for name in all_errors:
    error_ax.plot(actual_future_times, all_errors[name]["errors_over_time"], label=f"{name}")

error_ax.set_title(f"Prediction Error Over Time (Noise Level σ = {noise_level:.1f} px)")
error_ax.set_xlabel("Time (seconds)")
error_ax.set_ylabel("Error (pixels)")
error_ax.legend()
error_ax.grid(True)

plt.tight_layout()
plt.show()

# Print detailed error metrics for each trajectory type
print(f"Detailed Error Metrics (Noise Level σ = {noise_level:.1f} px):")
print("-" * 50)
for name, metrics in all_errors.items():
    print(f"Movement Type: {name}")
    print(f"  Mean Error: {metrics['mean_error']:.2f} pixels")
    print(f"  RMSE: {metrics['rmse']:.2f} pixels")
    print(f"  Maximum Error: {metrics['max_error']:.2f} pixels")
    print("-" * 50)