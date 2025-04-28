import math
import numpy as np
from scipy.optimize import curve_fit, minimize

def predict_future(time_position_history, future_times=None, num_points=50, sine_amplitude_threshold=1.0):
    """
    Given a history of [time, x, y] points, fit all three models (straight, circle, sine)
    to the history and predict future positions.
    
    Parameters:
        time_position_history: List of [time, x, y] points or numpy array of shape (n, 3)
        future_times: List of specific future time points to predict at
        num_points: Number of future points to predict if future_times not provided
        sine_amplitude_threshold: Minimum amplitude for a sine wave model to be selected
    
    Returns:
        List of [x, y] predicted positions
    """
    if len(time_position_history) < 2:
        return []
    
    # Convert to numpy array if it's not already
    history_array = np.array(time_position_history)
    
    # Extract time, x, y components
    times = history_array[:, 0]
    positions = history_array[:, 1:3]
    
    # Determine future times if not provided
    if future_times is None:
        dt = np.mean(np.diff(times)) if len(times) > 1 else 0.1
        last_time = times[-1]
        future_times = np.array([last_time + dt * (i+1) for i in range(num_points)])
    else:
        future_times = np.array(future_times)
    
    # Fit each model on the entire history
    straight_params, err_straight = fit_straight(times, positions)
    circle_params, err_circle = fit_circle(times, positions)
    sine_params, err_sine = fit_parametric_sine(times, positions)
    
    # Print errors for debugging
    print(f"Model errors - Straight: {err_straight:.2f}, Circle: {err_circle:.2f}, Sine: {err_sine:.2f}")
    
    errors = {"straight": err_straight, "circle": err_circle, "sine": err_sine}
    best_model = min(errors, key=errors.get)
    
    # Check if sine model is selected, but has very low amplitude
    if best_model == "sine":
        x0, vx, Ax, y0, vy, Ay, freq, phase = sine_params
        total_amplitude = np.sqrt(Ax**2 + Ay**2)  # Calculate total amplitude
        
        if total_amplitude < sine_amplitude_threshold:
            print(f"Sine model selected but amplitude ({total_amplitude:.2f}) below threshold ({sine_amplitude_threshold}).")
            print(f"Defaulting to straight line model instead.")
            best_model = "straight"
        else:
            print(f"Sine model selected with amplitude: {total_amplitude:.2f}")
    
    print(f"Best model: {best_model}")
    
    if best_model == "straight":
        return predict_straight_future(straight_params, future_times)
    elif best_model == "circle":
        return predict_circle_future(circle_params, future_times)
    else:
        return predict_parametric_sine_future(sine_params, future_times)

###########################
# Model Fitting Functions #
###########################

def fit_straight(times, positions):
    """
    Fit a straight-line model (constant velocity) to the history.
    We assume the model p(t) = a + b*t
    
    Parameters:
        times: Array of time values
        positions: Array of [x, y] positions of shape (n, 2)
    
    Returns:
        parameters (a_x, b_x, a_y, b_y) and the sum-of-squared error
    """
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Fit linear regression (np.polyfit returns [slope, intercept])
    b_x, a_x = np.polyfit(times, x, 1)
    b_y, a_y = np.polyfit(times, y, 1)
    
    predicted_x = a_x + b_x * times
    predicted_y = a_y + b_y * times
    
    error = np.sum((x - predicted_x) ** 2 + (y - predicted_y) ** 2)
    return (a_x, b_x, a_y, b_y), error

def fit_circle_least_squares(positions):
    """
    Given an array of positions (shape: (n,2)), fit a circle using a simple algebraic method.
    Returns center coordinates (a, b_center) and radius r.
    """
    x = positions[:, 0]
    y = positions[:, 1]
    A = np.column_stack((2*x, 2*y, np.ones_like(x)))
    b = x**2 + y**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_center, c = sol
    r = math.sqrt(a*a + b_center*b_center + c)
    return a, b_center, r

def fit_circle(times, positions):
    """
    Fit a circular model to the history.
    
    Parameters:
        times: Array of time values
        positions: Array of [x, y] positions of shape (n, 2)
        
    Returns:
        parameters (a, b_center, r, theta0, omega) and the sum-of-squared error
    """
    # Fit circle to all positions:
    a, b_center, r = fit_circle_least_squares(positions)
    
    # Compute angles relative to the fitted center:
    angles = np.arctan2(positions[:, 1] - b_center, positions[:, 0] - a)
    angles = np.unwrap(angles)
    
    # Fit a linear model for angle as a function of time:
    omega, theta0 = np.polyfit(times, angles, 1)  # omega: slope, theta0: intercept
    
    predicted_x = a + r * np.cos(theta0 + omega * times)
    predicted_y = b_center + r * np.sin(theta0 + omega * times)
    
    error = np.sum((positions[:, 0] - predicted_x) ** 2 +
                   (positions[:, 1] - predicted_y) ** 2)
    
    return (a, b_center, r, theta0, omega), error

def parametric_sine_model(t, x0, vx, Ax, y0, vy, Ay, freq, phase):
    """
    A parametric model that can represent a sine wave in any direction:
    x(t) = x0 + vx*t + Ax*sin(2*pi*freq*t + phase)
    y(t) = y0 + vy*t + Ay*sin(2*pi*freq*t + phase)
    """
    x = x0 + vx * t + Ax * np.sin(2 * np.pi * freq * t + phase)
    y = y0 + vy * t + Ay * np.sin(2 * np.pi * freq * t + phase)
    return np.column_stack((x, y))

def fit_parametric_sine(times, positions):
    """
    Fit a parametric sine model that can handle any orientation.
    
    Parameters:
        times: Array of time values
        positions: Array of [x, y] positions of shape (n, 2)
        
    Returns:
        parameters (x0, vx, Ax, y0, vy, Ay, freq, phase) and the sum-of-squared error
    """
    # First, get linear velocity estimates (needed for initial guesses)
    vx, x0 = np.polyfit(times, positions[:, 0], 1)
    vy, y0 = np.polyfit(times, positions[:, 1], 1)
    
    # Estimate frequency from oscillations around the linear trend
    x_residuals = positions[:, 0] - (x0 + vx * times)
    y_residuals = positions[:, 1] - (y0 + vy * times)
    
    # Use FFT to estimate frequency (on the component with larger variation)
    if np.var(x_residuals) > np.var(y_residuals):
        residuals = x_residuals
    else:
        residuals = y_residuals
    
    # If we have enough points, try to estimate frequency
    if len(times) > 10:
        # Ensure times are evenly spaced
        dt = np.mean(np.diff(times))
        # Get FFT
        fft_vals = np.fft.rfft(residuals)
        fft_freqs = np.fft.rfftfreq(len(residuals), dt)
        # Find dominant frequency (skipping DC component at index 0)
        if len(fft_vals) > 1:
            dominant_idx = np.argmax(np.abs(fft_vals[1:])) + 1
            freq_guess = fft_freqs[dominant_idx]
        else:
            # Fallback if FFT has only one component
            freq_guess = 1.0 / (times[-1] - times[0]) * 2
    else:
        # Fallback for short time series
        freq_guess = 1.0 / (times[-1] - times[0]) * 2
    
    # Estimate amplitudes
    Ax_guess = np.std(x_residuals) * 1.414  # std * sqrt(2) for sine waves
    Ay_guess = np.std(y_residuals) * 1.414
    phase_guess = 0.0
    
    # Initial guess
    initial_params = [x0, vx, Ax_guess, y0, vy, Ay_guess, freq_guess, phase_guess]
    
    # Function to minimize (sum of squared errors)
    def error_func(params):
        x0, vx, Ax, y0, vy, Ay, freq, phase = params
        predicted = parametric_sine_model(times, x0, vx, Ax, y0, vy, Ay, freq, phase)
        return np.sum(np.sum((positions - predicted) ** 2, axis=1))
    
    # Bounds to keep parameters reasonable
    bounds = [
        (None, None),  # x0
        (None, None),  # vx
        (0, None),     # Ax (positive)
        (None, None),  # y0
        (None, None),  # vy
        (0, None),     # Ay (positive)
        (0.01, 15),    # freq (reasonable range)
        (-np.pi, np.pi) # phase
    ]
    
    # Use minimize instead of curve_fit for more control
    try:
        result = minimize(error_func, initial_params, bounds=bounds, method='L-BFGS-B')
        params = result.x
        error = result.fun
    except Exception as e:
        print(f"Optimization failed: {e}")
        params = initial_params
        error = error_func(initial_params)
    
    return params, error

###############################
# Prediction (Extrapolation)  #
###############################

def predict_straight_future(params, future_times):
    """
    Given a fitted straight-line model with parameters (a_x, b_x, a_y, b_y),
    predict future positions at the specified future times.
    """
    a_x, b_x, a_y, b_y = params
    predicted_x = a_x + b_x * future_times
    predicted_y = a_y + b_y * future_times
    return list(zip(predicted_x, predicted_y))

def predict_circle_future(params, future_times):
    """
    Given a fitted circle model with parameters (a, b_center, r, theta0, omega),
    predict future positions at the specified future times.
    """
    a, b_center, r, theta0, omega = params
    predicted_x = a + r * np.cos(theta0 + omega * future_times)
    predicted_y = b_center + r * np.sin(theta0 + omega * future_times)
    return list(zip(predicted_x, predicted_y))

def predict_parametric_sine_future(params, future_times):
    """
    Given fitted parametric sine model parameters, predict future positions.
    """
    x0, vx, Ax, y0, vy, Ay, freq, phase = params
    predictions = parametric_sine_model(future_times, x0, vx, Ax, y0, vy, Ay, freq, phase)
    return list(zip(predictions[:, 0], predictions[:, 1]))