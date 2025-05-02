import math
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt # Added import

def predict_future_visual(time_position_history, future_times=None, num_points=50, sine_amplitude_threshold=1.0, circle_radius_threshold=1000.0):
    """
    Given a history of [time, x, y] points, fit all three models (straight, circle, sine)
    to the history and predict future positions. Plots initial guesses and final fits
    based ONLY on the historical time points.

    Parameters:
        time_position_history: List of [time, x, y] points or numpy array of shape (n, 3)
        future_times: List of specific future time points to predict at
        num_points: Number of future points to predict if future_times not provided
        sine_amplitude_threshold: Minimum amplitude for a sine wave model to be selected
        circle_radius_threshold: Maximum allowed radius for a circle model to be selected

    Returns:
        List of [x, y] predicted positions for the requested future_times
    """
    if len(time_position_history) < 2:
        return []

    # Convert to numpy array if it's not already
    history_array = np.array(time_position_history)

    # Extract time, x, y components
    times = history_array[:, 0]
    positions = history_array[:, 1:3]

    # Determine future times if not provided (for actual prediction return value)
    if future_times is None:
        dt = np.mean(np.diff(times)) if len(times) > 1 else 0.1
        last_time = times[-1]
        future_times_predict = np.array([last_time + dt * (i+1) for i in range(num_points)])
    else:
        future_times_predict = np.array(future_times)

    # --- Plot Initial Guesses (using only historical times) ---
    fig_guesses, axes_guesses = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig_guesses.suptitle('Initial Model Guesses (on History)', fontsize=16)
    # all_times_for_plot = np.concatenate((times, future_times)) # No longer needed for plotting

    # Straight Line Guess
    b_x_guess, a_x_guess = np.polyfit(times, positions[:, 0], 1)
    b_y_guess, a_y_guess = np.polyfit(times, positions[:, 1], 1)
    straight_guess_params = (a_x_guess, b_x_guess, a_y_guess, b_y_guess)
    # Predict only over historical times for plotting
    straight_guess_preds = predict_straight_future(straight_guess_params, times)
    straight_guess_preds_np = np.array(straight_guess_preds)
    axes_guesses[0].plot(positions[:, 0], positions[:, 1], 'bo', label='History')
    axes_guesses[0].plot(straight_guess_preds_np[:, 0], straight_guess_preds_np[:, 1], 'r--', label='Initial Guess')
    axes_guesses[0].set_title('Straight Line Guess')
    axes_guesses[0].legend()
    axes_guesses[0].grid(True)
    axes_guesses[0].set_ylabel('Y Position')

    # Circle Guess
    circle_guess_params = None # Initialize
    try:
        a_guess, b_center_guess, r_guess = fit_circle_least_squares(positions)
        angles_guess = np.unwrap(np.arctan2(positions[:, 1] - b_center_guess, positions[:, 0] - a_guess))
        omega_guess, theta0_guess = np.polyfit(times, angles_guess, 1)
        circle_guess_params = (a_guess, b_center_guess, r_guess, theta0_guess, omega_guess)
        # Predict only over historical times for plotting
        circle_guess_preds = predict_circle_future(circle_guess_params, times)
        circle_guess_preds_np = np.array(circle_guess_preds)
        axes_guesses[1].plot(positions[:, 0], positions[:, 1], 'bo', label='History')
        axes_guesses[1].plot(circle_guess_preds_np[:, 0], circle_guess_preds_np[:, 1], 'r--', label='Initial Guess')
        axes_guesses[1].set_title('Circle Guess')
        axes_guesses[1].legend()
        axes_guesses[1].grid(True)
        axes_guesses[1].set_ylabel('Y Position')
        axes_guesses[1].set_aspect('equal', adjustable='datalim')
    except Exception as e:
        axes_guesses[1].set_title(f'Circle Guess Failed: {e}')
        axes_guesses[1].plot(positions[:, 0], positions[:, 1], 'bo', label='History')
        axes_guesses[1].legend()
        axes_guesses[1].grid(True)
        axes_guesses[1].set_ylabel('Y Position')


    # Sine Guess
    sine_guess_params = None # Initialize
    try:
        sine_guess_params = get_sine_initial_guess(times, positions)
        # Predict only over historical times for plotting
        sine_guess_preds = predict_parametric_sine_future(sine_guess_params, times)
        sine_guess_preds_np = np.array(sine_guess_preds)
        axes_guesses[2].plot(positions[:, 0], positions[:, 1], 'bo', label='History')
        axes_guesses[2].plot(sine_guess_preds_np[:, 0], sine_guess_preds_np[:, 1], 'r--', label='Initial Guess')
        axes_guesses[2].set_title('Sine Guess')
        axes_guesses[2].legend()
        axes_guesses[2].grid(True)
        axes_guesses[2].set_xlabel('X Position')
        axes_guesses[2].set_ylabel('Y Position')
    except Exception as e:
        axes_guesses[2].set_title(f'Sine Guess Failed: {e}')
        axes_guesses[2].plot(positions[:, 0], positions[:, 1], 'bo', label='History')
        axes_guesses[2].legend()
        axes_guesses[2].grid(True)
        axes_guesses[2].set_xlabel('X Position')
        axes_guesses[2].set_ylabel('Y Position')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    # --- End Plot Initial Guesses ---

    # Fit each model on the entire history
    straight_params, err_straight = fit_straight(times, positions)
    circle_params, err_circle = fit_circle(times, positions)
    # Pass sine guess if available, otherwise fit_parametric_sine will calculate it
    sine_params, err_sine = fit_parametric_sine(times, positions, sine_guess_params)

    # Print errors for debugging
    print(f"Model errors - Straight: {err_straight:.2f}, Circle: {err_circle:.2f}, Sine: {err_sine:.2f}")

    errors = {"straight": err_straight, "circle": err_circle, "sine": err_sine}

    # --- Model Selection Logic ---
    valid_models = {}

    # Check Straight
    valid_models["straight"] = err_straight

    # Check Circle
    if circle_params is not None:
        a_c, b_center_c, r_c, theta0_c, omega_c = circle_params
        if r_c <= circle_radius_threshold:
            valid_models["circle"] = err_circle
            print(f"Circle model valid with radius: {r_c:.2f}")
        else:
            print(f"Circle model invalid: radius ({r_c:.2f}) exceeds threshold ({circle_radius_threshold}).")
            err_circle = float('inf') # Penalize invalid model

    # Check Sine
    if sine_params is not None:
        x0_s, vx_s, Ax_s, y0_s, vy_s, Ay_s, freq_s, phase_s = sine_params
        total_amplitude_s = np.sqrt(Ax_s**2 + Ay_s**2)
        if total_amplitude_s >= sine_amplitude_threshold:
            valid_models["sine"] = err_sine
            print(f"Sine model valid with amplitude: {total_amplitude_s:.2f}")
        else:
            print(f"Sine model invalid: amplitude ({total_amplitude_s:.2f}) below threshold ({sine_amplitude_threshold}).")
            err_sine = float('inf') # Penalize invalid model

    if not valid_models:
         print("Warning: No models were deemed valid based on thresholds. Defaulting to straight line.")
         best_model = "straight"
    else:
        best_model = min(valid_models, key=valid_models.get)


    print(f"Best model selected: {best_model}")

    # --- Plot Fitted Models (using only historical times) ---
    fig_fits, axes_fits = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig_fits.suptitle('Fitted Models (on History)', fontsize=16)

    # Straight Fit
    # Predict only over historical times for plotting
    straight_fit_preds = predict_straight_future(straight_params, times)
    straight_fit_preds_np = np.array(straight_fit_preds)
    axes_fits[0].plot(positions[:, 0], positions[:, 1], 'bo', label='History')
    axes_fits[0].plot(straight_fit_preds_np[:, 0], straight_fit_preds_np[:, 1], 'g-', label='Fitted Path')
    axes_fits[0].set_title(f'Straight Fit (Error: {err_straight:.2f}){" - BEST" if best_model == "straight" else ""}')
    axes_fits[0].legend()
    axes_fits[0].grid(True)
    axes_fits[0].set_ylabel('Y Position')

    # Circle Fit
    if circle_params is not None and err_circle != float('inf'):
        # Predict only over historical times for plotting
        circle_fit_preds = predict_circle_future(circle_params, times)
        circle_fit_preds_np = np.array(circle_fit_preds)
        axes_fits[1].plot(positions[:, 0], positions[:, 1], 'bo', label='History')
        axes_fits[1].plot(circle_fit_preds_np[:, 0], circle_fit_preds_np[:, 1], 'g-', label='Fitted Path')
        axes_fits[1].set_title(f'Circle Fit (Error: {err_circle:.2f}, R: {circle_params[2]:.1f}){" - BEST" if best_model == "circle" else ""}')
        axes_fits[1].legend()
        axes_fits[1].grid(True)
        axes_fits[1].set_ylabel('Y Position')
        axes_fits[1].set_aspect('equal', adjustable='datalim')
    else:
        axes_fits[1].set_title('Circle Fit Invalid or Failed')
        axes_fits[1].plot(positions[:, 0], positions[:, 1], 'bo', label='History')
        axes_fits[1].legend()
        axes_fits[1].grid(True)
        axes_fits[1].set_ylabel('Y Position')


    # Sine Fit
    if sine_params is not None and err_sine != float('inf'):
        # Predict only over historical times for plotting
        sine_fit_preds = predict_parametric_sine_future(sine_params, times)
        sine_fit_preds_np = np.array(sine_fit_preds)
        axes_fits[2].plot(positions[:, 0], positions[:, 1], 'bo', label='History')
        axes_fits[2].plot(sine_fit_preds_np[:, 0], sine_fit_preds_np[:, 1], 'g-', label='Fitted Path')
        total_amplitude_s = np.sqrt(sine_params[2]**2 + sine_params[5]**2)
        axes_fits[2].set_title(f'Sine Fit (Error: {err_sine:.2f}, Amp: {total_amplitude_s:.1f}){" - BEST" if best_model == "sine" else ""}')
        axes_fits[2].legend()
        axes_fits[2].grid(True)
        axes_fits[2].set_xlabel('X Position')
        axes_fits[2].set_ylabel('Y Position')
    else:
        axes_fits[2].set_title('Sine Fit Invalid or Failed')
        axes_fits[2].plot(positions[:, 0], positions[:, 1], 'bo', label='History')
        axes_fits[2].legend()
        axes_fits[2].grid(True)
        axes_fits[2].set_xlabel('X Position')
        axes_fits[2].set_ylabel('Y Position')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    # --- End Plot Fitted Models ---

    # Return predictions from the best model for the requested future times
    if best_model == "straight":
        return predict_straight_future(straight_params, future_times_predict)
    elif best_model == "circle":
        return predict_circle_future(circle_params, future_times_predict)
    else: # Best is Sine
        return predict_parametric_sine_future(sine_params, future_times_predict)

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
    Can raise LinAlgError if points are collinear.
    """
    x = positions[:, 0]
    y = positions[:, 1]
    A = np.column_stack((2*x, 2*y, np.ones_like(x)))
    b = x**2 + y**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None) # Can raise LinAlgError
    a, b_center, c = sol
    # Ensure the value inside sqrt is non-negative
    radius_sq = a*a + b_center*b_center + c
    if radius_sq < 0:
        # This can happen with noisy data or near-collinear points
        # Fallback: maybe return None or raise a specific error?
        # For now, let's return a very small radius to avoid math domain error
        print("Warning: Negative value in sqrt for circle radius calculation. Data might be noisy or collinear.")
        r = 1e-6 
    else:
        r = math.sqrt(radius_sq)
    return a, b_center, r

def fit_circle(times, positions):
    """
    Fit a circular model to the history.
    
    Parameters:
        times: Array of time values
        positions: Array of [x, y] positions of shape (n, 2)
        
    Returns:
        parameters (a, b_center, r, theta0, omega) and the sum-of-squared error, or (None, inf) if fit fails
    """
    try:
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
    except np.linalg.LinAlgError:
        print("Circle fit failed (likely collinear points).")
        return None, float('inf')
    except Exception as e:
        print(f"Circle fit failed with unexpected error: {e}")
        return None, float('inf')


def parametric_sine_model(t, x0, vx, Ax, y0, vy, Ay, freq, phase):
    """
    A parametric model that can represent a sine wave in any direction:
    x(t) = x0 + vx*t + Ax*sin(2*pi*freq*t + phase)
    y(t) = y0 + vy*t + Ay*sin(2*pi*freq*t + phase)
    """
    # Ensure t is a numpy array for vectorized operations
    t = np.asarray(t)
    angle_term = 2 * np.pi * freq * t + phase
    x = x0 + vx * t + Ax * np.sin(angle_term)
    y = y0 + vy * t + Ay * np.sin(angle_term)
    # Return shape depends on input t shape, handle scalar vs array
    if t.ndim == 0: # Scalar input
        return np.array([x, y])
    else: # Array input
        return np.column_stack((x, y))


def get_sine_initial_guess(times, positions):
    """ Calculates initial guess parameters for the sine model. """
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
    freq_guess = 1.0 # Default fallback
    if len(times) > 5: # Need a few points for FFT
        # Ensure times are evenly spaced for simple FFT freq calculation
        dt = np.mean(np.diff(times))
        if dt > 0: # Avoid division by zero
            # Get FFT
            fft_vals = np.fft.rfft(residuals)
            fft_freqs = np.fft.rfftfreq(len(residuals), dt)
            # Find dominant frequency (skipping DC component at index 0)
            if len(fft_vals) > 1:
                dominant_idx = np.argmax(np.abs(fft_vals[1:])) + 1
                # Basic check for validity
                if dominant_idx < len(fft_freqs):
                     freq_guess = abs(fft_freqs[dominant_idx]) # Use abs value
                else: # Fallback if index is out of bounds (shouldn't happen with rfft)
                     freq_guess = 1.0 / (times[-1] - times[0]) if (times[-1] - times[0]) > 0 else 1.0
            else:
                # Fallback if FFT has only one component (or timespan is zero)
                freq_guess = 1.0 / (times[-1] - times[0]) if (times[-1] - times[0]) > 0 else 1.0
        else:
             freq_guess = 1.0 # Fallback if dt is zero or negative
    elif len(times) > 1 and (times[-1] - times[0]) > 0:
        # Fallback for short time series (estimate based on half a cycle over the duration)
        freq_guess = 0.5 / (times[-1] - times[0])
    
    # Ensure freq_guess is positive and not too large
    freq_guess = max(0.01, min(freq_guess, 15.0))

    # Estimate amplitudes
    Ax_guess = np.std(x_residuals) * 1.414 if np.std(x_residuals) > 1e-6 else 0.1 # std * sqrt(2) for sine waves
    Ay_guess = np.std(y_residuals) * 1.414 if np.std(y_residuals) > 1e-6 else 0.1
    phase_guess = 0.0
    
    # Initial guess
    initial_params = [x0, vx, Ax_guess, y0, vy, Ay_guess, freq_guess, phase_guess]
    return initial_params


def fit_parametric_sine(times, positions, initial_params=None):
    """
    Fit a parametric sine model that can handle any orientation.
    
    Parameters:
        times: Array of time values
        positions: Array of [x, y] positions of shape (n, 2)
        initial_params: Optional pre-calculated initial guess
        
    Returns:
        parameters (x0, vx, Ax, y0, vy, Ay, freq, phase) and the sum-of-squared error, or (None, inf) if fit fails
    """
    if initial_params is None:
        initial_params = get_sine_initial_guess(times, positions)

    # Function to minimize (sum of squared errors)
    def error_func(params):
        x0, vx, Ax, y0, vy, Ay, freq, phase = params
        # Ensure freq is positive during optimization steps if method allows constraints
        # freq = abs(freq) # Or handle via bounds
        predicted = parametric_sine_model(times, x0, vx, Ax, y0, vy, Ay, freq, phase)
        # Ensure predicted has the same shape as positions
        if predicted.shape != positions.shape:
             # This might happen if times was scalar, handle appropriately
             # For fitting, times should always be an array, so this is unlikely
             return float('inf') 
        return np.sum(np.sum((positions - predicted) ** 2, axis=1))
    
    # Bounds to keep parameters reasonable
    # Allow negative amplitude initially, let optimizer decide direction via phase?
    # Or constrain Ax, Ay >= 0 and let phase handle it. Let's constrain >= 0.
    bounds = [
        (None, None),  # x0
        (None, None),  # vx
        (0, None),     # Ax (non-negative)
        (None, None),  # y0
        (None, None),  # vy
        (0, None),     # Ay (non-negative)
        (0.01, 15),    # freq (reasonable positive range)
        (-2*np.pi, 2*np.pi) # phase (allow full range, maybe wrap later)
    ]
    
    # Use minimize instead of curve_fit for more control
    try:
        # L-BFGS-B respects bounds
        result = minimize(error_func, initial_params, bounds=bounds, method='L-BFGS-B') 
        if result.success:
            params = result.x
            # Wrap phase to [-pi, pi] for consistency if desired
            # params[7] = (params[7] + np.pi) % (2 * np.pi) - np.pi 
            error = result.fun
            return params, error
        else:
            print(f"Sine optimization failed: {result.message}")
            # Return initial guess error if optimization failed
            return initial_params, error_func(initial_params) 
            
    except ValueError as ve:
         print(f"Sine optimization failed with ValueError (likely bounds/initial guess issue): {ve}")
         return None, float('inf')
    except Exception as e:
        print(f"Sine optimization failed with unexpected error: {e}")
        return None, float('inf')


###############################
# Prediction (Extrapolation)  #
###############################

def predict_straight_future(params, future_times):
    """
    Given a fitted straight-line model with parameters (a_x, b_x, a_y, b_y),
    predict future positions at the specified future times.
    """
    a_x, b_x, a_y, b_y = params
    future_times = np.asarray(future_times) # Ensure numpy array
    predicted_x = a_x + b_x * future_times
    predicted_y = a_y + b_y * future_times
    return list(zip(predicted_x, predicted_y))

def predict_circle_future(params, future_times):
    """
    Given a fitted circle model with parameters (a, b_center, r, theta0, omega),
    predict future positions at the specified future times.
    """
    a, b_center, r, theta0, omega = params
    future_times = np.asarray(future_times) # Ensure numpy array
    predicted_x = a + r * np.cos(theta0 + omega * future_times)
    predicted_y = b_center + r * np.sin(theta0 + omega * future_times)
    return list(zip(predicted_x, predicted_y))

def predict_parametric_sine_future(params, future_times):
    """
    Given fitted parametric sine model parameters, predict future positions.
    """
    x0, vx, Ax, y0, vy, Ay, freq, phase = params
    future_times = np.asarray(future_times) # Ensure numpy array
    predictions = parametric_sine_model(future_times, x0, vx, Ax, y0, vy, Ay, freq, phase)
    return list(zip(predictions[:, 0], predictions[:, 1]))