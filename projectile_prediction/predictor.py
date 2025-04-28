# predictor.py
import math
import numpy as np
from scipy.optimize import curve_fit

def predict_future(history, num_points=50, dt=0.1):
    """
    Given a history (list of (pos, vel) tuples), fit all three models (straight, circle, sine)
    to the history and compute their errors. The model with the lowest error is used to predict
    the next num_points positions.
    
    dt is the time interval (in seconds) between history samples.
    """
    if len(history) < 2:
        return []
    
    # Fit each model on the entire history:
    straight_params, err_straight = fit_straight(history, dt)
    circle_params, err_circle = fit_circle(history, dt)
    sine_params, err_sine = fit_sine(history)
    
    errors = {"straight": err_straight, "circle": err_circle, "sine": err_sine}
    best_model = min(errors, key=errors.get)
    # print("Model errors:", errors, "Chosen model:", best_model)
    
    if best_model == "straight":
        return predict_straight_future(history, straight_params, num_points, dt)
    elif best_model == "circle":
        return predict_circle_future(history, circle_params, num_points, dt)
    else:
        return predict_sine_future(history, sine_params, num_points)

###########################
# Model Fitting Functions #
###########################

def fit_straight(history, dt):
    """
    Fit a straight-line model (constant velocity) to the history.
    We assume the model p(t) = a + b*t, where t = index*dt.
    
    Returns parameters (a_x, b_x, a_y, b_y) and the sum-of-squared error.
    """
    n = len(history)
    t = np.arange(n) * dt
    positions = np.array([p[0] for p in history])  # shape: (n,2)
    x = positions[:, 0]
    y = positions[:, 1]
    # Fit linear regression (np.polyfit returns [slope, intercept])
    b_x, a_x = np.polyfit(t, x, 1)
    b_y, a_y = np.polyfit(t, y, 1)
    predicted_x = a_x + b_x * t
    predicted_y = a_y + b_y * t
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

def fit_circle(history, dt):
    """
    Fit a circular model to the history.
    
    First, fit a circle to all the positions (using a least-squares circle fit).
    Then, compute the angle (theta) of each point relative to the circle center and 
    fit a linear model theta(t) = theta0 + omega*t (with t = index*dt).
    
    Returns parameters (a, b_center, r, theta0, omega) and the sum-of-squared error.
    """
    n = len(history)
    t = np.arange(n) * dt
    positions = np.array([p[0] for p in history])
    # Fit circle to all positions:
    a, b_center, r = fit_circle_least_squares(positions)
    # Compute angles relative to the fitted center:
    angles = np.arctan2(positions[:, 1] - b_center, positions[:, 0] - a)
    angles = np.unwrap(angles)
    # Fit a linear model for angle as a function of time:
    omega, theta0 = np.polyfit(t, angles, 1)  # omega: slope, theta0: intercept
    predicted_x = a + r * np.cos(theta0 + omega * t)
    predicted_y = b_center + r * np.sin(theta0 + omega * t)
    error = np.sum((positions[:, 0] - predicted_x) ** 2 +
                   (positions[:, 1] - predicted_y) ** 2)
    return (a, b_center, r, theta0, omega), error

def fit_sine(history):
    """
    Fit a sine model to the history.
    We assume the model: y = D + A * sin(omega*x + phi),
    where x is taken directly from the positions.
    
    Returns parameters (D, A, omega, phi) and the sum-of-squared error.
    """
    positions = np.array([p[0] for p in history])
    x_data = positions[:, 0]
    y_data = positions[:, 1]
    
    def sine_model(x, D, A, omega, phi):
        return D + A * np.sin(omega * x + phi)
    
    D_guess = np.mean(y_data)
    A_guess = (np.max(y_data) - np.min(y_data)) / 2
    # A reasonable guess for omega: one full cycle over the span of x_data.
    span = np.max(x_data) - np.min(x_data)
    omega_guess = 2 * math.pi / span if span != 0 else 2 * math.pi
    phi_guess = 0
    initial_guess = [D_guess, A_guess, omega_guess, phi_guess]
    
    try:
        popt, _ = curve_fit(sine_model, x_data, y_data, p0=initial_guess)
        D, A, omega, phi = popt
    except Exception:
        D, A, omega, phi = initial_guess
    predicted_y = sine_model(x_data, D, A, omega, phi)
    error = np.sum((y_data - predicted_y) ** 2)
    return (D, A, omega, phi), error

###############################
# Prediction (Extrapolation)  #
###############################

def predict_straight_future(history, params, num_points, dt):
    """
    Given a fitted straight-line model with parameters (a_x, b_x, a_y, b_y),
    predict future positions for time steps t = n, n+1, ... (with dt spacing).
    """
    n = len(history)
    t_future = np.arange(n, n + num_points) * dt
    a_x, b_x, a_y, b_y = params
    predicted_x = a_x + b_x * t_future
    predicted_y = a_y + b_y * t_future
    return list(zip(predicted_x, predicted_y))

def predict_circle_future(history, params, num_points, dt):
    """
    Given a fitted circle model with parameters (a, b_center, r, theta0, omega),
    predict future positions using t = n, n+1, ... (with dt spacing).
    """
    n = len(history)
    t_future = np.arange(n, n + num_points) * dt
    a, b_center, r, theta0, omega = params
    predicted_x = a + r * np.cos(theta0 + omega * t_future)
    predicted_y = b_center + r * np.sin(theta0 + omega * t_future)
    return list(zip(predicted_x, predicted_y))

def predict_sine_future(history, params, num_points):
    """
    Given a fitted sine model with parameters (D, A, omega, phi), predict future positions.
    Since the sine model gives y as a function of x, we extrapolate x linearly.
    """
    positions = np.array([p[0] for p in history])  # shape: (n, 2)
    # Extract the x-coordinate of the last position:
    x_last = positions[-1, 0]
    
    # Estimate average horizontal increment from history:
    if len(positions) > 1:
        dx = np.mean(np.diff(positions[:, 0]))
    else:
        dx = 10  # fallback value
        
    future_x = np.array([x_last + dx * i for i in range(1, num_points + 1)])
    D, A, omega, phi = params
    future_y = D + A * np.sin(omega * future_x + phi)
    return list(zip(future_x, future_y))