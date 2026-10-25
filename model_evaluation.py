import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def validate_aerodynamic_bounds(features_data):
    """
    Validates that input features fall within physically possible aerodynamic bounds.

    Parameters
    ----------
    features_data : array-like
        The input dataset containing aerodynamic features (Frequency, Angle of Attack, 
        Chord length, Free-stream velocity, and Displacement thickness).

    Raises
    ------
    ValueError
        If unsupported inputs (e.g., negative physical dimensions or NaNs) are detected.
    """
    data_array = np.asarray(features_data)
    # Feature indices: 0=Frequency, 1=Angle of Attack, 2=Chord, 3=Velocity, 4=Displacement Thickness
    if (data_array[:, 0] <= 0).any():
        raise ValueError("Invalid Input: Frequency must be greater than 0 Hz.")
    if (data_array[:, 2] <= 0).any():
        raise ValueError("Invalid Input: Chord length must be strictly positive.")
    if (data_array[:, 3] <= 0).any():
        raise ValueError("Invalid Input: Free-stream velocity must be greater than 0 m/s.")
    if (data_array[:, 4] <= 0).any():
        raise ValueError("Invalid Input: Suction side displacement thickness must be strictly positive.")
    
    # Check for missing values (NaNs)
    if np.isnan(data_array).any():
        raise ValueError("Invalid Input: Dataset contains missing values (NaNs).")

# Fetch airfoil data
airfoil = fetch_ucirepo(id=291)
features = airfoil.data.features
targets = airfoil.data.targets.values.ravel()

# Validate raw data before processing
validate_aerodynamic_bounds(features)

# Split and scale data
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.2, random_state=42
)
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR(kernel='rbf', C=100, gamma='scale'),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True)
}

model_results = {}

# Warm-up run to initialize NumPy/SciPy libraries
_ = LinearRegression().fit(features_train, targets_train)

# Train and evaluate
for model_name, model_instance in models.items():
    start_time = time.perf_counter()
    if model_name in ["SVR", "Neural Network"]:
        model_instance.fit(features_train_scaled, targets_train)
        predictions = model_instance.predict(features_test_scaled)
    else:
        model_instance.fit(features_train, targets_train)
        predictions = model_instance.predict(features_test)
    
    model_results[model_name] = {
        'predictions': predictions,
        'root_mean_squared_error': root_mean_squared_error(targets_test, predictions),
        'mean_absolute_error': mean_absolute_error(targets_test, predictions),
        'r_squared_value': r2_score(targets_test, predictions),
        'training_time': time.perf_counter() - start_time
    }

def calculate_points(value, min_val, max_val, lower_is_better=True):
    """
    Calculates a Min-Max scaled proportional score between 0.0 and 1.0.

    Parameters
    ----------
    value : float
        The specific metric value to be scored.
    min_val : float
        The minimum bound of the metric across all evaluated models.
    max_val : float
        The maximum bound of the metric across all evaluated models.
    lower_is_better : bool, optional
        Indicates if a lower metric value represents better performance 
        (default is True for metrics like RMSE, MAE, and Time).

    Returns
    -------
    float
        The normalized score bounded between 0.0 and 1.0.
    """
    if max_val == min_val:
        return 1.0
    if lower_is_better:
        return (max_val - value) / (max_val - min_val)
    else:
        return (value - min_val) / (max_val - min_val)

# Extract min/max boundaries
rmses = [res['root_mean_squared_error'] for res in model_results.values()]
maes = [res['mean_absolute_error'] for res in model_results.values()]
r2s = [res['r_squared_value'] for res in model_results.values()]
times = [res['training_time'] for res in model_results.values()]

for model_name, res in model_results.items():
    res['rmse_points'] = calculate_points(res['root_mean_squared_error'], min(rmses), max(rmses), True)
    res['mae_points'] = calculate_points(res['mean_absolute_error'], min(maes), max(maes), True)
    res['r_squared_points'] = calculate_points(res['r_squared_value'], min(r2s), max(r2s), False)
    res['time_points'] = calculate_points(res['training_time'], min(times), max(times), True)

best_total_score = -1
best_model_name = ""

print("--- Model Performance Comparison ---")
for model_name, result_data in model_results.items():
    total_points = (result_data['rmse_points'] + 
                    result_data['mae_points'] + 
                    result_data['r_squared_points'] + 
                    result_data['time_points'])
    
    percentage_score = (total_points / 4.0) * 100
    result_data['total_score'] = percentage_score
    
    if percentage_score > best_total_score:
        best_total_score = percentage_score
        best_model_name = model_name

for model_name, result_data in model_results.items():
    marker = "<-- BEST" if model_name == best_model_name else ""
    print(f"{model_name.ljust(17)} | RMSE: {result_data['root_mean_squared_error']:5.2f} | "
          f"MAE: {result_data['mean_absolute_error']:5.2f} | R²: {result_data['r_squared_value']:5.3f} | "
          f"Time: {result_data['training_time']:5.4f}s | Score: {result_data['total_score']:5.2f}% {marker}")

# Plot 2x2 grid
figure, axes = plt.subplots(2, 2, figsize=(12, 10))
axes_flat = axes.flatten()

for index, (model_name, result_data) in enumerate(model_results.items()):
    axis = axes_flat[index]
    is_best_model = (model_name == best_model_name)
    
    axis.scatter(targets_test, result_data['predictions'], alpha=0.4, color='purple' if is_best_model else 'steelblue')
    axis.plot([min(targets_test), max(targets_test)], [min(targets_test), max(targets_test)], color='red', linestyle='--')
    axis.set_xlabel("Actual SPL (dB)")
    axis.set_ylabel("Predicted SPL (dB)")
    
    axis.set_title(f"{model_name}\nRMSE: {result_data['root_mean_squared_error']:.2f} | Score: {result_data['total_score']:.2f}%", 
                 color='green' if is_best_model else 'black', 
                 fontweight='bold' if is_best_model else 'normal')

plt.tight_layout()
plt.show()