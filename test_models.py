import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from ucimlrepo import fetch_ucirepo
from model_evaluation import validate_aerodynamic_bounds

def test_data_loading():
    """Positive test: Verify the Airfoil dataset loads with exactly 5 features."""
    airfoil = fetch_ucirepo(id=291)
    features = airfoil.data.features
    targets = airfoil.data.targets
    
    assert features.shape[1] == 5, "Dataset should have exactly 5 features"
    assert len(features) > 0, "Dataset should not be empty"
    assert len(features) == len(targets), "Features and targets must have matching row counts"

def test_invalid_data_type():
    """Negative test: Ensure the model raises a ValueError when fed string data."""
    model = LinearRegression()
    invalid_features = np.array([["text", "data"]])
    invalid_targets = np.array([1])
    
    with pytest.raises(ValueError):
        model.fit(invalid_features, invalid_targets)

def test_aerodynamic_bounds_validation():
    """Negative test: Ensure ValueError is raised for physically impossible aerodynamic inputs."""
    # Feature indices: 0=Freq, 1=AoA, 2=Chord, 3=Velocity, 4=Disp. Thickness
    
    # Test 1: Negative frequency
    invalid_freq_data = np.array([[-500, 5.0, 0.1, 31.7, 0.002]])
    with pytest.raises(ValueError, match="Frequency must be greater than 0 Hz"):
        validate_aerodynamic_bounds(invalid_freq_data)
        
    # Test 2: Negative chord length
    invalid_chord_data = np.array([[1000, 5.0, -0.1, 31.7, 0.002]])
    with pytest.raises(ValueError, match="Chord length must be strictly positive"):
        validate_aerodynamic_bounds(invalid_chord_data)