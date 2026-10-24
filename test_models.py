import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from ucimlrepo import fetch_ucirepo

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