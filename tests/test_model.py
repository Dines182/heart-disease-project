# tests/test_model.py
import sys
import os
sys.path.append(os.path.abspath('../src'))  # Add src to Python path
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.model import load_data, scale_features, train_models, evaluate_model, plot_roc


@pytest.fixture
def small_dataset(tmp_path):
    """Create a tiny synthetic heart dataset for testing."""
    data = {
        "age": [60, 55, 40, 50],
        "sex": [1, 0, 1, 0],
        "cp": [3, 2, 1, 0],
        "trestbps": [130, 140, 120, 135],
        "chol": [240, 250, 200, 210],
        "fbs": [1, 0, 0, 1],
        "restecg": [0, 1, 1, 0],
        "thalach": [150, 160, 170, 155],
        "exang": [0, 0, 1, 1],
        "oldpeak": [1.0, 2.0, 1.5, 0.5],
        "slope": [0, 1, 2, 1],
        "ca": [0, 1, 0, 2],
        "thal": [2, 3, 1, 2],
        "target": [1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "heart.csv"
    df.to_csv(file_path, index=False)
    return df, file_path


def test_load_data(small_dataset):
    """Ensure load_data splits features and target correctly."""
    _, path = small_dataset
    X_train, X_test, y_train, y_test = load_data(path)
    assert not X_train.empty
    assert "target" not in X_train.columns
    assert set(y_train.unique()).issubset({0, 1})


def test_scale_features(small_dataset):
    """Check scaling keeps shape and transforms values."""
    _, path = small_dataset
    X_train, X_test, y_train, y_test = load_data(path)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    assert X_train_scaled.shape == X_train.shape
    assert np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-1)


def test_train_models(small_dataset):
    """Check models are trained and return correct types."""
    _, path = small_dataset
    X_train, X_test, y_train, y_test = load_data(path)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    log_reg, dt, rf = train_models(X_train_scaled, X_train, y_train)

    assert isinstance(log_reg, LogisticRegression)
    assert isinstance(dt, DecisionTreeClassifier)
    assert isinstance(rf, RandomForestClassifier)


def test_evaluate_model(capsys):
    """Check evaluate_model prints accuracy and report."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    evaluate_model(y_true, y_pred, "Dummy Model")

    captured = capsys.readouterr()
    assert "Accuracy" in captured.out
    assert "Confusion Matrix" in captured.out
    assert "Classification Report" in captured.out


def test_plot_roc(monkeypatch):
    """Check plot_roc runs without showing a window."""
    y_test = np.array([0, 1, 0, 1])
    X_test_scaled = np.array([[0.1], [0.9], [0.2], [0.8]])

    model = LogisticRegression()
    model.classes_ = np.array([0, 1])  # fake fitted model
    model.predict_proba = lambda X: np.array([[1-x[0], x[0]] for x in X])

    monkeypatch.setattr(plt, "show", lambda: None)  # avoid GUI popup

    try:
        plot_roc(y_test, model, X_test_scaled)
    except Exception as e:
        pytest.fail(f"plot_roc crashed with {e}")


if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train models
    log_reg, dt, rf = train_models(X_train_scaled, X_train, y_train)

    # Evaluate models
    evaluate_model(y_test, log_reg.predict(X_test_scaled), "Logistic Regression")
    evaluate_model(y_test, dt.predict(X_test), "Decision Tree")
    evaluate_model(y_test, rf.predict(X_test), "Random Forest")

    # Plot ROC for Logistic Regression
    plot_roc(y_test, log_reg, X_test_scaled)
