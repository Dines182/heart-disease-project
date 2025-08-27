# tests/test_eda.py
import sys
import os
sys.path.append(os.path.abspath('../src'))  # Add src to Python path
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from src.eda import load_data, run_eda

@pytest.fixture
def sample_df(tmp_path):
    """Create a small dummy heart dataset"""
    data = {
        "age": [63, 45, 56],
        "sex": [1, 0, 1],
        "cp": [3, 2, 1],
        "trestbps": [145, 130, 120],
        "chol": [233, 250, 210],
        "fbs": [1, 0, 0],
        "restecg": [0, 1, 1],
        "thalach": [150, 170, 165],
        "exang": [0, 0, 1],
        "oldpeak": [2.3, 1.5, 0.5],
        "slope": [0, 2, 1],
        "ca": [0, 0, 1],
        "thal": [1, 2, 3],
        "target": [1, 0, 1]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "heart.csv"
    df.to_csv(file_path, index=False)
    return df, file_path

def test_load_data(sample_df):
    """Test if load_data reads CSV correctly"""
    df, path = sample_df
    loaded_df = load_data(path)
    assert not loaded_df.empty
    assert "age" in loaded_df.columns
    assert "target" in loaded_df.columns

def test_run_eda_no_crash(sample_df, monkeypatch):
    """Ensure run_eda runs without crashing (mock plt.show)"""
    df, _ = sample_df

    # Mock plt.show to avoid blocking
    monkeypatch.setattr(plt, "show", lambda: None)

    try:
        run_eda(df)
    except Exception as e:
        pytest.fail(f"run_eda crashed with error: {e}")

print("Test executed Succesfully")