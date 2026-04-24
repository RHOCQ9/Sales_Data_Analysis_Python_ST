import pytest
import pandas as pd
import numpy as np
from models.auto_ml import AutoML

@pytest.fixture
def regression_data():
    # Need at least 5-10 rows to have meaningful test split (20%)
    data = {
        "quantity": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "order_id": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
    return pd.DataFrame(data)

@pytest.fixture
def insufficient_data():
    return pd.DataFrame({"quantity": [1], "order_id": [10]})

def test_regression_success(regression_data):
    # TC-05: Entrenamiento Exitoso de Regresión
    automl = AutoML(regression_data)
    result = automl.train_regression(target="order_id", features=["quantity"])
    
    assert "mae" in result
    assert "r2" in result
    assert result["r2"] > 0.0  # Should have some predictive power since it's a perfect linear relation
    assert "predictions" in result
    assert "real" in result
    # Nuevas keys para SHAP
    assert "model" in result
    assert "X_train" in result

def test_insufficient_data(insufficient_data):
    # TC-06: Datos Insuficientes para Entrenamiento
    automl = AutoML(insufficient_data)
    
    # Requirement: "El sistema muestra un mensaje de error indicando que se requieren más datos"
    with pytest.raises(ValueError) as excinfo:
        automl.train_regression(target="order_id", features=["quantity"])
    
    assert "requieren más datos para el entrenamiento" in str(excinfo.value).lower()
