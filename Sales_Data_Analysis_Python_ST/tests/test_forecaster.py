import pytest
import pandas as pd
from models.forecaster import SalesForecaster

@pytest.fixture
def sample_sales_data():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "total_sales": [100, 150, 130, 170]
    })

def test_train_and_forecast_daily(sample_sales_data):
    forecaster = SalesForecaster(sample_sales_data)
    result = forecaster.train_and_forecast("date", "total_sales", periods=3, freq='D')
    
    assert "historical" in result
    assert "forecast" in result
    assert len(result["forecast"]) == 3
    assert result["mae"] >= 0
    assert result["r2"] <= 1.0

def test_train_and_forecast_monthly():
    data = pd.DataFrame({
        "date": ["2024-01-15", "2024-02-15", "2024-03-15"],
        "total_sales": [1000, 2000, 1500]
    })
    forecaster = SalesForecaster(data)
    result = forecaster.train_and_forecast("date", "total_sales", periods=2, freq='M')
    
    assert len(result["forecast"]) == 2

def test_insufficient_data():
    data = pd.DataFrame({"date": ["2024-01-01"], "total_sales": [100]})
    forecaster = SalesForecaster(data)
    with pytest.raises(ValueError, match="Se requieren más datos para el entrenamiento"):
        forecaster.train_and_forecast("date", "total_sales", periods=3)

def test_insufficient_grouped_data():
    # Fechas duplicadas terminan agrupándose en 1 solo periodo de fecha, lo cual fallaría en la regresión
    data = pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "total_sales": [100, 200]})
    forecaster = SalesForecaster(data)
    with pytest.raises(ValueError, match="Se requieren más datos para el entrenamiento después de agrupar por fecha"):
        forecaster.train_and_forecast("date", "total_sales", periods=3)

def test_missing_columns(sample_sales_data):
    forecaster = SalesForecaster(sample_sales_data)
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        forecaster.train_and_forecast("fechas", "total_sales", periods=2)

def test_invalid_frequency(sample_sales_data):
    forecaster = SalesForecaster(sample_sales_data)
    with pytest.raises(ValueError, match="Frecuencia no soportada"):
        forecaster.train_and_forecast("date", "total_sales", periods=2, freq='Y')
