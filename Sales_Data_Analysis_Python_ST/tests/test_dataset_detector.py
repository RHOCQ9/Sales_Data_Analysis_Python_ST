import pytest
import pandas as pd
from analysis.dataset_detector import DatasetDetector

@pytest.fixture
def mixed_dataframe():
    """DataFrame con tipos mixtos de columnas"""
    data = {
        "id": [1, 2, 3, 4],
        "price": [100.5, 200.0, 300.75, 400.25],
        "product": ["Laptop", "Mouse", "Teclado", "Monitor"],
        "region": ["Norte", "Sur", "Centro", "Este"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "quantity": [2, 5, 3, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_dataframe():
    """DataFrame vacío"""
    return pd.DataFrame()

@pytest.fixture
def numeric_only_dataframe():
    """DataFrame solo con columnas numéricas"""
    data = {
        "col1": [1, 2, 3],
        "col2": [4.5, 5.6, 6.7],
        "col3": [10, 20, 30]
    }
    return pd.DataFrame(data)

def test_detect_numeric_columns(mixed_dataframe):
    """TC001: Detectar columnas numéricas correctamente"""
    detector = DatasetDetector(mixed_dataframe)
    numeric_cols = detector.detect_numeric()
    
    assert "id" in numeric_cols
    assert "price" in numeric_cols
    assert "quantity" in numeric_cols
    assert len(numeric_cols) == 3
    
    # Verificar que NO incluye columnas no numéricas
    assert "product" not in numeric_cols
    assert "region" not in numeric_cols
    assert "date" not in numeric_cols

def test_detect_numeric_empty_dataframe(empty_dataframe):
    """TC001 (edge case): Detectar columnas numéricas en DataFrame vacío"""
    detector = DatasetDetector(empty_dataframe)
    numeric_cols = detector.detect_numeric()
    
    assert numeric_cols == []
    assert len(numeric_cols) == 0

def test_detect_categorical_columns(mixed_dataframe):
    """TC002: Detectar columnas categóricas (object/string)"""
    detector = DatasetDetector(mixed_dataframe)
    categorical_cols = detector.detect_categorical()
    
    assert "product" in categorical_cols
    assert "region" in categorical_cols
    assert len(categorical_cols) == 2
    
    # Verificar que NO incluye columnas numéricas o datetime
    assert "id" not in categorical_cols
    assert "price" not in categorical_cols
    assert "date" not in categorical_cols

def test_detect_datetime_columns(mixed_dataframe):
    """TC003: Detectar columnas datetime"""
    detector = DatasetDetector(mixed_dataframe)
    datetime_cols = detector.detect_datetime()
    
    assert "date" in datetime_cols
    assert len(datetime_cols) == 1
    
    # Verificar que NO incluye otros tipos
    assert "product" not in datetime_cols
    assert "price" not in datetime_cols

def test_dataset_summary_complete(mixed_dataframe):
    """TC004: Obtener resumen completo de tipos de columnas"""
    detector = DatasetDetector(mixed_dataframe)
    summary = detector.dataset_summary()
    
    # Verificar estructura del diccionario
    assert isinstance(summary, dict)
    assert "numeric" in summary
    assert "categorical" in summary
    assert "datetime" in summary
    
    # Verificar contenido
    assert len(summary["numeric"]) == 3
    assert len(summary["categorical"]) == 2
    assert len(summary["datetime"]) == 1
    
    # Verificar valores específicos
    assert "price" in summary["numeric"]
    assert "product" in summary["categorical"]
    assert "date" in summary["datetime"]

def test_dataset_summary_numeric_only(numeric_only_dataframe):
    """TC004 (edge case): Resumen con solo columnas numéricas"""
    detector = DatasetDetector(numeric_only_dataframe)
    summary = detector.dataset_summary()
    
    assert len(summary["numeric"]) == 3
    assert len(summary["categorical"]) == 0
    assert len(summary["datetime"]) == 0
