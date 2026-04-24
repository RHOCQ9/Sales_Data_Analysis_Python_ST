import pytest
import pandas as pd
import numpy as np
from analysis.eda_analyzer import EDAAnalyzer

@pytest.fixture
def numeric_dataframe():
    data = {
        "ventas": [100, 200, 300, 400, 500],
        "costo": [50, 100, 150, 200, 250],
        "cantidad": [1, 2, 3, 4, 5]
    }
    return pd.DataFrame(data)

@pytest.fixture
def single_variable_dataframe():
    return pd.DataFrame({"ventas": [100, 200, 300]})

def test_summary_statistics(numeric_dataframe):
    # TC-03: Cálculo de Estadísticas Descriptivas
    analyzer = EDAAnalyzer(numeric_dataframe)
    summary = analyzer.summary_statistics()
    
    # Expected: The result of analyzer.summary_statistics() should match pandas df.describe()
    pd.testing.assert_frame_equal(summary, numeric_dataframe.describe())
    
    # Check some values
    assert summary.loc["mean", "ventas"] == 300.0
    assert summary.loc["50%", "ventas"] == 300.0

def test_single_variable_correlation(single_variable_dataframe):
    # TC-04: Detección de Dataset con una Sola Variable
    analyzer = EDAAnalyzer(single_variable_dataframe)
    
    # Requirement: "El sistema informa que no es posible realizar un análisis de correlación"
    # Current implementation might just return a single cell matrix or similar.
    # We expect it to inform/raise an exception if we are to follow the prompt.
    
    with pytest.raises(ValueError) as excinfo:
        analyzer.correlation_matrix()
    
    assert "no es posible realizar un análisis de correlación" in str(excinfo.value).lower()

def test_basic_info(numeric_dataframe):
    # TC-09: Información Básica del Dataset
    analyzer = EDAAnalyzer(numeric_dataframe)
    info = analyzer.basic_info()
    assert info["rows"] == 5
    assert info["columns"] == 3
    assert "ventas" in info["column_names"]
