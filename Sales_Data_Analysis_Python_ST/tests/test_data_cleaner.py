import pytest
import pandas as pd
import numpy as np
from utils.data_cleaner import DataCleaner

@pytest.fixture
def messy_dataframe():
    data = {
        "PRECIO VENTA ": [100, 200, 300],
        " CANTIDAD": [1, 2, 3],
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def invalid_date_dataframe():
    data = {
        "date": ["2023-01-01", "not-a-date", "2023-01-03"],
        "price": [10, 20, 30]
    }
    return pd.DataFrame(data)

def test_normalize_columns(messy_dataframe):
    # TC-01: Normalización de Columnas
    cleaner = DataCleaner(messy_dataframe)
    cleaner.normalize_columns()
    
    expected_columns = ["precio venta", "cantidad", "date"]
    # We use lower() and strip() based on the requirement
    # Note: Current implementation in data_cleaner.py only does lower()
    assert list(cleaner.df.columns) == expected_columns

def test_handle_invalid_dates(invalid_date_dataframe):
    # TC-02: Manejo de Fechas Inválidas
    cleaner = DataCleaner(invalid_date_dataframe)
    
    # According to contextTest.txt: "El sistema detecta el error de formato y notifica al usuario"
    # This implies it shouldn't just crash but handle it. 
    # Current implementation uses pd.to_datetime which might raise an error or return NaT depending on errors parameter.
    with pytest.raises(Exception) as excinfo:
        cleaner.fix_data_types()
    
    assert "no pudo convertir la columna a tipo temporal" in str(excinfo.value).lower()

def test_remove_nulls():
    # TC-07: Eliminación de Nulos
    df = pd.DataFrame({"a": [1, None, 3]})
    cleaner = DataCleaner(df)
    cleaner.remove_nulls()
    assert len(cleaner.df) == 2

def test_remove_duplicates():
    # TC-08: Eliminación de Duplicados
    df = pd.DataFrame({"a": [1, 1, 2]})
    cleaner = DataCleaner(df)
    cleaner.remove_duplicates()
    assert len(cleaner.df) == 2

def test_clean_method(messy_dataframe):
    # TC-12 (Extra): Flujo completo de limpieza
    cleaner = DataCleaner(messy_dataframe)
    result = cleaner.clean()
    assert "precio venta" in result.columns
    assert result["date"].dtype == "datetime64[ns]"
