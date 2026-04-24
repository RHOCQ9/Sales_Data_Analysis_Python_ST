import pytest
import pandas as pd
import os
from loader.data_loader import DataLoader

@pytest.fixture
def sample_csv_path():
    """Ruta al archivo CSV de prueba existente"""
    return "data/sales.csv"

@pytest.fixture
def invalid_path():
    """Ruta a un archivo que no existe"""
    return "data/archivo_inexistente.csv"

def test_load_csv_file_success(sample_csv_path):
    """TC005: Cargar archivo CSV válido exitosamente"""
    loader = DataLoader()
    df = loader.load_csv(sample_csv_path)
    
    # Verificar que retorna un DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Verificar que tiene datos
    assert len(df) > 0
    
    # Verificar columnas esperadas del archivo sales.csv
    expected_columns = ["order_id", "date", "product", "region", "quantity", "price", "total_sales"]
    for col in expected_columns:
        assert col in df.columns

def test_load_csv_file_not_found(invalid_path):
    """TC005 (edge case): Error al cargar archivo CSV inexistente"""
    loader = DataLoader()
    
    with pytest.raises(FileNotFoundError):
        loader.load_csv(invalid_path)

def test_preview_data_default(sample_csv_path):
    """TC007: Preview de datos con valor por defecto (5 filas)"""
    loader = DataLoader()
    loader.load_csv(sample_csv_path)
    
    preview = loader.preview_data()
    
    # Verificar que retorna un DataFrame
    assert isinstance(preview, pd.DataFrame)
    
    # Verificar que retorna máximo 5 filas
    assert len(preview) <= 5

def test_preview_data_custom_rows(sample_csv_path):
    """TC007 (parametrizado): Preview con número custom de filas"""
    loader = DataLoader()
    loader.load_csv(sample_csv_path)
    
    # Probar con 3 filas
    preview_3 = loader.preview_data(rows=3)
    assert len(preview_3) <= 3
    
    # Probar con 10 filas (puede ser menor si el archivo tiene menos)
    preview_10 = loader.preview_data(rows=10)
    assert len(preview_10) <= 10

def test_get_loaded_dataframe(sample_csv_path):
    """TC008: Obtener DataFrame cargado previamente"""
    loader = DataLoader()
    
    # Cargar datos
    df_loaded = loader.load_csv(sample_csv_path)
    
    # Obtener DataFrame
    df_retrieved = loader.get_dataframe()
    
    # Verificar que son el mismo DataFrame
    assert isinstance(df_retrieved, pd.DataFrame)
    assert len(df_retrieved) == len(df_loaded)
    pd.testing.assert_frame_equal(df_retrieved, df_loaded)

def test_get_dataframe_before_loading():
    """TC008 (edge case): Obtener DataFrame antes de cargar datos"""
    loader = DataLoader()
    df = loader.get_dataframe()
    
    # Debe retornar None si no se ha cargado nada
    assert df is None
