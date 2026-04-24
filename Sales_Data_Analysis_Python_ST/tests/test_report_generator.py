import pytest
import pandas as pd
from utils.report_generator import ReportGenerator

@pytest.fixture
def complete_sales_dataframe():
    """DataFrame con todas las columnas esperadas para reportes"""
    data = {
        "order_id": [1, 2, 3],
        "product": ["Laptop", "Mouse", "Teclado"],
        "region": ["Norte", "Sur", "Centro"],
        "quantity": [2, 5, 3],
        "price": [800, 20, 50],
        "total_sales": [1600, 100, 150]
    }
    return pd.DataFrame(data)

@pytest.fixture
def partial_sales_dataframe():
    """DataFrame con solo algunas columnas (sin total_sales)"""
    data = {
        "order_id": [1, 2, 3],
        "product": ["Laptop", "Mouse", "Teclado"],
        "quantity": [2, 5, 3]
    }
    return pd.DataFrame(data)

def test_export_data_to_csv(complete_sales_dataframe):
    """TC009: Exportar DataFrame a CSV bytes con encoding UTF-8"""
    generator = ReportGenerator(complete_sales_dataframe)
    csv_bytes = generator.to_csv()
    
    # Verificar que retorna bytes
    assert isinstance(csv_bytes, bytes)
    
    # Verificar que puede decodificarse como UTF-8
    csv_string = csv_bytes.decode('utf-8')
    assert isinstance(csv_string, str)
    
    # Verificar que contiene las columnas
    assert "order_id" in csv_string
    assert "product" in csv_string
    assert "total_sales" in csv_string
    
    # Verificar que contiene datos
    assert "Laptop" in csv_string
    assert "1600" in csv_string

def test_csv_no_index(complete_sales_dataframe):
    """TC009 (validación): CSV generado no debe incluir el índice del DataFrame"""
    generator = ReportGenerator(complete_sales_dataframe)
    csv_bytes = generator.to_csv()
    csv_string = csv_bytes.decode('utf-8')
    
    # Verificar que no hay columna de índice (Unnamed: 0)
    assert "Unnamed" not in csv_string

def test_generate_text_summary_complete(complete_sales_dataframe):
    """TC010: Generar reporte de texto con todas las columnas esperadas"""
    generator = ReportGenerator(complete_sales_dataframe)
    report_bytes = generator.to_summary_text()
    
    # Verificar que retorna bytes
    assert isinstance(report_bytes, bytes)
    
    # Decodificar a string
    report_string = report_bytes.decode('utf-8')
    
    # Verificar que contiene el encabezado
    assert "REPORTE DE VENTAS" in report_string
    
    # Verificar que contiene estadísticas de ventas
    assert "Ventas Totales:" in report_string
    assert "1850.00" in report_string  # 1600 + 100 + 150
    
    # Verificar que contiene cantidad total
    assert "Cantidad Total:" in report_string
    assert "10" in report_string  # 2 + 5 + 3
    
    # Verificar que contiene número de órdenes
    assert "Número de Órdenes:" in report_string
    assert "3" in report_string
    
    # Verificar que contiene ventas por producto
    assert "Ventas por Producto:" in report_string
    assert "Laptop" in report_string
    assert "Mouse" in report_string
    
    # Verificar que contiene ventas por región
    assert "Ventas por Región:" in report_string
    assert "Norte" in report_string
    assert "Sur" in report_string

def test_generate_text_summary_partial(partial_sales_dataframe):
    """TC010 (edge case): Generar reporte con columnas faltantes (sin total_sales)"""
    generator = ReportGenerator(partial_sales_dataframe)
    report_bytes = generator.to_summary_text()
    
    # Verificar que retorna bytes
    assert isinstance(report_bytes, bytes)
    
    # Decodificar a string
    report_string = report_bytes.decode('utf-8')
    
    # Verificar que contiene el encabezado
    assert "REPORTE DE VENTAS" in report_string
    
    # Debe contener cantidad total (existe en el DataFrame)
    assert "Cantidad Total:" in report_string
    assert "10" in report_string
    
    # Debe contener número de órdenes
    assert "Número de Órdenes:" in report_string
    
    # NO debe tener "Ventas Totales" (columna faltante)
    # pero tampoco debe fallar, solo omite esa sección

def test_report_encoding_utf8(complete_sales_dataframe):
    """TC009/TC010 (validación): Verificar encoding UTF-8 en reportes"""
    # Crear DataFrame con caracteres especiales en español
    data = {
        "product": ["Móvil", "Ratón", "Configuración"],
        "region": ["España", "México", "Perú"],
        "total_sales": [100, 200, 300],
        "quantity": [1, 2, 3]
    }
    df_special = pd.DataFrame(data)
    
    generator = ReportGenerator(df_special)
    
    # Probar CSV
    csv_bytes = generator.to_csv()
    csv_string = csv_bytes.decode('utf-8')
    assert "Móvil" in csv_string
    assert "Ratón" in csv_string
    assert "Configuración" in csv_string
    
    # Probar reporte de texto
    report_bytes = generator.to_summary_text()
    report_string = report_bytes.decode('utf-8')
    assert "España" in report_string
    assert "México" in report_string
    assert "Perú" in report_string
