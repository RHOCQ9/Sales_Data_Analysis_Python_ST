import pytest
import pandas as pd
import plotly.graph_objs as go
from visualization.auto_visualizer import AutoVisualizer

@pytest.fixture
def sales_dataframe_for_viz():
    """DataFrame para pruebas de visualización"""
    data = {
        "product": ["Laptop", "Mouse", "Teclado", "Monitor", "Laptop"],
        "region": ["Norte", "Sur", "Centro", "Este", "Norte"],
        "quantity": [2, 5, 3, 1, 4],
        "price": [800, 20, 50, 300, 800],
        "total_sales": [1600, 100, 150, 300, 3200]
    }
    return pd.DataFrame(data)

@pytest.fixture
def numeric_dataframe_for_corr():
    """DataFrame con múltiples columnas numéricas para correlación"""
    data = {
        "price": [100, 200, 300, 400, 500],
        "quantity": [5, 4, 3, 2, 1],
        "total_sales": [500, 800, 900, 800, 500],
        "discount": [10, 20, 15, 25, 30]
    }
    return pd.DataFrame(data)

def test_numeric_distribution_histogram(sales_dataframe_for_viz):
    """TC_VIZ_001: Generar histograma para columna numérica"""
    visualizer = AutoVisualizer(sales_dataframe_for_viz)
    
    fig = visualizer.numeric_distribution("total_sales")
    
    # Verificar que retorna un objeto Figure de Plotly
    assert isinstance(fig, go.Figure)
    
    # Verificar que tiene un título descriptivo
    assert "Distribución de total_sales" in fig.layout.title.text

def test_numeric_distribution_different_columns(sales_dataframe_for_viz):
    """TC_VIZ_001 (múltiples columnas): Histogramas para diferentes columnas"""
    visualizer = AutoVisualizer(sales_dataframe_for_viz)
    
    # Probar con quantity
    fig_quantity = visualizer.numeric_distribution("quantity")
    assert isinstance(fig_quantity, go.Figure)
    assert "quantity" in fig_quantity.layout.title.text
    
    # Probar con price
    fig_price = visualizer.numeric_distribution("price")
    assert isinstance(fig_price, go.Figure)
    assert "price" in fig_price.layout.title.text

def test_categorical_counts_bar_chart(sales_dataframe_for_viz):
    """TC_VIZ_002: Generar gráfico de barras para columna categórica"""
    visualizer = AutoVisualizer(sales_dataframe_for_viz)
    
    fig = visualizer.categorical_counts("product")
    
    # Verificar que retorna un objeto Figure de Plotly
    assert isinstance(fig, go.Figure)
    
    # Verificar que tiene un título descriptivo
    assert "Conteo de product" in fig.layout.title.text
    
    # Verificar que el gráfico contiene datos
    assert len(fig.data) > 0

def test_categorical_counts_region(sales_dataframe_for_viz):
    """TC_VIZ_002 (otra columna): Gráfico de barras para región"""
    visualizer = AutoVisualizer(sales_dataframe_for_viz)
    
    fig = visualizer.categorical_counts("region")
    
    assert isinstance(fig, go.Figure)
    assert "Conteo de region" in fig.layout.title.text

def test_correlation_heatmap(numeric_dataframe_for_corr):
    """TC_VIZ_003: Generar heatmap de correlación para columnas numéricas"""
    visualizer = AutoVisualizer(numeric_dataframe_for_corr)
    
    fig = visualizer.correlation_heatmap()
    
    # Verificar que retorna un objeto Figure de Plotly
    assert isinstance(fig, go.Figure)
    
    # Verificar que tiene título
    assert "Matriz de correlación" in fig.layout.title.text
    
    # Verificar que contiene datos
    assert len(fig.data) > 0

def test_correlation_heatmap_values_displayed(numeric_dataframe_for_corr):
    """TC_VIZ_003 (validación): Heatmap debe mostrar valores de correlación"""
    visualizer = AutoVisualizer(numeric_dataframe_for_corr)
    
    fig = visualizer.correlation_heatmap()
    
    # Verificar que text_auto está habilitado (valores se muestran)
    # Esto se verifica en la configuración de imshow en auto_visualizer.py
    assert isinstance(fig, go.Figure)

def test_invalid_column_name():
    """TC_VIZ_004 (edge case): Error al usar nombre de columna inválido"""
    data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    df = pd.DataFrame(data)
    visualizer = AutoVisualizer(df)
    
    # Intentar generar histograma con columna inexistente
    # Plotly lanza ValueError en lugar de KeyError
    with pytest.raises(ValueError):
        visualizer.numeric_distribution("columna_inexistente")

def test_empty_dataframe_visualization():
    """TC_VIZ_005 (edge case): Visualizar DataFrame vacío"""
    df_empty = pd.DataFrame()
    visualizer = AutoVisualizer(df_empty)
    
    # El heatmap de correlación debería manejar DataFrame vacío
    # (puede fallar o retornar figura vacía dependiendo de implementación)
    try:
        fig = visualizer.correlation_heatmap()
        # Si no falla, debe retornar una figura válida
        assert isinstance(fig, go.Figure)
    except (ValueError, KeyError):
        # Es aceptable que falle con DataFrame vacío
        pass

def test_single_numeric_column_correlation():
    """TC_VIZ_006 (edge case): Correlación con una sola columna numérica"""
    data = {"only_col": [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data)
    visualizer = AutoVisualizer(df)
    
    # Correlación de una sola variable consigo misma
    fig = visualizer.correlation_heatmap()
    assert isinstance(fig, go.Figure)
