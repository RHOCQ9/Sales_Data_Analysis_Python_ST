import pytest
import pandas as pd
from analysis.sales_analyzer import SalesAnalyzer

@pytest.fixture
def sales_dataframe():
    data = {
        "product": ["A", "B", "A", "C"],
        "total_sales": [100, 200, 150, 300],
        "region": ["North", "South", "North", "East"],
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    }
    return pd.DataFrame(data)

def test_total_sales(sales_dataframe):
    # TC-10: Cálculo de Ventas Totales
    analyzer = SalesAnalyzer(sales_dataframe)
    total = analyzer.total_sales()
    assert total == 750

def test_sales_by_product(sales_dataframe):
    # TC-11: Ventas por Producto
    analyzer = SalesAnalyzer(sales_dataframe)
    sales_prod = analyzer.sales_by_product()
    assert sales_prod["A"] == 250
    assert sales_prod["B"] == 200
    assert sales_prod["C"] == 300

def test_sales_by_region(sales_dataframe):
    analyzer = SalesAnalyzer(sales_dataframe)
    sales_reg = analyzer.sales_by_region()
    assert sales_reg["North"] == 250

def test_monthly_sales(sales_dataframe):
    analyzer = SalesAnalyzer(sales_dataframe)
    m_sales = analyzer.monthly_sales()
    assert len(m_sales) == 1

def test_top_products(sales_dataframe):
    analyzer = SalesAnalyzer(sales_dataframe)
    top = analyzer.top_products(n=2)
    assert len(top) == 2
    assert top.index[0] == "C"

def test_generate_summary(sales_dataframe):
    analyzer = SalesAnalyzer(sales_dataframe)
    summary = analyzer.generate_summary()
    assert summary["total_sales"] == 750
    assert "top_products" in summary
    assert "sales_by_region" in summary
