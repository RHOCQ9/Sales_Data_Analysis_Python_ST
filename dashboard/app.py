import sys
import os

# Agregar la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from loader.data_loader import DataLoader
from utils.data_cleaner import DataCleaner
from analysis.sales_analyzer import SalesAnalyzer
from analysis.eda_analyzer import EDAAnalyzer


from analysis.dataset_detector import DatasetDetector
from visualization.auto_visualizer import AutoVisualizer
from models.auto_ml import AutoML

st.title("📊 Plataforma de Análisis de Ventas")

# =============================
# CARGAR DATASET
# =============================

st.sidebar.header("Cargar dataset")

uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Usando dataset de ejemplo")
    loader = DataLoader()
    df = loader.load_csv("data/sales.csv")

# =============================
# LIMPIEZA DE DATOS
# =============================

cleaner = DataCleaner(df)
df = cleaner.clean()

analyzer = SalesAnalyzer(df)
# =============================
# Detector
# =============================

detector = DatasetDetector(df)

summary = detector.dataset_summary()

st.subheader("Tipos de variables")

st.write(summary)
# =============================
# Visualizacion automatica
# =============================

visualizer = AutoVisualizer(df)

for col in summary["numeric"]:
    fig = visualizer.numeric_distribution(col)
    st.plotly_chart(fig)

for col in summary["categorical"]:
    fig = visualizer.categorical_counts(col)
    st.plotly_chart(fig)

st.plotly_chart(visualizer.correlation_heatmap())
# =============================
# Machine Learning
# =============================
st.header("Machine Learning")

automl = AutoML(df)

target = st.selectbox("Variable objetivo", summary["numeric"])

features = st.multiselect(
    "Variables predictoras",
    [col for col in summary["numeric"] if col != target]
)

if features:

    results = automl.train_regression(target, features)

    col1, col2 = st.columns(2)

    col1.metric("MAE", round(results["mae"],2))
    col2.metric("R2", round(results["r2"],2))


# =============================
# VISTA PREVIA DEL DATASET
# =============================

st.subheader("Vista previa del dataset")
st.dataframe(df.head())

st.subheader("Información del dataset")

col1, col2 = st.columns(2)

col1.metric("Filas", df.shape[0])
col2.metric("Columnas", df.shape[1])

st.write("Columnas disponibles:")
st.write(list(df.columns))

# =============================
# FILTROS
# =============================

st.sidebar.header("Filtros")

if "region" in df.columns and "product" in df.columns:

    region = st.sidebar.multiselect(
        "Seleccionar región",
        df["region"].unique(),
        default=df["region"].unique()
    )

    product = st.sidebar.multiselect(
        "Seleccionar producto",
        df["product"].unique(),
        default=df["product"].unique()
    )

    df_filtered = df[
        (df["region"].isin(region)) &
        (df["product"].isin(product))
    ]

else:
    st.warning("El dataset no contiene columnas 'region' o 'product'.")
    df_filtered = df

# =============================
# MÉTRICAS
# =============================

if "total_sales" in df_filtered.columns and "quantity" in df_filtered.columns:

    st.subheader("Métricas Generales")

    col1, col2, col3 = st.columns(3)

    col1.metric("Ventas Totales", df_filtered["total_sales"].sum())
    col2.metric("Cantidad Total", df_filtered["quantity"].sum())
    col3.metric("Número de Órdenes", len(df_filtered))

# =============================
# VENTAS POR PRODUCTO
# =============================

if "product" in df_filtered.columns and "total_sales" in df_filtered.columns:

    st.subheader("Ventas por Producto")

    sales_product = df_filtered.groupby("product")["total_sales"].sum().reset_index()

    fig = px.bar(
        sales_product,
        x="product",
        y="total_sales",
        title="Ventas por Producto"
    )

    st.plotly_chart(fig)

# =============================
# VENTAS POR REGIÓN
# =============================

if "region" in df_filtered.columns and "total_sales" in df_filtered.columns:

    st.subheader("Ventas por Región")

    sales_region = df_filtered.groupby("region")["total_sales"].sum().reset_index()

    fig2 = px.pie(
        sales_region,
        values="total_sales",
        names="region",
        title="Distribución de Ventas"
    )

    st.plotly_chart(fig2)

# =============================
# VENTAS MENSUALES
# =============================

if "date" in df_filtered.columns and "total_sales" in df_filtered.columns:

    st.subheader("Ventas Mensuales")

    df_filtered["date"] = pd.to_datetime(df_filtered["date"])
    df_filtered["month"] = df_filtered["date"].dt.to_period("M").astype(str)

    monthly = df_filtered.groupby("month")["total_sales"].sum().reset_index()

    fig3 = px.line(
        monthly,
        x="month",
        y="total_sales",
        title="Evolución de Ventas"
    )

    st.plotly_chart(fig3)

# =============================
# TABLA DE DATOS
# =============================

st.subheader("Datos Filtrados")
st.dataframe(df_filtered)

# =============================
# EDA AUTOMÁTICO
# =============================

st.header("Análisis Exploratorio de Datos (EDA)")

eda = EDAAnalyzer(df_filtered)

st.subheader("Estadísticas descriptivas")
st.dataframe(eda.summary_statistics())

st.subheader("Valores faltantes")
st.dataframe(eda.missing_values())

# =============================
# MACHINE LEARNING
# =============================

st.header("Machine Learning Automático")

numeric_columns = df_filtered.select_dtypes(include=['number']).columns

if len(numeric_columns) > 1:

    target = st.selectbox(
        "Selecciona la variable a predecir",
        numeric_columns
    )

    features = st.multiselect(
        "Selecciona variables predictoras",
        [col for col in numeric_columns if col != target]
    )

    if len(features) > 0:

        X = df_filtered[features]
        y = df_filtered[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.subheader("Resultados del modelo")

        col1, col2 = st.columns(2)

        col1.metric("Error absoluto medio (MAE)", round(mae, 2))
        col2.metric("R² Score", round(r2, 3))

        results = pd.DataFrame({
            "Real": y_test,
            "Predicción": predictions
        })

        st.subheader("Comparación real vs predicción")

        st.dataframe(results)

else:
    st.warning("Se necesitan al menos dos columnas numéricas para entrenar un modelo.")