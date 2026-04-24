import sys
import os

# Agregar la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
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
from models.forecaster import SalesForecaster
from utils.report_generator import ReportGenerator
from utils.config_manager import ConfigManager

config_mgr = ConfigManager()

st.title("📊 Plataforma de Análisis de Ventas")

# =============================
# CARGAR DATASET
# =============================

st.sidebar.header("Cargar dataset")

uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo CSV",
    type=["csv"]
)

@st.cache_data(show_spinner="Cargando y limpiando datos...")
def load_and_clean_data(file_context, filename_key):
    if file_context is not None:
        file_context.seek(0)
        data = pd.read_csv(file_context)
    else:
        loader_instance = DataLoader()
        data = loader_instance.load_csv("data/sales.csv")
    
    cleaner_instance = DataCleaner(data)
    return cleaner_instance.clean()

filename_tracker = uploaded_file.name if uploaded_file else "default-sales-csv"
df = load_and_clean_data(uploaded_file, filename_tracker)

if uploaded_file is None:
    st.sidebar.info("Usando dataset de ejemplo")

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

    # --- Sección de Configuración Favorita ---
    st.sidebar.subheader("⭐ Configuración Favorita")
    
    saved_configs = config_mgr.load_all_configs()
    config_names = ["-- Seleccionar --"] + list(saved_configs.keys())
    
    selected_config = st.sidebar.selectbox("Cargar configuración", config_names)
    
    if selected_config != "-- Seleccionar --":
        if st.sidebar.button("Aplicar Configuración"):
            config_mgr.apply_config(selected_config)
            st.rerun()

    new_config_name = st.sidebar.text_input("Nombre de la configuración actual")
    if st.sidebar.button("Guardar Configuración"):
        if new_config_name and new_config_name.strip():
            clean_name = new_config_name.strip()
            current_config = {
                "filter_region": st.session_state.get("filter_region", df["region"].unique().tolist()),
                "filter_product": st.session_state.get("filter_product", df["product"].unique().tolist())
            }
            config_mgr.save_config(clean_name, current_config)
            st.sidebar.success(f"Configuración '{clean_name}' guardada.")
        else:
            st.sidebar.error("Por favor ingrese un nombre.")

    st.sidebar.divider()

    # --- Filtros Estándar ---
    region = st.sidebar.multiselect(
        "Seleccionar región",
        df["region"].unique(),
        default=df["region"].unique(),
        key="filter_region"
    )

    product = st.sidebar.multiselect(
        "Seleccionar producto",
        df["product"].unique(),
        default=df["product"].unique(),
        key="filter_product"
    )

    df_filtered = df[
        (df["region"].isin(region)) &
        (df["product"].isin(product))
    ]

else:
    st.warning("El dataset no contiene columnas 'region' o 'product'.")
    df_filtered = df

# =============================
# REPORTES
# =============================

st.sidebar.header("📥 Descargar Reportes")

report_gen = ReportGenerator(df_filtered)

st.sidebar.download_button(
    label="📄 Descargar Datos Filtrados (CSV)",
    data=report_gen.to_csv(),
    file_name="reporte_ventas_filtrado.csv",
    mime="text/csv"
)

st.sidebar.download_button(
    label="📝 Descargar Resumen del Reporte (TXT)",
    data=report_gen.to_summary_text(),
    file_name="resumen_reporte_ventas.txt",
    mime="text/plain"
)

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

        @st.cache_resource(show_spinner="Entrenando modelo AutoML...")
        def train_model(df_in, target_col, feature_cols):
            X_data = df_in[feature_cols]
            y_data = df_in[target_col]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_data, y_data,
                test_size=0.2,
                random_state=42
            )

            mod = LinearRegression()
            mod.fit(X_tr, y_tr)

            preds = mod.predict(X_te)
            m_mae = mean_absolute_error(y_te, preds)
            m_r2 = r2_score(y_te, preds)
            
            return mod, X_tr, y_te, preds, m_mae, m_r2

        model, X_train, y_test, predictions, mae, r2 = train_model(df_filtered, target, features)

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

        st.subheader("Importancia de Variables (SHAP)")
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer(X_train)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.summary_plot(shap_values, X_train, show=False)
        st.pyplot(fig)

else:
    st.warning("Se necesitan al menos dos columnas numéricas para entrenar un modelo.")

# =============================
# FORECASTING (PREDICCIÓN)
# =============================

st.header("Predicción de Ventas (Forecasting)")

if "date" in df_filtered.columns:
    numeric_for_forecast = df_filtered.select_dtypes(include=['number']).columns
    
    if len(numeric_for_forecast) > 0:
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            target_forecast = st.selectbox(
                "Métrica a predecir", 
                numeric_for_forecast,
                index=list(numeric_for_forecast).index("total_sales") if "total_sales" in numeric_for_forecast else 0
            )
        with col_f2:
            forecast_periods = st.slider("Periodos a predecir", min_value=1, max_value=30, value=7)
        with col_f3:
            forecast_freq = st.selectbox("Frecuencia", options=["Diaria", "Mensual"])
            
        freq_code = "D" if forecast_freq == "Diaria" else "M"
        
        @st.cache_data(show_spinner="Calculando pronósticos...")
        def compute_forecast(df_in, date_c, target_c, per, freq):
            f = SalesForecaster(df_in)
            return f.train_and_forecast(date_c, target_c, per, freq)
        
        try:
            forecast_results = compute_forecast(df_filtered, "date", target_forecast, forecast_periods, freq_code)
            
            st.subheader(f"Proyección {forecast_freq} de {target_forecast}")
            
            hist_df = forecast_results["historical"]
            fut_df = forecast_results["forecast"]
            
            # Preparar datos para gráfica unificada con Plotly
            hist_df['Tipo'] = 'Histórico'
            hist_df = hist_df.rename(columns={'historical_sales': 'Valor'})
            
            fut_df['Tipo'] = 'Predicción'
            fut_df = fut_df.rename(columns={'predicted_sales': 'Valor'})
            
            combined_df = pd.concat([hist_df[['date', 'Valor', 'Tipo']], fut_df[['date', 'Valor', 'Tipo']]])
            
            fig = px.line(
                combined_df, 
                x='date', 
                y='Valor', 
                color='Tipo', 
                title=f"Tendencia e Histórico de {target_forecast}",
                color_discrete_map={"Histórico": "blue", "Predicción": "orange"}
            )
            # Volver la traza de predicción punteada
            fig.update_traces(line=dict(dash="dot"), selector=dict(name="Predicción"))
            
            st.plotly_chart(fig, use_container_width=True)
            
            col_met1, col_met2 = st.columns(2)
            col_met1.metric("Error tendencia histórica (MAE)", round(forecast_results["mae"], 2))
            col_met2.metric("Ajuste de Tendencia (R²)", round(forecast_results["r2"], 3))
            
            with st.expander("Ver tabla de predicciones"):
                display_df = fut_df[['date', 'Valor']].copy()
                display_df['date'] = display_df['date'].astype(str)
                st.dataframe(display_df.rename(columns={'date': 'Fecha', 'Valor': 'Predicción'}))

        except ValueError as e:
            st.warning(str(e))
            
    else:
        st.warning("No hay columnas numéricas para predecir.")
else:
    st.warning("El dataset requiere una columna 'date' para utilizar el pronóstico de ventas.")