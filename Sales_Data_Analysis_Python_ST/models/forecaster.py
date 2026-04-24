import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

class SalesForecaster:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def train_and_forecast(self, date_col, target_col, periods, freq='D'):
        if date_col not in self.df.columns or target_col not in self.df.columns:
            raise ValueError(f"Faltan columnas requeridas: {date_col} o {target_col}")
        
        # Preparar los datos
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
        # Limpiar nulos para evitar errores matemáticos
        df_clean = self.df.dropna(subset=[date_col, target_col]).copy()
        
        if len(df_clean) < 2:
            raise ValueError("Se requieren más datos para el entrenamiento")
            
        # Agrupar según frecuencia
        if freq == 'D':
            grouped = df_clean.groupby(df_clean[date_col].dt.date)[target_col].sum().reset_index()
            grouped[date_col] = pd.to_datetime(grouped[date_col])
        elif freq == 'M':
            # Para meses tomaremos el final del mes para mejor representacion en graficas
            grouped = df_clean.groupby(df_clean[date_col].dt.to_period('M'))[target_col].sum().reset_index()
            grouped[date_col] = grouped[date_col].dt.to_timestamp()
        else:
            raise ValueError("Frecuencia no soportada. Use 'D' o 'M'.")
            
        if len(grouped) < 2:
            raise ValueError("Se requieren más datos para el entrenamiento después de agrupar por fecha")
            
        # Ordenar cronológicamente
        grouped = grouped.sort_values(by=date_col)
        
        # Crear variable independiente X (días desde el inicio)
        min_date = grouped[date_col].min()
        grouped['days_since_start'] = (grouped[date_col] - min_date).dt.days
        
        X_train = grouped[['days_since_start']]
        y_train = grouped[target_col]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calcular MAE y R2 (sobre la tendencia histórica)
        y_pred_train = model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred_train)
        r2 = r2_score(y_train, y_pred_train)
        
        # Generar fechas futuras
        last_date = grouped[date_col].max()
        if freq == 'D':
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, periods + 1)]
        elif freq == 'M':
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, periods + 1)]
            
        future_df = pd.DataFrame({date_col: future_dates})
        future_df['days_since_start'] = (future_df[date_col] - min_date).dt.days
        
        future_predictions = model.predict(future_df[['days_since_start']])
        # Evitar predicciones negativas en ventas (Límite cero)
        future_predictions = np.maximum(future_predictions, 0)
        
        future_df['predicted_sales'] = future_predictions
        
        # Histórico para cruce en la gráfica
        historical_df = grouped[[date_col, target_col]].copy()
        historical_df = historical_df.rename(columns={target_col: 'historical_sales'})
        
        return {
            "historical": historical_df,
            "forecast": future_df,
            "mae": mae,
            "r2": r2
        }
