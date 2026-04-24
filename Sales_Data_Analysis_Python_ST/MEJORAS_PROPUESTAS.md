# Propuestas de Mejora y Actualizaciones del Producto

Este documento detalla las ideas y adiciones estratégicas sugeridas para mejorar la plataforma de **Análisis de Ventas (Python/Streamlit)**, basándose en la arquitectura actual, sus reglas de negocio y flujos de aserción (descritos en `AGENTS.md`).

---

## 1. Potenciar los Modelos Predictivos (Machine Learning Avanzado)

### 1.1 Pronóstico de Series de Tiempo (Forecasting)
Actualmente, el proyecto implementa un módulo `AutoML` con Regresión Lineal para entender la relación entre columnas numéricas existentes de forma transversal. Sin embargo, en un producto de *ventas*, predecir el comportamiento temporal es vital.
- **Sugerencia:** Introducir herramientas de series de tiempo como **Prophet** (Meta) o **Statsmodels (ARIMA)**. Esto permitiría predecir las ventas a "X meses" hacia adelante basándose estrictamente en fechas y tendencias históricas, ofreciendo una métrica crucial de negocio.

### 1.2 Módulo de Explicabilidad de IA (XAI)
Mostrar únicamente el Error Absoluto Medio (MAE) y el R² dificulta la toma de decisiones para los líderes de venta menos técnicos.
- **Sugerencia:** Integrar gráficos de **SHAP** o **LIME**. Esto explicaría visualmente el impacto predictivo de cada campo (ej. *"El descuento explica el 30% del volumen del mes y empuja positivamente las estimaciones"*).

---

## 2. Optimización del Rendimiento (Streamlit Experience)

### 2.1 Uso Estratégico de System Caching
El diseño de Streamlit ejecuta los scripts de forma imperativa (de arriba abajo) ante cualquier actualización visual en pantalla (como accionar un filtro). Funciones costosas como `DataLoader.load_csv` y `DataCleaner.clean` recalculan todo.
- **Sugerencia:** Agregar el decorador `@st.cache_data` sobre las lógicas de carga, limpieza y EDA; y `@st.cache_resource` sobre la función de entrenamiento de machine learning. De esta manera, el modelo no se re-entrenará ni los datos pesados se re-cargarán a menos que se cambien activamente los parámetros fuente.

---

## 3. Funcionalidades de UX y Toma de Decisiones

### 3.1 Reportes Automatizados en PDF
La plataforma actualmente genera reportes base (en CSV y un resumen general en TXT) apoyándose en `ReportGenerator`.
- **Sugerencia:** Integrar un exportador PDF directivo (con dependencias como `reportlab` o `pdfkit`) que agrupe no solo las estadísticas descriptivas del Análisis Exploratorio de Datos (EDA), sino que además embeba los *charts* de PLotly en formato imagen como una "captura o informe" formal del Dashboard listo para presentar en una junta.

### 3.2 Feature Engineering Automático de Fechas
En el contexto de ventas, las métricas varían en función de ciclos (quincenas, vacaciones, puentes, fines de semana).
- **Sugerencia:** Proveer capacidades ampliadas al `DataCleaner` para que, tras detectar y validar en español las excepciones por formato de fecha (`Exception` documentado en test suite), autogenere columnas contextuales: `día_de_semana`, `es_fin_de_semana`, `trimestre`. Así el `AutoML` encontrará mayores factores explicativos de manera automática.

### 3.3 Manejo Graceful de Excepciones Visuales
Las lógicas actuales están muy bien estructuradas para arrojar excepciones detalladas en español por validación estricta (ej. *"Se requieren más datos para el entrenamiento"*, o para matrices de correlación inválidas en `EDAAnalyzer`). 
- **Sugerencia:** Capturar estos errores dentro del script `app.py` mediante bloques `try...except` e interactuar con notificaciones nativas de Streamlit (`st.error()`, `st.warning()`) para mostrar ese mismo texto de error, evitando que la aplicación rompa flujos o muestre trazados de error en rojo técnico (*stack traces*).

---

## 4. Automatización y Calidad de Código (DevOps / CI-CD)

### 4.1 Pipeline de Integración Continua (CI) en GitHub/GitLab
Con una suite de pruebas ya sólida (16 pruebas unitarias, validacion de errores previstos, cobertura +97% y análisis integral E2E mediante IA con TestSprite), es crucial proteger la rama `main`.
- **Sugerencia:** Construir manuales de *GitHub Actions workflows* que provisionen el entorno en Python, corran `pytest` exigiendo métricas fijas de cobertura, e invoquen los escenarios E2E de TestSprite automáticamente. 

### 4.2 Restricción Temprana del Esquema 
- **Sugerencia:** Implementar comprobación de contratos con Pydantic durante la carga de CSV para validar pre-condiciones y avisar velozmente al usuario si el archivo que sube no posee las columnas mínimas permitidas, bloqueando los módulos downstream de romperse a la mitad del proceso.
