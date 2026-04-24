# Resumen de Archivos de Prueba

A continuación se detalla la funcionalidad de cada archivo en el directorio `tests`, de forma clara y concisa (menor a 5 líneas por archivo):

- **`test_auto_ml.py`**
  Verifica el correcto entrenamiento de modelos de regresión y evalúa el manejo de errores (por ejemplo, cuando hay datos insuficientes) utilizando el módulo `AutoML`.

- **`test_auto_visualizer.py`**
  Comprueba que la herramienta genere exitosamente gráficos interactivos de `Plotly` (como distribuciones numéricas, barras categóricas y mapas de calor de correlación).

- **`test_config_manager.py`**
  Valida la creación, guardado, sobrescritura y carga de las configuraciones de usuario en archivos JSON, incluyendo el manejo de fallos si hay un archivo inválido.

- **`test_data_cleaner.py`**
  Asegura que el proceso de limpieza opere adecuadamente al eliminar nulos y duplicados, formatear fechas y normalizar los nombres de las columnas del DataFrame.

- **`test_data_loader.py`**
  Verifica que los archivos CSV se carguen adecuadamente, gestiona las excepciones para archivos inexistentes y prueba el funcionamiento de la vista previa de datos.

- **`test_dataset_detector.py`**
  Valida el análisis de estructura del dataset, probando que el algoritmo sea capaz de clasificar acertadamente las respectivas columnas como numéricas, categóricas o temporales.

- **`test_eda_analyzer.py`**
  Comprueba la ejecución del análisis exploratorio evaluando cálculos de estadísticas descriptivas, información general y el comportamiento frente a excepciones de correlaciones.

- **`test_forecaster.py`**
  Verifica el modelo de proyecciones de ventas futuras considerando las configuraciones de diferentes frecuencias temporales (diarias/mensuales) y testea la suficiencia de datos.

- **`test_report_generator.py`**
  Valida la correcta codificación (`utf-8`) y exportación final de los resultados estadísticos tanto en formato de base de datos plana CSV como en resúmenes legibles de texto.

- **`test_sales_analyzer.py`**
  Confirma y cerciora la exactitud de funciones de cálculos comerciales; analizando debidamente diversas métricas como totales en moneda, agrupamiento por regiones, y comparativas.
