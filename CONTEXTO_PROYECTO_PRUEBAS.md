# Documentación de Contexto del Proyecto: Sales Data Analysis Platform

Este documento proporciona el contexto detallado del proyecto para facilitar el llenado de la tabla de documentación de pruebas (Table 4).

---

## 0. Resumen del Proyecto
**Nombre:** Sales Data Analysis Platform (AppPrediccion)  
**Objetivo:** Proporcionar una herramienta interactiva para cargar, limpiar, analizar (EDA) y predecir tendencias de ventas utilizando Machine Learning (AutoML) y visualizaciones dinámicas.  
**Arquitectura:** Python 3.13 + Streamlit.  
**Ubicación del Código:** `Sales_Data_Analysis_Python_ST/`

---

## 1. Test Policy (Política de Pruebas)
*   **Estándar de Calidad:** Cobertura de código superior al 95%.
*   **Validación Estricta:** Todos los mensajes de error deben ser amigables para el usuario y estar en español.
*   **Aislamiento:** Uso obligatorio de entornos virtuales (venv) para garantizar la reproducibilidad.

## 2. Organizational Test Strategy (Estrategia de Pruebas)
*   **Nivel Unitario/Integración:** Uso de `pytest` para lógica de negocio (limpieza, análisis, ML).
*   **Nivel E2E (UI):** Uso de `TestSprite` (IA-powered) para validar flujos completos en el Dashboard de Streamlit.
*   **Enfoque:** Validar tanto el "camino feliz" como la gestión de excepciones (ej. formatos de fecha incorrectos).

## 3. Test Plan (Plan de Pruebas)
*   **Alcance:** Módulos de carga (`loader`), limpieza (`utils/data_cleaner`), análisis (`analysis/eda_analyzer`), predicción (`models/auto_ml`) y visualización (`dashboard`).
*   **Herramientas:** Pytest, Pytest-cov, TestSprite.

## 4. Test Status Report (Informe de Estado de Pruebas)
*   **Estado Actual:** 16 pruebas unitarias implementadas y pasando.
*   **Cobertura:** ~97% (Faltan 7 líneas en métodos de graficación de `eda_analyzer.py`).
*   **E2E:** Pruebas de persistencia de filtros y guardado de presets completadas.

## 5. Test Completion Report (Informe de Finalización de Pruebas)
*   **Resumen:** Se han validado con éxito las lógicas de normalización de columnas, conversión de tipos de datos, matrices de correlación y entrenamiento de modelos de regresión.

## 6. Test Design Specification (Especificación de Diseño de Pruebas)
*   **Funcional:** Verificación de limpieza de strings (`strip`, `lower`), detección de columnas numéricas.
*   **No Funcional:** Rendimiento de carga de datos y entrenamiento de modelos.

## 7. Test Case Specification (Especificación de Casos de Prueba)
*   **Ejemplos Clave:**
    *   TC_01: Error controlado al convertir fechas inválidas.
    *   TC_02: Error al intentar entrenar ML con menos de 2 registros.
    *   TC_03: Generación correcta de matriz de correlación con >1 columna numérica.

## 8. Test Procedure Specification (Especificación de Procedimientos de Prueba)
1.  Activar venv: `./venv/Scripts/activate`
2.  Configurar PYTHONPATH: `export PYTHONPATH=.`
3.  Ejecutar unitarias: `pytest tests/`
4.  Ejecutar E2E: Iniciar app (`streamlit run dashboard/app.py`) y ejecutar TestSprite.

## 9. Test Data Requirement (Requisitos de Datos de Prueba)
*   Archivos CSV/Excel con al menos una columna de fecha y columnas numéricas para predicción.
*   Dataset por defecto: `data/sales.csv`.

## 10. Test Environment Requirement (Requisitos del Entorno de Pruebas)
*   **S.O.:** Windows.
*   **Lenguaje:** Python 3.13.
*   **Librerías:** pandas, scikit-learn, streamlit, matplotlib, seaborn, plotly.

## 11. Test Data Readiness Report (Informe de Preparación de Datos)
*   El archivo `data/sales.csv` ha sido validado y normalizado para servir como base de pruebas de integración.

## 12. Test Environment Readiness Report (Informe de Preparación del Entorno)
*   Virtual environment configurado y dependencias instaladas según `requirements.txt`.
*   Variable de entorno `PYTHONPATH` verificada.

## 13. Test Execution Log (Registro de Ejecución de Pruebas)
*   Consultar `testsprite_tests/testsprite-mcp-test-report.md` para logs detallados de UI.
*   Logs de consola de `pytest` disponibles en cada ejecución local.

## 14. Test Incident Report (Informe de Incidentes de Pruebas)
*   **Incidente detectado:** Fallo en conversión de fechas cuando el separador no es estándar.
*   **Resolución:** Implementada excepción personalizada en `DataCleaner.fix_data_types()` con mensaje descriptivo en español.
