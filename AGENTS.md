# Agent Instructions for Sales Data Analysis Project

## Project Structure

This is a Python-based sales data analysis platform with a Streamlit UI. The actual project code lives in `Sales_Data_Analysis_Python_ST/` (nested subdirectory, not repo root).

**Working directory for all commands:** `Sales_Data_Analysis_Python_ST/`

Directory layout:
- `analysis/` - EDA and sales analysis modules
- `models/` - ML regression models (AutoML)
- `utils/` - data cleaning, config, report generation
- `loader/` - CSV/Excel loading
- `dashboard/` - Streamlit app entry point (`app.py`)
- `visualization/` - auto-visualization utilities
- `tests/` - pytest test suite
- `testsprite_tests/` - TestSprite AI UI end-to-end testing resources and reports
- `data/` - sample datasets (`sales.csv`)

## Environment Setup

**Critical:** This project uses a Python virtual environment (venv) on Windows.

All Python commands must use the venv interpreter:
```powershell
./venv/Scripts/python.exe -m <module>
```

Do NOT use system Python (`python` or `C:\Python313\python.exe`).

### PYTHONPATH Configuration

The project requires `PYTHONPATH=.` to resolve module imports. This is set via:
- `.env` file (loaded by VS Code and Streamlit)
- `.vscode/settings.json` for terminal sessions
- `sys.path.append()` in `dashboard/app.py` for Streamlit

When running commands, the PYTHONPATH is already configured if using the venv from the project root.

## Running Tests

### Unit Testing (Backend / Data Modules)

Basic test execution:
```powershell
cd Sales_Data_Analysis_Python_ST
./venv/Scripts/python.exe -m pytest tests/
```

With coverage report:
```powershell
./venv/Scripts/python.exe -m pytest --cov=. --cov-report=term-missing tests/
```

### End-to-End Frontend Testing (TestSprite)

The UI features (like custom filters in Streamlit) are tested using TestSprite:
1. Ensure the Streamlit dashboard is running locally (usually on port 8501).
2. Generate tests via the TestSprite tools.
3. TestSprite outputs its generated tests, configuration, and markdown reports under the `testsprite_tests/` folder.

**Test count:** 56 unit tests across 10 test files (test_auto_ml, test_auto_visualizer, test_config_manager, test_data_cleaner, test_data_loader, test_dataset_detector, test_eda_analyzer, test_forecaster, test_report_generator, test_sales_analyzer), plus E2E AI Tests for the Streamlit App.

**Current coverage:** ~97% (7 lines missing in eda_analyzer.py plotting methods)

## Running the Streamlit Dashboard

```powershell
cd Sales_Data_Analysis_Python_ST
./venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

The app loads `data/sales.csv` as default data if no file is uploaded.

## Testing Conventions

- Test fixtures in `tests/` provide sample DataFrames
- Tests validate both basic flows and error conditions (e.g., invalid dates, insufficient data)
- ML tests check MAE and R2 metrics from sklearn
- Follow existing `test_*.py` naming pattern
- Exception messages are user-facing Spanish (e.g., "Se requieren más datos para el entrenamiento")

## Key Quirks

1. **Date handling:** `DataCleaner.fix_data_types()` raises `Exception` with Spanish message if date conversion fails. Tests expect this via `pytest.raises()`.

2. **Column normalization:** `DataCleaner.normalize_columns()` applies `.str.strip().str.lower()` to column names.

3. **Single-variable correlation:** `EDAAnalyzer.correlation_matrix()` raises `ValueError` if fewer than 2 numeric columns exist.

4. **ML data requirement:** `AutoML.train_regression()` requires `len(df) >= 2` or raises `ValueError`.

5. **Streamlit Input Validation:** `st.text_input` will return a string with whitespaces if the user types spaces. Truthiness checks like `if my_input:` will pass on strings like `"   "`. **Always use `.strip()`** when validating Streamlit text inputs (e.g., `if my_input and my_input.strip():`).

6. **Context files:** `contextAgent.txt` and `contextTest.txt` contain original QA requirements in Spanish. Preserve these conventions when extending tests.

## Dependencies

Managed via `requirements.txt`:
- Core: pandas, numpy, scikit-learn
- Viz: matplotlib, seaborn, plotly, streamlit
- Testing: pytest, pytest-cov, coverage

Check `installed.txt` for full locked versions.

## Common Mistakes to Avoid

- Do NOT run pytest with system Python (it's not installed there)
- Do NOT forget `cd Sales_Data_Analysis_Python_ST` before commands
- Do NOT modify PYTHONPATH manually; use existing .env and settings.json
- Do NOT skip coverage when adding new code; target is >95%
- Do NOT write test assertions in English when error messages are in Spanish
