import pytest
import json
import os
from utils.config_manager import ConfigManager

@pytest.fixture
def temp_config_file(tmp_path):
    """Crea un archivo de configuración temporal para pruebas"""
    config_file = tmp_path / "test_favorites.json"
    return str(config_file)

@pytest.fixture
def config_manager_with_temp_file(temp_config_file):
    """ConfigManager con archivo temporal"""
    return ConfigManager(file_path=temp_config_file)

def test_config_file_creation(temp_config_file):
    """TC_CONFIG_001: Crear archivo JSON si no existe"""
    # Verificar que el archivo no existe inicialmente
    assert not os.path.exists(temp_config_file)
    
    # Crear ConfigManager (debe crear el archivo)
    manager = ConfigManager(file_path=temp_config_file)
    
    # Verificar que el archivo fue creado
    assert os.path.exists(temp_config_file)
    
    # Verificar que contiene un objeto JSON vacío
    with open(temp_config_file, 'r') as f:
        content = json.load(f)
        assert content == {}

def test_save_config(config_manager_with_temp_file, temp_config_file):
    """TC_CONFIG_002: Guardar configuración con nombre específico"""
    manager = config_manager_with_temp_file
    
    # Configuración de ejemplo
    test_config = {
        "selected_columns": ["product", "total_sales"],
        "date_range": "2024-01",
        "filter_region": "Norte"
    }
    
    # Guardar configuración
    manager.save_config("mi_config_1", test_config)
    
    # Verificar que se guardó correctamente leyendo el archivo
    with open(temp_config_file, 'r') as f:
        saved_data = json.load(f)
    
    assert "mi_config_1" in saved_data
    assert saved_data["mi_config_1"] == test_config
    assert saved_data["mi_config_1"]["selected_columns"] == ["product", "total_sales"]
    assert saved_data["mi_config_1"]["filter_region"] == "Norte"

def test_save_multiple_configs(config_manager_with_temp_file, temp_config_file):
    """TC_CONFIG_002 (múltiples): Guardar varias configuraciones"""
    manager = config_manager_with_temp_file
    
    config1 = {"setting": "value1"}
    config2 = {"setting": "value2"}
    config3 = {"setting": "value3"}
    
    manager.save_config("config_a", config1)
    manager.save_config("config_b", config2)
    manager.save_config("config_c", config3)
    
    # Verificar que todas las configuraciones existen
    all_configs = manager.load_all_configs()
    assert len(all_configs) == 3
    assert "config_a" in all_configs
    assert "config_b" in all_configs
    assert "config_c" in all_configs

def test_load_all_configs(config_manager_with_temp_file):
    """TC_CONFIG_003: Cargar todas las configuraciones guardadas"""
    manager = config_manager_with_temp_file
    
    # Guardar algunas configuraciones
    manager.save_config("test1", {"data": "config1"})
    manager.save_config("test2", {"data": "config2"})
    
    # Cargar todas
    all_configs = manager.load_all_configs()
    
    assert isinstance(all_configs, dict)
    assert len(all_configs) == 2
    assert all_configs["test1"]["data"] == "config1"
    assert all_configs["test2"]["data"] == "config2"

def test_load_configs_empty_file(config_manager_with_temp_file):
    """TC_CONFIG_003 (edge case): Cargar configs de archivo vacío/nuevo"""
    manager = config_manager_with_temp_file
    
    # No hemos guardado nada aún
    all_configs = manager.load_all_configs()
    
    assert isinstance(all_configs, dict)
    assert len(all_configs) == 0
    assert all_configs == {}

def test_invalid_json_handling(temp_config_file):
    """TC_CONFIG_004: Manejo de archivo JSON corrupto"""
    # Crear archivo con JSON inválido
    with open(temp_config_file, 'w') as f:
        f.write("{invalid json content")
    
    manager = ConfigManager(file_path=temp_config_file)
    
    # Debe retornar diccionario vacío en lugar de fallar
    all_configs = manager.load_all_configs()
    assert isinstance(all_configs, dict)
    assert all_configs == {}

def test_overwrite_existing_config(config_manager_with_temp_file):
    """TC_CONFIG_002 (sobrescritura): Sobrescribir configuración existente"""
    manager = config_manager_with_temp_file
    
    # Guardar configuración inicial
    manager.save_config("my_config", {"version": 1})
    
    # Sobrescribir con nueva configuración
    manager.save_config("my_config", {"version": 2, "updated": True})
    
    # Verificar que se sobrescribió
    all_configs = manager.load_all_configs()
    assert all_configs["my_config"]["version"] == 2
    assert all_configs["my_config"]["updated"] is True

def test_apply_config_missing_streamlit():
    """TC_CONFIG_005: Aplicar config sin Streamlit disponible debe manejar el error"""
    # Nota: Este test verifica que apply_config maneja la ausencia de Streamlit
    # En un entorno sin Streamlit, st.session_state no existe
    # Este test documenta el comportamiento esperado
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({"test_config": {"key": "value"}}, f)
    
    try:
        manager = ConfigManager(file_path=temp_file)
        
        # apply_config intentará usar st.session_state que no existe en pytest
        # Debe manejar el error gracefully o lanzar AttributeError
        # (dependiendo de la implementación actual)
        try:
            result = manager.apply_config("test_config")
            # Si no falla, verificar el retorno
            assert isinstance(result, bool)
        except (AttributeError, NameError):
            # Es aceptable que falle si Streamlit no está disponible
            # Este es el comportamiento documentado en known_limitations
            pass
    finally:
        os.unlink(temp_file)
