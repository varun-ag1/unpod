import importlib
import os
import sys


def _reload_config_module(monkeypatch):
    monkeypatch.setenv("SKIP_DB_CHECK", "1")
    monkeypatch.setenv("SETTINGS_FILE", "super_services.settings.qa")

    # Other tests monkeypatch this module in sys.modules; force a real reload.
    sys.modules.pop("super_services.libs.config", None)
    config_module = importlib.import_module("super_services.libs.config")
    if hasattr(config_module, "get_settings"):
        config_module.get_settings.cache_clear()
    return importlib.reload(config_module)


def test_settings_defaults_to_qa_when_settings_file_unset(monkeypatch):
    config_module = _reload_config_module(monkeypatch)
    monkeypatch.delenv("SETTINGS_FILE", raising=False)

    settings = config_module.Settings()

    assert settings.SETTINGS_MODULE == "super_services.settings.qa"
    assert os.environ["SETTINGS_FILE"] == "super_services.settings.qa"


def test_settings_uses_explicit_settings_file(monkeypatch):
    config_module = _reload_config_module(monkeypatch)
    monkeypatch.setenv("SETTINGS_FILE", "super_services.settings.prod")

    settings = config_module.Settings()

    assert settings.SETTINGS_MODULE == "super_services.settings.prod"
    assert settings.ENV_NAME == "PROD"
