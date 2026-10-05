from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIGS_DIR = BASE_DIR / "configs"

Environment = Literal["dev", "staging", "prod"]


class StorageConfig(BaseModel):
    """Where the medallion layers physically live.

    `mode: local` writes to a plain folder on disk — this is what lets
    processing/ be developed and tested without a real Azure account.
    `mode: azure` targets Blob Storage via the ABFS driver, which is
    what Spark on Databricks expects. Switching modes never changes
    processing code — only src/common/storage_paths.py, which is the
    single place that knows how to build a path for each mode.
    """
    mode: Literal["local", "azure", "databricks"] = "local"
    local_base_path: str = "./data"
    raw_path: str = "./data/raw"
    quarantine_path: str = "./data/quarantine"
    inference_path: str = "./data/inference"
    databricks_volume_path: str | None = None
    account_name: str | None = None


class MedallionConfig(BaseModel):
    bronze_container: str
    silver_container: str
    gold_container: str


class MLflowConfig(BaseModel):
    experiment_name: str
    tracking_uri: str | None = None
    registry_uri: str | None = None


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_format: bool = True


class AppConfig(BaseModel):
    environment: Environment
    storage: StorageConfig
    medallion: MedallionConfig
    mlflow: MLflowConfig
    logging: LoggingConfig


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict, override: dict) -> dict:

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_app_config(environment: Environment) -> AppConfig:
    base = _load_yaml(CONFIGS_DIR / "base.yaml")
    env_overrides = _load_yaml(CONFIGS_DIR / f"{environment}.yaml")
    merged = _merge(base, env_overrides)
    merged["environment"] = environment
    return AppConfig(**merged)


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Environment = "dev"

    azure_storage_connection_string: str | None = None

    mlflow_tracking_uri: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config() -> AppConfig:
    settings = get_settings()
    config = _load_app_config(settings.app_env)

    if settings.mlflow_tracking_uri:
        config.mlflow.tracking_uri = settings.mlflow_tracking_uri

    return config
