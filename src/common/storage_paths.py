from __future__ import annotations

from pathlib import Path

from src.common.config import AppConfig, get_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Layer = str


def layer_path(layer: Layer, *, config: AppConfig | None = None) -> str:
    config = config or get_config()
    container = getattr(config.medallion, f"{layer}_container")

    if config.storage.mode == "local":

        base_path = Path(config.storage.local_base_path)

        if not base_path.is_absolute():
            base_path = PROJECT_ROOT / base_path

        path = base_path / layer
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    if config.storage.mode == "databricks":
        if config.storage.databricks_volume_path is None:
            raise ValueError(
                "storage.databricks_volume_path must be set "
                "when storage.mode == 'databricks'"
            )

        return f"{config.storage.databricks_volume_path.rstrip('/')}/{layer}"

    if config.storage.account_name is None:
        raise ValueError(
            "storage.account_name must be set in the config when storage.mode == 'azure'"
        )

    # ABFS is the driver Spark uses to talk to Azure Blob Storage /
    # Data Lake Storage Gen2 — this is the URI format Databricks expects.
    return f"abfss://{container}@{config.storage.account_name}.dfs.core.windows.net/"
