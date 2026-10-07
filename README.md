# RetailRocket ML Pipeline

A batch Machine Learning pipeline built around the RetailRocket e-commerce event dataset.

The project covers the path from exploratory model development to a reproducible data processing and batch inference workflow. Model research is kept separate from the operational pipeline, with experiments and model development under `research/` and the production-oriented pipeline under `src/`.

The operational pipeline has been validated end-to-end on **Databricks**, using PySpark, Parquet, a Medallion Architecture, MLflow with Unity Catalog Model Registry, and batch inference.

```text
Raw Events
    ↓
Ingestion
    ↓
Bronze
    ↓
Silver
    ↓
Gold
    ↓
Batch Inference
    ↓
Prediction Dataset
```

## Project Structure

```text
.
├── configs/
│   ├── base.yaml
│   ├── dev.yaml
│   └── prod.yaml
│
├── orchestration/
│   ├── airflow.sh
│   └── airflow_home/
│       └── dags/
│           └── retailrocket_pipeline.py
│
├── research/
│   └── ...
│
├── src/
│   ├── common/
│   ├── ingestion/
│   ├── processing/
│   ├── inference/
│   └── jobs/
│       ├── ingestion.py
│       ├── bronze_to_silver.py
│       ├── silver_to_gold.py
│       └── inference.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Research

The `research/` directory contains the exploratory and model-development phase of the project.

It includes the work performed to understand the RetailRocket dataset, engineer candidate features, evaluate models, perform model selection, calibrate probabilities, and produce the final trained model.

This phase is intentionally separated from the operational pipeline. The production workflow does **not** retrain the model on every execution. Instead, it consumes the existing trained model registered in MLflow.

The prediction target is:

```text
Probability that a visitor makes a purchase within the next 7 days.
```

The final inference model uses:

```text
user_views_7d
user_addtocart_7d
user_transactions_30d
user_sessions_30d
avg_session_duration_30d
user_cart_items_avg_popularity_7d
user_cart_top_seller_ratio_7d
```

The model is packaged as an MLflow PyFunc model and is managed through the Unity Catalog Model Registry.

## Data Pipeline

### Ingestion

The ingestion stage reads the raw RetailRocket event data from Parquet and produces the normalized Bronze dataset.

### Bronze

The Bronze layer preserves the event-level data while applying the required schema and type normalization.

### Silver

The Bronze → Silver stage performs data quality validation.

Invalid records are identified through `_dq_error` and written to the quarantine path, while valid records continue through the pipeline.

### Gold

The Silver → Gold stage creates the feature dataset used for inference.

The pipeline generates visitor-level daily snapshots and derives behavioral features based on rolling time windows, including views, add-to-cart events, transactions, sessions, session duration, cart popularity, and seller-related behavior.

Sessions are reconstructed using a 30-minute inactivity threshold.

### Batch Inference

The inference stage reads the latest Gold snapshot, selects the required model features, loads the production model from MLflow, generates purchase probabilities, and writes the resulting prediction dataset as Parquet.

The output contains:

```text
visitorid
purchase_probability
```

The predictions are intended to be consumed by downstream Marketing or BI processes.

## MLflow

MLflow is used for model management and experiment tracking.

The production model is registered in Unity Catalog:

```text
workspace.default.RetailRocketPurchaseModel
```

Inference loads the model through the `production` alias:

```text
models:/workspace.default.RetailRocketPurchaseModel@production
```

This allows the production model version to be changed through the registry without modifying the inference code.

The MLflow configuration is environment-dependent and is managed through the files under `configs/`.

## Orchestration

The project contains two orchestration approaches.

### Apache Airflow

The `orchestration/` directory contains an Airflow DAG for local execution.

The workflow is:

```text
ingestion
    ↓
bronze_to_silver
    ↓
silver_to_gold
    ↓
inference
```

The local Airflow workflow has been successfully executed end-to-end.

The project also provides:

```text
orchestration/airflow.sh
```

as a helper for the local Airflow environment.

The repository does not include or claim a validated hosted Airflow deployment. The DAG could be adapted to a server-based Airflow environment, but that deployment has not been tested as part of this project.

### Databricks

Databricks is the **validated execution environment for the operational pipeline**.

The Databricks workflow uses the same shared pipeline logic through the entry points under `src/jobs/`:

```text
src/jobs/
├── ingestion.py
├── bronze_to_silver.py
├── silver_to_gold.py
└── inference.py
```

The pipeline has been executed successfully end-to-end in Databricks, from raw data ingestion through batch prediction generation.

The Databricks environment uses:

* PySpark
* Unity Catalog
* Unity Catalog Managed Volumes
* MLflow
* Unity Catalog Model Registry
* environment-specific YAML configuration

The Databricks Job itself is configured in the Databricks workspace. Its configuration is not currently versioned as a separate infrastructure definition in this repository.

## Configuration

Environment-specific configuration is stored under `configs/` and loaded through `src/common/config.py`.

The active environment is selected using:

```text
APP_ENV
```

The available environments are:

```text
dev
staging
prod
```

This keeps storage paths, MLflow settings, and other environment-specific configuration outside the pipeline implementation.

## Local Setup

The project requires Python 3.11 or newer.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Set the environment:

```bash
export APP_ENV=dev
```

The individual pipeline stages can be executed directly:

```bash
python -m src.jobs.ingestion
python -m src.jobs.bronze_to_silver
python -m src.jobs.silver_to_gold
python -m src.jobs.inference
```

Airflow can be run through the provided local setup:

```bash
./orchestration/airflow.sh
```

## Data

The RetailRocket dataset is not stored in the repository.

The operational pipeline uses the event data as its current source. The `item_properties` and `category_tree` datasets are not currently required by the production feature pipeline.

Generated data, model artifacts, MLflow runtime files, and other environment-specific files are excluded from version control.

## Current Status

| Component                            | Status                 |
| ------------------------------------ | ---------------------- |
| Research and model development       | Complete               |
| Data ingestion                       | Implemented            |
| Bronze layer                         | Implemented            |
| Silver layer and data quality        | Implemented            |
| Gold feature generation              | Implemented            |
| Batch inference                      | Implemented            |
| MLflow model packaging               | Implemented            |
| Unity Catalog model registry         | Implemented            |
| Local Airflow orchestration          | Validated              |
| Databricks pipeline                  | Validated end-to-end   |
| Hosted Airflow deployment            | Not validated          |
| Databricks Job configuration as code | Not currently included |

## Design

The project intentionally separates experimentation from operational execution:

```text
research/
    ↓
trained model
    ↓
MLflow / Unity Catalog
    ↓
src/
    ↓
batch inference
```

The model is not retrained during normal pipeline execution.

The system is designed for periodic batch prediction rather than online model serving. The output is a dataset intended for downstream analytical and marketing use cases.

## License

This project is licensed under the MIT License.
