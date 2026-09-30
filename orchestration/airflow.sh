#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_HOME="$PROJECT_ROOT/orchestration/airflow_home"
AIRFLOW_BIN="$PROJECT_ROOT/orchestration/.venv/bin/airflow"

export AIRFLOW_HOME
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$AIRFLOW_BIN" "$@"