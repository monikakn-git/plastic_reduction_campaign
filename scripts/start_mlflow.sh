#!/bin/sh
set -eu

: "${MLFLOW_TRACKING_USERNAME:=admin}"
: "${MLFLOW_TRACKING_PASSWORD:?Set MLFLOW_TRACKING_PASSWORD in the shared Render environment group}"
: "${MLFLOW_FLASK_SERVER_SECRET_KEY:?Render must generate MLFLOW_FLASK_SERVER_SECRET_KEY}"
: "${MLFLOW_BACKEND_STORE_URI:=sqlite:////var/data/mlflow.db}"
: "${MLFLOW_ARTIFACTS_DESTINATION:=/var/data/mlartifacts}"

export MLFLOW_AUTH_ADMIN_USERNAME="$MLFLOW_TRACKING_USERNAME"
export MLFLOW_AUTH_ADMIN_PASSWORD="$MLFLOW_TRACKING_PASSWORD"
mkdir -p "$MLFLOW_ARTIFACTS_DESTINATION"
chown mlflow:mlflow /var/data "$MLFLOW_ARTIFACTS_DESTINATION"

# Accept the public Render host and Render's private service hostname for API-to-MLflow traffic.
allowed_hosts="${RENDER_EXTERNAL_HOSTNAME:-localhost},plastic-reduction-mlflow-*:*,localhost:*,127.0.0.1:*"
exec gosu mlflow:mlflow mlflow server \
  --host 0.0.0.0 \
  --port "${PORT:-5000}" \
  --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
  --serve-artifacts \
  --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION" \
  --app-name basic-auth \
  --allowed-hosts "$allowed_hosts"
