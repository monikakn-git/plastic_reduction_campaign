#!/bin/sh
# Use this project's isolated Colima context when configured; preserve explicit overrides.
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
if [ -f "$PROJECT_ROOT/.local/docker/config.json" ]; then
  DOCKER_CONFIG=${DOCKER_CONFIG:-"$PROJECT_ROOT/.local/docker"}
  DOCKER_CONTEXT=${DOCKER_CONTEXT:-colima-mlops-el}
  export DOCKER_CONFIG DOCKER_CONTEXT
fi
exec docker "$@"
