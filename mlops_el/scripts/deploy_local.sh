#!/bin/sh
# Explicit local demo deployment. Never called by pull requests or untrusted runners.
set -eu
cd "$(dirname "$0")/.."
command -v docker >/dev/null 2>&1 || { echo 'Docker is required; container deployment has not run.' >&2; exit 1; }
MODEL_RELEASE_DIR=${MODEL_RELEASE_DIR:-"$PWD/releases/yolo11m-demo-v3"}
API_PORT=${API_PORT:-8011}
export MODEL_RELEASE_DIR API_PORT
python3 scripts/verify_phase2_release.py "$MODEL_RELEASE_DIR"
if [ "${SKIP_BUILD:-0}" = 1 ]; then
  test -n "${DEPLOY_IMAGE:-}" || { echo 'A tested image is required' >&2; exit 1; }
  test -n "${GITHUB_SHA:-}" || { echo 'An exact source commit is required' >&2; exit 1; }
  REVISION=$(sh scripts/docker_local.sh image inspect "$DEPLOY_IMAGE" --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')
  test "$REVISION" = "$GITHUB_SHA" || { echo 'Image commit differs from approved CI commit' >&2; exit 1; }
else
  sh scripts/docker_local.sh compose build
fi
sh scripts/docker_local.sh compose up -d --no-build --wait --wait-timeout 120
python3 -c "import urllib.request,json; r=json.load(urllib.request.urlopen('http://127.0.0.1:$API_PORT/ready',timeout=10)); assert r['status']=='ready'; print(r)"
