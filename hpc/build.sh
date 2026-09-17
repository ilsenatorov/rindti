#!/usr/bin/env bash
# Build the RINDTI environment image and push it to GHCR.
#
#   ./hpc/build.sh           # build + push :<short-sha> and :latest
#   ./hpc/build.sh --no-push # build only, for local testing
#
# One-off: after the first push, make the package PUBLIC at
# https://github.com/users/ilsenatorov/packages/container/rindti/settings
# The execute nodes pull anonymously; a private package fails with "manifest unknown".
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/ilsenatorov/rindti}"
cd "$(dirname "$0")/.."

TAG="$(git rev-parse --short HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
    TAG="${TAG}-dirty"
fi

echo "==> building ${IMAGE}:${TAG}"
docker build -f hpc/Dockerfile -t "${IMAGE}:${TAG}" -t "${IMAGE}:latest" .

if [[ "${1:-}" == "--no-push" ]]; then
    echo "==> built, not pushing (--no-push)"
    exit 0
fi

echo "==> logging in to ghcr.io"
gh auth token | docker login ghcr.io -u "$(gh api user --jq .login)" --password-stdin

echo "==> pushing"
docker push "${IMAGE}:${TAG}"
docker push "${IMAGE}:latest"

cat <<MSG

Pushed ${IMAGE}:${TAG} and ${IMAGE}:latest.

The submit files use :latest. If you want a run pinned to this exact image, set
    docker_image = ${IMAGE}:${TAG}
in the .sub file instead.
MSG
