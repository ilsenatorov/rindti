#!/usr/bin/env bash
# Build the RINDTI environment image and push it to GHCR.
#
#   ./hpc/build.sh           # build + push :<short-sha> and :latest
#   ./hpc/build.sh --no-push # build only, for local testing
#
# Needs a gh token with write:packages - the default `gh auth login` scopes do not
# include it and the push fails with "permission_denied: The token provided does not
# match expected scopes":
#     gh auth refresh -h github.com -s write:packages -s delete:packages
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
if ! gh auth status 2>&1 | grep -q "write:packages"; then
    echo "ERROR: your gh token lacks the write:packages scope; GHCR will reject the push." >&2
    echo "       run: gh auth refresh -h github.com -s write:packages -s delete:packages" >&2
    exit 1
fi
gh auth token | docker login ghcr.io -u "$(gh api user --jq .login)" --password-stdin

echo "==> pushing"
docker push "${IMAGE}:${TAG}"
docker push "${IMAGE}:latest"

# Record which commit :latest was built from. hpc/submit.sh diffs pyproject.toml and the
# Dockerfile against this, and warns when a job is about to run a stale environment - the
# repo is bind-mounted, so nothing else would reveal it.
git rev-parse HEAD > hpc/.image-tag
echo "==> wrote hpc/.image-tag ($(git rev-parse --short HEAD)); commit it"

cat <<MSG

Pushed ${IMAGE}:${TAG} and ${IMAGE}:latest.

The submit files use :latest. If you want a run pinned to this exact image, set
    docker_image = ${IMAGE}:${TAG}
in the .sub file instead.
MSG
