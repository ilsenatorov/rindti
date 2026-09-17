#!/usr/bin/env bash
# One-time setup on the submit node (conduit / conduit2).
#
#   ssh conduit
#   git clone git@github.com:ilsenatorov/rindti.git /tmp/rindti-bootstrap
#   bash /tmp/rindti-bootstrap/hpc/setup_cluster.sh
#
# Creates the scratch tree the submit files expect and clones the repo into it.
# Safe to re-run: existing directories and an existing clone are left alone.
set -euo pipefail

ROOT="${RINDTI_ROOT:-/scratch/chair_kalinina/$USER}"
# SSH, not HTTPS: the submit nodes cannot complete a TLS handshake to github.com
# ("gnutls_handshake() failed"), so an HTTPS clone works once at best and can never be
# pulled again. git@ works from a key in ~/.ssh with no agent forwarding.
REPO_URL="${RINDTI_REPO_URL:-git@github.com:ilsenatorov/rindti.git}"
BRANCH="${RINDTI_BRANCH:-hpc}"

mkdir -p "$ROOT"/{datasets,tb_logs,cache,runlogs}

if [[ -d "$ROOT/rindti/.git" ]]; then
    echo "==> $ROOT/rindti already a clone, pulling"
    git -C "$ROOT/rindti" pull --ff-only
else
    echo "==> cloning $REPO_URL ($BRANCH) into $ROOT/rindti"
    git clone --branch "$BRANCH" "$REPO_URL" "$ROOT/rindti"
fi

# datasets/ and tb_logs/ are gitignored, so pointing them at the sibling directories
# keeps the bulky outputs out of the clone while letting the configs keep using the
# relative paths they use locally (datasets/davis/results/...).
ln -sfn "$ROOT/datasets" "$ROOT/rindti/datasets"
ln -sfn "$ROOT/tb_logs" "$ROOT/rindti/tb_logs"

cat <<MSG

Ready.

  root:     $ROOT
  repo:     $ROOT/rindti
  submit:   cd $ROOT/rindti && condor_submit hpc/smoke.sub

Note the submit files default to RINDTI_ROOT=/scratch/chair_kalinina/s8ilsena. If your
root differs, edit the ROOT line in hpc/*.sub (it appears once per file, as
\$(root) at the top).
MSG
