#!/bin/bash
# Wrapper that every RINDTI condor job runs: set up the environment, then exec the
# real command.
#
#   executable = /scratch/chair_kalinina/s8ilsena/rindti/hpc/job.sh
#   arguments  = rindti-train config/dti/base.yaml ...
#
# This exists because the submit files cannot use `environment`: the cluster's
# JOB_TRANSFORM_AddHomeEnv only injects HOME when `environment` is unset or empty, and
# +WantGPUHomeMounted depends on that transform. So every variable is exported here
# instead.
set -euo pipefail

# $0 is /<root>/rindti/hpc/job.sh - derive the paths rather than hardcoding them, so the
# tree can be relocated or duplicated for a second checkout.
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HPC_DIR")"
ROOT="$(dirname "$REPO")"

cd "$REPO"

# The repo is not pip-installed in the image (it is mounted, not baked), so it reaches
# sys.path this way. /usr/local/bin/rindti-train relies on it.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Everything cacheable goes to scratch: the container filesystem is thrown away, and
# $HOME is the shared NFS home we would rather not fill with model weights.
export UV_CACHE_DIR="$ROOT/cache/uv"
# The image bakes UV_PYTHON_INSTALL_DIR=/opt/python, which is root-owned; HTCondor's
# docker universe runs the job as the submitting user, so uv cannot install an
# interpreter there. get_datasets.py needs a 3.11 one (see its PEP 723 header), so point
# uv at scratch, where it can download and then cache it across jobs.
export UV_PYTHON_INSTALL_DIR="$ROOT/cache/python"
export TORCH_HOME="$ROOT/cache/torch"   # fair-esm downloads its checkpoints here
export HF_HOME="$ROOT/cache/hf"
export MPLCONFIGDIR="$ROOT/cache/mpl"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$TORCH_HOME" "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# Condor captures stdout to a file; a progress bar refreshing 10x/second makes it huge.
export TQDM_MININTERVAL=10
export PYTHONUNBUFFERED=1

# Lightning/torch otherwise grab every core on the machine, not the ones we requested.
THREADS="${_CONDOR_REQUEST_CPUS:-${OMP_NUM_THREADS:-4}}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

echo "=== $(date -Is) on $(hostname) ==="
echo "repo:    $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "cwd:     $(pwd)"
echo "threads: $THREADS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "gpu:     none requested"
echo "cmd:     $*"
echo "=========================================="

exec "$@"
