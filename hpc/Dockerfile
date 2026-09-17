# Environment image for running RINDTI on the Saarland HPC (HTCondor, docker universe).
#
# The image carries *only* the environment - python deps, CUDA and MMseqs2. The repo
# itself is bind-mounted from shared storage at job time (see hpc/job.sh), so a code
# change needs a `git pull` on the cluster, not a rebuild.
#
# Build and push with hpc/build.sh.

# `base` (not `runtime`/`devel`): the CUDA libraries torch needs come from its own pip
# wheels. What this base is actually for is the NVIDIA_VISIBLE_DEVICES /
# NVIDIA_DRIVER_CAPABILITIES defaults that let the container runtime inject the driver -
# HTCondor submit files here cannot set `environment` (see hpc/README.md), so those have
# to come from the image.
FROM nvidia/cuda:12.9.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libgomp1 procps \
    && rm -rf /var/lib/apt/lists/*

# uv stays in the final image: workflow/scripts/get_datasets.py is a PEP 723 script with
# its own numpy<2 dependency set and is run with `uv run`.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

# MMseqs2, needed by the cluster_target split. Having it on PATH is what lets snakemake
# run *without* --software-deployment-method conda: the `conda:` directive in
# workflow/rules/data.smk is ignored when conda deployment is off and the rule falls back
# to the `mmseqs` on PATH. No conda in this image at all.
RUN mkdir -p /opt && curl -fsSL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz \
        | tar -xz -C /opt \
    && /opt/mmseqs/bin/mmseqs version

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/mmseqs/bin:$PATH

# Dependencies only - `-r pyproject.toml` installs the project's dependencies without
# installing the project, which is what we want since the code is mounted.
#
# NOT `uv sync --frozen`: uv.lock resolves torch 2.13.0 against the CUDA *13* wheels, and
# every GPU node in this pool reports a CUDA 12.8 (some 12.4) driver, which cannot load a
# CUDA 13 runtime. The cu129 index has the same torch 2.13.0 built against CUDA 12.9,
# which runs on any 12.x driver via CUDA minor-version compatibility.
COPY pyproject.toml README.md LICENSE /src/
COPY rindti/version.py /src/rindti/version.py
RUN uv venv --python 3.12 /opt/venv \
    && uv pip install --no-cache --torch-backend=cu129 \
        -r /src/pyproject.toml --extra workflow --extra esm \
    && rm -rf /src

# Replaces the console script that `uv pip install -e .` would normally provide. The
# mounted repo is put on sys.path via PYTHONPATH in hpc/job.sh.
RUN printf '#!/opt/venv/bin/python\nfrom rindti.cli import train_cli\n\ntrain_cli()\n' \
        > /usr/local/bin/rindti-train \
    && chmod +x /usr/local/bin/rindti-train

WORKDIR /workspace
