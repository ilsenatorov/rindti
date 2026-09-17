# Running RINDTI on the Saarland HPC (HTCondor)

The cluster (`conduit.hpc.uni-saarland.de` / `conduit2`) **only accepts containerised
jobs** — `SUBMIT_REQUIREMENT_UseDocker` rejects anything that is not `universe = docker`
or a local `.sif`. These files build the image and submit the three kinds of job.

The image carries **only the environment**. The repo is bind-mounted from `/scratch` at
job time, so a code change needs a `git pull` on the cluster, not a rebuild. Rebuild only
when `pyproject.toml` dependencies change.

## Layout

```
/scratch/chair_kalinina/$USER/
├── rindti/            # git clone - the code jobs run
│   ├── datasets  -> ../datasets
│   └── tb_logs   -> ../tb_logs
├── datasets/          # snakemake output (graph .pkl files)
├── tb_logs/           # lightning logs + checkpoints
├── cache/             # uv, torch/ESM weights, matplotlib, AlphaFold structures
└── runlogs/           # condor .out/.err/.log
```

## One-time setup

**1. Build and push the image** (locally, needs `docker` and an authenticated `gh`):

```bash
./hpc/build.sh
```

Then make the GHCR package **public**, once:
<https://github.com/users/ilsenatorov/packages/container/rindti/settings> → Change
visibility → Public. The execute nodes pull anonymously; a private package fails the job
with `manifest unknown`.

**2. Set up the cluster tree:**

```bash
ssh conduit
git clone https://github.com/ilsenatorov/rindti /tmp/rindti-bootstrap
bash /tmp/rindti-bootstrap/hpc/setup_cluster.sh
```

**3. Smoke test:**

```bash
cd /scratch/chair_kalinina/$USER/rindti
condor_submit hpc/smoke.sub
condor_q
cat ../runlogs/smoke.*.out    # should print torch version and a GPU name
```

## Running jobs

Always submit from the repo root (`/scratch/chair_kalinina/$USER/rindti`).

| Job | Queue file | What it runs |
|---|---|---|
| `hpc/download.sub` | `hpc/runs/datasets.txt` | `uv run workflow/scripts/get_datasets.py` — TDC tables + AlphaFold structures |
| `hpc/prepare.sub` | `hpc/runs/prepare_configs.txt` | `snakemake` — builds the graph dataset `.pkl` |
| `hpc/train.sub` | `hpc/runs/train_runs.txt` | `rindti-train` — one GPU per job |

Each submit file queues one job per non-comment line of its queue file, so the normal
loop is: edit the queue file, `condor_submit`. `train_runs.txt` ships with only comments —
fill it in or `condor_submit` will (correctly) report `0 job(s) submitted`.

A training job does not have to be a single run: `rindti.cli` expands any list-valued
config entry into a sweep variant and `config/dti/base.yaml` repeats each over `runs: 5`
seeds. Use the queue file for **datasets** and config lists for **hyperparameters** — that
keeps every job on one GPU and keeps the `describe_variant` run naming intact.

Collect results the same way as locally:

```bash
python -m rindti.utils.results --logdir tb_logs --output results.csv --summary true
```

## Monitoring

```bash
condor_q -nobatch                 # your queue
condor_q -better-analyze <id>     # why is it idle?
condor_tail -f <id>               # follow stdout
condor_ssh_to_job <id>            # shell inside the running container
condor_rm <id>
```

## Gotchas

- **Never add `environment = ...` to a submit file.** The cluster's
  `JOB_TRANSFORM_AddHomeEnv` injects `HOME` *only* when `environment` is unset or empty,
  and `+WantGPUHomeMounted` depends on it. All per-job variables go in `hpc/job.sh`.
- **`GPUs_Capability >= 7.5`** in `train.sub`/`smoke.sub` is load-bearing. The pool spans
  P100 (sm 6.0) to Blackwell RTX PRO 6000 (sm 12.0); the torch wheels in the image are
  built for `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`, so the P100 nodes and the V100
  node (`loki`) are excluded. The A100, H200 and Blackwell nodes all qualify.
- **The image does not use `uv.lock`.** The lock resolves torch 2.13.0 against the CUDA
  **13** wheels, and every GPU node here reports a CUDA 12.8 (some 12.4) driver, which
  cannot load a CUDA 13 runtime. The Dockerfile installs the same torch 2.13.0 from the
  `cu129` index instead, which runs on any 12.x driver via CUDA minor-version
  compatibility. Consequence: the image is pinned by tag, not by lockfile.
- **No conda in the image.** MMseqs2 is baked in and on `PATH`, so snakemake runs
  *without* `--software-deployment-method conda` and the `conda:` directive in
  `workflow/rules/data.smk` falls back to it.
- **Locally, `docker run` writes files as root.** HTCondor's docker universe passes
  `--user <your uid>`, so cluster jobs write as you; a local test run does not. If you
  test the image by hand against a working copy, expect root-owned `.snakemake/`,
  `data/` and `tb_logs/` afterwards - use a throwaway copy of the repo.
- **First pull on a node is slow** (the image is several GB and
  `DOCKER_IMAGE_CACHE_SIZE = 8`). Keep using `:latest` rather than a fresh tag per commit
  unless you need a run pinned.
