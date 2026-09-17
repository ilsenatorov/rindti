---
name: hpc-submit
description: Use when running RINDTI work on the Saarland HTCondor cluster (conduit/conduit2) - submitting or queueing a download, prepare, sweep, train or smoke job, pushing an experiment to the pool, checking on a submitted cluster id, or asking why a job is idle, failed, or ran the wrong code.
---

# Pushing RINDTI jobs to HPC

## Overview

Every submission goes through `hpc/submit.sh`. It is the reproducible action: same five
steps in the same order, refusing to submit when one fails, and recording what was
submitted from which commit.

```bash
./hpc/submit.sh <job> [runfile]
```

It runs from a local checkout (driving `conduit` over ssh) or from the clone on the
submit node, choosing by whether `condor_submit` is on `PATH`. **Do not call
`condor_submit` by hand** — see Why below.

## The jobs

| Job | Resources | Default queue file | Runs |
|---|---|---|---|
| `smoke` | 1 GPU | none | `hpc/smoke.py` — image, mounts, GPU |
| `download` | CPU | `hpc/runs/datasets.txt` | TDC tables + AlphaFold structures |
| `prepare` | CPU | `hpc/runs/prepare_configs.txt` | `snakemake` — config with **no lists** |
| `sweep` | CPU | `hpc/runs/prepare_sweeps.txt` | `run_snakemake.py` — config **with lists** |
| `train` | 1 GPU/line | `hpc/runs/train_runs.txt` | `rindti-train` |

`prepare` vs `sweep` is the one choice the script cannot make for you: snakemake does not
expand list-valued config entries, `run_snakemake.py` does.

## Steps

1. **Commit and push first.** The cluster runs what is pushed; the script refuses a dirty
   tree or an unpushed HEAD, then fast-forwards the cluster clone to it and checks the
   two match.
2. **Rebuild the image only if `pyproject.toml` or `hpc/Dockerfile` changed** — the repo
   is bind-mounted, not baked. `./hpc/build.sh`, then commit `hpc/.image-tag`. The script
   warns when this is overdue.
3. **Make sure the queue file exists on the cluster.** Train queue files carry dataset
   content hashes that only exist after a prepare job, so they are generated there, not
   committed:
   ```bash
   ssh conduit
   cd /scratch/chair_kalinina/$USER/rindti
   python workflow/scripts/dataset_index.py 'datasets/*/results/prepare_all/*.pkl' \
       --emit-runs config/dti/base.yaml --runs 5 > hpc/runs/train_main.txt
   ```
   `hpc/runs/README.md` has the other generators, including `./hpc/gen_model_ablation.sh`.
4. **Dry-run when the queue file is new**: `./hpc/submit.sh --dry-run train hpc/runs/train_main.txt`
   validates and expands the submit file without queueing anything.
5. **Submit.** The script prints the cluster id and the `condor_q` / `condor_tail` /
   `condor_rm` lines for it.

## Options

| Flag | Use when |
|---|---|
| `--dry-run` | validating a new queue file; queues nothing |
| `--host conduit2` | `conduit` is down |
| `--no-sync` | resubmitting against the commit already on the cluster |
| `--allow-dirty` | genuinely testing uncommitted code, knowing the run is unreproducible |

`--allow-dirty` is not the way past a failing check. A dirty-tree error means commit and
push; a commit-mismatch error means the branch the cluster tracks is behind.

## Afterwards

```bash
ssh conduit tail -f /scratch/chair_kalinina/$USER/runlogs/train.<cluster>.0.out
ssh conduit condor_q -better-analyze <cluster>     # idle: why nothing matches
```

Every submission is appended to `$root/runlogs/submissions.tsv` (time, cluster id, job,
commit, queue file, job count) with the queue file itself copied to
`$root/runlogs/queues/<cluster>.<job>.txt`. That pair is how a results row is traced back
to what produced it — read it before re-deriving anything by hand.

Collect results with:

```bash
python -m rindti.utils.results --logdir tb_logs --output results.csv --summary true
```

## Why not condor_submit directly

These are the failures the script exists to catch; all of them are silent.

| Mistake | What happens |
|---|---|
| Queue file has a `#` comment or blank line | Condor queues a job with `#` as its argument |
| Two train lines share dataset + `exp_name` | Both pick the same `version_N` and overwrite each other's runs |
| A comma inside the `extra` column | The line is read as 5 columns and corrupted |
| Empty `extra` column | The seed budget (`--set runs=N`) is missing |
| Empty queue file | `0 job(s) submitted`, easy to miss in the output |
| Cluster clone behind local HEAD | Jobs run code you are not looking at |
| Dependency change without a rebuild | Jobs run the old environment |

## Reference

- `hpc/README.md` — how the HTCondor setup is put together, and the gotchas
  (the `environment` trap, `GPUs_Capability >= 7.5`, no conda in the image)
- `hpc/EXPERIMENTS.md` — the campaign: which phase to run and how to tell it worked
- `hpc/runs/README.md` — queue file columns and the rules behind them
