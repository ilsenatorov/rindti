# Queue files

Each `.sub` file queues one job per line of its queue file. Columns are comma-separated
and map to the `queue <names> from` line in the submit file.

**These files cannot contain comments or blank lines.** HTCondor's `queue ... from`
passes every line through verbatim, including `#` ones — a commented line becomes a job
with `#` as its first argument. Keep them as pure data; the explanations live here.

Every submit file reads `$(runfile)`, which has a default but is meant to be overridden
per phase:

```bash
condor_submit -a 'runfile=hpc/runs/train_main.txt' hpc/train.sub
```

`condor_submit -a` inserts its argument immediately before the `queue` command, which is
why this overrides the default set higher up in the file.

| File | Used by | Columns |
|---|---|---|
| `datasets.txt` | `hpc/download.sub` | dataset name: `davis`, `kiba`, `glass`, `BindingDB`, `BindingDB_Kd`, `BindingDB_Ki`, `BindingDB_IC50` |
| `prepare_single.txt`, `prepare_configs.txt` | `hpc/prepare.sub` | one snakemake configfile, **no lists in it** |
| `prepare_sweeps.txt` | `hpc/sweep.sub` | one snakemake configfile **containing lists**, expanded by `run_snakemake.py` |
| `train_*.txt` | `hpc/train.sub` | configfile, dataset pickle, experiment name, extra `--set` overrides |

Paths are relative to the repo, which is the working directory `hpc/job.sh` cds into.

## The `extra` column

The fourth column of a train queue file holds arbitrary `--set key.path=value` overrides
for that job alone. It exists so that an ablation runs **one condor job per
configuration**, in parallel across the pool, instead of one job expanding a list-valued
config into 45 configurations run serially on a single GPU.

Three rules:

- **No commas.** `queue ... from` splits columns on commas, so a YAML list (`[a, b]`)
  would be read as two extra columns and corrupt the line. One job per value instead —
  which is the entire point of the column.
- **Never empty.** Every line carries at least `--set runs=N`. That is also how the two
  seed budgets are applied: `runs=5` for headline results, `runs=3` for ablations.
- **Only keys that already exist.** `rindti.cli.apply_override` rejects unknown keys, so
  a typo fails the job immediately rather than being silently ignored.

## `exp_name` must be unique per job

`rindti.cli` appends a `describe_variant` tag to the log directory **only when the config
file itself holds lists**. Overrides passed with `--set` produce a single variant, so
there is no tag, and two jobs sharing an `exp_name` and a dataset both call
`next_version()` on the same directory. They pick the same `version_N`, and since the
per-run seeds are a deterministic function of `seed` and `runs`, they then write into
identical leaf directories and overwrite each other.

So put the axis in the name: `davis_random_node=gatconv`, not `davis_random`. Nothing
downstream depends on the name — `rindti.utils.results` recovers the axis from the
`hparams.yaml` Lightning writes beside each run — it is purely about not colliding.

## Filling the train queue files

The dataset column is a content hash of the snakemake config and is only known once a
prepare job has run, so these files are generated on the cluster rather than committed:

```bash
# everything, for a look
python workflow/scripts/dataset_index.py 'datasets/*/results/prepare_all/*.pkl'

# the main benchmark: every split of every dataset, 5 seeds
python workflow/scripts/dataset_index.py 'datasets/*/results/prepare_all/*.pkl' \
    --emit-runs config/dti/base.yaml --runs 5 > hpc/runs/train_main.txt

# the pipeline ablation: random-split Davis only, 3 seeds
python workflow/scripts/dataset_index.py 'datasets/davis/results/prepare_all/*.pkl' \
    --where split=random --emit-runs config/dti/base.yaml --runs 3 \
    > hpc/runs/train_pipeline_ablation.txt

# the model ablation: 16 lines over three pickles
./hpc/gen_model_ablation.sh <random.pkl> <cluster_target.pkl> <edge_distance.pkl> \
    > hpc/runs/train_model_ablation.txt
```

See `train_runs.txt.example` for the shape. `train_runs.txt` ships empty and is the
scratch file for ad-hoc runs; submitting it reports
`WARNING: ... has only empty "queue" commands -- no jobs queued`, which is the expected
complaint.

The configfile column has to be one that *defines* `datamodule.filename` and
`datamodule.exp_name` — `--set` rejects keys the config does not already have. That means
`config/dti/base.yaml` or `config/dti/esm.yaml`, not `config/test/default_dti.yaml`.
