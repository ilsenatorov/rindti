# Queue files

Each `.sub` file queues one job per line of its queue file. Columns are comma-separated
and map to the `queue <names> from` line in the submit file.

**These files cannot contain comments or blank lines.** HTCondor's `queue ... from`
passes every line through verbatim, including `#` ones — a commented line becomes a job
with `#` as its first argument. Keep them as pure data; the explanations live here.

| File | Used by | Columns |
|---|---|---|
| `datasets.txt` | `hpc/download.sub` | dataset name: `davis`, `kiba`, `glass`, `BindingDB`, `BindingDB_Kd`, `BindingDB_Ki`, `BindingDB_IC50` |
| `prepare_configs.txt` | `hpc/prepare.sub` | path to a snakemake configfile under `config/snakemake/` |
| `train_runs.txt` | `hpc/train.sub` | configfile, dataset pickle, experiment name |

Paths are relative to the repo, which is the working directory `hpc/job.sh` cds into.

`train_runs.txt` ships **empty** — you have to fill it in, because the dataset pickle path
contains a hash of the snakemake config and is only known after a prepare job has run:

```bash
ls datasets/davis/results/prepare_all/*.pkl
```

See `train_runs.txt.example` for the shape. Submitting with an empty file reports
`WARNING: "hpc/train.sub" has only empty "queue" commands -- no jobs queued`, which is the
expected complaint.

The configfile column has to be one that *defines* `datamodule.filename` and
`datamodule.exp_name` - `--set` rejects keys the config does not already have. That means
`config/dti/base.yaml` (or a copy of it), not `config/test/default_dti.yaml`.
