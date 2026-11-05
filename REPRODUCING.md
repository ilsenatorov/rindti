# Reproducing the published results

Everything behind the tables of the RINDTI application note: what to run, in what order,
and where each number in the paper comes from. The quick start in [`README.md`](README.md)
is the five-minute version; this is the campaign.

Two repositories are involved:

| | What it holds |
|---|---|
| this one | the pipeline, the model, every config, the template library |
| [`rindti-paper`](https://github.com/ilsenatorov/rindti-paper) | the manuscript, the collected results (`data/results_summary.csv`), the table generators and the HPC runbooks |

**Read [What you cannot reproduce exactly](#what-you-cannot-reproduce-exactly) before
starting.** Some of it needs inputs that are fetched at run time and are not pinned.

---

## 0. What the numbers cost

The full campaign is roughly 120 GPU jobs and about 76 prepared datasets. Peak disk for
the PyG caches was ~264 GB, and a single large experiment's cache is ~4.4 GB. On one
machine this is weeks; it was run on an HTCondor pool.

You do not need all of it. Most claims in the paper can be checked from one dataset:
§3 builds Davis, §4 trains it, §5 scores it against the label-only floor.

> **One rule that will cost you a day if you ignore it.** Never run two `snakemake` or
> `run_snakemake.py` invocations against the same checkout at once. The lock lives in a
> single `.snakemake/locks/` directory shared by the whole checkout, not per dataset, so
> concurrent sweeps race on it and fail with unrelated-looking `FileNotFoundError` and
> `CyclicGraphException` tracebacks. Use separate checkouts, or run them one at a time.

## 1. Get the code

```sh
git clone https://github.com/ilsenatorov/rindti
cd rindti
git checkout v3.0.0
```

The published runs were made on the line that became v3.0.0. Two fixes that ran as
uncommitted edits on the cluster are now in the tree (the PyG cache key, and the
`max_iterations` cap in `align_template`); so are the 17 paper-table configs, the 14
per-dataset ablation clones, and the kinase template library, all of which previously
existed only in the paper repository or on cluster scratch.

## 2. Build the environment

```sh
uv sync --locked --extra workflow        # add --extra esm for the ESM-1b arm
```

`uv.lock` is the pinned environment and `uv lock --check` verifies it against
`pyproject.toml`. MMseqs2 is needed only for `cluster_target` splits; either put it on
`PATH` or run snakemake with `--software-deployment-method conda`, which builds it from
`workflow/envs/mmseqs.yaml`.

**The GPU campaign did not use this lockfile.** `hpc/Dockerfile` deliberately installs
`-r pyproject.toml --torch-backend=cu129` instead, because `uv.lock` resolves torch
2.13.0 against CUDA 13 wheels while the GPU pool's drivers are CUDA 12.x. The published
runs used torch 2.13.0+cu129. If your numbers differ in the last decimal, this is the
first place to look.

## 3. Build the datasets

Download the raw tables and AlphaFold structures. PyTDC pins `numpy<2` and cannot share
an environment with the training stack, so the downloader declares its dependencies
inline (PEP 723) and runs in its own environment:

```sh
uv run workflow/scripts/get_datasets.py davis --min_num_aa 100 --download_structures true
uv run workflow/scripts/get_datasets.py kiba --min_num_aa 100 --download_structures true
uv run workflow/scripts/get_datasets.py bindingdb_kd --min_num_aa 100 --download_structures true
```

`--min_num_aa 100` is what the campaign used; the script's own default is unbounded.

Davis lists gene names where AlphaFold is keyed by UniProt accession, so its structures
need reconciling by sequence:

```sh
python workflow/scripts/link_structures.py datasets/davis/resources            # dry run
python workflow/scripts/link_structures.py datasets/davis/resources --apply true
```

For the `template` and `bsite` rows, install the kinase template library (kinase panels
only — BindingDB_Kd spans too many families for this to mean anything):

```sh
for ds in davis kiba; do
  mkdir -p datasets/$ds/resources/templates
  cp resources/templates/*.pdb datasets/$ds/resources/templates/
done
```

Then build. Each command is a sweep over one config; run them one at a time:

```sh
python run_snakemake.py config/snakemake/benchmark_splits_davis.yaml --threads 8
python run_snakemake.py config/snakemake/ablation/structs.yaml --threads 8
python run_snakemake.py config/snakemake/extra/threshold_davis.yaml --threads 8
# ... and so on, per dataset and per axis
```

Check what you built before spending GPU time on it:

```sh
python workflow/scripts/dataset_stats.py 'datasets/*/results/prepare_all/*.pkl' --table dataset_stats.tsv
python workflow/scripts/dataset_index.py 'datasets/*/results/prepare_all/*.pkl' --table index.tsv
```

`dataset_stats.py` exits non-zero on a cold split that leaks an entity across the
train/test boundary, and on an empty `val`/`test`. Both are worth catching here rather
than after a GPU run.

## 4. Train

```sh
rindti-train config/dti/base.yaml \
    --set datamodule.filename=datasets/davis/results/prepare_all/<hash>.pkl \
    --set datamodule.exp_name=davis_random
```

The `<hash>` is the md5 of the config that built the dataset, which `dataset_index.py`
maps back for you. `exp_name` must be unique per job: it names the TensorBoard directory
and nothing else, since the dataset cache is keyed by the pickle alone.

**Seeds.** `config/dti/base.yaml` sets `seed: 42` and `runs: 5`. The per-run seeds are
derived, not listed: `rindti/cli.py` does `seed_everything(42)` and then
`random.sample(range(1, 100), runs)`, so they are reproducible from `(seed, runs)` and
recorded in each run's `hparams.yaml`. The main benchmark and ESM-1b arms use five
seeds, the ablations three.

Splits carry no separate seed and were built once; the uncertainty in the paper is over
training seeds on one split per protocol, not over split draws.

## 5. Score against the label-only floor

The pipeline writes `results/baseline/<split name>.json` for every split it builds, with
all four label-only baselines. To put a model's metrics against it:

```sh
python -m rindti.models.dti.baseline.run score \
    '{"auc": 0.887}' datasets/davis/results/baseline/<split name>.json
```

```
auc        model 0.887  floor 0.843  BOSS +0.280
```

That reports BOSS: the fraction of the gap between the floor and a perfect score that the
model closes. The floor is the strongest of the four baselines **per metric**, which
makes BOSS a lower bound. On AUROC the strongest is always the target–drug average; on
MCC it is usually the drug-only prior, by a wide margin.

## 6. Collect the results

```sh
python -m rindti.utils.results --logdir tb_logs --output results.csv --summary true
```

Then, in the paper repository, drop the dry runs and the partially written retry
versions — this step appears in no other document:

```sh
python scripts/filter_results.py results.csv data/results_summary.csv
python scripts/make_tables.py --out tables      # or: make tables
```

`data/results_summary.csv` is authoritative. `rindti_results/results.csv` in the paper
repository is a superseded Phase-9 export that nothing reads.

## 7. Table 1 (published models)

Table 1 compares *reported* numbers from published papers against label-only baselines on
those papers' own splits. The baselines are scripted; the published values are not.

```sh
python -I scripts/sota_baselines.py DATA_DIR > notes/sota_baselines.tsv
```

`DATA_DIR` holds split files downloaded from the original repositories; the layout and
the source URLs are in the script's docstring. The reported values in `notes/sota_bff.md`
were transcribed by hand from the papers and have no generator, so they cannot be
regenerated — only re-checked against the sources.

---

## Where each table comes from

| Table / figure | Generated by | From |
|---|---|---|
| Table 1 (published models) | `make_tables.py` | `notes/sota_bff.md` + `notes/sota_baselines.tsv` |
| Table 2 (architectures) | `make_tables.py` | `data/results_summary.csv`, `rindti_results/baselines*.txt`, `split_index.tsv` |
| Table 3 (reference model) | `make_tables.py` | the same three |
| Table 4 (preprocessing) | `make_tables.py` | the same three + `dataset_stats_all.tsv` |
| Supp. Table (parameters) | `make_tables.py` | `data/config_defaults/*.yaml` |
| Figure 1 | `assemble_fig1.py` | OPM 5TVN, downloaded at build time |
| Supp. figures | `plot_split_results.py`, `plot_representation.py` | `data/results_summary.csv` |

The tables in Supplementary Files 1, 3 and 4's ablation sections are typed by hand from
`data/results_summary.csv` and `dataset_stats_all.tsv`. They have no generator, which is
the weakest provenance link in the submission.

## What you cannot reproduce exactly

- **Split filenames have changed.** `split_data` and `parse_dataset` now put the seed in
  their output names, because a re-seeded run previously found the old files under the
  same name and silently reused them. A fresh build writes `prncn_15995058.tsv` where the
  archived `baselines.txt` and `split_index.tsv` record `prncn_adb5b2e2.tsv`. The seed was
  42 throughout, so the contents are identical — only the name differs. `prepare_all`
  names already hashed the seed and are unaffected, including the
  `nlpncnndlwr_83288203.pkl` that Supplementary File 2 cites.
- **Fetched at run time, not pinned:** the TDC tables, AlphaFold models, the ESM-1b
  checkpoint, MMseqs2 (`mmseqs.com/latest`), the base CUDA image, and OPM 5TVN for
  Figure 1. The container was pinned only by the floating tag `:latest`; no digest was
  recorded.
- **The derived split and cluster files are not in either repository.** They live on
  cluster scratch and are intended for the Zenodo archive.
- **Davis UniProt mapping.** The real 379/379 map is on the cluster; the paper repository
  carries only `data/davis_mapping_reconstructed.tsv`, whose own docstring says it is not
  a record of the run.
- **Training is not bit-deterministic.** No deterministic flags are set, so expect
  agreement to within the seed-to-seed spread the tables report, not exact equality.

## Arms that are absent on purpose

So you do not go looking for them:

- `prot_node=onehot` crashed during graph batching in the campaign and is not a valid
  negative result. The cause — one-hot rows built as int64 and read as labels — is fixed
  in this version, so the arm can now be run; the paper reports it as missing.
- `cluster_drug` and `drug` coincide on Davis, which has only 66 ligands.
- `template` and `bsite` exist for Davis and KIBA only.
- ESM-1b truncates at 1,022 residues, so for 70 of 379 Davis sequences the sequence arm
  and the graph arm are not seeing the same protein.
