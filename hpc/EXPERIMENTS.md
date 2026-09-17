# The experiment campaign, end to end

This is the runbook for the benchmark re-run: from an empty `/scratch` tree to
`results.csv`. It assumes the cluster setup already works — [`hpc/README.md`](README.md)
is the reference for *how* the HTCondor side is put together (image, mounts, the
`environment` trap, GPU capability requirement). This file is *what to run, in what
order, and how to tell it worked*.

## Why this is a full re-run

The `[Unreleased]` section of [`CHANGELOG.md`](../CHANGELOG.md) invalidates every number
the project has produced. The load-bearing ones:

- `trainer.test` evaluated the weights training *stopped* on, 30 epochs past the optimum,
  while `ModelCheckpoint` had the best ones and never loaded them. Every test metric was
  systematically pessimistic.
- `ChebConvNet` was measurably *exactly* affine, and GAT/Transformer were non-linear only
  through their attention softmax — so the `node.module` ablation compared a GNN against a
  linear model.
- `element_l1` and `element_l2` computed the same function to six decimal places; the
  `feat_method` ablation ran one arm twice under two names.
- `node.dropout` was inert in every module.
- `split_groups` sent any bin shorter than ten entirely to train, so small cold splits had
  an empty val and test.
- Cold-*drug* splits grouped on raw `Drug_ID` and did not deduplicate, while cold-*target*
  splits did — the two controlled leakage to different standards, in a pipeline whose
  entire purpose is controlling leakage.
- Regression labels for `unit: nM` were `log10(Kd)` rather than pKd, which `RM2` is not
  invariant to.

The changelog explicitly requires cold-split and regression datasets to be rebuilt. In
practice that means everything: start from raw data.

## Scope

| | |
|---|---|
| **Datasets** | Davis, KIBA, BindingDB_Kd |
| **Seeds** | 5 for headline results, 3 for ablations |
| **Blocks** | split benchmark · structure-vs-sequence · regression · model ablation · pipeline ablation |
| **Jobs** | 15 CPU (3 download + 12 build), 41 GPU (15 + 3 + 16 + 7) |

BindingDB_Kd rather than Ki or IC50: it is the same quantity Davis reports, so the two are
directly comparable. IC50 is assay-dependent and must not be pooled with either. GLASS is
out of scope for this campaign.

---

## Phase 0 — Image and smoke test

Rebuild the image **only if `pyproject.toml` dependencies changed** since the last push —
the repo is bind-mounted, not baked, so ordinary code changes need a `git pull` on the
cluster and nothing else.

```bash
# locally, and only if deps changed
gh auth refresh -h github.com -s write:packages -s delete:packages   # one-off
./hpc/build.sh
```

On the cluster:

```bash
ssh conduit          # or conduit2
cd /scratch/chair_kalinina/$USER/rindti && git pull
condor_submit hpc/smoke.sub
condor_q
cat ../runlogs/smoke.*.out
```

> **The remote must be SSH, not HTTPS.** HTTPS to github.com fails from the submit nodes
> with `gnutls_handshake() failed: The TLS connection was non-properly terminated`, while
> `git@github.com:` works from a key in `~/.ssh` with no agent forwarding. A clone made
> with the HTTPS URL will sit silently at whatever commit it was created at — check with
> `git remote -v` and fix it once:
>
> ```bash
> git remote set-url origin git@github.com:ilsenatorov/rindti.git
> ```
>
> `setup_cluster.sh` defaults to the HTTPS URL, so pass
> `RINDTI_REPO_URL=git@github.com:ilsenatorov/rindti.git` when bootstrapping.

**Worked if:** the output names a GPU, prints `cuda ok True`, a capability ≥ 7.5, and an
mmseqs version.

---

## Phase 1 — Download (CPU · `download.sub` · 3 jobs)

```bash
condor_submit -a 'runfile=hpc/runs/datasets.txt' hpc/download.sub
```

Queue file already holds `davis`, `kiba`, `BindingDB_Kd`.

All three write into the shared `$(root)/cache/structures`, so on a **cold cache submit
them one at a time** — three processes racing to fetch the same overlapping AlphaFold
models is wasted bandwidth at best. Once the cache is warm they are safe to run together.

These jobs are network-bound and slow. If an execute node cannot reach `tdcommons.ai` or
`alphafold.ebi.ac.uk`, run the command on the submit node instead.

### ⛔ Gate A — Davis structure naming

`get_datasets.py` requests AlphaFold models **by `Target_ID`** and then drops every
interaction whose target has no structure. AlphaFold DB is keyed by UniProt accession,
while TDC's Davis may key its tables by gene name (`AAK1`, `PHKG2`) — the exact mismatch
[`link_structures.py`](../workflow/scripts/link_structures.py) exists to repair. If it
bites, Davis silently becomes an **empty dataset with no error**.

```bash
wc -l datasets/davis/resources/tables/inter.tsv datasets/davis/resources/tables/prot.tsv
ls datasets/davis/resources/structures | wc -l
```

**Pass:** a few hundred targets and tens of thousands of interactions.
**Fail:** near zero.

Recovery, if it fails:

1. Re-run the download with `--download_structures false` so the full tables survive.
2. Obtain AlphaFold models for the corresponding UniProt accessions and drop them **flat**
   into `datasets/davis/resources/structures/` (nested directories are rejected with an
   explicit error).
3. Reconcile by sequence — it matches exact, then substring, then similarity above a
   threshold, and reports rather than guesses at anything below it:

```bash
python workflow/scripts/link_structures.py datasets/davis/resources              # dry run
python workflow/scripts/link_structures.py datasets/davis/resources --apply true # writes structure_mapping.tsv
```

**Do not start Phase 2 for Davis until this gate passes.**

KIBA and BindingDB_Kd use UniProt accessions and should not hit this.

---

## Phase 2 — Build datasets (CPU · `sweep.sub` + `prepare.sub` · 12 jobs)

Two submit files because snakemake does not expand lists; `run_snakemake.py` does.

```bash
condor_submit -a 'runfile=hpc/runs/prepare_sweeps.txt' hpc/sweep.sub    # 9 jobs, configs WITH lists
condor_submit -a 'runfile=hpc/runs/prepare_single.txt' hpc/prepare.sub  # 3 jobs, configs without
```

| Queue file | Builds |
|---|---|
| `prepare_sweeps.txt` | 3 × 5 split variants, plus the six one-factor ablation axes |
| `prepare_single.txt` | `davis_esm`, `davis_reg`, `kiba_reg` |

Yields **15 split datasets + 8 distinct ablation datasets + 3 singles**. The ablation
axes share their baseline arm with the Davis random split, and snakemake's caching means
that arm is built once.

Notes:

- The `cluster_target` splits run MMseqs2 over every target and dominate the wall clock;
  BindingDB_Kd's is the longest job in the phase.
- `davis_esm` downloads ESM-1b weights into `$(root)/cache/torch` on first run.
- `run_snakemake.py` exits non-zero listing the runs that failed, so a sweep cannot
  "finish" with half its datasets missing. **Check the exit status**, not just that the
  job left the queue.

### ⛔ Gate B — every dataset is sound

```bash
python workflow/scripts/dataset_stats.py datasets/*/results/prepare_all/*.pkl --table dataset_stats.tsv
echo "exit=$?"
```

Non-zero means one of the two failures worth catching before a GPU run rather than after:

- **leakage** — a cold split with an entity on both sides of the train/test boundary,
  which is precisely what similarity-aware splitting exists to prevent;
- **a degenerate split** — an empty `val` or `test`.

Keep `dataset_stats.tsv`: it is the supplementary dataset table, and it is what keeps the
target counts quoted in the manuscript tied to the graphs actually trained on.

Then build the lookup from hash back to config:

```bash
python workflow/scripts/dataset_index.py 'datasets/*/results/prepare_all/*.pkl' --table index.tsv
```

Also, for Phase 5: **grep the `davis_esm` job log for the truncation count** that
`prot_esm.py` prints. ESM-1b's context is 1022 residues while the structure arm uses the
whole chain, so that number is a confound in the comparison the two arms exist to make and
belongs in the write-up.

---

## Phase 3 — Baselines (CPU · minutes · no condor job)

How much of the score is label priors rather than structure. The cheapest result in the
campaign, and the floor every other number is read against.

```bash
for tsv in datasets/*/results/split_data/*.tsv; do
    echo "== $tsv"
    python -m rindti.models.dti.baseline.run max "$tsv"
    python -m rindti.models.dti.baseline.run prot_drug_max "$tsv" --which prot
    python -m rindti.models.dti.baseline.run prot_drug_max "$tsv" --which drug
    python -m rindti.models.dti.baseline.run prot_drug_max "$tsv" --which both
done | tee baselines.txt
```

Arguments are positional (`run.py <model> <filename>`), not `--model`/`--filename`.

---

## Phase 4 — Main benchmark (GPU · 15 jobs · 5 seeds)

The headline table: three datasets under five leakage regimes.

Phase 2 built the ablation datasets into the same directories, so the selection has to
say "every split, every other axis at its baseline". `--where` clauses are ANDed, which
expresses exactly that:

```bash
# inspect first: this must print exactly FIFTEEN rows (3 datasets x 5 splits)
python workflow/scripts/dataset_index.py 'datasets/*/results/prepare_all/*.pkl' \
    --where task=class --where structs=whole --where protfeat=distance \
    --where prot_node=label --where prot_edge=none --where drug_node=label \
    --where filtering=posneg --where sampling=none

# then the same selection, as queue lines
python workflow/scripts/dataset_index.py 'datasets/*/results/prepare_all/*.pkl' \
    --where task=class --where structs=whole --where protfeat=distance \
    --where prot_node=label --where prot_edge=none --where drug_node=label \
    --where filtering=posneg --where sampling=none \
    --emit-runs config/dti/base.yaml --runs 5 > hpc/runs/train_main.txt
```

Confirm the fifteen rows before generating the queue file.

Two things the flags will not let you do by accident: repeating a column (`--where
split=random --where split=drug`) matches nothing, because clauses are ANDed rather than
ORed; and collecting the flags into a shell variable is rejected rather than silently
ignored, because `zsh` does not word-split an expansion and argparse reads the resulting
single token as a path.

```bash
condor_submit -a 'runfile=hpc/runs/train_main.txt' hpc/train.sub
```

BindingDB_Kd is much larger than Davis. If its jobs turn out epoch-bound rather than
convergence-bound, raise the batch size (`--set datamodule.batch_size=512`, the precedent
being `config/dti/glass.yaml`) rather than cutting `max_epochs` — early stopping is
supposed to be what ends training.

---

## Phase 5 — Structure vs sequence, and regression (GPU · 3 jobs · 5 seeds)

Write `hpc/runs/train_arms.txt` with the three pickle paths from `index.tsv`:

```
config/dti/esm.yaml, <davis_esm.pkl>, davis_esm, --set runs=5
config/dti/base.yaml, <davis_reg.pkl>, davis_reg, --set runs=5 --set model.module=reg
config/dti/base.yaml, <kiba_reg.pkl>, kiba_reg, --set runs=5 --set model.module=reg
```

```bash
condor_submit -a 'runfile=hpc/runs/train_arms.txt' hpc/train.sub
```

- The **ESM arm** uses `config/dti/esm.yaml`, not `base.yaml`: the ESM featurisation is
  one mean-pooled 1280-d vector per protein with no `edge_index`, so the graph encoder
  raises `KeyError` on it. Its comparator is the `davis_random` job from Phase 4 — same
  split, same drug tower, protein tower swapped for `method: vector`.
- The **regression arms** override `model.module`, which `check_task_matches` then
  verifies against the pickle's `parse_dataset.task` — a `reg` dataset trained as `class`
  used to run BCE against continuous affinities and report AUROC on them without error.
  These report MSE, Pearson/Spearman, the Gonen–Heller concordance index and RM2, which
  are the metrics the DeepDTA → GraphDTA → DGraphDTA → GEFA lineage reports.

---

## Phase 6 — Model ablation (GPU · 16 jobs · 3 seeds)

One factor at a time around `config/dti/base.yaml`, on **two** splits — `random` and
`cluster_target`. Running both is the point: an architectural choice that helps under a
random split and evaporates under a cold one is the finding.

```bash
./hpc/gen_model_ablation.sh <davis_random.pkl> <davis_cluster_target.pkl> <davis_edgedist.pkl> \
    > hpc/runs/train_model_ablation.txt
condor_submit -a 'runfile=hpc/runs/train_model_ablation.txt' hpc/train.sub
```

| axis | baseline | ablated values |
|---|---|---|
| `model.feat_method` | concat | element_l1, mult |
| `model.prot.node.module` | ginconv | gatconv, chebconv, filmconv |
| `model.prot.pool.module` | mean | attention, set2set |

The baseline arm is **not** re-run — it is the corresponding Phase 4 job, at 5 seeds
rather than 3. Report `n` alongside mean and std; `results.py --summary true` emits that
column already.

`transformer` is handled separately, on the `edge_feats: distance` dataset, with a
`ginconv` job on that same dataset as its matched control. It is the only module that
consumes continuous edge attributes — GIN/GAT/Cheb ignore edges and FiLM wants discrete
relation types — so on the plain contact graph it would measure nothing it was chosen for,
and against a *different* dataset without the control the architecture change would be
confounded with the input change.

---

## Phase 7 — Pipeline ablation (GPU · 7 jobs · 3 seeds)

The dataset-construction axes, read against the same `davis_random` baseline.

```bash
python workflow/scripts/dataset_index.py 'datasets/davis/results/prepare_all/*.pkl' \
    --where split=random --emit-runs config/dti/base.yaml --runs 3 \
    > hpc/runs/train_pipeline_ablation.txt
condor_submit -a 'runfile=hpc/runs/train_pipeline_ablation.txt' hpc/train.sub
```

`--where split=random` is what separates these from the Phase 4 split datasets sitting in
the same directory.

The command generates **nine** lines; delete two before submitting, leaving seven:

- the **baseline** (every axis at its default) — Phase 4 already ran it, at 5 seeds;
- the **`prot_edge=distance`** line — Phase 6 runs that dataset, with the transformer that
  is the only module able to consume the edge attributes. Training it here with the
  default `ginconv` would ignore the edges and reproduce the baseline under a new name.

| axis | ablated values |
|---|---|
| `prots.features.node_feats` | onehot |
| `prots.features.edge_feats` | distance *(measured in Phase 6, with the transformer)* |
| `drugs.node_feats` | onehot, rich |
| `parse_dataset.filtering` | all, balanced |
| `parse_dataset.sampling` | under |
| `prots.structs.method` | plddt |

`sampling: over` is excluded on purpose: it fabricates negatives by relabelling other
targets' rows. `structs.method: bsite`/`template` need a template library the downloaded
datasets do not ship.

---

## Phase 8 — Collect

```bash
python -m rindti.utils.results --logdir tb_logs --output results.csv --summary true
```

Use the default `last` reduction for the reported numbers. Since `trainer.test` now loads
`ckpt_path="best"`, the test metrics in the final event file are *already* the
best-checkpoint numbers; taking `best_max` over a test series on top of that would be a
peeked maximum. `--reduction best_max` is for reading validation curves, not for the table.

`results.py` keeps only the hyperparameter columns that vary across the tree, so an
ablation comes out grouped by the axis automatically:

```
hp:model,feat_method hp:model,prot,node,module     mean      std  n
              concat                   ginconv 0.416667 0.117851  2
          element_l1                   gatconv 0.750000 0.353553  2
```

| Output | Feeds |
|---|---|
| `results.csv` | every results table |
| `dataset_stats.tsv` | the supplementary dataset table |
| `index.tsv` | reproducibility appendix: hash → config |
| `baselines.txt` | the prior-only floor |

---

## Monitoring

```bash
condor_q -nobatch                 # your queue
condor_q -better-analyze <id>     # why is it idle?
condor_tail -f <id>               # follow stdout
condor_ssh_to_job <id>            # shell inside the running container
condor_rm <id>
```

Logs land in `$(root)/runlogs/<job>.<cluster>.<proc>.{out,err}`.

## Failure playbook

| Symptom | Cause | Fix |
|---|---|---|
| Job idle forever | `GPUs_Capability >= 7.5` excludes the P100 and V100 nodes; the qualifying ones are busy | `condor_q -better-analyze`. Wait. Do **not** relax the requirement — the cu129 wheels have no sm_60/sm_70 kernels and the job would die at the first launch |
| `manifest unknown` | GHCR package went private | Make it public again in the package settings; execute nodes pull anonymously |
| `0 job(s) submitted` | Queue file empty, or `-a 'runfile=...'` did not take | `condor_q -l <id> \| grep -i args` to see which file was read. If the override is ignored, edit the `runfile` default in the `.sub` directly |
| Sweep job exits non-zero | One or more snakemake runs failed | It names the failing `logs/logN.txt`. Fix, re-run — snakemake skips what already exists |
| Two runs overwrote each other | Two jobs shared an `exp_name` and a dataset | Unique `exp_name` per job; see [runs/README.md](runs/README.md) |
| Training dies immediately on a `reg` dataset | `model.module` left at `class` | `--set model.module=reg`; `check_task_matches` is what caught it |
| Root-owned `.snakemake/`, `data/`, `tb_logs/` | A **local** `docker run` (HTCondor passes `--user`, a hand test does not) | Use a throwaway copy of the repo for local image tests |
| `git pull` hangs or fails with `gnutls_handshake()` | The clone uses the HTTPS remote, which the submit nodes cannot reach | `git remote set-url origin git@github.com:ilsenatorov/rindti.git` |
| `Disk quota exceeded` writing into **some** directories while others are fine and `df` shows terabytes free | Not a quota. BeeGFS stripes each directory over storage targets; the targets behind that directory are full | Probe with `echo x > <dir>/.p` per directory. A **newly created** directory gets fresh targets, so re-clone into a new path and swap it in. Raise the target exhaustion with the admins — it will recur as `datasets/` grows |
| A download job dies in `uv` before fetching anything | A dependency with no wheel for the image's Python is being source-built | Check the `.err` for `CalledProcessError`. `get_datasets.py` is pinned to `<3.12` for exactly this reason; see its PEP 723 header |

To re-run one line of a queue file, put that line in a file of its own and submit it —
there is no need to resubmit the rest.

## Two things that will bite

- **Never add `environment = ...` to a submit file.** The cluster's
  `JOB_TRANSFORM_AddHomeEnv` injects `HOME` only when `environment` is unset or empty, and
  `+WantGPUHomeMounted` depends on it. Per-job variables go in [`hpc/job.sh`](job.sh).
- **The image carries the environment, not the code.** A code change needs `git pull` on
  the cluster, not a rebuild.
