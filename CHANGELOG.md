# CHANGELOG

## [Unreleased]

### Added

- **An experiment runbook and the tooling it needs.** `hpc/EXPERIMENTS.md` takes the
  benchmark campaign from an empty `/scratch` tree to `results.csv` in eight phases, with
  two blocking gates: Davis structure naming (`get_datasets.py` fetches AlphaFold models
  by `Target_ID` and drops targets it cannot find, so a gene-name/UniProt mismatch yields
  an *empty dataset with no error*), and `dataset_stats.py`'s exit code before any GPU
  time is spent.

- **`hpc/train.sub` takes per-job `--set` overrides.** A fourth queue column, so an
  ablation is one condor job per configuration, parallel across the pool, rather than one
  job expanding a list-valued config into 45 configurations run *serially on one GPU*.
  It also carries the seed budget (`runs=5` for headline results, `runs=3` for ablations).

  The column brings a constraint worth knowing: `rindti.cli` only appends a
  `describe_variant` tag to the log directory when the **config file** holds lists, so
  `--set` produces no tag. Two jobs sharing an `exp_name` and a dataset both call
  `next_version()` on the same directory, pick the same `version_N`, and - since the seed
  list is a deterministic function of `seed` and `runs` - overwrite each other. Queue
  files therefore put the axis in `exp_name`, and `hpc/gen_model_ablation.sh` generates
  them that way.

- **`workflow/scripts/dataset_index.py`.** `prepare_all` names its output by a hash of the
  whole config, and the documented way to find a particular dataset was to `ls` the
  directory - unworkable at the two dozen pickles this campaign builds. The pickle already
  carries the config that built it, so this reads it back, prints only the axes that vary,
  and emits `train.sub` queue lines directly. Separate from `dataset_stats.py`, which
  unpickles every graph and exits non-zero on a bad split: right for a QA gate, wrong for
  a lookup.

- **`hpc/sweep.sub`**, for configs holding lists. `prepare.sub` runs `snakemake`, which
  does not expand them; `run_snakemake.py` does, and needs `--conda false` in the image.

- **`config/snakemake/ablation/`** - six one-factor-at-a-time pipeline ablation configs,
  which is the shape `config/snakemake/ablation.yaml` recommends in its own header but
  does not implement (it takes the full 216-run cross product). Plus
  `benchmark_splits_{kiba,bindingdb_kd}.yaml` beside the renamed
  `benchmark_splits_davis.yaml`.

  `test_snakemake.py` discovered shipped configs with a flat `os.listdir`, so it validated
  nothing in a subdirectory and handed the directory itself to `read_config` as though it
  were one. It now walks the tree, which also brings the six new configs under the
  "every shipped config validates against the schema" check.

### Changed

- Every submit file reads `$(runfile)`, so the queue file is selected per phase with
  `condor_submit -a 'runfile=...'` instead of by editing the `.sub` file.

### Fixed

- **`split_groups` sent short bins entirely to train.** It allocated
  `min(len(subset), int(bin_size * frac))` per bin, so the proportions were relative to
  `bin_size` rather than to the bin in hand. The final partial bin was always biased
  toward train, and a dataset with fewer than `bin_size` entities went *entirely* to it -
  leaving val and test empty, and training then early-stopping on a validation set that
  did not exist. Allocation now deals a shuffled pattern of split labels across each bin:
  identical for a full bin, unbiased for a short one. Five entities now split 3/1/1
  instead of 5/0/0. **Cold-split datasets must be rebuilt.**

- **Cold-drug splits did not deduplicate.** `target` collapsed exact duplicate sequences
  via `dedup_sequences` while `drug` grouped on the raw `Drug_ID`, so the same molecule
  under two IDs - or written two ways - could sit on both sides of the boundary. The two
  cold splits controlled leakage to different standards, in a pipeline whose point is
  controlling leakage. `dedup_smiles` compares RDKit canonical SMILES.

- **`DTIDataset` raced between concurrent jobs.** The cache key is the directory, so
  several model configs submitted against one dataset all found the cache cold and all
  wrote into the same `processed/`; `InMemoryDataset.save` is not atomic, so a loser
  could read a half-written `.pt`. `process()` is now guarded by an `O_CREAT | O_EXCL`
  lock, with stale-lock recovery if the holder dies.

- **ESM truncation is reported.** `prot_esm.py` cut sequences to ESM-1b's 1022-residue
  context silently, while the structure arm uses the whole chain - a confound in the
  comparison the two arms exist to make. It now prints how many sequences were truncated
  and by how much. Its scratch files also moved from a relative `./esms/` - not a declared
  output, not content-hashed, and on a cluster shared between concurrent jobs - into a
  per-invocation temporary directory.

- **`distance_based.py`'s standalone CLI never ran.** It called `.to_pickle()` on a dict.
  Its contact threshold also defaulted to 5 A where the workflow schema says 7, so a
  hand-built dataset silently differed from a pipeline-built one, and `edge_feats` was
  not exposed at all.

- **`drugs.node_feats` accepted values the schema rejects.** `prepare_drugs` asserted
  against `{label, onehot, rich, glycan, glycanone, IUPAC}` while the config schema allows
  only the first four, and `DTIDataset._get_datum` carried an unreachable `IUPAC` branch.

- **`trainer.test` evaluated the wrong weights.** `rindti-train` ran
  `trainer.test(model, datamodule)` straight after `fit`, which tests the weights
  training stopped on - `early_stop.patience` (30 by default) epochs past the optimum -
  while `ModelCheckpoint` had saved the best ones and never loaded them. Every reported
  test metric was systematically pessimistic and noisier than it should have been. Now
  `ckpt_path="best"`.

- **A task/model mismatch is now rejected.** `parse_dataset.task` (snakemake) and
  `model.module` (training) live in separate config files and nothing related them, so a
  `reg` dataset trained with `model.module: class` ran
  `binary_cross_entropy_with_logits` against continuous affinities and reported AUROC on
  them without error. `rindti.cli.check_task_matches` raises instead.

- **Regression labels for `unit: nM` are now pKd, not `log10(Kd)`.** The DeepDTA ->
  GraphDTA -> DGraphDTA -> GEFA lineage reports `pKd = 9 - log10(Kd in nM)`. MSE and the
  rank-based CI are invariant to that affine flip, but `RM2` is not - its `r0^2` is a
  regression forced through the origin, so it is offset-dependent - and `RM2` exists
  specifically to be compared against those papers. **Regression datasets built with
  `log: true` and `unit: nM` must be rebuilt.**

- **Convolution stacks were linear.** Only `GINConvNet` had a non-linearity between
  layers (inside the `GINConv` MLP). `ChebConvNet` was measurably *exactly* affine - a
  three-layer stack collapsed to a single linear operator - and `GatConvNet` and
  `TransformerNet` were non-linear only through their attention softmax, so a
  `node.module` ablation compared a real GNN against a linear model. All five modules now
  apply a `PReLU` after each non-final convolution
  (`rindti.layers.base_layer.interlayer_activations`), guarded by a test.

- **`element_l1` and `element_l2` were the same function.** `_element_l2` computed
  `sqrt((d - p) ** 2 + 1e-6)`, which equals `|d - p|` to six decimal places, so a
  `feat_method` ablation ran two identical arms and reported them as separate results.
  It is now the element-wise squared difference.

- **`node.dropout` was inert.** Every conv module except `TransformerNet` swallowed it in
  `**kwargs`; in `TransformerNet` it configured *attention* dropout rather than feature
  dropout. It is now applied as feature dropout between layers in all five.

- **`run_snakemake.py` swallowed failures and could not run on the cluster.** It
  discarded each subprocess's return code with output redirected to a log file, so a
  sweep could "finish" with half its datasets missing; and it hardcoded
  `--software-deployment-method conda`, which the HPC image (MMseqs2 on `PATH`, no conda)
  cannot satisfy. It now exits non-zero listing the failed runs, and takes `--conda`.
  The temporary config also moved out of the repo into `tempfile`.

### Removed

- `workflow/envs/pymol.yaml`, `workflow/scripts/create_pymol_scripts.py` and
  `workflow/report/pymol_png.rst`, along with the `pymol_scripts` and `pymol_logs`
  output directories.

- **Dead configs.** `config/dti/glylec.yaml` (`method: pretrained`, an encoder deleted in
  v2.0.0, plus a hardcoded `/scratch/SCRATCH_SAS/roman/...` path),
  `config/dti/hparams_search.yaml` (hardcoded `/home/ilya/...` path in a pre-hash naming
  scheme, and no `model.monitor`), `config/prot/{ec,pfam}/*` and
  `config/test/default_pfam.yaml` (pretraining, whose models and datamodules were removed
  in v2.0.0). `config/dti/glass.yaml` was repaired rather than removed - the quick-start
  references it - dropping its stale hardcoded pickle path, its debug-sized hidden
  dimensions and its leftover DiffPool `ratio`/`num_heads` keys.

- **`DiffPoolNet`, replaced by two sparse poolers.** It densified the batch - an
  `(B, max_nodes, max_nodes)` adjacency - so at `batch_size: 128` its peak allocation was
  0.31 GB for 300-residue proteins, 1.85 GB at 800 and **6.2 GB at 1500**, while
  whole-protein AlphaFold graphs reach well past that. The pooling ablation was therefore
  the one axis that could not run on the same structures as every other axis.
  `pool.module` is now `mean` (unchanged default), `attention` or `set2set`, measured at
  0.17 / 0.27 / 0.37 GB respectively at 1500 residues.

  `AttentionPool` additionally exposes `attention_weights`, one scalar per residue after a
  forward pass - the readout the interpretability analysis needs, which neither mean
  pooling nor DiffPool's soft many-to-many cluster assignments provided.

  With no pooler producing auxiliary losses, `BaseModel.collect_aux_loss` and the
  `train_aux_loss` logging it fed are gone too, as is `pool.max_nodes`, which existed
  only for the dense batching.

- **`feat_method: element_l2`.** It computed `sqrt((d - p) ** 2 + 1e-6)`, which is
  `element_l1` to six decimal places, so the ablation ran one arm twice under two names.
  `element_l1` is kept as the element-wise distance merge.

### Added

- **`workflow/scripts/dataset_stats.py`** — describes a built dataset: the attrition chain
  from the raw tables through AlphaFold coverage and filtering to the graphs actually
  trained on, label balance, split sizes and graph dimensions. `--table` writes one row per
  dataset. Exits non-zero on a cold split that leaks an entity across the train/test
  boundary, or on an empty `val`/`test` split.


- `nn.LayerNorm` on the joint drug/protein embedding before the MLP head. Both poolers
  L2-normalise, so each tower emits a unit vector and the merge operators landed on very
  different scales - the joint embedding's norm was ~1.41 for `concat` and the
  element-wise differences but ~0.088 for `mult`. The `feat_method` ablation was partly
  measuring input scale.

### Changed

- **Structure parsing no longer uses PyMOL.** `prots.structs.method` values `plddt`,
  `bsite` and `template` were implemented by generating a `.pml` script per protein and
  running it in a conda environment (`workflow/envs/pymol.yaml`: pymol-open-source,
  pymol-psico, tmalign, pulling in the third-party `speleo3` channel). They are now
  `workflow/scripts/parse_structs.py`, built on [biotite](https://www.biotite-python.org),
  which installs from PyPI with the `workflow` extra. MMseqs2 is the only external tool
  the pipeline still needs, and only for `split_data.method: cluster_target`.

  Two semantics changed with the port, so **graphs built with these methods differ from
  v2.0.0**:

  - `plddt` selects whole residues whose CA B-factor passes the threshold. The PyMOL
    selection was `b > threshold` with no `br.`, i.e. atom-level, so a residue could
    reach the graph without the CA that represents it - or lose its CA and vanish while
    its neighbours stayed.
  - `bsite` and `template` measure against the best-scoring template only. The old
    script loaded every template, superimposed them all onto the best one, and then
    selected against all of them at once, so a template that matched poorly still
    contributed residues to the binding site.

  Template ranking now uses biotite's `superimpose_structural_homologs`/`tm_score`, a
  TM-align-inspired heuristic rather than the TM-align binary psico called. Scores are
  close but not identical; they are only used to rank templates against each other.

- Template PDBs are declared as rule inputs instead of being globbed at runtime, so
  changing the template library now invalidates the parsed structures.

- An empty selection fails in the parsing rule with the threshold or radius named,
  rather than writing a zero-residue PDB and failing later during graph construction.

## [v2.0.0]

Modernization release. The project was revived after an extended pause; this
release brings the whole stack to current versions and fixes a bug that made the
model unable to report any classification metric.

### Fixed

- **Metrics were never computed.** The PyTorch Lightning 1.x → 2.x migration had left
  all three `on_*_epoch_end` hooks commented out, so `train/val/test` metrics were
  accumulated every step but never computed, logged or reset. Only losses reached the
  logger, and metric state grew without bound across epochs. Accuracy, AUROC and MCC
  are now reported again.
- `configure_optimizers` built per-encoder parameter groups and then discarded them,
  so `prot_lr` / `drug_lr` were silently ignored; `weight_decay` and `momentum` were
  never passed to the optimizer at all.
- `BaseModel.__init__` returned a value, raising `TypeError` on direct instantiation.
- `RegressionModel` constructed classification metrics and immediately overwrote them.
- `_set_class_metrics` hardcoded `task="binary"` while branching on `num_classes`,
  so the multiclass path silently produced binary metrics.
- `SnakemakeHelper` silently produced an empty protein list when structures were
  nested one directory too deep; it now fails with an explanatory error.
- Positional `Series[0]` indexing in `workflow/scripts/utils.py`, removed in pandas 3.
- `get_git_hash()` crashed outside a git checkout.
- AlphaFold DB structure URLs were pinned to a model version that no longer exists;
  the download now resolves the current URL through the API.
- `config/snakemake/binding_db.yaml` pointed at a mis-capitalized dataset directory.

### Changed

- Python ≥ 3.11; torch ≥ 2.6, torch-geometric ≥ 2.7, lightning ≥ 2.5,
  torchmetrics ≥ 1.7, snakemake ≥ 9.
- Packaging moved from `setup.py` + a 500-line `conda env export` to `pyproject.toml`
  with version ranges and a committed `uv.lock`.
- Imports moved from `pytorch_lightning` to `lightning.pytorch`.
- Datasets use PyG's `save`/`load` API instead of assigning `self.data`/`self.slices`
  and calling `torch.load` without `weights_only`; the dataset config is now a JSON
  sidecar rather than a third element of the `.pt` tuple.
- Layers, encoders and losses are plain `nn.Module`s instead of `LightningModule`s.
- The workflow targets the Snakemake ≥ 8 `SnakemakeApi`; `--use-conda` is now
  `--software-deployment-method conda`.
- `workflow/schemas/config.schema.yaml` now declares `required`, `enum` and
  `additionalProperties: false`, so invalid configs fail at validation time.
- `workflow/envs/pymol.yaml` rebuilt on conda-forge (`pymol-open-source`); the old
  2021 environment no longer solved.
- `get_datasets.py` declares its dependencies inline (PEP 723) and runs isolated,
  because PyTDC pins `numpy<2`.
- Training entry point is now the `rindti-train` console script.
- CI rewritten around `uv` with a Python 3.11/3.12 matrix.
- Tests that transitively depend on the pipeline are marked `snakemake`, so
  `-m "not snakemake"` really does skip them.
- Added `LICENSE` (MIT) and `CITATION.cff`.

### Removed

- Pretraining (`pretrain.py`, `PreTrainDataset`, `PreTrainDataModule`, the
  `only_prots` workflow branch) — it imported models deleted from the package.
- `hyperparameter.py`, which targeted three Ray Tune APIs that no longer exist.
- `PNAConvNet` (referenced attributes its constructor never created),
  `SweetNetEncoder` and `PretrainedEncoder` (already disabled), and `rindti.losses`.
- The `dash/` app, scratch notebooks, committed Sphinx build output, and dead configs.
- The RINerator protein path (`parse_rinerator.py` and its workflow rule). Proteins are
  now always residue-level contact graphs built from C-alpha distances
  (`distance_based.py`) or ESM-1b embeddings; no true residue interaction network is
  computed anywhere in the pipeline.

## [v1.5.0]

- Improved documentation building
- Added a page in documentation on quick start
- Added baseline models that only use labels and tests for its features

## [v1.4.0]

- Using codecov and pre-commit-ci now
- Removed interrogate from pre-commit
- Split github workflows into many smaller ones
