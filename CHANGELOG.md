# CHANGELOG

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

## [v1.5.0]

- Improved documentation building
- Added a page in documentation on quick start
- Added baseline models that only use labels and tests for its features

## [v1.4.0]

- Using codecov and pre-commit-ci now
- Removed interrogate from pre-commit
- Split github workflows into many smaller ones
