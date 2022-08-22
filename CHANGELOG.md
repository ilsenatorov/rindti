# CHANGELOG

## [v1.6.0] - planned

- DynamicBatchLoader in dataloaders
    - Field in `config.datamodule` (`dyn_sampler`: `True`/`False`)
- `DataModule.setup()`
    - Loading single `splits`
    - Use `transforms`
- New transformations
    - Field in `config` (`config.transform`)
        - Allows also for weighting of the different graph tasks
- Change of all encoders to output tuples of `(graph_embedding, node_embedding)`
- MLP can get list of different hidden dimensions
    - option hidden_dims: `[x, y, z]` in config
        - `hidden_dims` overrides `hidden_dim` but defaults to `None`
- New metric for overlap of predicted values of positive and negative samples
- New learning rate schedulers
    - `rindti.lr_schedules.LWCA.LinearWarmupCosineAnnealing`
    - `rindti.lr_schedules.LWCAWR.LinearWarmupCosineAnnealingWarmRestarts`
    - New name in `config`: `reduce_lr` → `lr_schedule`
- MLP not outputs the prediction and the last-layer as embedding of the input (for explainability)
- Tests for SweetNet and ESM encoders based on some Lectin-Glycan interactions

## [v1.5.0]

- Improved documentation building
- Added a page in documentation on quick start
- Added baseline models that only use labels and tests for its features

## [v1.4.0]

- Using codecov and pre-commit-ci now
- Removed interrogate from pre-commit
- Split github workflows into many smaller ones
