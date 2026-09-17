# One-factor-at-a-time pipeline ablation

Each file here is `config/snakemake/davis.yaml` with **one** axis turned into a list,
including that axis's baseline value. Run them with `hpc/sweep.sub` (or
`run_snakemake.py ... --conda false`); snakemake's caching means the shared baseline arm
is built once and reused by every other file.

This is the shape `config/snakemake/ablation.yaml` recommends in its own header comment
but does not implement: that file takes the full cross product of every axis at once
(216 datasets as shipped, 864 with `structs.method` enabled), which is not the experiment
anyone reports. It stays as the escape hatch for when the interaction between two axes is
actually the question.

| File | Axis | Values (baseline first) |
|---|---|---|
| `protfeat.yaml` | `prots.features.node_feats` | label, onehot |
| `edgefeat.yaml` | `prots.features.edge_feats` | none, distance |
| `drugfeat.yaml` | `drugs.node_feats` | label, onehot, rich |
| `filtering.yaml` | `parse_dataset.filtering` | posneg, all, balanced |
| `sampling.yaml` | `parse_dataset.sampling` | none, under |
| `structs.yaml` | `prots.structs.method` | whole, plddt |

`split_data.method` is deliberately absent: it is the main benchmark
(`benchmark_splits_*.yaml`), not an ablation axis, and every dataset here uses the
`random` split so that the ablation is read against the `davis_random` baseline.

Two exclusions, both deliberate:

- **`parse_dataset.sampling: over`** fabricates negatives by relabelling other targets'
  rows. See the warning at the top of `workflow/scripts/parse_dataset.py`.
- **`prots.structs.method: bsite` and `template`** need a template library in
  `<source>/templates`, which the downloaded datasets do not ship.
