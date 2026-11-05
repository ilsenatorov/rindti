# Paper-table configs

The dataset builds behind the published tables that the six one-factor axes in
`../ablation/` do not cover: extra splits for an existing arm, and the three
preprocessing rows that need a setting the ablation configs deliberately leave out.

Every file here names the table it feeds in its first line. They were run sequentially,
not concurrently — snakemake's lock lives in one `.snakemake/locks/` directory shared by
the whole checkout, so two invocations against one checkout collide regardless of which
dataset they build.

| Group | Files | What it adds |
|---|---|---|
| `edge_cluster_*` | 3 | The `edge_feats: distance` dataset on the two cluster splits. Only `model.prot.node.module: transformer` consumes edge attributes, so pair it with the transformer arm. |
| `esm_cluster_*` | 3 | ESM-1b vectors on the two cluster splits, extending the random-split `*_esm.yaml` runs. |
| `reg_splits_*` | 3 | Regression (`task: reg`, `filtering: all`, `log: true`) on the three reported splits. Note `filtering: all`: regression cannot use `posneg`, so these are **not** the same interaction set as the classification runs. |
| `sampling_over_*` | 3 | `sampling: over`, excluded from `../ablation/sampling.yaml` because it fabricates negatives by relabelling other targets' rows. |
| `structs_template_{davis,kiba}` | 2 | `template` and `bsite` region selection. Needs the template library — see `resources/templates/`. Kinase panels only; BindingDB_Kd spans many families. |
| `threshold_*` | 3 | Binarisation thresholds either side of each dataset's default, as a sensitivity check. |

`split_index.py` in the paper repo's `rindti_results/hpc/` reads this directory by name
(`config/snakemake/extra/*.yaml`) to map each built split file back to the config that
produced it.

## A caveat on reproducing these

The split filenames these configs produce changed after the campaign ran: `split_data`
and `parse_dataset` now carry the seed in their names, so a fresh build writes
`prncn_15995058.tsv` where the archived `baselines.txt` and `split_index.tsv` record
`prncn_adb5b2e2.tsv`. The seed was 42 throughout, so the **contents** are unchanged — only
the name. `prepare_all` names, including the one Supplementary File 2 cites, already
hashed the seed and are unaffected.
