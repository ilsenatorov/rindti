# RINDTI

[![CI](https://github.com/ilsenatorov/rindti/actions/workflows/ci.yaml/badge.svg)](https://github.com/ilsenatorov/rindti/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/ilsenatorov/rindti/branch/master/graph/badge.svg?token=KWEX1R7FVS)](https://codecov.io/gh/ilsenatorov/rindti)
[![Documentation](https://readthedocs.org/projects/rindti/badge/?version=master)](https://rindti.readthedocs.io/en/master/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Drug–target interaction (DTI) prediction from protein **structure contact graphs**.

Proteins are represented as graphs whose nodes are amino-acid residues and whose edges
encode spatial contacts derived from a 3D structure; drugs are represented as molecular
graphs. Both are embedded with graph neural networks, and the two embeddings are merged
into a joint representation that an MLP scores for interaction.

The repository covers the whole path from raw inputs to a trained model:

```
structures (PDB)  ─┐
                   ├─► snakemake pipeline ─► graph dataset ─► GNN training ─► metrics
interactions (TSV)─┘
```

## Installation

Requires Python ≥ 3.11.

```bash
git clone https://github.com/ilsenatorov/rindti
cd rindti
uv pip install -e ".[dev,workflow]"
```

On a CPU-only machine, add `--torch-backend=cpu` to get much smaller wheels.

One pipeline option needs an external tool that is **not** pip-installable:

| Feature | Requirement | How |
|---|---|---|
| `prots.structs.method` = `bsite` / `template` / `plddt` | PyMOL, psico, TMalign | Provided by `workflow/envs/pymol.yaml`; run snakemake with `--software-deployment-method conda` |

The default config (`structs.method: whole`, `features.method: distance`) does not need it.

## Quick start

The repository ships a small test dataset, so you can run the whole thing in a minute:

```bash
# 1. Build the graph dataset
snakemake -s workflow/Snakefile -j4 \
    --software-deployment-method conda \
    --configfile config/snakemake/test.yaml

# 2. Train
rindti-train config/test/default_dti.yaml

# 3. Inspect
tensorboard --logdir=tb_logs
```

## Running the benchmarks

`config/dti/base.yaml` holds shared training settings; override per run rather than
copying the file:

```bash
rindti-train config/dti/base.yaml \
    --set datamodule.filename=datasets/davis/results/prepare_all/<hash>.pkl \
    --set datamodule.exp_name=davis_random
```

Unknown keys are rejected, so a typo fails immediately instead of being ignored.

To build every split variant of a dataset, use the sweep config:

```bash
python run_snakemake.py config/snakemake/benchmark_splits.yaml --threads 8
```

Then turn the TensorBoard logs into a table:

```bash
python -m rindti.utils.results --logdir tb_logs --output results.csv --summary true
```

## Ablations

Any list in a config is expanded into a sweep, on both sides of the pipeline.

**Dataset construction** — `config/snakemake/ablation.yaml` covers filtering,
sampling, splitting, node/edge features and structure preprocessing. Each
combination builds a full dataset, so mind the cross product (216 as shipped):

```bash
python run_snakemake.py config/snakemake/ablation.yaml --threads 8
```

**Model** — `config/dti/ablation.yaml` covers the merge method, convolution and
pooling. One command runs every combination over `runs` seeds:

```bash
rindti-train config/dti/ablation.yaml \
    --set datamodule.filename=datasets/davis/results/prepare_all/<hash>.pkl
```

Each configuration logs to its own directory named by what it changed
(`node.module=gatconv-pool.module=diffpool`), and the collector reads the
`hparams.yaml` Lightning writes beside each run, keeping the hyperparameters that
vary as columns:

```
hp:model,feat_method hp:model,prot,node,module     mean      std  n
              concat                   ginconv 0.416667 0.117851  2
          element_l1                   gatconv 0.750000 0.353553  2
```

Both files take the full cross product. For a one-factor-at-a-time ablation —
usually what you want to report — vary one list at a time and re-run per axis.

### Binarization thresholds

`parse_dataset.threshold` is interpreted on the scale given by `parse_dataset.unit`:

| Dataset | unit | threshold | meaning |
|---|---|---|---|
| Davis, GLASS, BindingDB | `nM` | 100 | Kd/Ki ≤ 100 nM is a positive (pKd ≥ 7) |
| KIBA | `score` | 12.1 | KIBA score ≥ 12.1 is a positive |

The direction differs: for `nM` a lower value is a stronger interaction, for `score`
a higher one is. Getting this wrong yields a single-class dataset, which the
pipeline now rejects rather than silently emitting an empty file.

## Using your own data

To use your own data, point `source` in a config at a directory laid out as:

```
resources/
├── structures/*.pdb          # one structure per target, flat (not nested)
├── tables/inter.tsv          # Drug_ID, Target_ID, Y
├── tables/lig.tsv            # Drug_ID, Drug (SMILES)
├── tables/prot.tsv           # Target_ID, Target (sequence)
└── templates/*.pdb           # only for structs.method = bsite/template
```

`workflow/scripts/get_datasets.py` downloads and lays out Davis, KIBA, BindingDB
(via [PyTDC](https://tdcommons.ai)) and GLASS for you, pulling structures from
AlphaFold DB. PyTDC pins `numpy<2` and so cannot share an environment with the
training stack; the script declares its own dependencies inline (PEP 723), so run
it in an isolated environment:

```bash
uv run workflow/scripts/get_datasets.py davis --min_num_aa 100 --download_structures true
```

Structures must end up **flat** in `resources/structures/`, named by `Target_ID` —
the pipeline fails with an explicit error if it finds them nested in a subdirectory.

If the structures are named differently from the `Target_ID`s in your tables (Davis,
for instance, uses gene names in its tables and UniProt accessions for its AlphaFold
models), reconcile them by sequence:

```bash
python workflow/scripts/link_structures.py datasets/davis/resources            # dry run
python workflow/scripts/link_structures.py datasets/davis/resources --apply true
```

It writes `structure_mapping.tsv` so the rename is reversible, and reports anything
it could not match confidently instead of guessing.

## Testing

```bash
pytest -m "not gpu and not snakemake"   # fast unit tests
pytest -m "not gpu"                     # includes full pipeline integration tests
```

## Documentation

<https://rindti.readthedocs.io/en/master/>

## Citation

If you use RINDTI, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## Contributing

See the [contributing guide](CONTRIBUTE.md).

## License

MIT — see [LICENSE](LICENSE).
