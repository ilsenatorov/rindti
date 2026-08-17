# RINDTI

[![CI](https://github.com/ilsenatorov/rindti/actions/workflows/ci.yaml/badge.svg)](https://github.com/ilsenatorov/rindti/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/ilsenatorov/rindti/branch/master/graph/badge.svg?token=KWEX1R7FVS)](https://codecov.io/gh/ilsenatorov/rindti)
[![Documentation](https://readthedocs.org/projects/rindti/badge/?version=latest)](https://rindti.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Drug–target interaction (DTI) prediction from protein **residue interaction networks**.

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

Two pipeline options need external tools that are **not** pip-installable:

| Feature | Requirement | How |
|---|---|---|
| `prots.structs.method` = `bsite` / `template` / `plddt` | PyMOL, psico, TMalign | Provided by `workflow/envs/pymol.yaml`; run snakemake with `--software-deployment-method conda` |
| `prots.features.method` = `rinerator` | `rinerator` on `$PATH` | Install separately |

The default config (`structs.method: whole`, `features.method: distance`) needs neither.

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

Structures must end up **flat** in `resources/structures/` — the pipeline fails
with an explicit error if it finds them nested in a subdirectory.

## Testing

```bash
pytest -m "not gpu and not snakemake"   # fast unit tests
pytest -m "not gpu"                     # includes full pipeline integration tests
```

## Documentation

<https://rindti.readthedocs.io>

## Citation

If you use RINDTI, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## Contributing

See the [contributing guide](CONTRIBUTE.md).

## License

MIT — see [LICENSE](LICENSE).
