# RINDTI

![testing](https://github.com/ilsenatorov/rindti/actions/workflows/test.yaml/badge.svg)
![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ilsenatorov/rindti/master.svg)](https://results.pre-commit.ci/latest/github/ilsenatorov/rindti/master)

This repository aims to simplify the drug-target interaction prediction process which is based on protein residue interaction networks (RINs)

## Overview

The repository aims to go from a simple collections of inputs - structures of proteins, interactions data on drugs to a fully-function GNN model

## Installation

1. clone the repository with `git clone https://github.com/ilsenatorov/rindti`
2. change in the root directory with `cd rindti`
3. *(Optional)* install mamba with `conda install -n base -c conda-forge mamba`
4. create the conda environment with `mamba env create -f workflow/envs/main.yaml` (might take some time)
5. activate the environment with `conda activate rindti`
6. Test the installation with `pytest`

## Documentation

Check out the [documentation](https://rindti.readthedocs.io/en/master/) to get more information.

## Contributing

If you would like to contribute to the repository, please check out the [contributing guide](CONTRIBUTE.md).
