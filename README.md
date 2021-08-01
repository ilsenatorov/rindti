# RINDTI

This repository aims to simplify the drug-target interaction prediction process which is based on protein residue interaction networks (RINs)

## Overview

The repository aims to go from a simple collections of inputs - structures of proteins, interactions data on drugs to a fully-function GNN model

### Snakemake pipeline

Snakemake pipeline (located in `workflow` directory) is responsible for obtaining the data for the model, including:

* Calculating RINs of proteins
* Converting RINs to torch_geometric Data
* Converting molecular SMILES to torch_geometric Data
* Parsing and splitting the dataset

The result of the pipeline is a single pickle file, containing all the necessary for the construction of torch_geometric Dataset entry.

### GNN model

The GNN model is shipped as a package, located in the `rindti` directory.

Additionally, scripts that give an example of model training are provided (`train.py` and `pretrain_graphlog.py`), which can be used as plug-and-play, or as inspiration to create custom training approaches.

## Installation

1. clone the repository with `git clone https://github.com/ilsenatorov/rindti`
1. change in the root directory with `cd rindti`
1. unzip the gpcr data with `tar -xvf gpcr_data.tar.gz`
1. *(Optional)* install mamba with `conda install -n base -c conda-forge mamba`
1. create the conda environment with `mamba env create -f workflow/envs/main.yml` (might take some time)
1. activate the environment with `conda activate rindti`
1. clone rinerator repository into your home dir with `git clone --branch perf https://wibi-git.helmholtz-hzi.de/ske18/rinerator/ ~/rinerator`
1. install rinerator with `pip install ~/rinerator`

## Usage

### Pipeline

Once everything is installed, obtaining the dataset should be as easy as running `snakemake -j <num_cores> --use-conda`, which should calculate the RINs for all structures in the `resources/strucures` folder and parse the interaction data in `resources/drugs` folder.

A file in `results/prepare_all` folder will appear at the end of it, which is then used for the GNN training

Configuration file in `config/config.yml` can be used to change the parameters of the pipeline, such as data splits, parsing etc.

### Training

A model can be trained with a simple `train.py results/prepare_all/<pickled_data>.pkl`, assuming pipeline has been ran in advance.

### Pretraining

In order to pretrain the model using the GraphLoG approach, one can run the
