# RINDTI

![testing](https://github.com/ilsenatorov/rindti/actions/workflows/test.yaml/badge.svg)

This repository aims to simplify the drug-target interaction prediction process which is based on protein residue interaction networks (RINs)

## Overview

The repository aims to go from a simple collections of inputs - structures of proteins, interactions data on drugs to a fully-function GNN model

## Installation

1. clone the repository with `git clone https://github.com/ilsenatorov/rindti`
2. change in the root directory with `cd rindti`
3. *(Optional)* install mamba with `conda install -n base -c conda-forge mamba`
4. create the conda environment with `mamba env create -f workflow/envs/main.yaml` (might take some time)
5. activate the environment with `conda activate rindti`
6. clone rinerator repository into your home dir with `git clone --branch perf https://wibi-git.helmholtz-hzi.de/ske18/rinerator/ ~/rinerator`
7. install rinerator with `pip install ~/rinerator`

## Snakemake pipeline

Snakemake pipeline (located in `workflow` directory) is responsible for parsing all the data.
Usually the result is a single pickle file containing all the necessary information for training, testing etc.
The pipeline relies on the `resources` folder which should contain `resources/structures/` folder with pdb files for acquiring protein networks.
For DTI prediction the model also needs csv files in `resources/drugs`, more info will be provided later.
The pipeline is controlled by a config file `config/snakemake.yaml` which can be modified to change the output of the pipeline

### DTI data

In order to obtain the pickle file necessary for training the DTI model please ensure the `only_proteins` entry in the yaml config file is set to `false`.
Once pipeline is run with `snakemake -j <num_cores> --use-conda`, it will generate a pickle file in the `results/prepare_all/` directory.

### Pretraining data

In order to omit the drug data, one can set the `only_proteins` to `true` in the config file.
Running the pipeline with `snakemake -j <num_cores> --use-conda` will generate a pickle file in `results/prepare_proteins/` directory.

Since snakemake sometimes struggles with calculating DAG for really large pipelines, steps of this pipeline could be ran manually

* For rinerator-based networks please run the rinerator on all the necessary pdb files, then use the `workflow/scripts/prepare_proteins.py` script to generate the necessary pickle file.
* For distance-based networks please run the `workflow/scripts/distance_based.py` script to generate the necessary pickle file.

Information on both scripts can be found through `workflow/scripts/distance_based.py --help` or `workflow/scripts/prepare_proteins.py --help`


### GNN model

The GNN model is shipped as a package, located in the `rindti` directory.

Additionally, scripts that give an example of model training are provided (`train.py` and `pretrain.py`), which can be used as plug-and-play, or as inspiration to create custom training approaches.


### Training

A model can be trained with a simple `train.py results/prepare_all/<pickled_data>.pkl`, assuming pipeline has been ran in advance.

Three types of models are currently available - class, reg and noisy nodes model.

Both class and noisy nodes require the final label to be a bool value (0 or 1), however noisy nodes introduces corruption of nodes and auxiliary loss of predicting the original labels of such nodes.

### Pretraining

In order to pretrain the model one can use the `pretrain.py` script, which has three types of self-supervised pretraining models available

[GraphLog](https://arxiv.org/pdf/2106.04113.pdf) and [Infograph](https://arxiv.org/pdf/1808.06670.pdf) can be used directly with the result of `prepare_proteins` step of the workflow (located in the `results/prepare_proteins` directory).

Pfam siamese network will require adding the information on the protein family to that dataframe in order to function.
