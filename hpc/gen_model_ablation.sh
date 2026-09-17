#!/usr/bin/env bash
# Emit the model-ablation queue file for hpc/train.sub.
#
#   ./hpc/gen_model_ablation.sh <davis_random.pkl> <davis_cluster_target.pkl> <davis_edgedist.pkl> \
#       > hpc/runs/train_model_ablation.txt
#
# The three pickle paths come from hpc/runs/train_main.txt or, more directly:
#   python workflow/scripts/dataset_index.py 'datasets/davis/results/prepare_all/*.pkl'
#
# One-factor-at-a-time around config/dti/base.yaml, on TWO splits. Running both is the
# point: an architectural choice that helps under a random split and evaporates under a
# cold one is the result worth reporting, and one that only ever gets measured on a
# random split is the reason this codebase needed a changelog.
#
# The baseline arm of every axis is NOT emitted - it is the corresponding job from
# hpc/runs/train_main.txt, which already ran it at 5 seeds. Report n alongside mean/std;
# `results.py --summary true` emits that column.
#
# WHY exp_name CARRIES THE AXIS: rindti.cli only appends a describe_variant tag to the
# log directory when the CONFIG FILE holds lists. Overrides passed with --set produce a
# single variant, so no tag. Two concurrent jobs sharing an exp_name would then both call
# next_version() on the same directory, pick the same version_N, and - because the seed
# list is a deterministic function of `seed` and `runs` - write into identical leaf
# directories, silently overwriting each other. Unique exp_name per job is what prevents
# that. The ablation table itself does not depend on the name: results.py recovers the
# axis from the hparams.yaml Lightning writes beside each run.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <random.pkl> <cluster_target.pkl> <edge_distance.pkl>" >&2
    exit 2
fi

RANDOM_PKL="$1"
COLD_PKL="$2"
EDGE_PKL="$3"
CONFIG="config/dti/base.yaml"
RUNS="--set runs=3" # ablations get 3 seeds; headline results get 5

emit() { # <pickle> <split-tag> <axis-tag> <value> <override>
    echo "$CONFIG, $1, davis_$2_$3=$4, $RUNS $5"
}

for pair in "$RANDOM_PKL:random" "$COLD_PKL:coldtarget"; do
    pkl="${pair%:*}"
    tag="${pair##*:}"

    # How the two tower embeddings are merged. concat (the baseline) doubles the joint
    # embedding; the element-wise merges need both towers to share hidden_dim.
    for v in element_l1 mult; do
        emit "$pkl" "$tag" feat "$v" "--set model.feat_method=$v"
    done

    # Protein-side convolution. transformer is handled separately below: it is the only
    # module that consumes continuous edge attributes, so on this contact graph it would
    # measure nothing it was chosen for.
    for v in gatconv chebconv filmconv; do
        emit "$pkl" "$tag" node "$v" "--set model.prot.node.module=$v"
    done

    # Protein-side pooling. All three are sparse since DiffPool was removed.
    for v in attention set2set; do
        emit "$pkl" "$tag" pool "$v" "--set model.prot.pool.module=$v"
    done
done

# The edge-attribute arm, on the edge_feats: distance dataset from
# config/snakemake/ablation/edgefeat.yaml. ginconv here is the matched control: it ignores
# edges, so the transformer-vs-ginconv gap on the SAME dataset is the effect of actually
# consuming the distances, rather than a transformer-vs-gin architecture difference
# confounded with a change of input.
#
# Random split only: edgefeat.yaml builds the random split, so there is no cold-target
# counterpart to pair this against without another prepare run.
emit "$EDGE_PKL" random_edge node transformer "--set model.prot.node.module=transformer"
emit "$EDGE_PKL" random_edge node ginconv "--set model.prot.node.module=ginconv"
