#!/usr/bin/env bash
# Push a job to the HTCondor pool, reproducibly.
#
#   ./hpc/submit.sh smoke
#   ./hpc/submit.sh train hpc/runs/train_main.txt
#   ./hpc/submit.sh --dry-run sweep hpc/runs/prepare_sweeps.txt
#
# Runs the same five steps every time, in order, and refuses to submit if one fails:
#
#   1. the local tree is clean and HEAD is pushed        (skip: --allow-dirty)
#   2. the cluster clone is fast-forwarded to that HEAD  (skip: --no-sync)
#   3. the queue file parses under the rules in hpc/runs/README.md
#   4. condor_submit runs from the repo root on the cluster
#   5. the submission is recorded in $root/runlogs/submissions.tsv, with a
#      snapshot of the exact queue file next to it
#
# Step 5 is why this exists rather than a remembered condor_submit line: a cluster id on
# its own does not say which commit, which queue file, or which contents produced it, and
# train queue files are regenerated on the cluster rather than committed.
#
# Works from a local checkout (drives the submit node over ssh) or from the clone on the
# submit node itself (runs condor_submit directly); it picks by looking for condor_submit
# on PATH.
set -euo pipefail

usage() {
    cat >&2 <<USAGE
usage: $0 [options] <job> [runfile]

jobs:
  smoke                 one GPU job, proves the image and mounts work (no runfile)
  download              CPU, TDC tables + AlphaFold structures
  prepare               CPU, snakemake, one dataset per queued config
  sweep                 CPU, run_snakemake.py, one list-valued config per queued line
  train                 GPU, rindti-train, one job per queued line

runfile                 queue file, repo-relative (default: the one in hpc/<job>.sub)

options:
  --host HOST           submit node (default: \$RINDTI_HPC_HOST or conduit)
  --dry-run             validate and expand with condor_submit -dry-run, submit nothing
  --no-sync             do not git pull on the cluster (it must already match HEAD)
  --allow-dirty         skip the clean-and-pushed and matching-commit checks
  -h, --help            this
USAGE
    exit 2
}

HOST="${RINDTI_HPC_HOST:-conduit}"
DRY_RUN=false
SYNC=true
ALLOW_DIRTY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)        HOST="${2:?--host needs a value}"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --no-sync)     SYNC=false; shift ;;
        --allow-dirty) ALLOW_DIRTY=true; shift ;;
        -h|--help)     usage ;;
        --)            shift; break ;;
        -*)            echo "unknown option: $1" >&2; usage ;;
        *)             break ;;
    esac
done

[[ $# -ge 1 ]] || usage
JOB="$1"; shift
RUNFILE="${1:-}"

case "$JOB" in
    smoke|download|prepare|sweep|train) ;;
    *) echo "unknown job: $JOB" >&2; usage ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
SUB="hpc/${JOB}.sub"
[[ -f "$SUB" ]] || { echo "no such submit file: $SUB" >&2; exit 1; }

# The scratch tree is declared once, in hpc/common.sub. Parse it rather than repeating it
# here, so relocating the tree stays a one-line edit.
ROOT="$(sed -n 's/^root[[:space:]]*=[[:space:]]*//p' hpc/common.sub | head -1)"
[[ -n "$ROOT" ]] || { echo "could not read 'root = ...' from hpc/common.sub" >&2; exit 1; }
CLUSTER_REPO="$ROOT/rindti"

# The default queue file comes from the .sub itself, for the same reason.
if [[ -z "$RUNFILE" && "$JOB" != "smoke" ]]; then
    RUNFILE="$(sed -n 's/^runfile[[:space:]]*=[[:space:]]*//p' "$SUB" | head -1)"
    RUNFILE="${RUNFILE#\$(root)/rindti/}"
    echo "==> no runfile given, using the default from $SUB: $RUNFILE"
fi
if [[ "$JOB" == "smoke" && -n "$RUNFILE" ]]; then
    echo "smoke.sub queues a single job and reads no runfile; drop the argument" >&2
    exit 1
fi

# ---------------------------------------------------------------- where this runs

if command -v condor_submit >/dev/null 2>&1; then
    ON_SUBMIT_NODE=true
    echo "==> condor_submit found locally; running on the submit node"
else
    ON_SUBMIT_NODE=false
    echo "==> driving $HOST over ssh"
fi

# Run a command in the cluster clone, wherever that is. Stdin is closed: ssh would
# otherwise swallow whatever the caller is reading from.
remote() {
    if $ON_SUBMIT_NODE; then
        bash -c "cd $(printf '%q' "$CLUSTER_REPO") && $1" </dev/null
    else
        ssh -n -o BatchMode=yes "$HOST" "cd $(printf '%q' "$CLUSTER_REPO") && $1"
    fi
}

# Same, but feeding the far-side bash a script on stdin (`bash -s` heredocs below).
remote_script() {
    if $ON_SUBMIT_NODE; then
        bash -c "cd $(printf '%q' "$CLUSTER_REPO") && $1"
    else
        ssh -o BatchMode=yes "$HOST" "cd $(printf '%q' "$CLUSTER_REPO") && $1"
    fi
}

# ------------------------------------------------------------------ 1. local is pushed

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
HEAD_SHA="$(git rev-parse HEAD)"

if $ALLOW_DIRTY; then
    echo "==> --allow-dirty: not checking the local tree"
elif $ON_SUBMIT_NODE && [[ "$REPO" == "$CLUSTER_REPO" ]]; then
    echo "==> running in the cluster clone itself; nothing to push"
else
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "ERROR: local tree is dirty. The cluster runs what is pushed, not what is" >&2
        echo "       in your editor - commit and push, or re-run with --allow-dirty." >&2
        git status --short >&2
        exit 1
    fi
    if ! git rev-parse --verify --quiet "origin/$BRANCH" >/dev/null; then
        echo "ERROR: branch '$BRANCH' has no origin counterpart; push it first." >&2
        exit 1
    fi
    if [[ "$(git rev-parse "origin/$BRANCH")" != "$HEAD_SHA" ]]; then
        echo "ERROR: HEAD is not pushed to origin/$BRANCH; run: git push origin $BRANCH" >&2
        exit 1
    fi
    echo "==> local tree clean, $BRANCH at ${HEAD_SHA:0:7} is pushed"
fi

# A stale image is the quiet failure: the repo is bind-mounted, so a code change needs
# only a pull, but a dependency change needs ./hpc/build.sh. build.sh records the commit
# it built from in hpc/.image-tag.
if [[ -f hpc/.image-tag ]]; then
    IMAGE_SHA="$(cat hpc/.image-tag)"
    if git rev-parse --verify --quiet "$IMAGE_SHA" >/dev/null; then
        CHANGED="$(git diff --name-only "$IMAGE_SHA..HEAD" -- pyproject.toml hpc/Dockerfile)"
        if [[ -n "$CHANGED" ]]; then
            echo "WARNING: changed since the image was built (${IMAGE_SHA:0:7}):" >&2
            echo "$CHANGED" | sed 's/^/           /' >&2
            echo "         jobs run the old environment until you ./hpc/build.sh" >&2
        fi
    fi
fi

# ------------------------------------------------------------------- 2. sync the clone

if $SYNC; then
    echo "==> pulling the cluster clone"
    remote "git pull --ff-only"
else
    echo "==> --no-sync: leaving the cluster clone as it is"
fi

CLUSTER_SHA="$(remote "git rev-parse HEAD" | tr -d '\r')"
if [[ "$CLUSTER_SHA" != "$HEAD_SHA" ]] && ! $ALLOW_DIRTY; then
    echo "ERROR: cluster clone is at ${CLUSTER_SHA:0:7}, local HEAD is ${HEAD_SHA:0:7}." >&2
    echo "       The jobs would run different code than you are looking at." >&2
    echo "       Push the branch the cluster tracks, or re-run with --allow-dirty." >&2
    exit 1
fi
echo "==> cluster clone at ${CLUSTER_SHA:0:7}"

# ------------------------------------------------------------- 3. validate the runfile

# Checked on the cluster, not locally: train queue files carry dataset hashes that only
# exist once a prepare job has run, so they are generated there and never committed.
if [[ -n "$RUNFILE" ]]; then
    echo "==> validating $RUNFILE"
    remote_script "JOB=$(printf '%q' "$JOB") RUNFILE=$(printf '%q' "$RUNFILE") bash -s" <<'VALIDATE'
set -euo pipefail
fail() { echo "ERROR: $RUNFILE: $*" >&2; exit 1; }

[[ -f "$RUNFILE" ]] || fail "no such file on the cluster (generate it there first)"
[[ -s "$RUNFILE" ]] || fail "is empty; condor_submit would report '0 job(s) submitted'"

# `queue ... from` passes every line through verbatim, so a '#' line becomes a job whose
# first argument is '#', and a blank line becomes a job with empty arguments.
if grep -nE '^[[:space:]]*(#|$)' "$RUNFILE"; then
    fail "holds comments or blank lines; queue files must be pure data"
fi

n=0
while IFS= read -r line; do
    n=$((n + 1))
    ncols=$(awk -F, '{print NF}' <<<"$line")
    case "$JOB" in
        train)
            [[ $ncols -eq 4 ]] || fail "line $n has $ncols columns, expected 4 (configfile, dataset, expname, extra). A comma inside 'extra' splits it - use one job per value instead."
            cfg=$(cut -d, -f1 <<<"$line" | xargs)
            pkl=$(cut -d, -f2 <<<"$line" | xargs)
            extra=$(cut -d, -f4 <<<"$line" | xargs)
            [[ -f "$cfg" ]] || fail "line $n: config '$cfg' does not exist"
            [[ -f "$pkl" ]] || fail "line $n: dataset '$pkl' does not exist"
            [[ -n "$extra" ]] || fail "line $n: the extra column is empty; it must carry at least --set runs=N"
            ;;
        prepare|sweep)
            [[ $ncols -eq 1 ]] || fail "line $n has $ncols columns, expected 1 configfile"
            cfg=$(xargs <<<"$line")
            [[ -f "$cfg" ]] || fail "line $n: config '$cfg' does not exist"
            ;;
        download)
            [[ $ncols -eq 1 ]] || fail "line $n has $ncols columns, expected 1 dataset name"
            ;;
    esac
done < "$RUNFILE"

if [[ "$JOB" == "train" ]]; then
    # Two jobs sharing a dataset and an exp_name pick the same version_N and overwrite
    # each other - silently, because the per-run seeds are a function of seed and runs.
    dupes=$(cut -d, -f2,3 "$RUNFILE" | tr -d ' ' | sort | uniq -d)
    [[ -z "$dupes" ]] || fail "duplicate dataset+expname pairs would overwrite each other:
$dupes"
fi

echo "    $n line(s), ok"
VALIDATE
fi

# ------------------------------------------------------------------------- 4. submit

ARGS=""
[[ -n "$RUNFILE" ]] && ARGS="-a $(printf '%q' "runfile=$RUNFILE")"

if $DRY_RUN; then
    echo "==> dry run: expanding $SUB without submitting"
    remote "condor_submit -dry-run /dev/stdout $ARGS $SUB"
    echo "==> dry run only, nothing was queued"
    exit 0
fi

echo "==> submitting $SUB"
OUT="$(remote "condor_submit $ARGS $SUB" 2>&1 | tee /dev/stderr)"
CLUSTER_ID="$(sed -n 's/.*submitted to cluster \([0-9]*\).*/\1/p' <<<"$OUT" | tail -1)"

if [[ -z "$CLUSTER_ID" ]]; then
    echo "ERROR: condor_submit did not report a cluster id; nothing was recorded." >&2
    exit 1
fi

# ------------------------------------------------------------------------- 5. record

echo "==> recording submission $CLUSTER_ID"
remote_script "CLUSTER_ID=$(printf '%q' "$CLUSTER_ID") JOB=$(printf '%q' "$JOB") RUNFILE=$(printf '%q' "${RUNFILE:-none}") SHA=$(printf '%q' "$CLUSTER_SHA") ROOT=$(printf '%q' "$ROOT") bash -s" <<'RECORD'
set -euo pipefail
mkdir -p "$ROOT/runlogs/queues"
log="$ROOT/runlogs/submissions.tsv"
[[ -f "$log" ]] || printf 'submitted_at\tcluster_id\tjob\tcommit\trunfile\tnjobs\n' > "$log"
if [[ "$RUNFILE" != "none" && -f "$RUNFILE" ]]; then
    njobs=$(grep -c '' "$RUNFILE")
    cp "$RUNFILE" "$ROOT/runlogs/queues/${CLUSTER_ID}.${JOB}.txt"
else
    njobs=1
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$CLUSTER_ID" "$JOB" "$SHA" "$RUNFILE" "$njobs" >> "$log"
RECORD

SSH_PREFIX=""
$ON_SUBMIT_NODE || SSH_PREFIX="ssh $HOST "

cat <<MSG

Cluster $CLUSTER_ID - $JOB from ${CLUSTER_SHA:0:7}${RUNFILE:+ ($RUNFILE)}

  ${SSH_PREFIX}condor_q $CLUSTER_ID -nobatch        # queue
  ${SSH_PREFIX}condor_q -better-analyze $CLUSTER_ID # why is it idle?
  ${SSH_PREFIX}condor_tail -f $CLUSTER_ID.0         # follow stdout
  ${SSH_PREFIX}condor_rm $CLUSTER_ID                # cancel

  logs:   $ROOT/runlogs/$JOB.$CLUSTER_ID.*
  record: $ROOT/runlogs/submissions.tsv
MSG
