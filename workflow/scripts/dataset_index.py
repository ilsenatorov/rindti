"""Map built dataset pickles back to the configs that produced them.

``prepare_all`` names its output by an MD5 of the whole snakemake config, so a
directory of built datasets is a pile of indistinguishable hashes::

    datasets/davis/results/prepare_all/dlnnpprwl_1f2c9a04.pkl
    datasets/davis/results/prepare_all/dlnnpprwl_7b0e3d15.pkl

Training needs the path of a *particular* one ("Davis, cold-target split"), and with a
split sweep plus a pipeline ablation there are a couple of dozen to tell apart. The
pickle carries the full config it was built from (``prepare_all.py`` stores
``snakemake.config`` under the ``config`` key - the same thing
``rindti.cli.check_task_matches`` reads), so the mapping needs no hashing, just a read::

    python workflow/scripts/dataset_index.py 'datasets/*/results/prepare_all/*.pkl'
    python workflow/scripts/dataset_index.py 'datasets/*/*/prepare_all/*.pkl' --table index.tsv

and, to generate hpc/train.sub queue lines directly rather than by hand::

    python workflow/scripts/dataset_index.py 'datasets/davis/results/prepare_all/*.pkl' \
        --emit-runs config/dti/base.yaml --runs 5 --prefix davis

This is deliberately *not* part of ``dataset_stats.py``. That one unpickles every graph
to measure node and edge counts and exits non-zero on a leaking or degenerate split,
which is what you want from a QA gate and exactly what you do not want from a lookup
table you run in a shell substitution.
"""

import argparse
import glob
import os
import pickle
import sys

# The config keys that distinguish one built dataset from another, as
# (dotted path, short column name). Everything else in the config is either constant
# across a campaign or a location rather than a description.
AXES = [
    ("source", "source"),
    ("split_data.method", "split"),
    ("parse_dataset.task", "task"),
    ("parse_dataset.filtering", "filtering"),
    ("parse_dataset.sampling", "sampling"),
    ("parse_dataset.unit", "unit"),
    ("prots.structs.method", "structs"),
    ("prots.features.method", "protfeat"),
    ("prots.features.node_feats", "prot_node"),
    ("prots.features.edge_feats", "prot_edge"),
    ("drugs.node_feats", "drug_node"),
]


def _dig(config: dict, path: str):
    """Value at a dotted path, or None if any level is missing."""
    node = config
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _dataset_name(source: str) -> str:
    """``datasets/davis/resources`` -> ``davis``."""
    if not source:
        return "unknown"
    return os.path.basename(os.path.dirname(source.rstrip("/"))) or "unknown"


def describe(path: str) -> dict:
    """The distinguishing config values for one prepare_all pickle.

    Only the ``config`` key is needed, but pickle has no way to read one entry of a
    dict without materialising the rest, so this does pay for loading the graphs.
    """
    with open(path, "rb") as handle:
        config = pickle.load(handle)["config"]
    row = {"path": path}
    row.update({column: _dig(config, dotted) for dotted, column in AXES})
    row["dataset"] = _dataset_name(row["source"])
    return row


def varying(rows: list[dict]) -> list[str]:
    """Columns that actually differ across the pickles in hand.

    A table of twenty-six datasets is only readable if it shows the handful of axes
    the campaign varied, not all eleven. ``dataset`` and ``split`` are always kept:
    they are what a run is named after, even when only one value is present.
    """
    always = ["dataset", "split"]
    others = [
        column
        for _, column in AXES
        if column not in (*always, "source") and len({str(r.get(column)) for r in rows}) > 1
    ]
    return always + others


def exp_name(row: dict, columns: list[str], prefix: str = None) -> str:
    """A run name that says what this dataset is, for ``datamodule.exp_name``.

    ``davis_random``, or ``davis_random_drug_node=rich`` when the pickle also moves a
    pipeline-ablation axis. No commas and no spaces: the name goes into a column of a
    comma-separated condor queue file.
    """
    base = prefix or row["dataset"]
    parts = [base, str(row.get("split"))]
    parts += [f"{c}={row[c]}" for c in columns if c not in ("dataset", "split") and row.get(c) is not None]
    return "_".join(p for p in parts if p and p != "None").replace(" ", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataset_index.py",
        description="Map prepare_all pickles back to the configs that built them.",
        epilog="Example: dataset_index.py 'datasets/*/results/prepare_all/*.pkl' --table index.tsv",
    )
    # nargs="+" so a shell glob works, but each argument is also globbed here: the
    # useful invocation quotes the pattern so it survives into --emit-runs output.
    parser.add_argument("patterns", nargs="+", help="prepare_all/*.pkl paths or globs")
    parser.add_argument("--table", help="also write the index as TSV here")
    parser.add_argument(
        "--emit-runs",
        metavar="CONFIG",
        help="print hpc/train.sub queue lines using this training config",
    )
    parser.add_argument("--runs", type=int, default=5, help="seeds per job, for --emit-runs")
    parser.add_argument("--extra", default="", help="further --set overrides appended to each emitted line")
    parser.add_argument("--prefix", help="override the dataset part of the generated exp_name")
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help=(
            "keep only pickles whose COLUMN equals VALUE; repeatable. "
            "The pipeline ablation needs this: after the split sweep, a dataset "
            "directory holds both the five split variants and the ablation variants, "
            "and only the random-split ones are the ablation."
        ),
    )
    args = parser.parse_args()

    # argparse treats any token containing a space as a positional, so an unsplit
    # "--where a=b --where c=d" (a shell variable expanded under zsh, which does not
    # word-split) silently lands here as a glob pattern, matches nothing, and leaves the
    # filter unapplied - returning every dataset instead of erroring. Catch it.
    for pattern in args.patterns:
        if pattern.startswith("-"):
            raise SystemExit(
                f"{pattern!r} was parsed as a path, not an option. A token containing a "
                "space is always positional to argparse; pass each flag as its own "
                "argument rather than expanding one shell variable holding all of them."
            )

    paths = sorted(
        {p for pattern in args.patterns for p in glob.glob(pattern)} | {p for p in args.patterns if os.path.isfile(p)}
    )
    if not paths:
        raise SystemExit(f"No pickles matched {args.patterns}")

    rows = [describe(p) for p in paths]

    for clause in args.where:
        column, _, value = clause.partition("=")
        if not _:
            raise SystemExit(f"--where {clause!r} is not of the form COLUMN=VALUE")
        known = {c for _, c in AXES} | {"dataset"}
        if column not in known:
            raise SystemExit(f"--where: no such column {column!r}; known columns are {sorted(known)}")
        rows = [r for r in rows if str(r.get(column)) == value]
    if not rows:
        raise SystemExit(f"No pickles left after {args.where}")

    columns = varying(rows)

    if args.emit_runs:
        if "," in args.extra:
            raise SystemExit(
                "--extra must not contain a comma: train.sub's queue file is comma-separated, "
                "so a comma would be read as a new column."
            )
        for row in rows:
            extra = f"--set runs={args.runs}"
            if args.extra:
                extra += f" {args.extra}"
            print(f"{args.emit_runs}, {row['path']}, {exp_name(row, columns, args.prefix)}, {extra}")
        return

    header = ["path", *columns]
    widths = [max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in header]
    for line in (header, *(["-" * w for w in widths],), *([str(r.get(h, "")) for h in header] for r in rows)):
        print("  ".join(str(cell).ljust(w) for cell, w in zip(line, widths, strict=True)))

    if args.table:
        with open(args.table, "w") as handle:
            handle.write("\t".join(header) + "\n")
            for row in rows:
                handle.write("\t".join(str(row.get(h, "")) for h in header) + "\n")
        print(f"\nWrote {len(rows)} rows to {args.table}", file=sys.stderr)


if __name__ == "__main__":
    main()
