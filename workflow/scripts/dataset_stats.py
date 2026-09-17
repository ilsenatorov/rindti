"""Describe a built dataset: attrition, label balance, splits and graph sizes.

Every number the manuscript reports about a dataset comes from here, so that the
"442 kinases" in the text and the graphs actually trained on cannot drift apart. The
pipeline discards entities at five separate points - no AlphaFold model, `min_num_aa`,
`max_num_atoms`, binarization filtering, and the final join in `prepare_all` - and none
of them announced how much they removed.

    python workflow/scripts/dataset_stats.py datasets/davis/results/prepare_all/<hash>.pkl
    python workflow/scripts/dataset_stats.py <pkl> --output stats.json

The split section doubles as a sanity check, and the exit status is non-zero when either
fires. A cold-target split that leaves a target on both sides is the failure that
similarity-aware splitting exists to prevent, so `overlap` is asserted rather than
trusted. An empty val or test split is the other one: `split_data.split_groups` allocates
within bins of ten and takes `int(bin_size * frac)` from each, so a group column with
fewer than ten distinct values sends everything to train, and training then early-stops
on a validation set that does not exist.
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

# Cold splits and the ID column whose values must not cross the train/test boundary.
# `random` is absent deliberately: overlap is expected there and is not a defect.
COLD_SPLITS = {
    "target": "prot_id",
    "cluster_target": "prot_id",
    "drug": "drug_id",
    "cluster_drug": "drug_id",
}


def _sizes(frame: pd.DataFrame, key: str) -> dict:
    """Node and edge counts over one entity table, as a summary dict.

    ``edge_index`` is absent for the ESM featurisation, which is one pooled vector per
    protein rather than a graph, so edges are reported as None instead of raising.
    """
    nodes = np.array([graph["x"].size(0) for graph in frame["data"]])
    stats = {
        f"{key}_count": int(len(frame)),
        f"{key}_nodes_mean": float(nodes.mean()),
        f"{key}_nodes_median": float(np.median(nodes)),
        f"{key}_nodes_min": int(nodes.min()),
        f"{key}_nodes_max": int(nodes.max()),
    }
    if "edge_index" in frame["data"].iloc[0]:
        # Undirected graphs are stored with both directions, so halve for the count a
        # reader expects.
        edges = np.array([graph["edge_index"].size(1) / 2 for graph in frame["data"]])
        stats[f"{key}_edges_mean"] = float(edges.mean())
        stats[f"{key}_edges_median"] = float(np.median(edges))
        stats[f"{key}_degree_mean"] = float((2 * edges / nodes).mean())
    return stats


def _attrition(config: dict, final: pd.DataFrame) -> dict:
    """How many entities each stage removed, from the raw tables to the trained graphs.

    Reads the source tables and the ``parse_dataset`` output beside the pickle. Any
    stage whose file is missing is reported as None rather than aborting, so the tool
    still works on a dataset copied without its intermediates.
    """
    source = config["source"]
    tables = os.path.join(source, "tables")
    out = {}

    try:
        raw = pd.read_csv(os.path.join(tables, "inter.tsv"), sep="\t")
        out["raw_interactions"] = int(len(raw))
        out["raw_targets"] = int(raw["Target_ID"].nunique())
        out["raw_drugs"] = int(raw["Drug_ID"].nunique())
    except FileNotFoundError:
        out["raw_interactions"] = out["raw_targets"] = out["raw_drugs"] = None

    structures = os.path.join(source, "structures")
    out["targets_with_structure"] = (
        len([f for f in os.listdir(structures) if f.endswith(".pdb")]) if os.path.isdir(structures) else None
    )

    # parse_dataset's output: binarization, filtering and sampling applied, nothing joined.
    try:
        from snakemake_helper import Namer

        parsed = os.path.join(
            os.path.dirname(source.rstrip("/")),
            "results",
            "parse_dataset",
            Namer(8)(config["parse_dataset"]) + ".tsv",
        )
        out["interactions_after_filtering"] = int(len(pd.read_csv(parsed, sep="\t")))
    except (ImportError, FileNotFoundError, KeyError):
        out["interactions_after_filtering"] = None

    out["final_interactions"] = int(len(final))
    out["final_targets"] = int(final["prot_id"].nunique())
    out["final_drugs"] = int(final["drug_id"].nunique())
    return out


def _splits(final: pd.DataFrame, method: str) -> dict:
    """Split sizes, entity counts per split, and the cold-split leakage check."""
    out = {"split_method": method}
    total = len(final)
    for split in ("train", "val", "test"):
        rows = final[final["split"] == split]
        out[f"{split}_interactions"] = int(len(rows))
        out[f"{split}_fraction"] = float(len(rows) / total) if total else 0.0
        out[f"{split}_targets"] = int(rows["prot_id"].nunique())
        out[f"{split}_drugs"] = int(rows["drug_id"].nunique())

    out["empty_splits"] = [s for s in ("train", "val", "test") if out[f"{s}_interactions"] == 0]

    column = COLD_SPLITS.get(method)
    if column is not None:
        train = set(final[final["split"] == "train"][column])
        test = set(final[final["split"] == "test"][column])
        out["leakage_column"] = column
        out["leakage_overlap"] = int(len(train & test))
    return out


def _labels(final: pd.DataFrame, config: dict) -> dict:
    """Label balance for classification, or the value distribution for regression."""
    parse = config.get("parse_dataset", {})
    out = {"task": parse.get("task"), "unit": parse.get("unit")}
    labels = final["label"].to_numpy()
    if parse.get("task") == "reg":
        out.update(
            logged=parse.get("log"),
            label_mean=float(labels.mean()),
            label_std=float(labels.std()),
            label_min=float(labels.min()),
            label_max=float(labels.max()),
        )
    else:
        out.update(
            threshold=parse.get("threshold"),
            filtering=parse.get("filtering"),
            sampling=parse.get("sampling"),
            positives=int(labels.sum()),
            positive_rate=float(labels.mean()),
        )
    return out


def collect(filename: str) -> dict:
    """Every statistic for one ``prepare_all`` pickle, as a flat-ish dict."""
    with open(filename, "rb") as file:
        data = pickle.load(file)
    config = data["config"]
    final = pd.DataFrame(data["data"])

    stats = {"pickle": os.path.basename(filename), "source": config.get("source")}
    stats.update(_attrition(config, final))
    stats.update(_labels(final, config))
    stats.update(_splits(final, config.get("split_data", {}).get("method")))
    stats.update(_sizes(data["prots"], "prot"))
    stats.update(_sizes(data["drugs"], "drug"))
    stats["prot_features"] = config.get("prots", {}).get("features", {})
    stats["prot_structs"] = config.get("prots", {}).get("structs", {})
    stats["drug_features"] = {
        "node_feats": config.get("drugs", {}).get("node_feats"),
        "edge_feats": config.get("drugs", {}).get("edge_feats"),
        "max_num_atoms": config.get("drugs", {}).get("max_num_atoms"),
    }
    return stats


def _line(label: str, value, note: str = "") -> str:
    if value is None:
        shown = "n/a"
    elif isinstance(value, float):
        shown = f"{value:,.2f}"
    elif isinstance(value, int):
        shown = f"{value:,}"
    else:
        shown = str(value)
    return f"  {label:<38} {shown:>14}{'   ' + note if note else ''}"


def render(stats: dict) -> str:
    """Human-readable report, in the order the manuscript needs the numbers."""
    lines = [f"=== {stats['source']} ===", f"    {stats['pickle']}", "", "Attrition"]
    lines += [
        _line("raw interactions", stats["raw_interactions"]),
        _line("raw targets", stats["raw_targets"]),
        _line("raw drugs", stats["raw_drugs"]),
        _line("targets with an AlphaFold structure", stats["targets_with_structure"]),
        _line("interactions after filtering", stats["interactions_after_filtering"]),
        _line("final interactions", stats["final_interactions"]),
        _line("final targets", stats["final_targets"]),
        _line("final drugs", stats["final_drugs"]),
        "",
        "Labels",
        _line("task", stats["task"]),
    ]
    if stats["task"] == "reg":
        unit = "log10-transformed" if stats.get("logged") else stats.get("unit")
        lines += [
            _line("scale", unit),
            _line("mean +- std", f"{stats['label_mean']:.2f} +- {stats['label_std']:.2f}"),
            _line("range", f"{stats['label_min']:.2f} .. {stats['label_max']:.2f}"),
        ]
    else:
        direction = "<" if stats.get("unit") == "nM" else ">="
        lines += [
            _line("binarization", f"Y {direction} {stats['threshold']} ({stats.get('unit')})"),
            _line("filtering / sampling", f"{stats['filtering']} / {stats['sampling']}"),
            _line("positives", stats["positives"], f"({stats['positive_rate']:.1%})"),
        ]

    lines += ["", f"Splits ({stats['split_method']})"]
    for split in ("train", "val", "test"):
        lines.append(
            _line(
                f"{split} interactions",
                stats[f"{split}_interactions"],
                f"({stats[f'{split}_fraction']:.1%})  "
                f"{stats[f'{split}_targets']} targets, {stats[f'{split}_drugs']} drugs",
            )
        )
    if "leakage_overlap" in stats:
        overlap = stats["leakage_overlap"]
        verdict = "OK" if overlap == 0 else "LEAKAGE"
        lines.append(_line(f"train/test {stats['leakage_column']} overlap", overlap, f"<- {verdict}"))
    if stats["empty_splits"]:
        lines.append(_line("empty splits", ", ".join(stats["empty_splits"]), "<- DEGENERATE"))

    for kind, label in (("prot", "Protein graphs"), ("drug", "Drug graphs")):
        unit = "residues" if kind == "prot" else "atoms"
        lines += ["", f"{label} ({stats[f'{kind}_count']:,} unique)"]
        lines.append(
            _line(
                unit,
                f"{stats[f'{kind}_nodes_mean']:.1f} mean",
                f"median {stats[f'{kind}_nodes_median']:.0f}, "
                f"range {stats[f'{kind}_nodes_min']}..{stats[f'{kind}_nodes_max']}",
            )
        )
        if f"{kind}_edges_mean" in stats:
            lines.append(
                _line(
                    "edges",
                    f"{stats[f'{kind}_edges_mean']:.1f} mean",
                    f"mean degree {stats[f'{kind}_degree_mean']:.1f}",
                )
            )
    return "\n".join(lines)


def main(pickles: list[str], output: str = None, table: str = None) -> None:
    """Report dataset statistics for one or more ``prepare_all`` pickles.

    Exits non-zero if any cold split leaks an entity across the train/test boundary.
    """
    collected = [collect(path) for path in pickles]
    print("\n\n".join(render(stats) for stats in collected))

    if output:
        with open(output, "w") as file:
            json.dump(collected if len(collected) > 1 else collected[0], file, indent=2)
        print(f"\nWrote {output}")
    if table:
        pd.DataFrame(collected).to_csv(table, sep="\t", index=False)
        print(f"Wrote {table}")

    problems = 0
    for stats in collected:
        if stats.get("leakage_overlap", 0):
            problems += 1
            print(
                f"\nLEAKAGE: {stats['pickle']} ({stats['split_method']}) shares "
                f"{stats['leakage_overlap']} {stats['leakage_column']} between train and test",
                file=sys.stderr,
            )
        if stats["empty_splits"]:
            problems += 1
            print(
                f"\nDEGENERATE: {stats['pickle']} ({stats['split_method']}) has an empty "
                f"{'/'.join(stats['empty_splits'])} split. split_groups allocates "
                "int(10 * frac) per bin of ten, so a dataset with fewer than ten distinct "
                "groups sends everything to train.",
                file=sys.stderr,
            )
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    if "snakemake" in globals():
        with open(snakemake.output.stats, "w") as handle:  # noqa: F821
            json.dump(collect(snakemake.input.pickle), handle, indent=2)  # noqa: F821
    else:
        # argparse rather than jsonargparse: the point of this tool is to take a shell
        # glob over a whole results directory, and `nargs="+"` accepts that directly
        # where a jsonargparse list positional wants comma-separated values.
        from argparse import ArgumentParser

        parser = ArgumentParser(
            prog="dataset_stats.py",
            description="Describe built datasets: attrition, labels, splits, graph sizes.",
            epilog="Example: dataset_stats.py datasets/davis/results/prepare_all/*.pkl --table supp1.tsv",
        )
        parser.add_argument("pickles", nargs="+", help="prepare_all/<hash>.pkl paths")
        parser.add_argument("--output", help="also write all statistics as JSON here")
        parser.add_argument("--table", help="also write one row per dataset as TSV here")
        main(**vars(parser.parse_args()))
