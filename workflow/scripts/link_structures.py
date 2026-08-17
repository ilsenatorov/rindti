"""Reconcile structure filenames with the Target_IDs used in the interaction tables.

The pipeline expects ``<source>/structures/<Target_ID>.pdb``. Datasets frequently
violate this: Davis, for instance, keys its tables by gene name (``PHKG2``, ``TAK1``)
while its structures were downloaded from AlphaFold DB and are named by UniProt
accession (``P57058``). The overlap is then zero and ``prepare_all`` produces an
empty dataset with no indication of why.

This script rebuilds the correspondence from the one thing both sides agree on --
the amino-acid sequence -- and renames the structures accordingly.

Matching proceeds in three passes, most to least trustworthy:

1. ``exact``     - the table sequence equals the structure's CA sequence.
2. ``substring`` - the table sequence is contained in the structure's (a construct
                   or kinase-domain fragment of the full-length model).
3. ``similar``   - highest sequence similarity above ``--threshold``. This is what
                   recovers point mutants such as ``FGFR3(G697C)``, whose sequence
                   differs from the wild-type structure by a single residue.

Anything below the threshold is reported and left alone rather than guessed at: an
incorrect pairing silently attaches a target's interactions to the wrong protein,
which is far worse than a missing one.

Usage::

    python workflow/scripts/link_structures.py datasets/davis/resources          # dry run
    python workflow/scripts/link_structures.py datasets/davis/resources --apply
"""

import difflib
import shutil
from pathlib import Path

import pandas as pd

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

UNMATCHED_DIRNAME = "unmatched"


def sequence_from_pdb(path: Path) -> str:
    """One-letter CA sequence of a PDB file, in file order."""
    return "".join(
        THREE_TO_ONE.get(line[17:20].strip(), "X")
        for line in path.read_text().splitlines()
        if line.startswith("ATOM") and line[12:16].strip() == "CA"
    )


def build_mapping(targets: dict, structures: dict, threshold: float = 0.95) -> pd.DataFrame:
    """Map each Target_ID to a structure stem.

    Args:
        targets: ``{Target_ID: sequence}`` from ``tables/prot.tsv``.
        structures: ``{file stem: CA sequence}`` from ``structures/*.pdb``.
        threshold: minimum similarity for the fuzzy pass.

    Returns:
        A frame with columns ``Target_ID, structure, method, score``. ``structure``
        is None where nothing was confidently matched.
    """
    by_exact = {}
    for stem, seq in structures.items():
        by_exact.setdefault(seq, stem)

    claimed, rows = set(), []
    for target_id, seq in targets.items():
        seq = seq.upper()

        stem = by_exact.get(seq)
        if stem is not None and stem not in claimed:
            claimed.add(stem)
            rows.append((target_id, stem, "exact", 1.0))
            continue

        stem = next(
            (s for s, ps in structures.items() if s not in claimed and seq in ps),
            None,
        )
        if stem is not None:
            claimed.add(stem)
            rows.append((target_id, stem, "substring", 1.0))
            continue

        best_score, best_stem, best_bound = 0.0, None, 0.0
        for candidate, ps in structures.items():
            if candidate in claimed or abs(len(ps) - len(seq)) > 0.25 * max(len(seq), 1):
                continue
            matcher = difflib.SequenceMatcher(None, seq, ps, autojunk=False)
            # real_quick_ratio/quick_ratio are cheap upper bounds; only pay for the
            # exact ratio when the bound says a match is still possible. Keep the
            # best bound so the report can show how near the near-misses were.
            bound = min(matcher.real_quick_ratio(), matcher.quick_ratio())
            best_bound = max(best_bound, bound)
            if bound < threshold:
                continue
            score = matcher.ratio()
            if score > best_score:
                best_score, best_stem = score, candidate

        if best_stem is not None and best_score >= threshold:
            claimed.add(best_stem)
            rows.append((target_id, best_stem, "similar", round(best_score, 4)))
        else:
            # Report the upper bound rather than 0.0: it says how close the nearest
            # candidate came, which is what you need to pick a new --threshold.
            rows.append((target_id, None, "unmatched", round(max(best_score, best_bound), 4)))

    return pd.DataFrame(rows, columns=["Target_ID", "structure", "method", "score"])


def apply_mapping(structures_dir: Path, mapping: pd.DataFrame) -> None:
    """Rename structures to their Target_ID; park unmatched ones out of the way.

    Unmatched structures are moved into ``structures/unmatched/`` so that the
    pipeline's non-recursive glob stops seeing them - otherwise they would enter the
    DAG as phantom proteins that no interaction refers to.
    """
    matched = mapping.dropna(subset=["structure"])

    staging = structures_dir / ".renaming"
    staging.mkdir(exist_ok=True)
    for row in matched.itertuples():
        shutil.move(structures_dir / f"{row.structure}.pdb", staging / f"{row.Target_ID}.pdb")

    leftovers = sorted(structures_dir.glob("*.pdb"))
    if leftovers:
        parked = structures_dir / UNMATCHED_DIRNAME
        parked.mkdir(exist_ok=True)
        for path in leftovers:
            shutil.move(path, parked / path.name)

    for path in staging.glob("*.pdb"):
        shutil.move(path, structures_dir / path.name)
    staging.rmdir()


def link(resources: str, threshold: float = 0.95, apply: bool = False) -> None:
    """Match structures to targets by sequence and optionally rename them.

    Args:
        resources: dataset resources directory, containing ``tables/`` and ``structures/``.
        threshold: minimum sequence similarity for the fuzzy pass.
        apply: actually rename the files. Without it, only the mapping is written.
    """
    resources = Path(resources)
    structures_dir = resources / "structures"
    prot_table = resources / "tables" / "prot.tsv"

    prot = pd.read_csv(prot_table, sep="\t")
    targets = dict(zip(prot["Target_ID"], prot["Target"], strict=True))
    structures = {p.stem: sequence_from_pdb(p) for p in sorted(structures_dir.glob("*.pdb"))}
    print(f"{len(targets)} targets in {prot_table}, {len(structures)} structures in {structures_dir}")

    already = sum(1 for t in targets if t in structures)
    if already == len(targets):
        print("Every target already has a structure named after it; nothing to do.")
        return

    mapping = build_mapping(targets, structures, threshold)
    out = resources / "structure_mapping.tsv"
    mapping.to_csv(out, sep="\t", index=False)

    counts = mapping["method"].value_counts().to_dict()
    print(f"Mapping written to {out}: {counts}")

    unmatched = mapping[mapping["method"] == "unmatched"]
    if not unmatched.empty:
        print(f"\n{len(unmatched)} targets could not be matched above {threshold}:")
        for row in unmatched.itertuples():
            print(f"  {row.Target_ID:<18} best similarity {row.score}")
        print(
            "\nThese are usually isoforms or aliases (TRKA/NTRK1, TIE2/TEK, FAK/PTK2).\n"
            "Lower --threshold to include them, or map them by hand in the TSV.\n"
            "Their interactions will be dropped by prepare_all's inner join."
        )

    if not apply:
        print("\nDry run. Re-run with --apply to rename the structures.")
        return

    apply_mapping(structures_dir, mapping)
    renamed = int((mapping["method"] != "unmatched").sum())
    print(f"\nRenamed {renamed} structures; parked the rest in {structures_dir / UNMATCHED_DIRNAME}/.")
    print(f"Reversible via {out}.")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(link)
