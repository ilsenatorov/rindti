"""Grouping functions for similarity-controlled dataset splits.

A random split, and even a cold-target split on raw IDs, leaves near-identical
entities on both sides of the train/test boundary: Davis alone contains exact
duplicate kinase sequences under different Target_IDs. A model can then score a test
protein it has effectively already trained on, which inflates every reported number.

These helpers produce a *grouping column*; the split itself is unchanged and still
happens in ``split_data.split_groups``, which already accepts any column.

Nothing here drops rows. ``prepare_all.py`` filters interactions against the protein
and drug pickles, so every original ID must survive - only the split assignment is
shared across the members of a group.
"""

from __future__ import annotations

import sys


def dedup_sequences(sequences: dict[str, str]) -> dict[str, str]:
    """Map each ID to a canonical representative of its exact-duplicate group.

    This is the cheap half of similarity control, and it applies even to a plain
    ``method: target`` split, where no clustering runs at all. GEFA reports Davis
    collapsing from 442 to 361 distinct targets this way.

    The representative is the alphabetically first ID sharing the sequence, so the
    grouping is reproducible across builds rather than dependent on file order.
    """
    groups: dict[str, list[str]] = {}
    for id_ in sorted(sequences):
        groups.setdefault(sequences[id_], []).append(id_)
    return {id_: members[0] for members in groups.values() for id_ in members}


def _fingerprints(smiles: dict[str, str], radius: int = 2, n_bits: int = 2048):
    """ECFP4 fingerprints, in a stable ID order.

    Returns ``(ids, fps, failed)``. Unparseable SMILES are reported rather than
    silently dropped - they still need a group, and the caller gives them their own.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    ids, fps, failed = [], [], []
    for id_ in sorted(smiles):
        mol = Chem.MolFromSmiles(smiles[id_])
        if mol is None:
            failed.append(id_)
            continue
        ids.append(id_)
        fps.append(generator.GetFingerprint(mol))
    return ids, fps, failed


def cluster_drugs(smiles: dict[str, str], cutoff: float = 0.6, log_every: int = 10_000) -> dict[str, str]:
    """Group drugs by ECFP4/Tanimoto similarity, returning ID -> representative ID.

    Leader (sphere-exclusion) clustering: walk the molecules in a fixed order and
    attach each to the first leader it is within ``cutoff`` of, or make it a new
    leader. Every comparison is against the leaders only, never the full matrix.

    That matters at scale. GLASS holds 165,691 ligands, i.e. ~1.4e10 unique pairs; the
    condensed float32 distance matrix that ``scipy.cluster.hierarchy`` would need is
    about 55 GB, so agglomerative clustering is simply not an option there.

    Iteration order is the sorted ID order, not dict or file order, so the same input
    yields the same clusters on every build - the split seed would otherwise no longer
    describe the experiment.
    """
    from rdkit.DataStructs import BulkTanimotoSimilarity

    if not 0 < cutoff <= 1:
        raise ValueError(f"drug_similarity must be in (0, 1], got {cutoff}")

    ids, fps, failed = _fingerprints(smiles)
    if failed:
        print(f"{len(failed)} SMILES could not be parsed, each gets its own cluster: {failed[:5]}", file=sys.stderr)

    leader_ids: list[str] = []
    leader_fps: list = []
    assignment: dict[str, str] = {}
    for n, (id_, fp) in enumerate(zip(ids, fps, strict=True)):
        if leader_fps:
            similarities = BulkTanimotoSimilarity(fp, leader_fps)
            best = max(range(len(similarities)), key=similarities.__getitem__)
            if similarities[best] >= cutoff:
                assignment[id_] = leader_ids[best]
                continue
        leader_ids.append(id_)
        leader_fps.append(fp)
        assignment[id_] = id_
        if log_every and n and n % log_every == 0:
            print(f"clustered {n}/{len(ids)} molecules into {len(leader_ids)} clusters", file=sys.stderr)

    # An unparseable molecule is its own cluster: it cannot be shown similar to
    # anything, so the conservative choice is to assume it is not.
    for id_ in failed:
        assignment[id_] = id_
    print(f"{len(ids) + len(failed)} molecules -> {len(leader_ids) + len(failed)} clusters", file=sys.stderr)
    return assignment


def write_fasta(sequences: dict[str, str], path: str) -> dict[str, str]:
    """Write a FASTA with surrogate headers, returning surrogate -> original ID.

    Real Target_IDs are not safe as FASTA headers: Davis uses names like
    ``RSK1(KinDom.1-N-terminal)``, and MMseqs2 truncates a header at the first
    whitespace and reports its own idea of an accession. Numbering the entries in
    sorted ID order sidesteps all of it and keeps the mapping reproducible.
    """
    mapping = {}
    with open(path, "w") as fasta:
        for n, id_ in enumerate(sorted(sequences)):
            surrogate = f"s{n}"
            mapping[surrogate] = id_
            fasta.write(f">{surrogate}\n{sequences[id_]}\n")
    return mapping


def parse_mmseqs_clusters(path: str, mapping: dict[str, str]) -> dict[str, str]:
    """Read an MMseqs2 ``*_cluster.tsv`` into original-ID -> representative-ID."""
    assignment = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            representative, member = line.rstrip("\n").split("\t")[:2]
            assignment[mapping[member]] = mapping[representative]
    missing = set(mapping.values()) - assignment.keys()
    if missing:
        raise ValueError(f"MMseqs2 did not assign {len(missing)} sequences to a cluster, e.g. {sorted(missing)[:5]}")
    return assignment
