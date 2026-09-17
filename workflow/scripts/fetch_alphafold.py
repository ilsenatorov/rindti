# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.2",
#     "requests>=2.32",
#     "tqdm>=4.66",
#     "jsonargparse>=4.28",
# ]
# ///
"""Fetch AlphaFold models for a dataset whose tables are keyed by gene name.

``get_datasets.py`` asks AlphaFold DB for one model per ``Target_ID``. That works when
the ``Target_ID`` *is* a UniProt accession (KIBA, BindingDB) and fails completely when it
is a gene name, which is what Davis uses: all 379 lookups miss, every interaction is then
dropped for having no structure, and the script writes three header-only tables and exits
0. An empty dataset with no error is the worst possible failure, so this script exists to
supply the missing half of the mapping.

    uv run workflow/scripts/fetch_alphafold.py datasets/davis/resources            # dry run
    uv run workflow/scripts/fetch_alphafold.py datasets/davis/resources --apply true

For each ``Target_ID`` it resolves a reviewed human UniProt accession by gene name, and
then **verifies that accession against the sequence already in the table** before
accepting it. The verification is the point, not a formality: ``TAK1`` alone returns both
MAP3K7 (O43318) and NR2C2 (P49116) from UniProt, and Davis carries mutants and isolated
kinase domains (``BRAF(V600E)``, ``JAK1(JH2domain-pseudokinase)``) whose sequences differ
from the full-length model. Anything below ``--threshold`` is reported and skipped rather
than guessed at - a wrong pairing silently attaches a target's interactions to the wrong
protein, which is far worse than a missing target.

Structures are written as ``<Target_ID>.pdb``, which is what the pipeline expects, so
``link_structures.py`` is not needed afterwards. The accession chosen for each target is
recorded in ``uniprot_mapping.tsv``.
"""

import difflib
import shutil
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD = "https://alphafold.ebi.ac.uk/api/prediction"

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


def _session() -> requests.Session:
    """A session that retries, because this network is not reliable."""
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


# Davis names a handful of targets in a way no UniProt query resolves: a lab alias with
# no synonym entry (MRCKA, PFTAIRE2), or an alias whose nearest match is the wrong
# paralog - "AMPK-alpha1" finds PRKAA2 (alpha-2), "S6K1" finds an unrelated entry. These
# are only *suggestions*: each still has to beat --threshold against the table's own
# sequence, so a wrong line here is rejected rather than believed.
ALIASES = {
    "AMPK-alpha1": "PRKAA1",
    "AMPK-alpha2": "PRKAA2",
    "DLK": "MAP3K12",
    "MRCKA": "CDC42BPA",
    "MRCKB": "CDC42BPB",
    "PFTAIRE2": "CDK15",
    "PKAC-alpha": "PRKACA",
    "PKAC-beta": "PRKACB",
    "S6K1": "RPS6KB1",
}


def gene_candidates(target_id: str) -> list[str]:
    """Gene names to try for one ``Target_ID``, most to least specific.

    Davis decorates gene names three ways: a parenthesised mutation or domain
    (``BRAF(V600E)``, ``JAK1(JH1domain-catalytic)``), a trailing ``p`` for the
    phosphorylated form (``ABL1p``), and a hyphenated isoform (``CSNK1A1L``). The
    decorations describe the *construct*, not a different gene, so they are stripped.
    """
    base = target_id.split("(")[0].strip()
    candidates = []
    for key in (target_id, base):
        if key in ALIASES:
            candidates.append(ALIASES[key])
    candidates.append(base)
    if base.endswith("p") and len(base) > 2:
        candidates.append(base[:-1])
    if "-" in base:
        candidates.append(base.split("-")[0])
    seen = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]


def _search(session: requests.Session, query: str) -> list[dict]:
    """One UniProt query, as (accession, sequence) records."""
    url = f"{UNIPROT}?query={urllib.parse.quote(query)}&fields=accession,sequence&format=json&size=10"
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return [
            {"accession": entry["primaryAccession"], "sequence": entry["sequence"]["value"]}
            for entry in response.json().get("results", [])
        ]
    except (requests.RequestException, ValueError, KeyError):
        return []


def uniprot_hits(session: requests.Session, gene: str) -> list[dict]:
    """Candidate UniProt entries for a name, broadening until something turns up.

    Davis labels targets the way papers do, not the way UniProt does, so an exact gene
    symbol match finds only some of them:

    - ``p38-alpha``, ``IKK-beta`` and ``MRCKA`` are common aliases of MAPK14, IKBKB and
      CDC42BPA. ``gene:`` matches synonyms where ``gene_exact:`` does not.
    - ``PFCDPK1(Pfalciparum)`` and ``PFPK5(Pfalciparum)`` are *Plasmodium* kinases, so the
      human filter has to come off before they can be found at all.

    Broadening a search cannot pair a target with the wrong protein on its own: every
    candidate returned here still has to beat ``--threshold`` against the sequence already
    in the table, which is what rejects e.g. ``DLK`` against delta-like protein 1.
    """
    queries = [
        f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        f"gene:{gene} AND organism_id:9606 AND reviewed:true",
        f"gene:{gene} AND reviewed:true",
        f"protein_name:{gene} AND reviewed:true",
    ]
    hits: dict[str, dict] = {}
    for query in queries:
        for hit in _search(session, query):
            hits.setdefault(hit["accession"], hit)
        if hits:
            break
    return list(hits.values())


def score(table_sequence: str, candidate: str) -> float:
    """How much of the shorter sequence the two share.

    Coverage of the shorter sequence, not ``SequenceMatcher.ratio()``. The ratio is
    symmetric and so punishes a construct for being short: Davis ships isolated kinase
    domains, and a 300-residue domain sitting inside a 1000-residue entry scores ~0.6 on
    the ratio while being a perfect match for the part it covers. EPHB2, NEK4 and
    FGFR3(G697C) all score 1.00 here against 0.63, 0.70 and 0.75 by ratio, while ``DLK``
    against delta-like protein 1 - a genuinely wrong pairing - stays at 0.20 either way.

    ``autojunk=False`` is equally load-bearing. SequenceMatcher's default treats any
    character in more than 1% of a sequence of length >= 200 as junk, which in a
    20-letter amino-acid alphabet is every residue: ABL2 against its own UniProt entry
    scores 0.0009 with the default and 0.9698 without. link_structures.py already passes
    it for the same reason.
    """
    if not table_sequence or not candidate:
        return 0.0
    if table_sequence == candidate:
        return 1.0
    if table_sequence in candidate or candidate in table_sequence:
        return 0.99
    matcher = difflib.SequenceMatcher(None, table_sequence, candidate, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / min(len(table_sequence), len(candidate))


def resolve(session: requests.Session, target_id: str, sequence: str) -> tuple[str | None, float, str]:
    """Best accession for one target, with its score and the gene that produced it."""
    best = (None, 0.0, "")
    for gene in gene_candidates(target_id):
        for hit in uniprot_hits(session, gene):
            value = score(sequence, hit["sequence"])
            if value > best[1]:
                best = (hit["accession"], value, gene)
        if best[1] >= 0.999:  # exact hit, no point trying a less specific gene name
            break
    return best


def fetch_model(session: requests.Session, accession: str, destination: Path) -> bool:
    """Download one AlphaFold model, resolving the file URL through the API."""
    try:
        meta = session.get(f"{ALPHAFOLD}/{accession}", timeout=30)
        if not meta.ok or not meta.json():
            return False
        pdb = session.get(meta.json()[0]["pdbUrl"], timeout=60)
        pdb.raise_for_status()
        destination.write_text(pdb.text)
        return True
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return False


def run(source: str, apply: bool = False, threshold: float = 0.8, limit: int = 0) -> None:
    """Resolve gene names to UniProt and fetch the matching AlphaFold models.

    Args:
        source: dataset resources directory, e.g. ``datasets/davis/resources``.
        apply: actually download. The default only reports what would be fetched.
        threshold: minimum sequence agreement to accept an accession.
        limit: stop after this many targets, for a quick trial. 0 means all.
    """
    root = Path(source)
    table = pd.read_csv(root / "tables" / "prot.tsv", sep="\t")
    if limit:
        table = table.head(limit)
    structures = root / "structures"
    session = _session()

    rows, unresolved = [], []
    cache: dict[str, Path] = {}
    if apply:
        structures.mkdir(parents=True, exist_ok=True)

    for record in tqdm(table.itertuples(index=False), total=len(table), desc="targets"):
        target_id, sequence = record.Target_ID, record.Target
        accession, value, gene = resolve(session, target_id, sequence)
        if accession is None or value < threshold:
            unresolved.append((target_id, accession, round(value, 3)))
            continue
        rows.append({"Target_ID": target_id, "accession": accession, "gene": gene, "score": round(value, 3)})

        if not apply:
            continue
        destination = structures / f"{target_id}.pdb"
        if destination.exists():
            continue
        # Several Target_IDs share a gene - JAK1(JH1domain-catalytic) and
        # JAK1(JH2domain-pseudokinase) are one accession - so fetch each model once.
        if accession in cache:
            shutil.copyfile(cache[accession], destination)
        elif fetch_model(session, accession, destination):
            cache[accession] = destination
        else:
            unresolved.append((target_id, accession, "alphafold fetch failed"))
        time.sleep(0.05)

    mapping = pd.DataFrame(rows)
    print(f"\nresolved {len(mapping)} of {len(table)} targets")
    if not mapping.empty:
        print(mapping.head(10).to_string(index=False))
    if unresolved:
        print(f"\n{len(unresolved)} unresolved (reported, not guessed):", file=sys.stderr)
        for entry in unresolved[:25]:
            print(f"  {entry}", file=sys.stderr)

    if apply and not mapping.empty:
        mapping.to_csv(root / "uniprot_mapping.tsv", sep="\t", index=False)
        found = len(list(structures.glob("*.pdb")))
        print(f"\nwrote {found} structures to {structures}")
        print(f"mapping recorded in {root / 'uniprot_mapping.tsv'}")
        if found == 0:
            raise SystemExit("no structures downloaded - refusing to report success")
    elif not apply:
        print("\ndry run; pass --apply true to download")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(run)
