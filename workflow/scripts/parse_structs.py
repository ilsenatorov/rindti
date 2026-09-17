"""Reduce a protein structure to the residues a graph should be built from.

Replaces the old PyMOL/psico/TMalign pipeline, which generated a ``.pml`` script per
protein and ran it in a conda environment. Everything here is `biotite`, a plain PyPI
wheel, so the workflow no longer needs a conda environment for structure parsing.

Three methods, all of which return a subset of the input as whole residues:

``plddt``
    Keep residues whose CA B-factor exceeds a threshold. For AlphaFold models the
    B-factor column holds pLDDT, so this drops the parts of a prediction that the
    predictor itself is not confident about.
``bsite``
    Superimpose the best-matching template onto the query and keep the residues near
    that template's bound ligand - the binding site, transferred from a homolog.
``template``
    The same alignment, but keep the residues near the template's backbone, i.e. the
    part of the query the template actually covers.

The graph downstream is built from CA-CA distances (``distance_based.py``), so the
coordinate frame of the output is irrelevant; only which residues survive matters.
The query is nevertheless left in its original frame - the template is moved onto it,
not the other way round - so the output can still be overlaid on the input.
"""

import sys

import biotite.structure as struc
import numpy as np
from biotite.structure.io.pdb import PDBFile


def read_structure(path: str) -> struc.AtomArray:
    """Read the first model of a PDB file, keeping the B-factor column.

    ``b_factor`` is an optional annotation in biotite and has to be asked for. The
    ``plddt`` method is a threshold on it, and carrying it through means the parsed
    structure still reports the model confidence it was filtered on.
    """
    return PDBFile.read(path).get_structure(model=1, extra_fields=["b_factor"])


def write_structure(path: str, structure: struc.AtomArray) -> None:
    """Write an atom array back out as a PDB."""
    pdb = PDBFile()
    pdb.set_structure(structure)
    pdb.write(path)


def _whole_residues(structure: struc.AtomArray, indices) -> np.ndarray:
    """Expand atom indices to a mask over every atom of the residues they belong to.

    This is PyMOL's ``br.`` operator. Without it a radius selection cuts residues in
    half, and the old ``plddt`` script did exactly that - it thresholded atoms, not
    residues, so a residue could reach the graph without the CA that represents it.
    """
    indices = np.unique(np.asarray(list(indices), dtype=int))
    if indices.size == 0:
        return np.zeros(structure.array_length(), dtype=bool)
    return struc.get_residue_masks(structure, indices).any(axis=0)


def select_plddt(structure: struc.AtomArray, threshold: float) -> struc.AtomArray:
    """Keep whole residues whose CA B-factor is above ``threshold``."""
    ca = np.flatnonzero((structure.atom_name == "CA") & (structure.b_factor > threshold))
    return structure[_whole_residues(structure, ca)]


def largest_peptide_chain(structure: struc.AtomArray) -> struc.AtomArray:
    """The amino-acid atoms of the chain with the most residues.

    ``superimpose_structural_homologs`` aligns peptide chains and expects the two
    structures to have the same number of them, so both sides are reduced to a single
    chain before alignment. The resulting transformation is rigid and is applied to
    the whole template afterwards, so nothing is lost by aligning on one chain.
    """
    peptide = structure[struc.filter_amino_acids(structure)]
    if peptide.array_length() == 0:
        raise ValueError("structure contains no amino acids")
    chains = np.unique(peptide.chain_id)
    if len(chains) == 1:
        return peptide
    sizes = [struc.get_residue_count(peptide[peptide.chain_id == c]) for c in chains]
    return peptide[peptide.chain_id == chains[int(np.argmax(sizes))]]


def align_template(query: struc.AtomArray, template: struc.AtomArray) -> tuple[struc.AtomArray, float]:
    """Superimpose ``template`` onto ``query``, returning the moved template and its TM-score.

    ``superimpose_structural_homologs`` is biotite's TM-align-inspired heuristic: it
    aligns the two structures in a structural alphabet, then refines with a
    TM-score-based substitution matrix. Scores are close to, but not identical with,
    the TM-align binary the PyMOL version called through psico. They are only used to
    rank templates against each other, so the absolute value does not matter.
    """
    query_chain = largest_peptide_chain(query)
    template_chain = largest_peptide_chain(template)
    _, transform, query_idx, template_idx = struc.superimpose_structural_homologs(query_chain, template_chain)
    score = struc.tm_score(query_chain, transform.apply(template_chain), query_idx, template_idx)
    return transform.apply(template), score


def best_template(query: struc.AtomArray, template_paths: list[str]) -> tuple[struc.AtomArray, str, float]:
    """Pick the template that aligns best to ``query`` and return it in the query's frame.

    The old PyMOL script loaded every template, aligned them all onto the best-scoring
    one, and then selected against *all* of them at once - so a poorly matching
    template still contributed residues to the binding site. Only the best template is
    used here, which is what the method is supposed to mean.
    """
    if not template_paths:
        raise ValueError("no templates given; `bsite` and `template` need PDBs in <source>/templates")
    best, best_path, best_score = None, None, -np.inf
    for path in template_paths:
        fitted, score = align_template(query, read_structure(path))
        print(f"template {path}: TM-score {score:.3f}", file=sys.stderr)
        if score > best_score:
            best, best_path, best_score = fitted, path, score
    print(f"best template: {best_path} (TM-score {best_score:.3f})", file=sys.stderr)
    return best, best_path, best_score


def organic_atoms(structure: struc.AtomArray) -> struc.AtomArray:
    """The ligand atoms of a structure, as PyMOL's ``organic`` selection understood them.

    Everything that is not an amino acid, not solvent and not a lone ion - so a bound
    drug or cofactor, but not the waters and sulfates that crystallography leaves
    behind and that would smear the binding site across the whole surface.
    """
    mask = ~struc.filter_amino_acids(structure) & ~struc.filter_solvent(structure)
    mask &= ~struc.filter_monoatomic_ions(structure)
    return structure[mask]


def select_near(structure: struc.AtomArray, coord: np.ndarray, radius: float) -> struc.AtomArray:
    """Keep whole residues of ``structure`` with an atom within ``radius`` of ``coord``."""
    if len(coord) == 0:
        raise ValueError("nothing to measure the distance to")
    neighbours = struc.CellList(structure, radius).get_atoms(coord, radius)
    return structure[_whole_residues(structure, neighbours[neighbours != -1])]


def parse_structure(
    struct_path: str,
    method: str,
    params: dict,
    template_paths: list[str] | None = None,
) -> struc.AtomArray:
    """Apply one of the structure-parsing methods to a single PDB."""
    structure = read_structure(struct_path)
    if method == "plddt":
        result = select_plddt(structure, params["plddt"]["threshold"])
    elif method in ("bsite", "template"):
        template, _, _ = best_template(structure, template_paths or [])
        anchor = organic_atoms(template) if method == "bsite" else template[template.atom_name == "CA"]
        if anchor.array_length() == 0:
            what = "organic ligand" if method == "bsite" else "CA"
            raise ValueError(f"the best template has no {what} atoms, so `{method}` cannot select anything")
        result = select_near(structure, anchor.coord, params[method]["radius"])
    else:
        raise ValueError(f"unknown structure method {method!r}")
    if result.array_length() == 0:
        raise ValueError(
            f"`{method}` selected no atoms in {struct_path}. The old pipeline wrote an empty PDB here and "
            "failed much later, while building the graph; check the threshold or radius instead."
        )
    print(
        f"{struct_path}: {struc.get_residue_count(result)} of {struc.get_residue_count(structure)} residues kept",
        file=sys.stderr,
    )
    return result


if __name__ == "__main__":
    if "snakemake" in globals():
        write_structure(
            snakemake.output.struct,
            parse_structure(
                snakemake.input.struct,
                snakemake.params.method,
                snakemake.params.other_params,
                list(snakemake.input.templates),
            ),
        )
    else:
        from jsonargparse import CLI

        def run(
            struct: str,
            output: str,
            method: str = "plddt",
            threshold: float = 70,
            radius: float = 5,
            templates: list[str] = None,
        ):
            """Parse a single structure outside of snakemake."""
            params = {"plddt": {"threshold": threshold}, "bsite": {"radius": radius}, "template": {"radius": radius}}
            write_structure(output, parse_structure(struct, method, params, templates or []))

        CLI(run)
