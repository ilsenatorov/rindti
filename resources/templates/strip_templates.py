"""Strip each template PDB to one kinase chain (kinase domain) plus its one ligand residue.

1ATP is cut to the PKA kinase core (40-300), dropping the PKA-specific N-terminal aA helix
and C-tail, so all three templates cover only the conserved kinase domain (1IEP and 1AQ1
are kinase-domain constructs already).

Usage: python strip_templates.py <dir of raw RCSB PDBs> <output dir>
"""

import sys

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

# (entry, chain, ligand residue, residue range to keep)
TEMPLATES = [
    ("1IEP", "A", "STI", (0, 9999)),
    ("1AQ1", "A", "STU", (0, 9999)),
    ("1ATP", "E", "ATP", (40, 300)),
]


def main(raw: str, out: str) -> None:
    for pdb, chain, lig, (lo, hi) in TEMPLATES:
        s = PDBFile.read(f"{raw}/{pdb}.pdb").get_structure(model=1, extra_fields=["b_factor"])
        s = s[s.chain_id == chain]
        s = s[(struc.filter_amino_acids(s) & (s.res_id >= lo) & (s.res_id <= hi)) | (s.res_name == lig)]
        pdbf = PDBFile()
        pdbf.set_structure(s)
        pdbf.write(f"{out}/{pdb.lower()}.pdb")

        # Whatever parse_structs.organic_atoms will call "ligand" in this template. If a
        # revised RCSB entry brings a second heteroatom along, bsite would silently
        # measure distances to the wrong thing, so fail here instead.
        org = s[~struc.filter_amino_acids(s) & ~struc.filter_solvent(s) & ~struc.filter_monoatomic_ions(s)]
        print(pdb, "residues:", struc.get_residue_count(s), "ligand atoms:", org.array_length(), set(org.res_name))
        assert set(org.res_name) == {lig}, org.res_name


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
