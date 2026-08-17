from rdkit.Chem.rdchem import ChiralType


def list_to_dict(list_object: list):
    """Convert list to dict"""
    return {val: i for i, val in enumerate(list_object)}


encd = {
    "prot": {
        "node": list_to_dict(
            [
                "ala",
                "arg",
                "asn",
                "asp",
                "cys",
                "gln",
                "glu",
                "gly",
                "his",
                "ile",
                "leu",
                "lys",
                "met",
                "phe",
                "pro",
                "ser",
                "thr",
                "trp",
                "tyr",
                "val",
            ]
        ),
    },
    "drug": {
        "node": list_to_dict(["other", 6, 7, 8, 9, 16, 17, 35, 15, 53, 5, 11, 14, 34]),
        # TRIPLE used to be absent, which silently dropped every molecule
        # containing a triple bond (nitriles, alkynes) from the dataset.
        "edge": list_to_dict(["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]),
    },
    "glycan": {
        "other": [0, 0, 0],
        6: [1, 0, 0],  # carbon
        7: [0, 1, 0],  # nitrogen
        8: [0, 0, 1],  # oxygen
    },
    "chirality": {
        ChiralType.CHI_OTHER: [0, 0, 0],
        ChiralType.CHI_TETRAHEDRAL_CCW: [
            1,
            1,
            0,
        ],  # counterclockwise rotation of polarized light -> rotate light to the left
        ChiralType.CHI_TETRAHEDRAL_CW: [
            1,
            0,
            1,
        ],  # clockwise rotation of polarized light -> rotate light to the right
        ChiralType.CHI_UNSPECIFIED: [0, 0, 0],
    },
}
