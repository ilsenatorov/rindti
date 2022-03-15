import os

from tqdm import tqdm

threetoone = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}


def pdb_to_sequence(pdb_filename: str) -> str:
    """Extract sequence from PDB file and return it as a string."""
    sequence = ""
    with open(pdb_filename, "r") as file:
        for line in file.readlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                sequence += threetoone[line[17:20].strip()]
    return sequence


def bulk_pdb_to_fasta(pdb_dir: str, max_length: int = None) -> str:
    """Extract sequences from all PDB files in a directory and return them as a FASTA string"""
    fasta = ""
    for filename in tqdm(os.listdir(pdb_dir)):
        if filename.endswith(".pdb"):
            fasta += ">" + filename[:-4] + "\n"
            seq = pdb_to_sequence(os.path.join(pdb_dir, filename))
            if max_length:
                seq = seq[:max_length]
            fasta += seq + "\n"
    return fasta


if __name__ == "__main__":
    from jsonargparse import CLI

    def run(pdb_dir: str, output: str, max_length: int = None) -> None:
        """Run the bulk_pdb_to_fasta function on all PDB files in a directory and write the result to a file."""
        fasta = bulk_pdb_to_fasta(pdb_dir, max_length=max_length)
        with open(output, "w") as file:
            file.write(fasta)

    CLI(run)
