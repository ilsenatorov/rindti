import pandas as pd
import os
from Bio import PDB

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

result = ''

if snakemake.config['clustalo']['sequence_source'] == 'csv':
    df = pd.read_csv(snakemake.input.targ)
    with open(snakemake.output.sequences, 'w') as file:
        for name, row in df.iterrows():
            result += '>{name}\n{sequence}\n'.format(
                name=row['UniProt ID'],
                sequence=row['FASTA Sequence']
            )
elif snakemake.config['clustalo']['sequence_source'] == 'structures':
    parser = PDB.PDBParser()
    for struct in snakemake.input.targ:
        name = struct.split('/')[-1].split('.')[0]
        result += '>{name}\n'.format(name=name)
        data = parser.get_structure(name, struct)
        for residue in data[0]['A'].get_residues():
            result += d3to1[residue.resname]
        result += '\n'

with open(snakemake.output.sequences, 'w') as file:
    file.write(result)
