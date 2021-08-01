import pandas as pd
from rdkit import Chem
from rdkit.Chem.QED import default


def get_druglikeness(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return default(mol)
    except Exception as e:
        print(e)
        return 0


lig = pd.read_csv(snakemake.input.lig, sep='\t')
lig['druglikeness'] = lig['Canonical SMILES'].apply(get_druglikeness)
lig.to_csv(snakemake.output.lig, index=False)
