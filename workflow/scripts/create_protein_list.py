import os

import pandas as pd

l = pd.Series([x[:-4] for x in os.listdir("resources/structures")], name="UniProt ID")
pd.DataFrame(l).reset_index().to_csv(snakemake.output.protein_list, index=False)
