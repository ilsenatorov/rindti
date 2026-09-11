"""Write protein sequences to FASTA for MMseqs2, with surrogate headers.

Separate from the clustering rule so that the MMseqs2 binary - which lives in its own
conda environment - only ever has to run a shell command.
"""

import pandas as pd
from cluster import write_fasta

if __name__ == "__main__":
    prots = pd.read_csv(snakemake.input.seqs, sep="\t").set_index("Target_ID")
    mapping = write_fasta(prots["Target"].to_dict(), snakemake.output.fasta)
    pd.Series(mapping, name="Target_ID").rename_axis("surrogate").to_csv(snakemake.output.mapping, sep="\t")
