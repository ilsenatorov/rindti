"""Featurise proteins as mean-pooled ESM-1b embeddings instead of contact graphs.

This is the sequence arm of the structure-versus-sequence comparison, so it has to be
matched with `model.prot.method: vector` - `GraphEncoder` cannot consume a lone vector.
"""

import os
import tempfile

import pandas as pd
import torch
from extract_esm import create_parser
from extract_esm import main as extract_main

# ESM-1b's positional embeddings stop at 1024 tokens, two of which are the start and end
# tokens. Longer chains have to be cut.
MAX_ESM_LENGTH = 1022


def truncation_report(sequences: dict[str, str]) -> tuple[int, int]:
    """How many sequences exceed the model's context, and by how much at worst.

    Truncation used to be silent. The structure arm builds its graph from the whole
    chain, so in the comparison this featurisation is meant to be matched against, a
    truncated protein is a confound rather than a detail - it has to be reported and
    disclosed, not discovered later.
    """
    excess = [len(seq) - MAX_ESM_LENGTH for seq in sequences.values() if len(seq) > MAX_ESM_LENGTH]
    return len(excess), max(excess, default=0)


def generate_esm(prot: pd.DataFrame, workdir: str) -> pd.DataFrame:
    """Mean-pooled layer-33 representations, one row per protein.

    ``workdir`` is created by the caller and is per-invocation. The FASTA and the
    per-protein ``.pt`` files used to be written to a relative ``./esms/``, which is not
    a declared snakemake output, is not content-hashed, and on a cluster resolves to
    whatever the job's working directory happens to be - so two concurrent jobs
    overwrote each other's embeddings.
    """
    prot_ids, seqs = zip(*prot["Target"].to_dict().items(), strict=True)

    truncated, worst = truncation_report(prot["Target"].to_dict())
    if truncated:
        print(
            f"WARNING: {truncated} of {len(prot_ids)} sequences exceed ESM-1b's "
            f"{MAX_ESM_LENGTH}-residue context and are truncated (longest by {worst} residues). "
            "The structure arm uses the whole chain, so disclose this when comparing them."
        )

    fasta_path = os.path.join(workdir, "prots.fasta")
    with open(fasta_path, "w") as fasta:
        for prot_id, seq in zip(prot_ids, seqs, strict=True):
            fasta.write(f">{prot_id}\n{seq[:MAX_ESM_LENGTH]}\n")

    esm_args = create_parser().parse_args(
        ["esm1b_t33_650M_UR50S", fasta_path, workdir, "--repr_layers", "33", "--include", "mean"]
    )
    extract_main(esm_args)

    prot["data"] = [
        {"x": torch.load(os.path.join(workdir, f"{prot_id}.pt"))["mean_representations"][33].unsqueeze(0)}
        for prot_id in prot_ids
    ]
    return prot


if __name__ == "__main__":
    prots = pd.read_csv(snakemake.input.seqs, sep="\t").set_index("Target_ID")
    # Cleaned up on exit: the embeddings are an intermediate, and the pickle is the output.
    with tempfile.TemporaryDirectory(prefix="rindti_esm_") as workdir:
        prots = generate_esm(prots, workdir)
    prots.to_pickle(snakemake.output.pickle)
