import os

import esm
import pandas as pd
import torch
from extract_esm import create_parser
from extract_esm import main as extract_main


def generate_esm_python(prot: pd.DataFrame) -> pd.DataFrame:
    """Return esms."""

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    prot.set_index("Target_ID", inplace=True)
    data = [(k, v) for k, v in prot["Target"].to_dict().items()]

    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(
            token_representations[i, 1 : len(seq) + 1].mean(0)
        )
    data = [{"x": x} for x in sequence_representations]
    prot["data"] = data
    prot = prot.to_dict("index")
    return prot


def generate_esm_script(prot: pd.DataFrame) -> pd.DataFrame:
    """Create an ESM script for btach processing."""
    prot_ids, seqs = list(zip(*[(k, v) for k, v in prot["Target"].to_dict().items()]))
    os.makedirs("./esms", exist_ok=True)
    with open("./esms/prots.fasta", "w") as fasta:
        for prot_id, seq in zip(prot_ids, seqs):
            fasta.write(f">{prot_id}\n{seq[:1022]}\n")

    esm_parser = create_parser()
    esm_args = esm_parser.parse_args(
        [
            "esm1b_t33_650M_UR50S",
            "esms/prots.fasta",
            "esms/",
            "--repr_layers",
            "33",
            "--include",
            "mean",
        ]
    )
    extract_main(esm_args)
    data = []
    for prot_id in prot_ids:
        data.append(
            {
                "x": torch.load(f"./esms/{prot_id}.pt")["mean_representations"][
                    33
                ].unsqueeze(0)
            }
        )
    # os.rmdir("./esms")
    prot["data"] = data
    # prot = prot.to_dict("index")
    return prot


if __name__ == "__main__":
    prots = pd.read_csv(snakemake.input.seqs, sep="\t").set_index("Target_ID")
    prots = generate_esm_script(prots)
    prots.to_pickle(snakemake.output.pickle)
