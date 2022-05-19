import os

import esm
import pandas as pd
import torch
from extract_esm import create_parser
from extract_esm import main as extract_main


def generate_esm_python(prot_ids:list, seqs:list, batch_size:int=16) -> dict:
    """Return esms."""
    # TODO refactor
    esms = {}
    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    seq_data = [(pid, seq[:1022]) for pid, seq in zip(prot_ids, seqs)]
    for b in range(0, len(seq_data), batch_size):
        # print(f"\r{b}/{len(seq_data)}", end="")
        _, _, batch_tokens = batch_converter(seq_data[b : b + batch_size])

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, (pid, seq) in enumerate(seq_data[b : b + batch_size]):
            esms[pid] = token_representations[i, 1 : len(seq) + 1].mean(0)
    # print()
    return esms


def generate_esm_script(prot_ids:list, seqs:list) -> dict:
    """Generate ESM script."""
    # TODO refactor
    if not os.path.exists("./esms"):
        os.makedirs("./esms", exist_ok=True)
        with open("./esms/prots.fasta", "w") as fasta:
            for prot_id, seq in zip(prot_ids, seqs):
                fasta.write(f">{prot_id}\n{seq[:1022]}\n")

        esm_parser = create_parser()
        esm_args = esm_parser.parse_args(
            ["esm1b_t33_650M_UR50S", "esms/prots.fasta", "esms/", "--repr_layers", "33", "--include", "mean"]
        )
        # print("start")
        extract_main(esm_args)
        # print("finish")

    embeds = {}
    for prot_id in prot_ids:
        repres = torch.load(f"./esms/{prot_id}.pt")["mean_representations"][33]
        embeds[prot_id] = repres
    # os.rmdir("./esms")
    return embeds


class ESMEncoder:
    """Encodes protein data with ESM."""
    def __init__(self, prot_ids, seqs, batch_size=32):
        # self.esms = generate_esm_python(prot_ids, seqs, batch_size)
        self.esms = generate_esm_script(prot_ids, seqs, batch_size)

    def __call__(self, prot_id):
        return {"x": self.esms[prot_id]}


def extract_name(protein_sif: str) -> str:
    """Extract the protein name from the sif filename"""
    return protein_sif.split("/")[-1].split("_")[0]


if __name__ == "__main__":
    prots = pd.read_csv(snakemake.input.seqs, sep="\t")
    esm_encoder = ESMEncoder(prots["Target_ID"], prots["AASeq"])
    prots["data"] = prots["Target_ID"].apply(esm_encoder)
    prots.set_index("Target_ID", inplace=True)
    prots.to_pickle(snakemake.output.pickle)
