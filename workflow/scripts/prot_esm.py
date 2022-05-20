import esm
import pandas as pd
import torch


def generate_esm_python(prot: pd.DataFrame) -> pd.DataFrame:
    """Return esms."""

    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    prot.set_index("Target_ID", inplace=True)
    data = [(k, v) for k, v in prot["Target"].to_dict().items()]

    _, _, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
    data = [{"x": x} for x in sequence_representations]
    prot["data"] = data
    return prot


if __name__ == "__main__":
    prots = pd.read_csv(snakemake.input.seqs, sep="\t")
    prots = generate_esm_python(prots)
    prots.to_pickle(snakemake.output.pickle)
