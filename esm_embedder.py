import os
import pickle
import sys

import esm
import numpy as np
import torch
import umap
from Bio import SeqIO
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

"""
Usage:
fist argument: 
    filename/-path of the plot
next up to four arguments: 
    filenames/-paths of fasta files starting with the ground distribution

Example call: 
python esm_embedder.py lectins_esm lectins.fasta ricin.fasta
"""

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

seqs = []
sizes = [0]
embeds = []
names = []

for fasta_file in sys.argv[2:]:
    print(f"Reading {fasta_file}")
    count = 0
    names.append(fasta_file[:-6])
    if os.path.exists(fasta_file[:-5] + "pkl"):
        pkl_embeds = pickle.load(open(fasta_file[:-5] + "pkl", "rb"))
        sizes.append(len(pkl_embeds) + sizes[-1])
        embeds += pkl_embeds
    else:
        for fasta in SeqIO.parse(open(sys.argv[1]), "fasta"):
            count += 1
            print(f"\r\tSeq. Nr. {count}", end="")
            idx, seq = fasta.id, str(fasta.seq)
            _, _, batch_tokens = batch_converter([(idx, seq[:1022])])

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_rep = results["representations"][33]

            seqs.append(seq)
            embeds.append(token_rep[0, 1: len(seq) + 1].mean(0).numpy())

        print("\rDone")
        sizes.append(count + sizes[-1])

        with open(fasta_file[:-5] + "pkl", "wb") as store:
            pickle.dump(embeds[sizes[-2]:], store)

tsne_embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(
    np.array(embeds)
)
umap_embeds = umap.UMAP().fit_transform(
    np.array(embeds)
)

fig, ax = plt.subplots(1, 2, figsize=(12, 8))
colors = ["lightgray", "r", "g", "b"]
for i in range(1, len(sizes)):
    ax[0].scatter(tsne_embeds[sizes[i-1]:sizes[i], 0], tsne_embeds[sizes[i-1]:sizes[i]:, 1], color=colors[i - 1], marker="o", s=20, label=names[i-1])
    ax[1].scatter(umap_embeds[sizes[i-1]:sizes[i], 0], umap_embeds[sizes[i-1]:sizes[i]:, 1], color=colors[i - 1], marker="o", s=20, label=names[i-1])
ax[0].title.set_text("ESM/tSNE embedding")
ax[1].title.set_text("ESM/UMAP embedding")
plt.legend()
plt.savefig(f"{sys.argv[1]}.png")
plt.clf()
