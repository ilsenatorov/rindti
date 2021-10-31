import argparse
import os

import pandas as pd
import yaml
from joblib import Parallel, delayed
from prepare_proteins import ProteinEncoder
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str)
parser.add_argument("outdir", type=str)
parser.add_argument("-j", type=int, default=128)
args = parser.parse_args()

for node in ["label"]:
    for edge in ["none"]:
        filename = "pfam_{}_{}.pkl".format(node, edge)
        print("working on " + filename)
        if filename not in os.listdir(args.outdir):
            print("Processing!")
            proteins = pd.Series(list(os.listdir(args.folder)), name="ID")
            proteins = pd.DataFrame(proteins)
            proteins["sif"] = proteins["ID"].apply(lambda x: os.path.join(args.folder, x, x + "_h.sif"))
            proteins.set_index("ID", inplace=True)
            prot_encoder = ProteinEncoder(node, edge)
            data = Parallel(n_jobs=args.j)(delayed(prot_encoder)(i) for i in tqdm(proteins["sif"]))
            proteins["data"] = data
            proteins.to_pickle(os.path.join(args.outdir, filename))
        else:
            print("It's present!")
