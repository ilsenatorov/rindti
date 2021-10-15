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

for node in ["label", "onehot"]:
    for edge in ["label", "onehot", "none"]:
        proteins = pd.Series(list(os.listdir(args.folder)), name="ID")
        proteins = pd.DataFrame(proteins)
        proteins["sif"] = proteins["ID"].apply(lambda x: os.path.join(args.folder, x, x + "_h.sif"))
        proteins.set_index("ID", inplace=True)
        prot_encoder = ProteinEncoder(node, edge)
        data = Parallel(n_jobs=args.j)(delayed(prot_encoder)(i) for i in tqdm(proteins["sif"]))
        proteins["data"] = data
        proteins.to_pickle(os.path.join(args.outdir, "pfam_{}_{}.pkl".format(node, edge)))
