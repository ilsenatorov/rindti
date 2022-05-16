import os
from typing import Tuple

import esm
import numpy as np
import pandas as pd
import torch
from extract_esm import create_parser
from extract_esm import main as extract_main
from utils import list_to_dict, onehot_encode

node_encoding = list_to_dict(
    [
        "ala",
        "arg",
        "asn",
        "asp",
        "cys",
        "gln",
        "glu",
        "gly",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "pro",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
    ]
)

edge_encoding = list_to_dict(["cnt", "combi", "hbond", "pept", "ovl"])


def generate_esm_python(prot_ids, seqs, batch_size):
    esms = {}
    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    seq_data = [(pid, seq[:1022]) for pid, seq in zip(prot_ids, seqs)]
    for b in range(0, len(seq_data), batch_size):
        print(f"\r{b}/{len(seq_data)}", end="")
        batch_labels, batch_strs, batch_tokens = batch_converter(seq_data[b : b + batch_size])

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, (pid, seq) in enumerate(seq_data[b : b + batch_size]):
            esms[pid] = token_representations[i, 1 : len(seq) + 1].mean(0)
    print()
    return esms


def generate_esm_script(prot_ids, seqs, batch_size):
    if not os.path.exists("./esms"):
        os.makedirs("./esms", exist_ok=True)
        with open("./esms/prots.fasta", "w") as fasta:
            for prot_id, seq in zip(prot_ids, seqs):
                fasta.write(f">{prot_id}\n{seq[:1022]}\n")

        esm_parser = create_parser()
        esm_args = esm_parser.parse_args(
            ["esm1b_t33_650M_UR50S", "esms/prots.fasta", "esms/", "--repr_layers", "33", "--include", "mean"]
        )
        print("start")
        extract_main(esm_args)
        print("finish")

    embeds = {}
    for prot_id in prot_ids:
        repres = torch.load(f"./esms/{prot_id}.pt")["mean_representations"][33]
        embeds[prot_id] = repres
    # os.rmdir("./esms")
    return embeds


class ESMEncoder:
    def __init__(self, prot_ids, seqs, batch_size=32):
        # self.esms = generate_esm_python(prot_ids, seqs, batch_size)
        self.esms = generate_esm_script(prot_ids, seqs, batch_size)

    def __call__(self, prot_id):
        return {"x": self.esms[prot_id]}


class ProteinEncoder:
    def __init__(self, node_feats: str, edge_feats: str):
        self.node_feats = node_feats
        self.edge_feats = edge_feats

    def encode_residue(self, residue: str) -> np.array:
        """Fully encode residue - one-hot and node_feats
        Args:
            residue (str): One-letter residue name
        Returns:
            np.array: Concatenated node_feats and one-hot encoding of residue name
        """
        residue = residue.lower()
        if self.node_feats == "label":
            return node_encoding[residue] + 1
        elif self.node_feats == "onehot":
            return onehot_encode(node_encoding[residue], len(node_encoding))
        else:
            raise ValueError("Unknown node_feats type!")

    def parse_sif(self, filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse a single sif file
        Args:
            filename (str): SIF file location
        Returns:
            Tuple[DataFrame, DataFrame]: nodes, edges DataFrames
        """
        nodes = []
        edges = []
        if not os.path.exists(filename):
            return None, None
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                splitline = line.split()
                if len(splitline) != 3:
                    continue
                node1, edgetype, node2 = splitline
                node1split = node1.split(":")
                node2split = node2.split(":")
                if len(node1split) != 4:
                    continue
                if len(node2split) != 4:
                    continue
                chain1, resn1, x1, resaa1 = node1split
                chain2, resn2, x2, resaa2 = node2split
                if x1 != "_" or x2 != "_":
                    continue
                if resaa1.lower() not in node_encoding or resaa2.lower() not in node_encoding:
                    continue
                resn1 = int(resn1)
                resn2 = int(resn2)
                if resn1 == resn2:
                    continue
                edgesplit = edgetype.split(":")
                if len(edgesplit) != 2:
                    continue
                node1 = {"chain": chain1, "resn": resn1, "resaa": resaa1}
                node2 = {"chain": chain2, "resn": resn2, "resaa": resaa2}
                edgetype, _ = edgesplit
                edge1 = {
                    "resn1": resn1,
                    "resn2": resn2,
                    "type": edgetype,
                }
                edge2 = {
                    "resn1": resn2,
                    "resn2": resn1,
                    "type": edgetype,
                }
                nodes.append(node1)
                nodes.append(node2)
                edges.append(edge1)
                edges.append(edge2)
        nodes = pd.DataFrame(nodes).drop_duplicates()
        try:
            nodes = nodes.sort_values("resn").reset_index(drop=True).reset_index().set_index("resn")
        except Exception as e:
            print(nodes)
            print(filename)
            print(e)
            return None, None
        for node in nodes.index:
            if (node - 1) in nodes.index:
                edges.append({"resn1": node, "resn2": node - 1, "type": "pept"})
                edges.append({"resn2": node, "resn1": node - 1, "type": "pept"})
        edges = pd.DataFrame(edges).drop_duplicates()
        node_idx = nodes["index"].to_dict()
        edges["node1"] = edges["resn1"].apply(lambda x: node_idx[x])
        edges["node2"] = edges["resn2"].apply(lambda x: node_idx[x])
        return nodes, edges

    def encode_nodes(self, nodes: pd.DataFrame) -> torch.Tensor:
        """Given dataframe of nodes create node node_feats
        Args:
            nodes (pd.DataFrame): nodes dataframe from parse_sif
        Returns:
            torch.Tensor: Tensor of node node_feats [n_nodes, *]
        """
        nodes.drop_duplicates(inplace=True)
        node_attr = [self.encode_residue(x) for x in nodes["resaa"]]
        node_attr = np.asarray(node_attr)
        if self.node_feats == "label":
            node_attr = torch.tensor(node_attr, dtype=torch.long)
        else:
            node_attr = torch.tensor(node_attr, dtype=torch.float32)
        return node_attr

    def encode_edges(self, edges: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given dataframe of edges, create edge index and edge node_feats
        Args:
            edges (pd.DataFrame): edges dataframe from parse_sif
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: edge index [2,n_edges], edge attributes [n_edges, *]
        """
        if self.edge_feats == "none":
            edges.drop("type", axis=1, inplace=True)
        edges.drop_duplicates(inplace=True)
        edge_index = edges[["node1", "node2"]].astype(int).values
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        if self.edge_feats == "none":
            return edge_index, None
        edge_feats = edges["type"].apply(lambda x: edge_encoding[x])
        if self.edge_feats == "label":
            edge_feats = torch.tensor(edge_feats, dtype=torch.long)
            return edge_index, edge_feats
        elif self.edge_feats == "onehot":
            edge_feats = edge_feats.apply(onehot_encode, count=len(edge_encoding))
            edge_feats = torch.tensor(edge_feats, dtype=torch.float)
            return edge_index, edge_feats

    def __call__(self, protein_sif: str) -> dict:
        """Fully process the protein
        Args:
            protein_sif (str): File location for sif file
        Returns:
            dict: standard format with x for node node_feats, edge_index for edges etc
        """
        try:
            nodes, edges = self.parse_sif(protein_sif)
            if nodes is None:
                return np.nan
            node_attr = self.encode_nodes(nodes)
            edge_index, edge_feats = self.encode_edges(edges)
            return dict(
                x=node_attr,
                edge_index=edge_index,
                edge_feats=edge_feats,
                # index_mapping=nodes["index"].to_dict(),
            )
        except Exception as e:
            print(protein_sif)
            print(e)
            return np.nan


def extract_name(protein_sif: str) -> str:
    """Extract the protein name from the sif filename"""
    return protein_sif.split("/")[-1].split("_")[0]


if __name__ == "__main__":
    if "snakemake" in globals():
        if snakemake.config["prepare_prots"]["node_feats"] == "esm":
            prots = pd.read_csv(snakemake.input.seqs, sep="\t")
            esm_encoder = ESMEncoder(prots["Target_ID"], prots["AASeq"])
            prots["data"] = prots["Target_ID"].apply(esm_encoder)
            prots.set_index("Target_ID", inplace=True)
        else:
            prots = pd.Series(list(snakemake.input.rins), name="sif")
            prots = pd.DataFrame(prots)
            prots["ID"] = prots["sif"].apply(extract_name)
            prots.set_index("ID", inplace=True)
            prot_encoder = ProteinEncoder(
                snakemake.config["prepare_prots"]["node_feats"], snakemake.config["prepare_prots"]["edge_feats"]
            )
            prots["data"] = prots["sif"].apply(prot_encoder)
        prots.to_pickle(snakemake.output.protein_pickle)
    else:
        import argparse

        from joblib import Parallel, delayed
        from tqdm import tqdm

        parser = argparse.ArgumentParser(description="Prepare protein data from rinerator")
        parser.add_argument("--sifs", nargs="+", required=True, help="Rinerator output folders")
        parser.add_argument("--output", required=True, help="Output pickle file")
        parser.add_argument("--node_feats", type=str, default="label")
        parser.add_argument("--edge_feats", type=str, default="none")
        parser.add_argument("--threads", type=int, default=1, help="Number of threads to use")
        args = parser.parse_args()

        prots = pd.DataFrame(pd.Series(args.sifs, name="sif"))
        prots["ID"] = prots["sif"].apply(extract_name)
        prots.set_index("ID", inplace=True)
        prot_encoder = ProteinEncoder(args.node_feats, args.edge_feats)
        data = Parallel(n_jobs=args.threads)(delayed(prot_encoder)(i) for i in tqdm(prots["sif"]))
        prots["data"] = data
        prots.to_pickle(args.output)
