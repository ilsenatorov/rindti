import numpy as np
import pandas as pd
from scipy import sparse

from .base_baseline import BaseBaseline


def kmer_matrix(sequences: list[str], k: int = 3) -> sparse.csr_matrix:
    """Binary presence matrix of the k-mers in each sequence, one row per sequence."""
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for row, seq in enumerate(sequences):
        kmers = {seq[i : i + k] for i in range(len(seq) - k + 1)}
        for kmer in kmers:
            rows.append(row)
            cols.append(vocab.setdefault(kmer, len(vocab)))
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(sequences), max(len(vocab), 1)))


def ecfp_matrix(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """ECFP4 bit matrix, the fingerprint ``cluster_drug`` splits on.

    An unparseable SMILES gets an all-zero row, so it is similar to nothing.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for row, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out[row] = generator.GetFingerprintAsNumPy(mol)
    return out


def tanimoto(a, b) -> np.ndarray:
    """Tanimoto similarity between every row of binary matrix ``a`` and of ``b``.

    Two empty rows have no shared bits and no union; they score 0, not NaN.
    """
    inter = a @ b.T
    inter = np.asarray(inter.todense() if sparse.issparse(inter) else inter, dtype=np.float32)
    na = np.asarray(a.sum(axis=1)).reshape(-1, 1)
    nb = np.asarray(b.sum(axis=1)).reshape(1, -1)
    union = na + nb - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class KNN(BaseBaseline):
    """Similarity-weighted k nearest training pairs.

    A test pair (p, d) is scored against every training pair (p', d') by
    ``S_prot(p, p') * S_drug(d, d')``, where both are Tanimoto similarities: on 3-mer
    sets for proteins and on ECFP4 for drugs. The prediction is the similarity-weighted
    mean label of the ``k`` highest-scoring training pairs.

    Unlike ``Max`` and ``ProtDrugMax`` it reads features, so it is not part of the
    label-only floor that ``report`` writes and BOSS is measured against.

    If no training pair has any similarity, the prediction falls back to the train mean.
    """

    def __init__(
        self,
        prots: str | dict,
        drugs: str | dict,
        k: int = 10,
        kmer: int = 3,
        task: str = "class",
        chunk: int = 512,
        **kwargs,
    ):
        super().__init__(prob=False, task=task)
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.sequences = self._table(prots, "Target_ID", "Target")
        self.smiles = self._table(drugs, "Drug_ID", "Drug")
        self.k = k
        self.kmer = kmer
        self.chunk = chunk

    @staticmethod
    def _table(source: str | dict, id_col: str, value_col: str) -> dict:
        if isinstance(source, dict):
            return source
        table = pd.read_csv(source, sep="\t", usecols=[id_col, value_col])
        return dict(zip(table[id_col], table[value_col], strict=True))

    def _lookup(self, ids, table: dict, kind: str) -> list[str]:
        missing = [i for i in ids if i not in table]
        if missing:
            raise KeyError(f"{len(missing)} {kind} IDs are not in the {kind} table, e.g. {missing[:5]}")
        return [table[i] for i in ids]

    def fit(self, train: pd.DataFrame):
        """Fit the model to the training dataframe. Has to have 'Drug_ID', 'Target_ID' and 'Y' columns."""
        self.train_prots = sorted(train["Target_ID"].unique())
        self.train_drugs = sorted(train["Drug_ID"].unique())
        prot_index = {p: i for i, p in enumerate(self.train_prots)}
        drug_index = {d: i for i, d in enumerate(self.train_drugs)}
        self.pair_prot = train["Target_ID"].map(prot_index).to_numpy()
        self.pair_drug = train["Drug_ID"].map(drug_index).to_numpy()
        self.y = train["Y"].to_numpy(dtype=np.float64)
        self.fallback = float(self.y.mean())

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Apply prediction to the whole test dataframe."""
        test_prots = sorted(test["Target_ID"].unique())
        test_drugs = sorted(test["Drug_ID"].unique())

        # The k-mer vocabulary has to be shared, so both sides go through one matrix.
        prot_seqs = self._lookup(self.train_prots + test_prots, self.sequences, "protein")
        kmers = kmer_matrix(prot_seqs, self.kmer)
        n = len(self.train_prots)
        s_prot = tanimoto(kmers[n:], kmers[:n])

        fps_train = ecfp_matrix(self._lookup(self.train_drugs, self.smiles, "drug"))
        fps_test = ecfp_matrix(self._lookup(test_drugs, self.smiles, "drug"))
        s_drug = tanimoto(fps_test, fps_train)

        t_prot = test["Target_ID"].map({p: i for i, p in enumerate(test_prots)}).to_numpy()
        t_drug = test["Drug_ID"].map({d: i for i, d in enumerate(test_drugs)}).to_numpy()
        k = min(self.k, len(self.y))
        preds = np.empty(len(test))
        for start in range(0, len(test), self.chunk):
            stop = start + self.chunk
            # (chunk, n_train_pairs) similarity of each test pair to each training pair.
            w = s_prot[t_prot[start:stop]][:, self.pair_prot] * s_drug[t_drug[start:stop]][:, self.pair_drug]
            top = np.argpartition(-w, k - 1, axis=1)[:, :k]
            w_top = np.take_along_axis(w, top, axis=1)
            y_top = self.y[top]
            total = w_top.sum(axis=1)
            weighted = (w_top * y_top).sum(axis=1)
            preds[start:stop] = np.divide(weighted, total, out=np.full(len(total), self.fallback), where=total > 0)
        test["pred"] = preds
        return test
