"""Regression tests for the pure helpers in workflow/scripts."""

import pandas as pd
import pytest
from cluster import cluster_drugs, dedup_sequences, parse_mmseqs_clusters, write_fasta
from distance_based import Structure, encode_residue
from encd import encd
from split_data import GROUP_COL, add_group_column, split_groups, split_random
from utils import onehot_encode


class TestOnehotEncode:
    """`t[position - 1] = 1` made index 0 wrap onto the last slot."""

    def test_first_class_sets_first_slot(self):
        assert onehot_encode(0, 4) == [1, 0, 0, 0]

    def test_last_class_sets_last_slot(self):
        assert onehot_encode(3, 4) == [0, 0, 0, 1]

    def test_first_and_last_do_not_collide(self):
        assert onehot_encode(0, 4) != onehot_encode(3, 4)

    def test_unknown_is_all_zeros(self):
        assert onehot_encode(None, 4) == [0, 0, 0, 0]

    def test_out_of_range_rejected(self):
        with pytest.raises(IndexError):
            onehot_encode(4, 4)

    def test_every_residue_gets_a_distinct_vector(self):
        vocab = encd["prot"]["node"]
        vectors = {tuple(onehot_encode(i, len(vocab))) for i in vocab.values()}
        assert len(vectors) == len(vocab)


class TestEncodeResidue:
    def test_label_is_one_based(self):
        """0 is reserved as the embedding's padding/unknown index."""
        assert encode_residue("ala", "label") == encd["prot"]["node"]["ala"] + 1

    def test_unknown_residue_label(self):
        """Non-standard residues used to raise KeyError on encd['prot']['node']['unk']."""
        assert encode_residue("xyz", "label") == 0

    def test_unknown_residue_onehot(self):
        assert encode_residue("xyz", "onehot") == [0] * len(encd["prot"]["node"])

    def test_case_insensitive(self):
        assert encode_residue("ALA", "label") == encode_residue("ala", "label")


class TestMultiChain:
    """Residue numbering restarts per chain, so keying on number alone collapsed chains."""

    @staticmethod
    def _atom(chain, num, name="ALA", x=0.0):
        # Columns per the PDB spec: name 13-16, resName 18-20, chainID 22, resSeq 23-26.
        return f"ATOM  {1:>5}  CA  {name} {chain}{num:>4}    {x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00 50.00           C\n"

    def test_two_chains_stay_distinct(self, tmp_path):
        pdb = tmp_path / "two_chains.pdb"
        pdb.write_text(
            self._atom("A", 1, x=0.0)
            + self._atom("A", 2, x=1.0)
            + self._atom("B", 1, x=2.0)
            + self._atom("B", 2, x=3.0)
        )
        struct = Structure(str(pdb), "label")
        assert len(struct.residues) == 4, "chains A and B both number from 1"
        assert struct.get_coords().shape == (4, 3)

    def test_single_chain_unaffected(self, tmp_path):
        pdb = tmp_path / "one_chain.pdb"
        pdb.write_text(self._atom("A", 1) + self._atom("A", 2))
        assert len(Structure(str(pdb), "label").residues) == 2


class TestSplitRandom:
    """`val_frac` is a fraction of the whole, but was applied to the post-train remainder."""

    @pytest.fixture
    def inter(self):
        return pd.DataFrame({"Drug_ID": [f"D{i}" for i in range(1000)], "Y": [0, 1] * 500})

    @pytest.mark.parametrize(
        ("train_frac", "val_frac"),
        [(0.7, 0.2), (0.8, 0.1), (0.6, 0.2)],
    )
    def test_realised_fractions_match_config(self, inter, train_frac, val_frac):
        out = split_random(inter, train_frac=train_frac, val_frac=val_frac)
        counts = out["split"].value_counts(normalize=True)
        assert counts["train"] == pytest.approx(train_frac, abs=0.02)
        assert counts["val"] == pytest.approx(val_frac, abs=0.02)
        assert counts["test"] == pytest.approx(1 - train_frac - val_frac, abs=0.02)

    def test_nothing_lost_or_duplicated(self, inter):
        out = split_random(inter, train_frac=0.7, val_frac=0.2)
        assert len(out) == len(inter)
        assert set(out["Drug_ID"]) == set(inter["Drug_ID"])


class TestDedupSequences:
    """W6: Davis carries exact duplicate sequences under different Target_IDs, so a
    plain cold-target split trains and tests on the same protein."""

    def test_duplicates_share_a_representative(self):
        groups = dedup_sequences({"A": "MKV", "B": "MKV", "C": "AAA"})
        assert groups["A"] == groups["B"]
        assert groups["C"] != groups["A"]

    def test_every_id_survives(self):
        """Rows must not be dropped: prepare_all filters interactions against the
        protein pickle, so a lost Target_ID silently shrinks the dataset."""
        seqs = {"A": "MKV", "B": "MKV", "C": "AAA"}
        assert dedup_sequences(seqs).keys() == seqs.keys()

    def test_representative_is_deterministic(self):
        """Alphabetically first, not insertion-ordered - the clustering happens
        upstream of the split seed, so it has to be reproducible on its own."""
        assert dedup_sequences({"B": "MKV", "A": "MKV"})["B"] == "A"
        assert dedup_sequences({"A": "MKV", "B": "MKV"})["B"] == "A"


class TestClusterDrugs:
    """Leader clustering over ECFP4/Tanimoto. Never materialises the full matrix:
    GLASS has 165,691 ligands, i.e. ~55 GB of condensed float32 distances."""

    ETHANOL = "CCO"
    PROPANOL = "CCCO"
    BENZENE = "c1ccccc1"

    def test_identical_molecules_cluster_together(self):
        groups = cluster_drugs({"a": self.ETHANOL, "b": self.ETHANOL}, cutoff=0.6)
        assert groups["a"] == groups["b"]

    def test_dissimilar_molecules_do_not(self):
        groups = cluster_drugs({"a": self.ETHANOL, "b": self.BENZENE}, cutoff=0.6)
        assert groups["a"] != groups["b"]

    def test_cutoff_of_one_only_merges_identical(self):
        groups = cluster_drugs({"a": self.ETHANOL, "b": self.PROPANOL}, cutoff=1.0)
        assert groups["a"] != groups["b"]

    def test_unparseable_smiles_gets_its_own_cluster(self):
        """It cannot be shown similar to anything, and it must not be dropped."""
        groups = cluster_drugs({"a": self.ETHANOL, "bad": "not a molecule"}, cutoff=0.6)
        assert groups["bad"] == "bad"
        assert groups.keys() == {"a", "bad"}

    def test_deterministic_across_input_order(self):
        mols = {"a": self.ETHANOL, "b": self.PROPANOL, "c": self.BENZENE}
        forward = cluster_drugs(mols, cutoff=0.4)
        backward = cluster_drugs(dict(reversed(list(mols.items()))), cutoff=0.4)
        assert forward == backward

    def test_rejects_a_nonsense_cutoff(self):
        with pytest.raises(ValueError, match="drug_similarity"):
            cluster_drugs({"a": self.ETHANOL}, cutoff=0)


class TestMmseqsIO:
    """Target_IDs contain dots and parentheses (Davis `RSK1(KinDom.1-N-terminal)`),
    which MMseqs2 mangles, so FASTA headers are surrogates mapped back afterwards."""

    def test_roundtrip_through_surrogate_ids(self, tmp_path):
        seqs = {"RSK1(KinDom.1-N-terminal)": "MKV", "plain": "MKV"}
        fasta = tmp_path / "p.fasta"
        mapping = write_fasta(seqs, str(fasta))
        assert set(mapping.values()) == set(seqs)
        assert "(" not in fasta.read_text().split("\n")[0]

        clusters = tmp_path / "c.tsv"
        surrogates = sorted(mapping)
        clusters.write_text("".join(f"{surrogates[0]}\t{s}\n" for s in surrogates))
        assignment = parse_mmseqs_clusters(str(clusters), mapping)
        assert len(set(assignment.values())) == 1
        assert assignment.keys() == seqs.keys()

    def test_unassigned_sequence_is_an_error(self, tmp_path):
        """Silently dropping a target would shrink the dataset downstream."""
        seqs = {"A": "MKV", "B": "AAA"}
        fasta = tmp_path / "p.fasta"
        mapping = write_fasta(seqs, str(fasta))
        clusters = tmp_path / "c.tsv"
        clusters.write_text("s0\ts0\n")
        with pytest.raises(ValueError, match="did not assign"):
            parse_mmseqs_clusters(str(clusters), mapping)


class TestGroupedSplit:
    """W5/W6 reduce to: attach a grouping column, then split on it. `split_groups`
    already accepts any column, so nothing about the split itself changes."""

    @pytest.fixture
    def inter(self):
        # 60 targets, paired up into 30 groups.
        targets = [f"T{i}" for i in range(60)]
        return pd.DataFrame({"Target_ID": targets * 5, "Y": [0, 1] * 150})

    @pytest.fixture
    def assignment(self):
        return {f"T{i}": f"T{i - i % 2}" for i in range(60)}

    def test_no_group_spans_two_splits(self, inter, assignment):
        """The guarantee the cold-cluster split exists to provide."""
        out = split_groups(add_group_column(inter, "Target_ID", assignment), col_name=GROUP_COL)
        assert (out.groupby(GROUP_COL)["split"].nunique() == 1).all()

    def test_nothing_lost_or_duplicated(self, inter, assignment):
        out = split_groups(add_group_column(inter, "Target_ID", assignment), col_name=GROUP_COL)
        assert len(out) == len(inter)
        assert set(out["Target_ID"]) == set(inter["Target_ID"])

    def test_unknown_id_becomes_its_own_group(self, inter):
        """An ID missing from the assignment must not be dropped or crash."""
        out = add_group_column(inter, "Target_ID", {"T0": "T0"})
        assert out.loc[out["Target_ID"] == "T5", GROUP_COL].unique().tolist() == ["T5"]
        assert len(out) == len(inter)

    def test_all_three_splits_are_populated(self, inter, assignment):
        out = split_groups(add_group_column(inter, "Target_ID", assignment), col_name=GROUP_COL)
        assert set(out["split"]) == {"train", "val", "test"}
