"""Regression tests for the pure helpers in workflow/scripts."""

import pandas as pd
import pytest
from distance_based import Structure, encode_residue
from encd import encd
from split_data import split_random
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
