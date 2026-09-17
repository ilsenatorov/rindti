"""Tests for the gene-name -> UniProt -> AlphaFold recovery path.

The scoring function is the whole safety mechanism here: it is what stops a target's
interactions being attached to the wrong protein when a gene alias resolves to a
plausible-looking but different entry. Both of its non-obvious properties are pinned
below, because both were wrong in the first draft and neither is visible by reading.
"""

import pytest
from fetch_alphafold import ALIASES, gene_candidates, score


class TestScore:
    def test_identical_is_one(self):
        assert score("MKKFFDSRRE", "MKKFFDSRRE") == 1.0

    def test_empty_is_zero(self):
        assert score("", "MKKF") == 0.0
        assert score("MKKF", "") == 0.0

    def test_construct_inside_full_length_scores_high(self):
        """Davis ships isolated kinase domains; they must not be penalised for length."""
        domain = "ACDEFGHIKLMNPQRSTVWY" * 15
        full = "M" * 400 + domain + "K" * 400
        assert score(domain, full) >= 0.99

    def test_amino_acid_alphabet_does_not_collapse_the_score(self):
        """difflib's autojunk heuristic treats every residue as junk above 200 chars.

        Any character occurring in more than 1% of a sequence of length >= 200 is
        discarded, and a 20-letter alphabet guarantees that for all of them. With the
        default, two near-identical proteins score ~0.001.
        """
        a = ("ACDEFGHIKLMNPQRSTVWY" * 30)[:600]
        b = a[:300] + "W" + a[301:]  # one substitution
        assert score(a, b) > 0.9

    def test_unrelated_sequences_stay_low(self):
        """The rejection that matters: a wrong gene must not clear the threshold."""
        a = "ACDEFGHIKLM" * 40
        b = "WWWWYYYYPPP" * 40
        assert score(a, b) < 0.5

    def test_coverage_is_measured_against_the_shorter_sequence(self):
        short = "ACDEFGHIKLMNPQRSTVWY" * 10
        long = short + "".join("GSGS" for _ in range(200))
        assert score(short, long) >= 0.99


class TestGeneCandidates:
    @pytest.mark.parametrize(
        "target_id,expected_first",
        [
            ("AAK1", "AAK1"),
            ("BRAF(V600E)", "BRAF"),  # parenthesised mutation
            ("JAK1(JH1domain-catalytic)", "JAK1"),  # parenthesised domain
        ],
    )
    def test_decorations_are_stripped(self, target_id, expected_first):
        assert gene_candidates(target_id)[0] == expected_first

    def test_phospho_suffix_is_offered_as_a_fallback(self):
        # ABL1p is the phosphorylated form of ABL1, not a different gene.
        assert "ABL1" in gene_candidates("ABL1p")

    def test_alias_is_tried_before_the_literal_name(self):
        # "S6K1" finds an unrelated entry; RPS6KB1 is the real symbol.
        assert gene_candidates("S6K1")[0] == "RPS6KB1"

    def test_candidates_are_unique(self):
        for target_id in ("AAK1", "ABL1p", "S6K1", "BRAF(V600E)"):
            candidates = gene_candidates(target_id)
            assert len(candidates) == len(set(candidates))

    def test_every_alias_maps_to_a_plausible_symbol(self):
        for alias, symbol in ALIASES.items():
            assert symbol and symbol != alias
            assert symbol.isupper() or any(c.isdigit() for c in symbol)
