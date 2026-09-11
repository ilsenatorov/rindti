"""Regression tests for config naming.

`Namer` decides every artifact filename in the workflow, so a name that depends on
anything other than the config's *content* silently orphans previously built datasets.
"""

import pytest
from snakemake_helper import Namer


@pytest.fixture
def namer():
    return Namer(8)


class TestOrderIndependence:
    """`json.dumps` and `dict` iteration both follow insertion order.

    YAML preserves the order keys were written in, so swapping two lines in a config
    file - a change with no semantic content whatsoever - used to produce a different
    hash, a different filename, and a rebuild of everything downstream.
    """

    def test_flat_key_order_does_not_change_the_name(self, namer):
        a = {"method": "random", "train": 0.7, "val": 0.2}
        b = {"val": 0.2, "method": "random", "train": 0.7}
        assert namer(a) == namer(b)

    def test_nested_key_order_does_not_change_the_name(self, namer):
        a = {"prots": {"features": {"method": "distance", "node_feats": "label"}}}
        b = {"prots": {"features": {"node_feats": "label", "method": "distance"}}}
        assert namer(a) == namer(b)

    def test_letter_prefix_is_order_independent_too(self, namer):
        """The prefix is built by iterating the flattened config, not just the hash."""
        a = {"aaa": "xylophone", "zzz": "banana"}
        b = {"zzz": "banana", "aaa": "xylophone"}
        assert namer(a).split("_")[0] == namer(b).split("_")[0]


class TestNamesStillDiscriminate:
    """Order-independence must not collapse genuinely different configs together."""

    def test_a_changed_value_changes_the_name(self, namer):
        a = {"method": "random", "train": 0.7}
        b = {"method": "target", "train": 0.7}
        assert namer(a) != namer(b)

    def test_a_changed_number_changes_the_name(self, namer):
        """Numbers contribute no letter, so only the hash can distinguish these."""
        a = {"method": "random", "train": 0.7}
        b = {"method": "random", "train": 0.8}
        assert namer(a) != namer(b)

    def test_swapped_values_between_keys_change_the_name(self, namer):
        """Same multiset of values, different assignment - the hash must catch it."""
        a = {"one": "alpha", "two": "beta"}
        b = {"one": "beta", "two": "alpha"}
        assert namer(a) != namer(b)


class TestIgnoredKeys:
    """`source`/`target` are paths, so they contribute no letter to the readable prefix.

    They do still enter the hash. That is deliberate: `davis.yaml` and
    `bindingdb_kd.yaml` are identical apart from `source`, so dropping it would give
    them the same name.
    """

    @pytest.mark.parametrize("key", ["source", "target"])
    def test_location_keys_contribute_no_letter(self, namer, key):
        base = {"method": "random"}
        assert namer({**base, key: "/somewhere/else"}).split("_")[0] == "r"

    @pytest.mark.parametrize("key", ["source", "target"])
    def test_location_keys_still_change_the_hash(self, namer, key):
        base = {"method": "random"}
        assert namer(base) != namer({**base, key: "/somewhere/else"})

    def test_get_name_and_explain_name_agree(self, namer, capsys):
        """The two used to disagree about whether `target` contributes a letter."""
        config = {"method": "random", "source": "a/b", "target": "c/d"}
        explained = namer.explain_name(config)
        capsys.readouterr()
        assert explained == namer.get_name(config)


class TestHashCutoff:
    def test_cutoff_limits_the_hash_length(self):
        config = {"method": "random"}
        assert len(Namer(8)(config).split("_")[1]) == 8
        assert len(Namer(None)(config).split("_")[1]) == 32
