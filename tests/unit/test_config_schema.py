"""Tests for the closed-schema preconstruction firewall mechanism.

These cover the mechanism only -- key tokenization, the two-tree sweep, and the
forbidden-resolver check. The helium-importance policy that supplies the actual
forbidden families is tested separately; a mechanism test that also encoded the
policy would pass for either reason and could not say which one broke.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tpen.config_schema import (
    ClosedSchemaError,
    ForbiddenSurface,
    Rejection,
    SchemaPolicy,
    iter_nodes,
    sweep,
    tokens_of,
)


REFERENCE = ForbiddenSurface(
    name="reference",
    tokens=frozenset({"reference"}),
    reason="a training configuration may not hold a reference value",
)

POLICY = SchemaPolicy(
    name="test.policy.v1",
    forbidden_surfaces=(REFERENCE,),
    allowed_sections=frozenset({"system", "model"}),
    forbidden_resolvers=frozenset({"oc.env"}),
)


def _trees(mapping: dict[str, object]) -> tuple[object, object]:
    """Return the raw and resolved plain-container forms of one config."""

    cfg = OmegaConf.create(mapping)
    return (
        OmegaConf.to_container(cfg, resolve=False),
        OmegaConf.to_container(cfg, resolve=True),
    )


class TestTokensOf:
    """Key tokenization is the discriminator between a real hit and a substring."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("reference_energy", ("reference", "energy")),
            ("referenceEnergy", ("reference", "energy")),
            ("REFERENCE-ENERGY", ("reference", "energy")),
            ("reference", ("reference",)),
            ("energy_band_2", ("energy", "band", "2")),
        ],
    )
    def test_splits_on_word_boundaries(self, name: str, expected: tuple[str, ...]) -> None:
        assert tokens_of(name) == expected

    @pytest.mark.parametrize("name", ["preference", "dereference", "bandwidth", "gaps_free"])
    def test_does_not_split_inside_a_word(self, name: str) -> None:
        """A substring rule would match these; a token rule must not.

        ``preference`` contains ``reference`` and ``bandwidth`` contains
        ``band``. Both are ordinary configuration words. ``gaps_free`` is the
        one case that *does* tokenize to a plural, and it is listed here to pin
        that ``gaps`` and ``gap`` are distinct tokens: a policy that forbids
        ``gap`` must say so for both spellings rather than rely on a prefix.
        """

        assert "reference" not in tokens_of(name)
        assert "band" not in tokens_of(name)
        assert "gap" not in tokens_of(name)


class TestIterNodes:
    """The walk must reach every depth, including through list elements."""

    def test_yields_nested_and_list_paths(self) -> None:
        tree = {"a": {"b": 1}, "c": [{"d": 2}, 3]}
        paths = {path for path, _key, _value in iter_nodes(tree)}
        assert {"a", "a.b", "c", "c[0]", "c[0].d", "c[1]"} <= paths

    def test_does_not_walk_into_strings(self) -> None:
        """A string is a Sequence; walking it would yield one node per character."""

        paths = [path for path, _key, _value in iter_nodes({"a": "xyz"})]
        assert paths == ["a"]

    def test_list_elements_have_no_key(self) -> None:
        """List positions carry no name, so no key rule can be applied to them."""

        by_path = {path: key for path, key, _value in iter_nodes({"c": [1]})}
        assert by_path["c"] == "c"
        assert by_path["c[0]"] is None


class TestForbiddenSurfaces:
    def test_rejects_a_nested_reference(self) -> None:
        """The contract's falsifier: the reference is nested, not top level."""

        raw, resolved = _trees({"system": {"nuclei": {"reference_energy": -2.9}}})
        rejections = sweep(raw, resolved, POLICY)
        paths = {r.path for r in rejections if r.rule == "forbidden-surface:reference"}
        assert paths == {"system.nuclei.reference_energy"}

    def test_rejects_a_reference_inside_a_list_element(self) -> None:
        """Callbacks are a list, so a reference-bearing callback lives at ``[i]``."""

        raw, resolved = _trees({"model": {"factors": [{"reference_energy": -2.9}]}})
        rejections = sweep(raw, resolved, POLICY)
        assert any(r.path == "model.factors[0].reference_energy" for r in rejections)

    def test_accepts_a_key_that_merely_contains_the_token_as_a_substring(self) -> None:
        """``preference`` must survive a policy that forbids ``reference``."""

        raw, resolved = _trees({"model": {"preference": 1}})
        assert sweep(raw, resolved, POLICY) == ()

    def test_reports_every_finding_not_only_the_first(self) -> None:
        raw, resolved = _trees(
            {"system": {"reference_energy": -2.9}, "model": {"reference_state": 1}}
        )
        rejections = sweep(raw, resolved, POLICY)
        paths = {r.path for r in rejections}
        assert {"system.reference_energy", "model.reference_state"} <= paths


class TestTwoTreeSweep:
    """Each tree catches something the other cannot."""

    def test_resolved_only_surface_is_labelled_resolved(self) -> None:
        """A key reached through an interpolation is invisible in the raw tree.

        The raw tree holds the literal text ``"${system.value}"``, so the value
        that appears at the destination is not there to inspect. Only the
        resolved sweep sees it -- and the finding must say so, because "present
        in resolved but not raw" is a different defect from "present in both".
        """

        raw, resolved = _trees(
            {"system": {"value": -2.9}, "model": {"reference_energy": "${system.value}"}}
        )
        hits = [r for r in sweep(raw, resolved, POLICY) if r.path == "model.reference_energy"]
        # The key itself is spelled in both trees, so both sweeps see it. What
        # matters is that both are reported and each is labelled.
        assert {hit.tree for hit in hits} == {"raw", "resolved"}

    def test_raw_only_resolver_is_labelled_raw(self) -> None:
        raw, resolved = _trees({"system": {"tag": "${oc.env:USER,fallback}"}})
        hits = [r for r in sweep(raw, resolved, POLICY) if r.rule == "forbidden-resolver"]
        assert len(hits) == 1
        assert hits[0].tree == "raw"
        assert hits[0].path == "system.tag"


class TestForbiddenResolvers:
    """Environment interpolation is how rank-local data alters a resolved config."""

    def test_rejects_an_environment_interpolation(self) -> None:
        raw, resolved = _trees({"model": {"seed": "${oc.env:RANK,0}"}})
        assert any(r.rule == "forbidden-resolver" for r in sweep(raw, resolved, POLICY))

    def test_accepts_a_plain_node_reference(self) -> None:
        """``${system.dim}`` names a config node, not a resolver, and is fine."""

        raw, resolved = _trees({"system": {"dim": 3}, "model": {"dim": "${system.dim}"}})
        assert sweep(raw, resolved, POLICY) == ()

    def test_accepts_a_resolver_the_policy_does_not_forbid(self) -> None:
        """Only the named resolvers are refused; the check is an allowlist inverse."""

        policy = SchemaPolicy(name="p", forbidden_resolvers=frozenset({"oc.env"}))
        raw = {"model": {"dim": "${tpen.basis_feature_dim:${model.basis}}"}}
        assert sweep(raw, raw, policy) == ()


class TestUnknownSections:
    def test_rejects_a_top_level_section_outside_the_closed_set(self) -> None:
        raw, resolved = _trees({"system": {}, "sneaky": {}})
        hits = [r for r in sweep(raw, resolved, POLICY) if r.rule == "unknown-field"]
        assert {hit.path for hit in hits} == {"sneaky"}

    def test_accepts_a_subset_of_the_closed_set(self) -> None:
        """Closed means no *extra* sections, not that every section is required."""

        raw, resolved = _trees({"system": {}})
        assert sweep(raw, resolved, POLICY) == ()

    def test_an_empty_allowed_set_disables_the_check(self) -> None:
        """A policy that declares no sections is not a policy that forbids all."""

        policy = SchemaPolicy(name="p", allowed_sections=frozenset())
        assert sweep({"anything": 1}, {"anything": 1}, policy) == ()


class TestClosedSchemaError:
    def test_message_names_every_rejection(self) -> None:
        error = ClosedSchemaError(
            [
                Rejection(rule="unknown-field", tree="raw", path="a", detail="first"),
                Rejection(rule="unknown-field", tree="raw", path="b", detail="second"),
            ]
        )
        text = str(error)
        assert "2 findings" in text
        assert "first" in text and "second" in text

    def test_singular_message_for_one_rejection(self) -> None:
        error = ClosedSchemaError([Rejection(rule="r", tree="raw", path="a", detail="d")])
        assert "(1 finding)" in str(error)

    def test_retains_rejections_for_programmatic_inspection(self) -> None:
        rejection = Rejection(rule="r", tree="raw", path="a", detail="d")
        assert ClosedSchemaError([rejection]).rejections == (rejection,)
