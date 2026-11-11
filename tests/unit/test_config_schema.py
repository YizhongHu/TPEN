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
    iter_interpolations,
    iter_nodes,
    split_resolver,
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


class TestIterInterpolations:
    """The brace walker, tested apart from the policy that consumes it.

    WHY THIS EXISTS AS A SEPARATE CLASS: the sweep can only report resolvers it
    is shown, so a sweep test that passes says the sweep is fine about the forms
    the enumerator happens to hand it. Measuring the enumerator directly is what
    makes the coverage claim about FORMS rather than about one form.
    """

    def test_a_plain_expression_yields_itself(self) -> None:
        assert list(iter_interpolations("${system.spatial_dim}")) == ["system.spatial_dim"]

    def test_text_without_an_interpolation_yields_nothing(self) -> None:
        assert list(iter_interpolations("outputs/run")) == []

    def test_a_nested_expression_yields_outer_then_inner(self) -> None:
        assert list(iter_interpolations("${oc.select:missing,${oc.env:X}}")) == [
            "oc.select:missing,${oc.env:X}",
            "oc.env:X",
        ]

    def test_siblings_are_yielded_in_source_order(self) -> None:
        assert list(iter_interpolations("a ${x:1} b ${y:2} c")) == ["x:1", "y:2"]

    def test_a_sibling_after_a_nested_expression_is_still_found(self) -> None:
        """Pins the resume point: the scan must continue past the OUTER brace.

        A walker that resumed after the inner expression's closing brace would
        re-enter the outer one; a walker that resumed at the inner one's start
        would loop. Neither shows up on a string with only one interpolation.
        """

        assert list(iter_interpolations("${oc.select:a,${oc.env:X}} then ${now:%S}")) == [
            "oc.select:a,${oc.env:X}",
            "oc.env:X",
            "now:%S",
        ]

    def test_an_unterminated_expression_yields_its_remainder(self) -> None:
        """Malformed must not be the one shape the firewall does not look at."""

        assert list(iter_interpolations("${oc.env:X")) == ["oc.env:X"]


class TestNestedForbiddenResolvers:
    """A forbidden resolver nested inside a permitted one still runs.

    EVERY FORM BELOW EXCEPT THE FIRST IS DRAWN FROM THE SPACE THE OLD
    INSTRUMENT MISSED, not from the space it handled. The rule was written
    against ``${oc.select:missing,${oc.env:X}}`` alone; the rest of the matrix
    was applied to it afterwards and is what makes this a coverage measurement
    rather than a restatement of the exhibit.

    MEASURED against the superseded pattern ``\\$\\{([^}]*)\\}``, which stops at
    the first closing brace: it captured the outer expression's text up to that
    brace, read the resolver as ``oc.select`` (or as nothing at all, for the
    unterminated form) and reported no finding. **All five nested arms
    discriminate.** The sibling arm in :class:`TestIterInterpolations` does NOT
    -- the old pattern found those already -- and it is kept as a non-regression
    control rather than counted as coverage.
    """

    @pytest.mark.parametrize(
        "expression",
        [
            # The exhibit the rule was written from.
            "${oc.select:missing,${oc.env:X}}",
            # Depth three: a walker that only looked one level in would miss it.
            "${oc.select:a,${oc.select:b,${oc.env:X}}}",
            # The OUTER expression is not a resolver at all -- it is a node
            # reference whose PATH is interpolated.
            "${${oc.env:X}}",
            # Nested inside a quoted argument.
            "${oc.select:'${oc.env:X}'}",
            # Nested, then unterminated. Malformed on the outside, forbidden on
            # the inside.
            "${oc.select:a,${oc.env:X}",
        ],
    )
    def test_rejects_a_forbidden_resolver_at_any_nesting_depth(self, expression: str) -> None:
        raw = {"model": {"seed": expression}}
        rejections = sweep(raw, {"model": {"seed": 0}}, POLICY)
        assert any(r.rule == "forbidden-resolver" for r in rejections), expression

    @pytest.mark.parametrize(
        "expression",
        [
            # Nested, but the inner expression is a plain node reference.
            "${oc.select:missing,${system.dim}}",
            # Nested node references only, two deep.
            "${oc.select:${system.dim},${model.dim}}",
            # A permitted resolver with no nesting at all.
            "${oc.select:missing,fallback}",
        ],
    )
    def test_accepts_nesting_that_reaches_no_forbidden_resolver(self, expression: str) -> None:
        """The over-restriction control: brace walking must not refuse by shape.

        A rule that refused every nested expression would pass every test in the
        class above while making legitimate configuration unwritable, and the
        failure would first appear as a run that cannot start.
        """

        raw = {"system": {"dim": 3}, "model": {"dim": expression}}
        rejections = sweep(raw, {"system": {"dim": 3}, "model": {"dim": 3}}, POLICY)
        assert [r for r in rejections if r.rule == "forbidden-resolver"] == []


class TestTheResolverColonIsFoundAtDepthZero:
    """The first colon in the text is not necessarily this expression's colon.

    ``str.partition`` read
    ``model.${oc.select:model.missing,embedding}.channels`` as resolver
    ``model.${oc.select`` -- but that colon belongs to the NESTED ``oc.select``,
    and the outer expression is a plain node reference whose PATH is computed.
    MEASURED: OmegaConf resolves that form to ``32`` against the shipped
    config, so the uncheckable-resolver guard was refusing valid configuration.
    """

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("oc.env:X", ("oc.env", True)),
            ("system.dim", ("", False)),
            # The resolver NAME is computed: a resolver call, and unqualifiable.
            ("oc.${leaf}:VAR,0", ("oc.${leaf}", True)),
            # A node PATH is computed and the nested part carries its own colon:
            # NOT a resolver call at this depth.
            ("model.${oc.select:model.missing,embedding}.channels", ("", False)),
            ("oc.select:missing,${oc.env:X}", ("oc.select", True)),
        ],
    )
    def test_splits_at_its_own_colon(self, expression: str, expected: tuple) -> None:
        assert split_resolver(expression) == expected

    def test_a_node_path_with_a_nested_permitted_resolver_is_accepted(self) -> None:
        """The over-restriction control, and it is a refusal that really happened."""

        raw = {"model": {"dim": "${model.${oc.select:model.missing,embedding}.channels}"}}
        rejections = sweep(raw, {"model": {"dim": 32}}, POLICY)
        assert [r for r in rejections if r.rule == "uncheckable-resolver"] == []

    def test_an_interpolated_resolver_name_is_still_refused(self) -> None:
        """Non-regression: narrowing the guard must not lose what it was for."""

        raw = {"model": {"dim": "${oc.${leaf}:VAR,0}"}}
        rejections = sweep(raw, {"model": {"dim": 0}}, POLICY)
        assert any(r.rule == "uncheckable-resolver" for r in rejections)

    def test_a_forbidden_resolver_nested_in_a_computed_node_path_is_still_found(self) -> None:
        """The outer being a node reference must not exempt what is inside it."""

        raw = {"model": {"dim": "${model.${oc.env:ARCH}.channels}"}}
        rejections = sweep(raw, {"model": {"dim": 32}}, POLICY)
        assert any(r.rule == "forbidden-resolver" for r in rejections)


class TestEscapedInterpolations:
    """An escaped opener runs no resolver, so reporting one is a FALSE refusal.

    EVERY EXPECTATION BELOW WAS MEASURED AGAINST OMEGACONF, not read off its
    documentation and not reasoned from the syntax. Resolving each raw string
    gave:

    - ``\\${oc.env:VAR,0}`` -> the literal text ``${oc.env:VAR,0}``; the
      environment is never touched.
    - ``${oc.select:missing,"\\${oc.env:VAR,0}"}`` -> the same literal text;
      still no environment read.
    - ``\\\\${oc.env:VAR}`` -> ``\\7`` with the variable set. An EVEN number of
      backslashes is an escaped BACKSLASH followed by a REAL interpolation, so
      parity is the test rather than presence.
    - ``\\${oc.select:a,${oc.env:VAR}}`` -> ``${oc.select:a,7}``. The outer is
      literal text and the INNER RAN.

    That last one is why the walker resumes just past the escaped opener rather
    than past the whole escaped expression. A fix that skipped to the closing
    brace would pass every other arm here and silently admit a live
    ``oc.env``.
    """

    def test_an_escaped_opener_yields_nothing(self) -> None:
        assert list(iter_interpolations(r"\${oc.env:VAR,0}")) == []

    def test_an_escaped_opener_inside_a_real_one_is_not_reported(self) -> None:
        found = list(iter_interpolations(r'${oc.select:missing,"\${oc.env:VAR,0}"}'))
        assert found == [r'oc.select:missing,"\${oc.env:VAR,0}"']

    def test_a_real_interpolation_inside_an_escaped_one_is_still_found(self) -> None:
        """The arm that separates a correct fix from a plausible one."""

        assert list(iter_interpolations(r"\${oc.select:a,${oc.env:VAR}}")) == ["oc.env:VAR"]

    def test_an_escaped_backslash_leaves_a_real_interpolation(self) -> None:
        assert list(iter_interpolations("\\\\${oc.env:VAR}")) == ["oc.env:VAR"]

    def test_the_policy_accepts_an_escaped_forbidden_resolver(self) -> None:
        """The over-restriction control at policy level.

        This is a REAL refusal the first version of this rule produced: a config
        carrying an escaped, literal, never-executed ``oc.env`` could not be
        written at all. Over-restriction does not fail a test, it fails a run.
        """

        raw = {"model": {"note": r'${oc.select:missing,"\${oc.env:VAR,0}"}'}}
        rejections = sweep(raw, {"model": {"note": "x"}}, POLICY)
        assert [r for r in rejections if r.rule == "forbidden-resolver"] == []

    def test_the_policy_still_refuses_a_live_resolver_inside_an_escaped_one(self) -> None:
        raw = {"model": {"note": r"\${oc.select:a,${oc.env:VAR}}"}}
        rejections = sweep(raw, {"model": {"note": "x"}}, POLICY)
        assert any(r.rule == "forbidden-resolver" for r in rejections)


class TestUncheckableResolvers:
    """A resolver whose NAME is built by another interpolation.

    ``${oc.${leaf}:VAR,0}`` with ``leaf: env`` IS ``oc.env`` by the time
    OmegaConf runs it. A static check sees ``oc.${leaf}``, matches nothing, and
    admits it -- so the forbidden-resolver list is bypassed without ever naming
    a forbidden resolver. Refused rather than admitted: a list can only refuse
    names it can read, and an allowlist that cannot read the name it is checking
    is not a check.
    """

    @pytest.mark.parametrize(
        "expression",
        [
            # The exhibit: the resolver name's tail is interpolated.
            "${oc.${leaf}:VAR,0}",
            # Composed inside a permitted outer resolver, which is how it would
            # actually be written to look innocuous.
            "${oc.select:missing,${oc.${leaf}:VAR,0}}",
            # The WHOLE name is one interpolation, with no literal part at all.
            "${${name}:VAR}",
        ],
    )
    def test_rejects_a_resolver_whose_name_is_interpolated(self, expression: str) -> None:
        raw = {"model": {"seed": expression}}
        rejections = sweep(raw, {"model": {"seed": 0}}, POLICY)
        assert any(r.rule == "uncheckable-resolver" for r in rejections), expression

    @pytest.mark.parametrize(
        "expression",
        [
            # A node reference whose PATH is interpolated. No colon, no
            # resolver, and it reads configuration rather than process state.
            "${model.${arch}.channels}",
            "${${section}.dim}",
        ],
    )
    def test_accepts_a_node_reference_whose_path_is_interpolated(self, expression: str) -> None:
        """The over-restriction control, and it is the one that matters here.

        A rule that refused every interpolation containing a nested ``${``
        would pass every arm above while forbidding layered configuration
        outright. The discriminator is the COLON: an expression with no colon
        names a config node, not a resolver.
        """

        raw = {"system": {"dim": 3}, "model": {"dim": expression}}
        rejections = sweep(raw, {"system": {"dim": 3}, "model": {"dim": 3}}, POLICY)
        assert [r for r in rejections if r.rule == "uncheckable-resolver"] == [], expression


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
