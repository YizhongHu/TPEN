"""Closed-schema preconstruction firewall for TPEN run configurations.

This module owns the *mechanism*: how a run configuration is swept for
forbidden surfaces and unknown fields before anything is constructed. It does
not own any study's policy -- the helium-importance train schema declares which
surfaces are forbidden and which sections are closed, and passes that policy in.

Why the sweep runs over two trees
---------------------------------
A forbidden value can hide in either of two places, and checking one tree
cannot see the other:

``raw``
    The configuration as written, with interpolations left as literal
    ``"${...}"`` text. A key that only *appears* after resolution is invisible
    here, but the interpolation *expressions* themselves are visible -- and an
    expression is how rank-local data enters a configuration
    (``${oc.env:RANK}``). Only the raw tree can catch that.

``resolved``
    The configuration Hydra will actually instantiate from. A forbidden literal
    reached through a chain of interpolations, or supplied by a nested default,
    is visible only here.

So :func:`sweep` runs over both and labels every rejection with the tree it
came from.

Why token matching rather than substring matching
-------------------------------------------------
``"reference" in "preference"`` is true, and ``"band" in "bandwidth"`` is true.
A substring rule would reject innocent keys and -- worse -- would look correct
while doing so, because its rejections are still *rejections* and nobody
inspects a firewall's false positives as closely as its false negatives. Keys
are therefore split into lowercase alphanumeric tokens and matched whole:
``energy_band`` and ``energyBand`` both tokenize to ``("energy", "band")`` and
match, while ``bandwidth`` tokenizes to ``("bandwidth",)`` and does not.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ClosedSchemaError",
    "canonical_digest",
    "ForbiddenSurface",
    "Rejection",
    "SchemaPolicy",
    "iter_nodes",
    "sweep",
    "sweep_environment",
    "sweep_raw",
    "sweep_resolved",
    "tokens_of",
]


# Split on every run of characters that is not a lowercase letter or digit.
# The name is lowercased first, so ``camelCase`` boundaries are recovered by the
# separate pass in :func:`tokens_of` rather than by this pattern.
_NON_TOKEN = re.compile(r"[^0-9a-z]+")

# Insert a separator at a lower-to-upper transition so ``referenceEnergy``
# tokenizes the same way ``reference_energy`` does. Applied before lowercasing.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# The opening delimiter of an OmegaConf interpolation, e.g. ``${oc.env:RANK}``
# or ``${system.spatial_dim}``. Only the raw tree contains these; the resolved
# tree has had them replaced by values.
#
# A REGEX CANNOT DO THIS JOB, which is why only the opener is one. The obvious
# pattern ``\$\{([^}]*)\}`` stops at the FIRST closing brace, so on
# ``${oc.select:missing,${oc.env:X}}`` it captures
# ``oc.select:missing,${oc.env`` and reports the resolver as ``oc.select`` --
# the nested ``oc.env`` is consumed into the argument text of the outer
# expression and never examined. Brace nesting is not a regular language, so
# :func:`iter_interpolations` walks the braces instead.
_INTERPOLATION_OPEN = "${"


def tokens_of(name: object) -> tuple[str, ...]:
    """Split a configuration key into lowercase word tokens.

    Parameters
    ----------
    name : object
        A configuration key or other identifier. Coerced with :func:`str`.

    Returns
    -------
    tuple of str
        The non-empty lowercase tokens of ``name``, in order.

    Examples
    --------
    >>> tokens_of("reference_energy")
    ('reference', 'energy')
    >>> tokens_of("referenceEnergy")
    ('reference', 'energy')
    >>> tokens_of("bandwidth")
    ('bandwidth',)
    """

    spaced = _CAMEL_BOUNDARY.sub("_", str(name))
    return tuple(token for token in _NON_TOKEN.split(spaced.lower()) if token)


@dataclass(frozen=True)
class ForbiddenSurface:
    """One named family of configuration keys a closed schema refuses.

    Parameters
    ----------
    name : str
        Short identifier for the family, e.g. ``"reference"``. Reported on every
        rejection so a failure names the rule it broke rather than only the key.
    tokens : frozenset of str
        Key tokens that trigger the rejection. A key matches when *any* of its
        tokens is in this set; see the module docstring for why matching is on
        whole tokens.
    reason : str
        Why the family is forbidden. Included verbatim in the rejection message,
        because the person who trips a firewall is rarely the person who wrote it.
    """

    name: str
    tokens: frozenset[str]
    reason: str

    def matches(self, key: object) -> bool:
        """Return whether ``key`` belongs to this forbidden family."""

        return bool(self.tokens.intersection(tokens_of(key)))


@dataclass(frozen=True)
class Rejection:
    """One reason a configuration was refused.

    Attributes
    ----------
    rule : str
        Which check failed, e.g. ``"forbidden-surface:reference"`` or
        ``"unknown-field"``.
    tree : str
        ``"raw"`` or ``"resolved"`` -- which of the two swept trees the offending
        node was found in. A surface present in only one of them is a real and
        reportable difference, so the tree is part of the finding rather than
        incidental.
    path : str
        Dotted path to the offending node, with list positions as ``[i]``.
    detail : str
        Human-readable explanation.
    """

    rule: str
    tree: str
    path: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.tree}] {self.path}: {self.rule} -- {self.detail}"


class ClosedSchemaError(ValueError):
    """A run configuration violated its closed schema.

    Carries *every* rejection rather than only the first. A firewall that fails
    on one finding at a time turns a single fix-and-rerun cycle into as many
    cycles as there are violations, and each cycle on this project may be a
    cluster job.

    Attributes
    ----------
    rejections : tuple of Rejection
        All findings, in discovery order.
    """

    def __init__(self, rejections: Sequence[Rejection]) -> None:
        self.rejections = tuple(rejections)
        joined = "\n  ".join(str(rejection) for rejection in self.rejections)
        count = len(self.rejections)
        plural = "" if count == 1 else "s"
        super().__init__(f"closed schema rejected the configuration ({count} finding{plural}):\n  {joined}")


@dataclass(frozen=True)
class SchemaPolicy:
    """The study-specific policy a closed schema enforces.

    Parameters
    ----------
    name : str
        Policy identifier, e.g. ``"tpen.hi.train.v1"``. This is the string a
        configuration opts in with, so it is a durable external identifier.
    forbidden_surfaces : tuple of ForbiddenSurface
        Key families refused anywhere at any depth in either tree.
    allowed_sections : frozenset of str
        Closed set of permitted top-level section names. A top-level key outside
        this set is an unknown field.
    forbidden_resolvers : frozenset of str
        Interpolation resolver names refused in the raw tree. These are the
        mechanism by which process-local facts (environment, clock, randomness)
        would otherwise reach a resolved configuration and make it differ
        between ranks. A DENYLIST, and see ``allowed_resolvers`` for when that
        is the wrong instrument.
    allowed_resolvers : frozenset of str or None
        When not ``None``, the CLOSED set of resolver names a configuration may
        call; every other resolver call is refused. ``None`` leaves the policy
        on ``forbidden_resolvers`` alone.

        A DENYLIST CANNOT WORK HERE AND THIS PROJECT HAS THE MEASUREMENTS.
        ``oc.decode`` was refused by name because it evaluates arbitrary text;
        ``oc.create`` does the same job, is not on any list, and additionally
        RE-RESOLVES interpolations in its output -- so a config carrying
        ``${oc.create:...}`` around a YAML string whose unicode escapes hide the
        interpolation opener reached the environment while every raw sweep
        passed. Refusing that one name would leave the next sibling. The set of
        resolvers that can evaluate text is open-ended and grows with OmegaConf;
        the set a study actually needs is small, closed, and known.
    """

    name: str
    forbidden_surfaces: tuple[ForbiddenSurface, ...] = ()
    allowed_sections: frozenset[str] = field(default_factory=frozenset)
    forbidden_resolvers: frozenset[str] = field(default_factory=frozenset)
    allowed_resolvers: frozenset[str] | None = None


def iter_nodes(tree: Any, prefix: str = "") -> Iterator[tuple[str, Any, Any]]:
    """Walk a plain container tree, yielding every keyed node at every depth.

    Parameters
    ----------
    tree : Any
        A nested structure of mappings, sequences, and scalars -- the output of
        ``OmegaConf.to_container``. Not a ``DictConfig``: containers are walked
        so that no access can trigger interpolation as a side effect.
    prefix : str, optional
        Dotted path prefix for the current node.

    Yields
    ------
    path : str
        Dotted path to the node, with list positions rendered as ``[i]``.
    key : Any
        The mapping key the node was reached through, or ``None`` for a list
        element (which has a position but no name to match rules against).
    value : Any
        The node itself.

    Notes
    -----
    Strings are sequences, so they are excluded from the sequence branch
    explicitly; otherwise every string would be walked character by character.
    """

    if isinstance(tree, Mapping):
        for key, value in tree.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, key, value
            yield from iter_nodes(value, path)
    elif isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        for index, value in enumerate(tree):
            path = f"{prefix}[{index}]"
            yield path, None, value
            yield from iter_nodes(value, path)


def _sweep_forbidden_surfaces(tree: Any, label: str, policy: SchemaPolicy) -> list[Rejection]:
    """Reject every key belonging to a forbidden family, at any depth."""

    rejections: list[Rejection] = []
    for path, key, _value in iter_nodes(tree):
        if key is None:
            continue
        for surface in policy.forbidden_surfaces:
            if surface.matches(key):
                rejections.append(
                    Rejection(
                        rule=f"forbidden-surface:{surface.name}",
                        tree=label,
                        path=path,
                        detail=f"key {key!r} is a {surface.name} surface; {surface.reason}",
                    )
                )
    return rejections


def _sweep_unknown_sections(tree: Any, label: str, policy: SchemaPolicy) -> list[Rejection]:
    """Reject top-level sections outside the policy's closed set."""

    if not policy.allowed_sections or not isinstance(tree, Mapping):
        return []
    return [
        Rejection(
            rule="unknown-field",
            tree=label,
            path=str(key),
            detail=(
                f"section {str(key)!r} is not in the closed schema; "
                f"permitted sections are {sorted(policy.allowed_sections)}"
            ),
        )
        for key in tree
        if str(key) not in policy.allowed_sections
    ]


def iter_interpolations(text: str) -> Iterator[str]:
    """Yield every interpolation expression in ``text``, at every nesting depth.

    Parameters
    ----------
    text : str
        A raw configuration value, which may contain zero or more
        ``${...}`` expressions, each of which may itself contain more.

    Yields
    ------
    str
        The body of one interpolation, without its ``${`` and ``}``. Outer
        expressions are yielded before the expressions nested inside them, and
        siblings in source order.

    Notes
    -----
    NESTING IS THE WHOLE POINT. ``${oc.select:missing,${oc.env:X}}`` is two
    expressions, not one: the outer selects, and the inner reads the process
    environment. A caller that sees only the outer one sees a resolver nobody
    forbids while the forbidden resolver runs inside it.

    An UNTERMINATED expression yields the remainder of the string AND is
    scanned in turn. That is deliberate: a value ending in ``${oc.env:X`` is
    malformed and OmegaConf will refuse it, but refusing it HERE, by name, beats
    letting a malformed expression be the one shape a firewall does not look at.
    The recursion is not decoration -- ``${oc.select:a,${oc.env:X}`` is
    malformed on the OUTSIDE and carries a well-formed forbidden resolver
    inside, and an earlier version of this function reported only the outer
    text.

    Examples
    --------
    >>> list(iter_interpolations("${system.spatial_dim}"))
    ['system.spatial_dim']
    >>> list(iter_interpolations("${oc.select:missing,${oc.env:X}}"))
    ['oc.select:missing,${oc.env:X}', 'oc.env:X']
    """

    index = 0
    length = len(text)
    while True:
        start = text.find(_INTERPOLATION_OPEN, index)
        if start < 0:
            return
        # ESCAPED OPENERS RUN NO RESOLVER, so reporting one is a false refusal.
        # MEASURED against OmegaConf rather than read off the documentation:
        # ``\${oc.env:VAR,0}`` resolves to the literal text ``${oc.env:VAR,0}``
        # and never touches the environment. An ODD number of preceding
        # backslashes escapes the opener; an EVEN number is an escaped backslash
        # followed by a REAL interpolation (``\\${oc.env:VAR}`` resolved to
        # ``\7`` with the variable set), so parity is the test, not presence.
        backslashes = 0
        probe = start - 1
        while probe >= 0 and text[probe] == "\\":
            backslashes += 1
            probe -= 1
        if backslashes % 2 == 1:
            # Resume just past THIS opener rather than past the whole escaped
            # expression, and that distinction is load-bearing: a real
            # interpolation nested inside an escaped one still resolves.
            # ``\${oc.select:a,${oc.env:VAR}}`` came back as ``${oc.select:a,7}``
            # -- the outer is literal text, the inner RAN. Skipping to the
            # closing brace would admit exactly that.
            index = start + len(_INTERPOLATION_OPEN)
            continue
        # Walk forward counting braces so the expression ends at ITS OWN closing
        # brace rather than at the first one belonging to something nested.
        depth = 0
        cursor = start
        end = -1
        while cursor < length:
            if text.startswith(_INTERPOLATION_OPEN, cursor):
                depth += 1
                cursor += len(_INTERPOLATION_OPEN)
                continue
            if text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    end = cursor
                    break
            cursor += 1
        if end < 0:
            # Unterminated. Yield the remainder AND recurse into it: an
            # expression can be malformed on the OUTSIDE while carrying a
            # well-formed forbidden resolver inside, as in
            # ``${oc.select:a,${oc.env:X}``. Returning after the single yield
            # reported only the outer text and let the inner resolver through.
            remainder = text[start + len(_INTERPOLATION_OPEN) :]
            yield remainder
            yield from iter_interpolations(remainder)
            return
        expression = text[start + len(_INTERPOLATION_OPEN) : end]
        yield expression
        # Recurse into the body, which is where a nested resolver hides. The
        # scan then resumes AFTER this expression so siblings are found too.
        yield from iter_interpolations(expression)
        index = end + 1


def split_resolver(expression: str) -> tuple[str, bool]:
    """Split an interpolation body at ITS OWN resolver colon.

    Parameters
    ----------
    expression : str
        One interpolation body, as yielded by :func:`iter_interpolations`.

    Returns
    -------
    resolver : str
        The text before the expression's own colon, stripped. Empty when there
        is none.
    is_resolver_call : bool
        Whether the expression is a resolver call at all. ``False`` means it is
        a node reference such as ``${system.dim}``.

    Notes
    -----
    THE FIRST COLON IN THE TEXT IS NOT NECESSARILY THIS EXPRESSION'S COLON.
    ``str.partition`` reads ``model.${oc.select:model.missing,embedding}.channels``
    as resolver ``model.${oc.select`` -- but that colon belongs to the NESTED
    ``oc.select``, and the outer expression is a plain node reference whose PATH
    happens to be computed. MEASURED: OmegaConf resolves that form to ``32``
    against the shipped config, so refusing it refuses valid configuration.

    Only a colon at nesting depth ZERO is this expression's own, so the scan
    steps over nested ``${...}`` rather than counting characters.
    """

    depth = 0
    cursor = 0
    length = len(expression)
    while cursor < length:
        if expression.startswith(_INTERPOLATION_OPEN, cursor):
            depth += 1
            cursor += len(_INTERPOLATION_OPEN)
            continue
        character = expression[cursor]
        if character == "}" and depth > 0:
            depth -= 1
        elif character == ":" and depth == 0:
            return expression[:cursor].strip(), True
        cursor += 1
    return "", False


def _sweep_forbidden_resolvers(raw_tree: Any, policy: SchemaPolicy) -> list[Rejection]:
    """Reject interpolations that read process-local state, in the raw tree only.

    A resolved tree cannot be checked for this: by then ``${oc.env:RANK}`` has
    become the value it produced, which is indistinguishable from a literal that
    was always there. This is precisely the check that makes a resolved
    configuration a pure function of the file and its overrides, and therefore
    identical on every rank.

    NESTED expressions are swept, not only outermost ones. A forbidden resolver
    nested inside a permitted one runs exactly as it would on its own:
    ``${oc.select:missing,${oc.env:X}}`` resolves to the environment value
    whenever the selected key is absent, so two ranks with different
    environments resolve one file differently -- which is the property this
    check exists to protect. See :func:`iter_interpolations` for why the sweep
    walks braces instead of matching a pattern.
    """

    if not policy.forbidden_resolvers and policy.allowed_resolvers is None:
        return []

    rejections: list[Rejection] = []
    for path, _key, value in iter_nodes(raw_tree):
        if not isinstance(value, str):
            continue
        for expression in iter_interpolations(value):
            # ``oc.env:RANK`` -> resolver ``oc.env``; a plain node reference such
            # as ``system.spatial_dim`` has no colon and no resolver name.
            resolver, is_resolver_call = split_resolver(expression)
            if not is_resolver_call:
                # No colon AT THIS EXPRESSION'S OWN NESTING DEPTH: a plain
                # node reference such as ``${system.dim}``, which names no
                # resolver. This stays true when the reference PATH is itself
                # interpolated -- ``${model.${arch}.channels}`` reads
                # configuration, never process-local state -- including when the
                # nested part is a permitted resolver CALL carrying its own
                # colon. Any resolver nested inside is reached by recursion and
                # judged on its own terms; see :func:`split_resolver`.
                continue
            if _INTERPOLATION_OPEN in resolver:
                # THE RESOLVER NAME IS COMPUTED AT RESOLVE TIME, so it cannot be
                # compared against any list here. ``${oc.${leaf}:VAR,0}`` with
                # ``leaf: env`` is ``oc.env`` by the time OmegaConf runs it, and
                # a static check sees only ``oc.${leaf}``. Unqualifiable is
                # refused rather than admitted: an allowlist that cannot read
                # the name it is checking is not a check.
                rejections.append(
                    Rejection(
                        rule="uncheckable-resolver",
                        tree="raw",
                        path=path,
                        detail=(
                            f"interpolation ${{{expression}}} builds its RESOLVER NAME from "
                            f"another interpolation ({resolver!r}), so which resolver runs is "
                            "decided at resolution time and cannot be qualified before it. "
                            "Spell the resolver literally. This is refused rather than "
                            "admitted because the forbidden-resolver list can only refuse "
                            "names it can read"
                        ),
                    )
                )
                continue
            if resolver in policy.forbidden_resolvers:
                rejections.append(
                    Rejection(
                        rule="forbidden-resolver",
                        tree="raw",
                        path=path,
                        detail=(
                            f"interpolation ${{{expression}}} reads process-local state through "
                            f"resolver {resolver!r}; the resolved configuration must be a pure "
                            "function of the config file and its overrides so that every rank "
                            "resolves it identically"
                        ),
                    )
                )
                # The named reason is the more useful one, so it stands alone.
                # Reporting the generic refusal beside it would be two findings
                # for one expression, which reads as two defects.
                continue
            if policy.allowed_resolvers is not None and resolver not in policy.allowed_resolvers:
                admitted = sorted(policy.allowed_resolvers)
                rejections.append(
                    Rejection(
                        rule="unadmitted-resolver",
                        tree="raw",
                        path=path,
                        detail=(
                            f"interpolation ${{{expression}}} calls resolver {resolver!r}, which "
                            f"is not in this schema's closed set of admitted resolvers "
                            f"({admitted or 'none'}). Resolver calls are ADMITTED, not merely "
                            "screened: the set that can evaluate text or read process-local "
                            "state is open-ended and grows with OmegaConf, while the set a "
                            "study needs is small and known. Plain node references such as "
                            "${section.key} are unaffected -- they name configuration, not a "
                            "resolver"
                        ),
                    )
                )
                continue
    return rejections


def canonical_digest(resolved_tree: Any) -> str:
    """Return a stable content digest of a resolved configuration.

    Parameters
    ----------
    resolved_tree : Any
        The configuration as plain containers, interpolations resolved.

    Returns
    -------
    str
        A hex SHA-256 over a canonical JSON rendering: keys sorted, no
        insignificant whitespace.

    Notes
    -----
    This is the "canonical resolved input" two ranks compare. Key order is
    normalized because a mapping's iteration order is not part of a
    configuration's meaning, and leaving it in would make the digest report a
    difference where there is none.

    The digest is only as strong as the guarantee that resolution is
    deterministic. That guarantee comes from the forbidden-resolver check: with
    ``oc.env`` and ``now`` refused, resolution is a pure function of the file
    and its overrides. Without that check this digest would faithfully report
    two different values and prove nothing about why.
    """

    rendered = json.dumps(resolved_tree, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def sweep_environment(env: Mapping[str, str], policy: SchemaPolicy) -> tuple[Rejection, ...]:
    """Reject forbidden surfaces present in a process environment.

    Parameters
    ----------
    env : Mapping of str to str
        The launch environment, e.g. ``os.environ``. Passed in rather than read
        here so a test can supply a deterministic environment; a check that
        reads the ambient process state cannot be tested for the case it
        exists to catch.
    policy : SchemaPolicy
        The study policy to enforce.

    Returns
    -------
    tuple of Rejection
        One finding per offending variable name, in sorted name order.

    Notes
    -----
    Variable *names* are matched, not values. A name is the only part of an
    environment variable that says what it means; matching values would reject
    any variable that happened to hold a number near the reference.

    The whole environment is scanned rather than a project-prefixed subset.
    That is deliberate and follows the reference-energy firewall, which forbids
    a reference in the training launch environment and extends the same rule to
    "apparently unused fields" -- an unread variable is exactly such a field. A
    false positive here is loud and is fixed by unsetting one variable; a false
    negative is a silent science violation. Token matching is what keeps the
    false-positive rate low enough for that trade to be the right one.
    """

    rejections: list[Rejection] = []
    for name in sorted(env):
        for surface in policy.forbidden_surfaces:
            if surface.matches(name):
                rejections.append(
                    Rejection(
                        rule=f"forbidden-environment:{surface.name}",
                        tree="environment",
                        path=name,
                        detail=(
                            f"environment variable {name!r} is a {surface.name} surface; "
                            f"{surface.reason}. Unset it before launching training -- its "
                            "value is not read here, and an unread variable is still a "
                            "forbidden field"
                        ),
                    )
                )
    return tuple(rejections)


def sweep_raw(raw_tree: Any, policy: SchemaPolicy) -> tuple[Rejection, ...]:
    """Collect the violations visible before interpolations are resolved.

    Parameters
    ----------
    raw_tree : Any
        The configuration as plain containers, interpolations unresolved.
    policy : SchemaPolicy
        The study policy to enforce.

    Returns
    -------
    tuple of Rejection
        Every raw-tree finding, in discovery order.

    Notes
    -----
    Separated from :func:`sweep_resolved` because resolution can *fail*, and a
    configuration that fails to resolve must still report what the raw tree
    already showed. Folding both sweeps into one resolution-dependent step
    would let a broken interpolation suppress an unrelated forbidden reference
    -- the finding and the thing that hides it would share a failure domain.
    """

    rejections: list[Rejection] = []
    rejections.extend(_sweep_unknown_sections(raw_tree, "raw", policy))
    rejections.extend(_sweep_forbidden_surfaces(raw_tree, "raw", policy))
    rejections.extend(_sweep_forbidden_resolvers(raw_tree, policy))
    return tuple(rejections)


def sweep_resolved(resolved_tree: Any, policy: SchemaPolicy) -> tuple[Rejection, ...]:
    """Collect the violations visible only after interpolations are resolved.

    Parameters
    ----------
    resolved_tree : Any
        The configuration as plain containers, interpolations resolved.
    policy : SchemaPolicy
        The study policy to enforce.

    Returns
    -------
    tuple of Rejection
        Every resolved-tree finding, in discovery order.

    Notes
    -----
    Sections and surfaces are re-checked here rather than trusted from the raw
    sweep, because resolution can introduce a key the raw tree never spelled.
    Forbidden resolvers are not re-checked: by this point the interpolation has
    become the value it produced and is indistinguishable from a literal.
    """

    rejections: list[Rejection] = []
    rejections.extend(_sweep_unknown_sections(resolved_tree, "resolved", policy))
    rejections.extend(_sweep_forbidden_surfaces(resolved_tree, "resolved", policy))
    return tuple(rejections)


def sweep(raw_tree: Any, resolved_tree: Any, policy: SchemaPolicy) -> tuple[Rejection, ...]:
    """Collect every closed-schema violation in a configuration.

    Parameters
    ----------
    raw_tree : Any
        The configuration as plain containers with interpolations unresolved.
    resolved_tree : Any
        The same configuration with interpolations resolved.
    policy : SchemaPolicy
        The study policy to enforce.

    Returns
    -------
    tuple of Rejection
        Every finding, in discovery order. Empty when the configuration passes.

    Notes
    -----
    This function only *reports*. Raising is the caller's decision, so a caller
    that wants to audit a legacy configuration can list its violations without
    being stopped by the first one. A caller whose configuration might not
    resolve should use :func:`sweep_raw` and :func:`sweep_resolved` directly.
    """

    return sweep_raw(raw_tree, policy) + sweep_resolved(resolved_tree, policy)
