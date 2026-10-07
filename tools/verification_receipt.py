from __future__ import annotations

"""Produce and falsify self-contained TPEN verification receipts.

This module deliberately uses only the standard library and supports Python
3.8 as well as newer interpreters. The wrapper runs inside scheduler jobs,
possibly before or outside TPEN's ``uv`` environment, so it must not require a
newer Python than the project's test suite itself.
"""

import argparse
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple


SENTINEL = "TPEN-VERIFICATION-RECEIPT"
VERSION = "v1"
OBSERVATION_ENV = "TPEN_VERIFICATION_OBSERVATION_PATH"
NONCE_ENV = "TPEN_VERIFICATION_NONCE"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_COUNT_FIELDS = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
_SELECTION_OPTION_NAMES = (
    "keyword",
    "markexpr",
    "stepwise",
    "stepwise_skip",
    "last_failed",
    "failed_first",
    "maxfail",
    "exitfirst",
    "collectonly",
    "ignore",
    "ignore_glob",
    "deselect",
    "file_or_dir",
)
# This is intentionally the only optional absence. Pytest represents -x by
# setting maxfail, so an exitfirst attribute is not required when that alias
# is absent. Every other selection-bearing observation must be found.
_OPTIONAL_ABSENT_OPTIONS = {"exitfirst"}
_SELECTION_LONG = {
    "--ignore",
    "--ignore-glob",
    "--deselect",
    "--lf",
    "--last-failed",
    "--ff",
    "--failed-first",
    "--exitfirst",
    "--maxfail",
    "--collect-only",
    "--co",
}
# Closed set of pytest options whose separate next token is a value, not a
# positional test target. Unknown options intentionally retain the loud,
# fail-closed behaviour of the final bare-token rule.
_VALUE_OPTIONS = {
    "-p",
    "-n",
    "-c",
    "-o",
    "-W",
    "--rootdir",
    "--junitxml",
    "--junit-xml",
    "--durations",
    "--basetemp",
    "--log-file",
    "--color",
    "--capture",
    "--tb",
}
_ATTACHED_VALUE_SHORT = ("-p", "-n", "-c", "-o", "-W")
_CLUSTER_FLAGS = set("qvrsfxlakm")


def _shell_token(token: str) -> str:
    """Return one argv token in a compact, unambiguous display form."""

    return shlex.quote(token)


def _is_python_program(token: str) -> bool:
    """Whether *token* names a Python interpreter executable."""

    name = os.path.basename(token)
    return name == "python" or name.startswith("python3") or name.startswith("python2")


def _is_pytest_program(token: str) -> bool:
    """Whether *token* names a pytest executable or module."""

    name = os.path.basename(token)
    return name == "pytest" or name.startswith("pytest-")


def split_pytest_command(command: Sequence[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """Split a command into pytest arguments and an optional parse error.

    Parameters
    ----------
    command
        Complete command vector supplied after the wrapper's ``--``.

    Returns
    -------
    tuple[list[str] | None, str | None]
        Arguments after a pytest executable/module prefix, or an error that
        names the unrecognized command.
    """

    values = list(command)
    if values and _is_pytest_program(values[0]):
        return values[1:], None
    if len(values) >= 3 and _is_python_program(values[0]) and values[1] == "-m" and values[2] == "pytest":
        return values[3:], None

    command_name = values[0] if values else "<empty command>"
    return None, "command_unrecognized=" + _shell_token(command_name)


def classify_pytest_args(pytest_args: Sequence[str]) -> Tuple[str, List[str]]:
    """Classify pytest arguments and enumerate pre-flight disqualifiers.

    This classifier is advisory only. The pytest-side observer is authoritative
    for the receipt because selection can arrive through config, environment,
    plugins, or collection hooks without appearing in argv.
    """

    values = list(pytest_args)
    disqualifying: List[str] = []
    index = 0
    after_double_dash = False
    while index < len(values):
        token = values[index]
        if after_double_dash:
            disqualifying.append(token)
            index += 1
            continue
        if token == "--":
            after_double_dash = True
            index += 1
            continue

        if token in ("-k", "-m"):
            disqualifying.append(token)
            if index + 1 < len(values):
                disqualifying.append(values[index + 1])
                index += 2
            else:
                index += 1
            continue
        if (
            token.startswith("-k=")
            or token.startswith("-m=")
            or (token.startswith("-k") and len(token) > 2)
            or (token.startswith("-m") and len(token) > 2)
        ):
            disqualifying.append(token)
            index += 1
            continue

        if token in _VALUE_OPTIONS:
            index += 2 if index + 1 < len(values) else 1
            continue
        if token.startswith("--") and any(token.startswith(option + "=") for option in _VALUE_OPTIONS):
            index += 1
            continue
        if token.startswith("-") and not token.startswith("--"):
            if any(token.startswith(option) and len(token) > len(option) for option in _ATTACHED_VALUE_SHORT):
                index += 1
                continue
            # Only a genuine short-flag cluster can carry -x/-k/-m here.
            # Values such as -Wignore::RuntimeWarning and -ocache_dir=...
            # are not clusters and must not cause a pre-flight false refusal.
            if set(token[1:]).issubset(_CLUSTER_FLAGS) and any(
                flag in token[1:] for flag in ("x", "k", "m")
            ):
                disqualifying.append(token)
            index += 1
            continue

        if token.startswith("-"):
            index += 1
            continue

        # Pytest treats a bare path, package, file, or node id as a target.
        disqualifying.append(token)
        index += 1

    if disqualifying:
        return "SELECTED", disqualifying
    return "UNSELECTED", []


def classify_command(command: Sequence[str]) -> Dict[str, object]:
    """Classify a complete post-wrapper command, including its prefix."""

    pytest_args, error = split_pytest_command(command)
    if error is not None:
        return {
            "selection": "SELECTED",
            "selection_tokens": [],
            "pytest_args": None,
            "command_error": error,
        }
    selection, tokens = classify_pytest_args(pytest_args or [])
    return {
        "selection": selection,
        "selection_tokens": tokens,
        "pytest_args": list(pytest_args or []),
        "command_error": None,
    }


def _git_probe() -> Dict[str, object]:
    """Resolve in-process git facts, marking every command failure explicitly."""

    result: Dict[str, object] = {
        "head": "UNKNOWN",
        "branch": "UNKNOWN",
        "tracked_clean": "NO",
        "git_ok": False,
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        branch = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return result

    if head.returncode != 0 or branch.returncode != 0 or status.returncode != 0:
        return result
    resolved_head = head.stdout.strip()
    if not _HEX40.fullmatch(resolved_head):
        return result
    result["head"] = resolved_head
    result["branch"] = branch.stdout.strip() or "DETACHED"
    result["tracked_clean"] = "YES" if not status.stdout else "NO"
    result["git_ok"] = True
    return result


def _raw_tokens(value: Any) -> List[str]:
    """Convert pytest's list-or-string config values to argv tokens."""

    if value is None:
        return []
    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError:
            return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _unique(values: Sequence[str]) -> List[str]:
    """Keep source diagnostics stable and non-repeating."""

    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _sources_from_tokens(tokens: Sequence[str], prefix: str) -> List[str]:
    """Name the source and option that made a raw token stream selected."""

    selection, disqualifying = classify_pytest_args(tokens)
    if selection != "SELECTED":
        return []
    option_tokens = [token for token in disqualifying if token.startswith("-")]
    if option_tokens:
        return _unique([prefix + token for token in option_tokens])
    return [prefix + disqualifying[0]] if disqualifying else [prefix + "present"]


def _override_sources(invocation_args: Sequence[str]) -> List[str]:
    """Expose selection hidden in ``-o addopts=...`` or ``testpaths=...``."""

    sources: List[str] = []
    index = 0
    values = list(invocation_args)
    while index < len(values):
        token = values[index]
        override: Optional[str] = None
        if token in ("-o", "--override-ini") and index + 1 < len(values):
            override = values[index + 1]
            index += 2
        else:
            index += 1
        if not override or "=" not in override:
            continue
        name, value = override.split("=", 1)
        if name == "addopts":
            sources.extend(_sources_from_tokens(_raw_tokens(value), "ini:addopts:"))
        elif name == "testpaths" and value.strip():
            sources.append("ini:testpaths")
    return _unique(sources)


def _option_has_value(value: Any) -> bool:
    """Whether a resolved pytest option represents an active selection."""

    if value is None or value is False or value == 0 or value == "":
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _selection_from_observation(observation: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Derive selection only from pytest-side observations."""

    sources: List[str] = []
    invocation = [str(item) for item in observation["invocation_args"]]
    sources.extend(_sources_from_tokens(invocation, "argv:"))
    sources.extend(_override_sources(invocation))

    env_addopts = observation["env_pytest_addopts"]
    if env_addopts is not None and str(env_addopts).strip():
        env_sources = _sources_from_tokens(_raw_tokens(env_addopts), "PYTEST_ADDOPTS:")
        sources.extend(env_sources or ["PYTEST_ADDOPTS:present"])

    ini_addopts = observation["ini_addopts"]
    sources.extend(_sources_from_tokens(_raw_tokens(ini_addopts), "ini:addopts:"))

    ini_testpaths = _raw_tokens(observation["ini_testpaths"])
    if ini_testpaths:
        sources.append("ini:testpaths")

    options = observation["options"]
    absent_options = [
        name
        for name in _SELECTION_OPTION_NAMES
        if options[name]["state"] == "ABSENT" and name not in _OPTIONAL_ABSENT_OPTIONS
    ]
    if absent_options:
        return "UNKNOWN", ["option_absent:" + name for name in absent_options]
    for name in _SELECTION_OPTION_NAMES:
        if _option_has_value(options[name]["value"]):
            sources.append("option:" + name)

    deselected = observation["deselected"]
    if deselected:
        sources.append("deselected:" + str(deselected))
    sources = _unique(sources)
    return ("SELECTED", sources) if sources else ("UNSELECTED", [])


def _valid_nonnegative_int(value: Any) -> bool:
    """Whether a structured count is an integer and not negative."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _read_observation(path: Optional[str], nonce: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Read and validate the nonce-bound pytest observation file."""

    if not path:
        return None, "observation_missing"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            observation = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None, "observation_unreadable"
    if not isinstance(observation, dict):
        return None, "observation_malformed"
    if observation.get("version") != 1:
        return None, "observation_version"
    if observation.get("nonce") != nonce:
        return None, "nonce_mismatch"
    if not isinstance(observation.get("invocation_args"), list):
        return None, "observation_invocation_args"
    if not all(isinstance(item, str) for item in observation["invocation_args"]):
        return None, "observation_invocation_args"
    if "ini_addopts" not in observation or "ini_testpaths" not in observation:
        return None, "observation_ini"
    if not isinstance(observation.get("options"), dict):
        return None, "observation_options"
    if not set(_SELECTION_OPTION_NAMES).issubset(set(observation["options"])):
        return None, "observation_options"
    for name in _SELECTION_OPTION_NAMES:
        option = observation["options"][name]
        if not isinstance(option, dict) or option.get("state") not in {"FOUND", "ABSENT"}:
            return None, "observation_options"
        if "value" not in option:
            return None, "observation_options"
        if option["state"] == "FOUND":
            if not isinstance(option.get("attribute"), str):
                return None, "observation_options"
        elif option.get("attribute") is not None:
            return None, "observation_options"
    if observation.get("env_pytest_addopts") is not None and not isinstance(
        observation["env_pytest_addopts"], str
    ):
        return None, "observation_environment"
    if not _valid_nonnegative_int(observation.get("collected")):
        return None, "observation_collected"
    if not _valid_nonnegative_int(observation.get("deselected")):
        return None, "observation_deselected"
    stats = observation.get("stats")
    if not isinstance(stats, dict):
        return None, "observation_stats"
    if not all(_valid_nonnegative_int(stats.get(field)) for field in _COUNT_FIELDS):
        return None, "observation_stats"
    return observation, None


def _unknown_counts(counts: Dict[str, Optional[int]]) -> bool:
    """Whether any authoritative receipt count is unavailable."""

    return any(counts.get(field) is None for field in ("collected",) + _COUNT_FIELDS)


def _receipt_facts(
    command: Sequence[str],
    classification: Dict[str, object],
    before: Dict[str, object],
    after: Dict[str, object],
    pytest_exit: int,
    observation: Optional[Dict[str, Any]],
    observation_error: Optional[str],
    command_error: Optional[str] = None,
) -> Dict[str, object]:
    """Build canonical facts from the observer and both Git probes."""

    head = str(before["head"])
    head_after = str(after["head"])
    stable = "YES" if bool(before["git_ok"]) and bool(after["git_ok"]) and head == head_after else "NO"
    if observation is None:
        selection = "UNKNOWN"
        selection_sources = [observation_error or "observation_missing"]
        counts: Dict[str, Optional[int]] = {field: None for field in ("collected",) + _COUNT_FIELDS}
    else:
        selection, selection_sources = _selection_from_observation(observation)
        counts = {"collected": observation["collected"]}
        counts.update({field: observation["stats"][field] for field in _COUNT_FIELDS})

    provenance = (
        "OK"
        if observation is not None
        and selection != "UNKNOWN"
        and stable == "YES"
        and bool(before["git_ok"])
        else "BROKEN"
    )
    reasons: List[str] = []
    if provenance != "OK":
        reasons.append("provenance")
    if selection != "UNSELECTED":
        reasons.append("selection_unknown" if selection == "UNKNOWN" else "selection")
    if stable != "YES":
        reasons.append("head_unstable")
    if before["tracked_clean"] != "YES":
        reasons.append("tracked_dirty_before")
    if after["tracked_clean"] != "YES":
        reasons.append("tracked_dirty_after")
    if _unknown_counts(counts):
        reasons.append("counts_unknown")
    if pytest_exit != 0:
        reasons.append("pytest_exit")
    if command_error is not None:
        reasons.append(command_error)

    tracked_clean = "YES" if before["tracked_clean"] == "YES" and after["tracked_clean"] == "YES" else "NO"
    return {
        "provenance": provenance,
        "head": head,
        "head_after": head_after,
        "head_stable": stable,
        "tracked_clean": tracked_clean,
        "tracked_clean_before": str(before["tracked_clean"]),
        "tracked_clean_after": str(after["tracked_clean"]),
        "branch": str(before["branch"]),
        "selection": selection,
        "selection_tokens": selection_sources,
        "baseline_eligible": "YES" if not reasons else "NO",
        "baseline_reasons": reasons,
        "pytest_exit": pytest_exit,
        "collected": counts.get("collected"),
        "passed": counts.get("passed"),
        "failed": counts.get("failed"),
        "errors": counts.get("errors"),
        "skipped": counts.get("skipped"),
        "xfailed": counts.get("xfailed"),
        "xpassed": counts.get("xpassed"),
        "argv": list(command),
        "advisory_selection": classification["selection"],
        "observation_error": observation_error,
        "selection_options": observation["options"] if observation is not None else None,
    }


def _display_number(value: object) -> str:
    """Render a receipt number or its required unknown marker."""

    return "UNKNOWN" if value is None else str(value)


def format_receipt(facts: Dict[str, object]) -> str:
    """Serialize canonical facts into the one-line greppable receipt."""

    selection_tokens = ",".join(_shell_token(str(token)) for token in facts["selection_tokens"])
    reasons = ",".join(_shell_token(str(reason)) for reason in facts["baseline_reasons"])
    selection = str(facts["selection"])
    if selection_tokens:
        selection += "[" + selection_tokens + "]"
    baseline = str(facts["baseline_eligible"]) + "[" + reasons + "]"
    fields = [
        SENTINEL,
        VERSION,
        "provenance=" + str(facts["provenance"]),
        "head=" + str(facts["head"]),
        "head_after=" + str(facts["head_after"]),
        "head_stable=" + str(facts["head_stable"]),
        "tracked_clean=" + str(facts["tracked_clean"]),
        "tracked_clean_before=" + str(facts["tracked_clean_before"]),
        "tracked_clean_after=" + str(facts["tracked_clean_after"]),
        "branch=" + _shell_token(str(facts["branch"])),
        "selection=" + selection,
        "baseline_eligible=" + baseline,
        "pytest_exit=" + str(facts["pytest_exit"]),
        "collected=" + _display_number(facts["collected"]),
        "passed=" + _display_number(facts["passed"]),
        "failed=" + _display_number(facts["failed"]),
        "errors=" + _display_number(facts["errors"]),
        "skipped=" + _display_number(facts["skipped"]),
        "xfailed=" + _display_number(facts["xfailed"]),
        "xpassed=" + _display_number(facts["xpassed"]),
        "argv=" + shlex.join([str(value) for value in facts["argv"]]),
    ]
    return " ".join(fields)


def _write_json(path: str, facts: Dict[str, object]) -> None:
    """Write the same canonical facts used for the text receipt."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(facts, handle, sort_keys=True)
        handle.write("\n")


def _observer_command(command: Sequence[str]) -> List[str]:
    """Inject the observer plugin after the pytest command prefix."""

    pytest_args, error = split_pytest_command(command)
    if error is not None or pytest_args is None:
        return list(command)
    if _is_pytest_program(command[0]):
        return [command[0], "-p", "_verification_observer"] + list(command[1:])
    return list(command[:3]) + ["-p", "_verification_observer"] + list(command[3:])


def _run_command(options: argparse.Namespace, command: Sequence[str]) -> int:
    """Run pytest, forward raw output, and emit one authoritative receipt."""

    classification = classify_command(command)
    command_error = classification["command_error"]
    if options.claim_baseline and (command_error is not None or classification["selection"] != "UNSELECTED"):
        reason = str(command_error) if command_error is not None else "selection=" + ",".join(
            _shell_token(str(token)) for token in classification["selection_tokens"]
        )
        print("--claim-baseline refused before pytest: " + reason, file=sys.stderr)
        return 2

    before = _git_probe()
    after = before
    pytest_exit = 127
    observation_path: Optional[str] = None
    nonce = secrets.token_hex(16)
    try:
        if command_error is not None:
            print("verification receipt refused: " + str(command_error), file=sys.stderr)
        else:
            try:
                descriptor, observation_path = tempfile.mkstemp(
                    prefix=".tpen-verification-observation-", suffix=".json"
                )
                os.close(descriptor)
                os.unlink(observation_path)
                environment = os.environ.copy()
                environment[OBSERVATION_ENV] = observation_path
                environment[NONCE_ENV] = nonce
                tool_directory = os.path.dirname(os.path.abspath(__file__))
                existing_pythonpath = environment.get("PYTHONPATH")
                environment["PYTHONPATH"] = tool_directory + (
                    os.pathsep + existing_pythonpath if existing_pythonpath else ""
                )
                process = subprocess.run(
                    _observer_command(command),
                    capture_output=True,
                    text=True,
                    check=False,
                    env=environment,
                )
                if process.stdout:
                    sys.stdout.write(process.stdout)
                if process.stderr:
                    sys.stdout.write(process.stderr)
                pytest_exit = process.returncode
            except (OSError, subprocess.SubprocessError) as error:
                print("pytest invocation failed: " + str(error), file=sys.stderr)
    finally:
        after = _git_probe()
        observation, observation_error = _read_observation(observation_path, nonce)
        facts = _receipt_facts(
            command,
            classification,
            before,
            after,
            pytest_exit,
            observation,
            observation_error,
            command_error,
        )
        print(format_receipt(facts))
        if options.receipt_json:
            try:
                _write_json(options.receipt_json, facts)
            except OSError as error:
                print("receipt JSON write failed: " + str(error), file=sys.stderr)
                return_code = 1
            else:
                return_code = 0
        else:
            return_code = 0
        if observation_path:
            try:
                os.unlink(observation_path)
            except OSError:
                pass
    if pytest_exit != 0:
        return pytest_exit
    return return_code


def _parse_receipt_line(line: str) -> Optional[Dict[str, str]]:
    """Parse a receipt line, preserving quoted variable-length fields."""

    if not line.startswith(SENTINEL + " "):
        return None
    argv_marker = line.find(" argv=", line.find(" xpassed="))
    if argv_marker < 0:
        return None
    prefix = line[:argv_marker]
    argv = line[argv_marker + len(" argv=") :]
    header = SENTINEL + " " + VERSION + " "
    if not prefix.startswith(header):
        return None
    field_names = [
        "provenance",
        "head",
        "head_after",
        "head_stable",
        "tracked_clean",
        "tracked_clean_before",
        "tracked_clean_after",
        "branch",
        "selection",
        "baseline_eligible",
        "pytest_exit",
        "collected",
        "passed",
        "failed",
        "errors",
        "skipped",
        "xfailed",
        "xpassed",
    ]
    cursor = len(header)
    values: Dict[str, str] = {"argv": argv}
    for index, field_name in enumerate(field_names):
        marker = field_name + "="
        if not prefix.startswith(marker, cursor):
            return None
        value_start = cursor + len(marker)
        if index + 1 < len(field_names):
            next_marker = " " + field_names[index + 1] + "="
            value_end = prefix.find(next_marker, value_start)
            if value_end < 0:
                return None
        else:
            value_end = len(prefix)
        values[field_name] = prefix[value_start:value_end]
        cursor = value_end + (1 if index + 1 < len(field_names) else 0)
    if set(values) != set(field_names) | {"argv"}:
        return None
    if values["provenance"] not in {"OK", "BROKEN"}:
        return None
    for field in ("head", "head_after"):
        if values[field] != "UNKNOWN" and not _HEX40.fullmatch(values[field]):
            return None
    for field in ("head_stable", "tracked_clean", "tracked_clean_before", "tracked_clean_after"):
        if values[field] not in {"YES", "NO"}:
            return None
    if not (
        values["selection"] == "UNSELECTED"
        or values["selection"].startswith("SELECTED[")
        and values["selection"].endswith("]")
        or values["selection"].startswith("UNKNOWN[")
        and values["selection"].endswith("]")
    ):
        return None
    if not (
        values["baseline_eligible"] == "YES[]"
        or values["baseline_eligible"].startswith("NO[")
        and values["baseline_eligible"].endswith("]")
    ):
        return None
    try:
        int(values["pytest_exit"])
    except ValueError:
        return None
    for field in ("collected",) + _COUNT_FIELDS:
        if values[field] != "UNKNOWN":
            try:
                if int(values[field]) < 0:
                    return None
            except ValueError:
                return None
    return values


def _receipt_state(value: str) -> str:
    """Return YES or NO from the structured baseline field."""

    return "YES" if value.startswith("YES[") else "NO"


def _check_consistency(receipt: Dict[str, str]) -> List[str]:
    """Re-derive receipt implications instead of trusting its labels."""

    failures: List[str] = []
    counts_unknown = any(receipt[field] == "UNKNOWN" for field in ("collected",) + _COUNT_FIELDS)
    unknown_provenance = (
        receipt["head"] == "UNKNOWN"
        or receipt["head_after"] == "UNKNOWN"
        or receipt["branch"] == "UNKNOWN"
        or receipt["selection"].startswith("UNKNOWN[")
        or counts_unknown
    )
    if receipt["provenance"] == "OK" and unknown_provenance:
        failures.append("provenance=OK beside UNKNOWN evidence")
    if receipt["head_stable"] == "YES" and receipt["head"] != receipt["head_after"]:
        failures.append("head differs while head_stable=YES")
    if receipt["provenance"] == "OK" and receipt["head_stable"] != "YES":
        failures.append("provenance=OK requires head_stable=YES")
    derived_tracked_clean = (
        "YES"
        if receipt["tracked_clean_before"] == "YES" and receipt["tracked_clean_after"] == "YES"
        else "NO"
    )
    if receipt["tracked_clean"] != derived_tracked_clean:
        failures.append("tracked_clean disagrees with before/after fields")

    reasons: List[str] = []
    if receipt["provenance"] != "OK":
        reasons.append("provenance")
    if receipt["selection"] != "UNSELECTED":
        reasons.append("selection")
    if receipt["head_stable"] != "YES":
        reasons.append("head_unstable")
    if receipt["head"] != receipt["head_after"]:
        reasons.append("head_mismatch")
    if receipt["tracked_clean_before"] != "YES":
        reasons.append("tracked_dirty_before")
    if receipt["tracked_clean_after"] != "YES":
        reasons.append("tracked_dirty_after")
    if counts_unknown:
        reasons.append("counts_unknown")
    if int(receipt["pytest_exit"]) != 0:
        reasons.append("pytest_exit")
    expected = "YES" if not reasons else "NO"
    if _receipt_state(receipt["baseline_eligible"]) != expected:
        failures.append("baseline_eligible disagrees with receipt evidence")
    return failures


def _check_command(options: argparse.Namespace) -> int:
    """Fail closed when a log does not prove the requested conditions."""

    try:
        if options.log == "-":
            text = sys.stdin.read()
        else:
            with open(options.log, "r", encoding="utf-8") as handle:
                text = handle.read()
    except OSError as error:
        print("verification receipt check failed: cannot read log: " + str(error), file=sys.stderr)
        return 1

    receipt_lines = [line for line in text.splitlines() if line.startswith(SENTINEL)]
    if not receipt_lines:
        print("verification receipt check failed: no receipt line present", file=sys.stderr)
        return 1
    receipts = [_parse_receipt_line(line) for line in receipt_lines]
    if any(receipt is None for receipt in receipts):
        print("verification receipt check failed: malformed receipt line", file=sys.stderr)
        return 1
    parsed = [receipt for receipt in receipts if receipt is not None]
    first = parsed[0]
    if any(receipt != first for receipt in parsed[1:]):
        print("verification receipt check failed: receipt lines disagree", file=sys.stderr)
        return 1
    failures = _check_consistency(first)
    if first["provenance"] != "OK":
        failures.append("provenance=" + first["provenance"])
    if options.require_baseline and first["baseline_eligible"] != "YES[]":
        failures.append("baseline_eligible=" + first["baseline_eligible"])
    if options.expect_head is not None and first["head"] != options.expect_head:
        failures.append("head expected " + options.expect_head + ", got " + first["head"])
    if failures:
        print("verification receipt check failed: " + "; ".join(_unique(failures)), file=sys.stderr)
        return 1
    return 0


def _run_options(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    """Parse wrapper options while preserving the command after bare ``--``."""

    values = list(argv)
    try:
        separator = values.index("--")
    except ValueError:
        raise SystemExit("run requires a bare -- before the command")
    parser = argparse.ArgumentParser(prog="verification_receipt.py run")
    parser.add_argument("--claim-baseline", action="store_true")
    parser.add_argument("--receipt-json")
    options = parser.parse_args(values[:separator])
    command = values[separator + 1 :]
    if not command:
        raise SystemExit("run requires a command after --")
    return options, command


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the selected receipt subcommand."""

    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        print("usage: verification_receipt.py {run,check}", file=sys.stderr)
        return 2
    if values[0] == "run":
        try:
            options, command = _run_options(values[1:])
        except SystemExit as error:
            print(str(error), file=sys.stderr)
            return 2
        return _run_command(options, command)
    if values[0] == "check":
        parser = argparse.ArgumentParser(prog="verification_receipt.py check")
        parser.add_argument("--require-baseline", action="store_true")
        parser.add_argument("--expect-head")
        parser.add_argument("log")
        try:
            options = parser.parse_args(values[1:])
        except SystemExit as error:
            return int(error.code)
        return _check_command(options)
    print("unknown subcommand: " + values[0], file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
