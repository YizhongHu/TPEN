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
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple


SENTINEL = "TPEN-VERIFICATION-RECEIPT"
VERSION = "v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_COUNT_FIELDS = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
_SUMMARY_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<word>passed|failed|errors?|skipped|xfailed|xpassed)\b"
)
_COLLECTED_RE = re.compile(r"\bcollected\s+(?P<count>\d+)\s+items?\b", re.IGNORECASE)
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
    "--durations",
    "--basetemp",
    "--log-file",
}


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
    """Classify pytest arguments and enumerate every disqualifying token.

    Parameters
    ----------
    pytest_args
        Arguments after the pytest executable or ``python -m pytest`` prefix.

    Returns
    -------
    tuple[str, list[str]]
        ``("UNSELECTED", [])`` for a whole-suite invocation, otherwise
        ``("SELECTED", tokens)`` where ``tokens`` are exact argv tokens that
        caused the classification.
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
        if token.startswith("-k=") or token.startswith("-m="):
            disqualifying.append(token)
            index += 1
            continue

        if token in _VALUE_OPTIONS:
            if index + 1 < len(values):
                index += 2
            else:
                index += 1
            continue
        if token.startswith("--") and any(token.startswith(option + "=") for option in _VALUE_OPTIONS):
            index += 1
            continue

        matched_long = None
        for option in _SELECTION_LONG:
            if token == option or token.startswith(option + "="):
                matched_long = option
                break
        if matched_long is not None:
            disqualifying.append(token)
            takes_value = matched_long in {"--ignore", "--ignore-glob", "--deselect", "--maxfail"}
            if token == matched_long and takes_value and index + 1 < len(values):
                disqualifying.append(values[index + 1])
                index += 2
            else:
                index += 1
            continue

        # Handle clustered short flags such as -xq. Pytest's -k and -m
        # spellings are also caught here when supplied in a cluster.
        if token.startswith("-") and not token.startswith("--") and len(token) > 1:
            if any(flag in token[1:] for flag in ("k", "m", "x")):
                disqualifying.append(token)
                index += 1
                continue
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


def parse_pytest_summary(output: str) -> Dict[str, Optional[int]]:
    """Parse counts from the terminal summary line in pytest output.

    Parameters
    ----------
    output
        Combined pytest stdout and stderr.

    Returns
    -------
    dict[str, int | None]
        Count fields. All fields are ``None`` when no summary line is found;
        a known summary line gives absent categories the value zero.
    """

    summary_matches = []
    for line in output.splitlines():
        matches = list(_SUMMARY_RE.finditer(line))
        if matches:
            summary_matches = matches
    counts: Dict[str, Optional[int]] = {field: None for field in _COUNT_FIELDS}
    collected: Optional[int] = None
    collected_matches = list(_COLLECTED_RE.finditer(output))
    if collected_matches:
        collected = int(collected_matches[-1].group("count"))
    if not summary_matches:
        counts["collected"] = collected
        return counts

    for field in _COUNT_FIELDS:
        counts[field] = 0
    for match in summary_matches:
        field = match.group("word")
        if field == "error":
            field = "errors"
        counts[field] = int(match.group("count"))
    if collected is None:
        collected = sum(int(counts[field] or 0) for field in _COUNT_FIELDS)
    counts["collected"] = collected
    return counts


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


def _unknown_counts(counts: Dict[str, Optional[int]]) -> bool:
    """Whether any receipt count is unavailable."""

    return any(counts.get(field) is None for field in ("collected",) + _COUNT_FIELDS)


def _receipt_facts(
    command: Sequence[str],
    classification: Dict[str, object],
    before: Dict[str, object],
    after: Dict[str, object],
    pytest_exit: int,
    counts: Dict[str, Optional[int]],
    command_error: Optional[str] = None,
) -> Dict[str, object]:
    """Build the canonical facts shared by text and JSON receipts."""

    head = str(before["head"])
    head_after = str(after["head"])
    stable = "YES" if bool(before["git_ok"]) and bool(after["git_ok"]) and head == head_after else "NO"
    provenance = "OK" if stable == "YES" and bool(before["git_ok"]) else "BROKEN"
    reasons: List[str] = []
    if provenance != "OK":
        reasons.append("provenance")
    if classification["selection"] != "UNSELECTED":
        reasons.append("selection")
    if stable != "YES":
        reasons.append("head_unstable")
    if before["tracked_clean"] != "YES":
        reasons.append("tracked_dirty")
    if _unknown_counts(counts):
        reasons.append("counts_unknown")
    if pytest_exit != 0:
        reasons.append("pytest_exit")
    if command_error is not None:
        reasons.append(command_error)

    return {
        "provenance": provenance,
        "head": head,
        "head_after": head_after,
        "head_stable": stable,
        "tracked_clean": str(before["tracked_clean"]),
        "branch": str(before["branch"]),
        "selection": str(classification["selection"]),
        "selection_tokens": list(classification["selection_tokens"]),
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
    }


def _display_number(value: object) -> str:
    """Render a receipt number or its required unknown marker."""

    return "UNKNOWN" if value is None else str(value)


def format_receipt(facts: Dict[str, object]) -> str:
    """Serialize canonical facts into the one-line greppable receipt."""

    selection_tokens = ",".join(_shell_token(str(token)) for token in facts["selection_tokens"])
    reasons = ",".join(str(reason) for reason in facts["baseline_reasons"])
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


def _run_command(options: argparse.Namespace, command: Sequence[str]) -> int:
    """Run pytest, forward its output, and emit exactly one receipt line."""

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
    output = ""
    try:
        if command_error is not None:
            print("verification receipt refused: " + str(command_error), file=sys.stderr)
        else:
            try:
                process = subprocess.run(list(command), capture_output=True, text=True, check=False)
                output = process.stdout + process.stderr
                if process.stdout:
                    sys.stdout.write(process.stdout)
                if process.stderr:
                    sys.stdout.write(process.stderr)
                pytest_exit = process.returncode
            except (OSError, subprocess.SubprocessError) as error:
                print("pytest invocation failed: " + str(error), file=sys.stderr)
    finally:
        after = _git_probe()
        counts = parse_pytest_summary(output)
        facts = _receipt_facts(command, classification, before, after, pytest_exit, counts, command_error)
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
    if pytest_exit != 0:
        return pytest_exit
    return return_code


def _parse_receipt_line(line: str) -> Optional[Dict[str, str]]:
    """Parse a receipt line, preserving the variable-length argv field."""

    if not line.startswith(SENTINEL + " "):
        return None
    xpassed_marker = line.find(" xpassed=")
    argv_marker = line.find(" argv=", xpassed_marker)
    if xpassed_marker < 0 or argv_marker < 0:
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
    required = set(field_names) | {"argv"}
    if set(values) != required:
        return None
    if values["provenance"] not in {"OK", "BROKEN"}:
        return None
    if values["head"] != "UNKNOWN" and not _HEX40.fullmatch(values["head"]):
        return None
    if values["head_after"] != "UNKNOWN" and not _HEX40.fullmatch(values["head_after"]):
        return None
    if values["head_stable"] not in {"YES", "NO"}:
        return None
    if values["tracked_clean"] not in {"YES", "NO"}:
        return None
    if not (
        values["selection"] == "UNSELECTED"
        or values["selection"].startswith("SELECTED[")
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
    for field_name in ("collected",) + _COUNT_FIELDS:
        if values[field_name] != "UNKNOWN":
            try:
                int(values[field_name])
            except ValueError:
                return None
    return values


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
    failures: List[str] = []
    if first.get("provenance") != "OK":
        failures.append("provenance=" + first.get("provenance", "<missing>"))
    if options.require_baseline and first.get("baseline_eligible") != "YES[]":
        failures.append("baseline_eligible=" + first.get("baseline_eligible", "<missing>"))
    if options.expect_head is not None and first.get("head") != options.expect_head:
        failures.append("head expected " + options.expect_head + ", got " + first.get("head", "<missing>"))
    if failures:
        print("verification receipt check failed: " + "; ".join(failures), file=sys.stderr)
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
