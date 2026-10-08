from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "tools" / "check_authoritative_edit.py"
SPEC = importlib.util.spec_from_file_location("check_authoritative_edit", MODULE_PATH)
assert SPEC and SPEC.loader
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)

ITEM_ID = "11111111-1111-4111-8111-111111111111"
ROOT_ID = GUARD.DEFAULT_PROJECT_ROOT_ID


def _run(cwd: Path, *args: str) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _pre39_interpreters() -> tuple[list[tuple[Path, str]], list[str]]:
    candidates = []
    configured = os.environ.get("TPEN_GUARD_PRE39_PYTHON")
    if configured:
        candidates.append(("TPEN_GUARD_PRE39_PYTHON", configured))
    candidates.extend((name, name) for name in ("python3.8", "python3.7", "python3", "python"))

    found: list[tuple[Path, str]] = []
    attempts: list[str] = []
    seen: set[Path] = set()
    version_code = "import sys; print('%d.%d.%d' % sys.version_info[:3])"
    for label, candidate in candidates:
        executable = shutil.which(candidate)
        if executable is None:
            attempts.append(f"{label}: not found")
            continue
        realpath = Path(executable).resolve()
        result = subprocess.run(
            [str(realpath), "-c", version_code],
            check=False,
            capture_output=True,
            text=True,
        )
        version = result.stdout.strip()
        attempts.append(f"{label}: {realpath} -> {version or result.stderr.strip() or 'no version'}")
        if result.returncode != 0 or not version:
            continue
        try:
            parsed = tuple(int(part) for part in version.split("."))
        except ValueError:
            continue
        if (3, 7) <= parsed[:2] < (3, 9) and realpath not in seen:
            seen.add(realpath)
            found.append((realpath, version))
    return found, attempts


def _guard_args(repo: Path, api_url: str) -> list[str]:
    return [
        "--cwd",
        str(repo),
        "--api-url",
        api_url,
        "--item",
        ITEM_ID,
        "--project-root-id",
        ROOT_ID,
    ]


def _item(
    *,
    item_type: str = "implementation-slice",
    role: str = "work",
    claimed: bool = True,
    body: str = "scope",
) -> dict:
    return {
        "id": ITEM_ID,
        "type": item_type,
        "role": role,
        "isClaimed": claimed,
        "notes": [{"key": "acceptance-contract", "role": "queue", "body": body}],
    }


@contextmanager
def _api(
    *,
    item: dict | None = None,
    root_id: str = ROOT_ID,
    active_claim_ids: set[str] | None = None,
    count_file: Path | None = None,
) -> Iterator[str]:
    payload = item or _item()
    if active_claim_ids is None:
        active_claim_ids = {ITEM_ID} if payload.get("isClaimed") is True else set()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if count_file is not None:
                previous = count_file.read_text() if count_file.exists() else ""
                count_file.write_text(previous + "hit\n")
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == f"/api/v1/items/{ITEM_ID}" and parsed.query == "include=notes":
                body = payload
            elif parsed.path == "/api/v1/items":
                query = urllib.parse.parse_qs(parsed.query)
                expected = {
                    "rootId": [ROOT_ID],
                    "role": ["work"],
                    "type": ["implementation-slice"],
                    "claimStatus": ["claimed"],
                    "page": ["1"],
                    "pageSize": ["100"],
                }
                if query != expected:
                    self.send_error(400)
                    return
                items = [payload] if ITEM_ID in active_claim_ids else []
                body = {
                    "items": items,
                    "page": 1,
                    "pageSize": 100,
                    "totalItems": len(items),
                    "hasMore": False,
                }
            elif parsed.path == f"/api/v1/items/{ITEM_ID}/breadcrumbs":
                body = [{"id": root_id}, {"id": ITEM_ID}]
            else:
                self.send_error(404)
                return
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@contextmanager
def _fake_git_path() -> Iterator[tuple[Path, Path]]:
    """Put an instrumented git wrapper first on PATH.

    Yields
    ------
    tuple[pathlib.Path, pathlib.Path]
        The temporary directory containing the wrapper and its log path.
    """
    with tempfile.TemporaryDirectory(prefix="guard-git-instrument-") as raw:
        directory = Path(raw)
        log = directory / "git.log"
        real_git = shutil.which("git")
        assert real_git
        wrapper = directory / "git"
        wrapper.write_text(f"#!/bin/sh\nprintf '%s\\n' git >> {log}\nexec {real_git} \"$@\"\n")
        wrapper.chmod(0o755)
        yield directory, log


class AuthoritativeEditGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "repo"
        self.repo.mkdir()
        _run(self.repo, "git", "init", "-q", "-b", "codex/guard-test")
        _run(self.repo, "git", "remote", "add", "origin", "https://github.com/YizhongHu/TPEN.git")
        (self.repo / "tracked.txt").write_text("clean\n")
        _run(self.repo, "git", "add", "tracked.txt")
        _run(
            self.repo,
            "git",
            "-c",
            "user.name=Guard Test",
            "-c",
            "user.email=guard@example.test",
            "commit",
            "-qm",
            "fixture",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def check(self, api_url: str) -> dict[str, str]:
        return GUARD.check_launch(self.repo, ITEM_ID, api_url, ROOT_ID)

    def test_allows_clean_agent_branch_with_live_work_claim(self) -> None:
        (self.repo / "untracked-research-data").write_text("preserve me\n")
        expected_head = _run(self.repo, "git", "rev-parse", "HEAD")
        with _api() as api_url:
            receipt = self.check(api_url)
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(receipt["itemId"], ITEM_ID)
        self.assertEqual(receipt["branch"], "codex/guard-test")
        self.assertEqual(receipt["head"], expected_head)
        self.assertEqual(_run(self.repo, "git", "rev-parse", "HEAD"), expected_head)

    def test_rejects_non_agent_branch(self) -> None:
        _run(self.repo, "git", "branch", "-m", "dev")
        with _api() as api_url, self.assertRaisesRegex(GUARD.GuardFailure, "agent branch"):
            self.check(api_url)

    def test_rejects_dirty_tracked_tree(self) -> None:
        (self.repo / "tracked.txt").write_text("dirty\n")
        with _api() as api_url, self.assertRaisesRegex(GUARD.GuardFailure, "tracked worktree is dirty"):
            self.check(api_url)

    def test_rejects_invalid_item_contract(self) -> None:
        cases = [
            (_item(item_type="research"), "item must have type"),
            (_item(body="  "), "non-empty queue acceptance-contract"),
        ]
        for item, message in cases:
            with self.subTest(message=message):
                with _api(item=item) as api_url, self.assertRaisesRegex(GUARD.GuardFailure, message):
                    self.check(api_url)

    def test_rejects_expired_work_claim_residue(self) -> None:
        with _api(active_claim_ids=set()) as api_url:
            with self.assertRaisesRegex(GUARD.GuardFailure, "active unexpired claim"):
                self.check(api_url)

    def test_rejects_terminal_claim_residue(self) -> None:
        with _api(item=_item(role="terminal"), active_claim_ids={ITEM_ID}) as api_url:
            with self.assertRaisesRegex(GUARD.GuardFailure, "item must be in work role"):
                self.check(api_url)

    def test_rejects_no_claim(self) -> None:
        with _api(item=_item(claimed=False), active_claim_ids=set()) as api_url:
            with self.assertRaisesRegex(GUARD.GuardFailure, "active unexpired claim"):
                self.check(api_url)

    def test_rejects_item_outside_tpen_root(self) -> None:
        with _api(root_id="22222222-2222-4222-8222-222222222222") as api_url:
            with self.assertRaisesRegex(GUARD.GuardFailure, "must belong to TPEN project root"):
                self.check(api_url)

    def test_rejects_unreachable_server(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardFailure, "Task Orchestrator read failed"):
            self.check("http://127.0.0.1:1")

    def test_rejects_non_loopback_or_unsafe_api_url(self) -> None:
        for api_url in [
            "https://127.0.0.1:3001",
            "http://example.com:3001",
            "http://127.0.0.1:3001/api/v1",
            "http://user:secret@127.0.0.1:3001",
        ]:
            with self.subTest(api_url=api_url):
                with self.assertRaisesRegex(GUARD.GuardFailure, "Task Orchestrator API"):
                    self.check(api_url)

    def test_current_interpreter_passes_full_guard_and_reaches_instruments(self) -> None:
        """Prove the supported-runtime control independently and completely.

        The control asserts the full success envelope and instruments both git
        and the Task Orchestrator API. Keeping it separate from discovery means
        an absent old interpreter cannot report this control as skipped.
        """
        with _fake_git_path() as (git_directory, git_log), tempfile.TemporaryDirectory() as raw:
            api_log = Path(raw) / "api.log"
            expected_head = _run(self.repo, "git", "rev-parse", "HEAD")
            with _api(count_file=api_log) as api_url:
                result = subprocess.run(
                    [sys.executable, str(MODULE_PATH), *_guard_args(self.repo, api_url)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "PATH": f"{git_directory}:{os.environ['PATH']}"},
                )
                expected = {
                    "status": "ok",
                    "itemId": ITEM_ID,
                    "projectRootId": ROOT_ID,
                    "cwd": str(self.repo.resolve()),
                    "branch": "codex/guard-test",
                    "head": expected_head,
                }
                self.assertTrue(git_log.exists() and git_log.read_text())
                self.assertTrue(api_log.exists() and api_log.read_text())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout), expected)
        self.assertEqual(_run(self.repo, "git", "rev-parse", "HEAD"), expected_head)

    def test_pre39_subprocess_refuses_before_git_or_api(self) -> None:
        """Exercise every CLI shape under a real pre-3.9 interpreter.

        The skip is decided before any refusal assertions. Every subcase must
        produce the same structured interpreter refusal and must avoid both
        repository and API instruments.
        """
        interpreters, attempts = _pre39_interpreters()
        if not interpreters:
            self.skipTest(
                "NOT EXERCISED: no real pre-3.9 interpreter; candidates tried: "
                + "; ".join(attempts)
            )

        with tempfile.TemporaryDirectory() as raw:
            api_log = Path(raw) / "api.log"
            with _api(count_file=api_log) as api_url:
                for interpreter, version in interpreters:
                    with _fake_git_path() as (git_directory, git_log):
                        cases = [
                            ("valid", _guard_args(self.repo, api_url)),
                            (
                                "missing-item",
                                [
                                    "--cwd",
                                    str(self.repo),
                                    "--api-url",
                                    api_url,
                                    "--project-root-id",
                                    ROOT_ID,
                                ],
                            ),
                            ("unknown-option", ["--unknown"]),
                            ("help", ["--help"]),
                            (
                                "nonexistent-cwd",
                                ["--cwd", str(self.repo / "missing"), "--item", ITEM_ID],
                            ),
                            (
                                "unsafe-api",
                                ["--api-url", "http://example.com:3001", "--item", ITEM_ID],
                            ),
                        ]
                        env = {**os.environ, "PATH": f"{git_directory}:{os.environ['PATH']}"}
                        baseline_git = ""
                        baseline_api = ""
                        for name, args in cases:
                            with self.subTest(interpreter=str(interpreter), version=version, case=name):
                                result = subprocess.run(
                                    [str(interpreter), str(MODULE_PATH), *args],
                                    check=False,
                                    capture_output=True,
                                    text=True,
                                    env=env,
                                )
                                self.assertEqual(result.returncode, 1, result.stderr)
                                self.assertEqual(result.stdout, "")
                                verdict = json.loads(result.stderr)
                                self.assertEqual(set(verdict), {"status", "reason"})
                                self.assertEqual(verdict["status"], "blocked")
                                for expected in (
                                    version,
                                    "3.9+",
                                    "no guard precondition was evaluated",
                                    "uv run --no-project python tools/check_authoritative_edit.py",
                                ):
                                    self.assertIn(expected, verdict["reason"])
                                for forbidden in ("Traceback", "AttributeError", "removesuffix"):
                                    self.assertNotIn(forbidden, result.stderr)
                                self.assertEqual(
                                    git_log.read_text() if git_log.exists() else "", baseline_git
                                )
                                self.assertEqual(
                                    api_log.read_text() if api_log.exists() else "", baseline_api
                                )

    def test_simulated_main_and_check_launch_gate_boundary(self) -> None:
        """Pin gate placement at both entry points with simulated versions.

        Notes
        -----
        These are injected version tuples, not executions under old Python
        interpreters. Rejected versions must reach neither parser, repository
        validation, cwd resolution, nor API work; accepted versions must reach
        a downstream sentinel.
        """
        current_version = tuple(sys.version_info[:3])
        for version in ((3, 7, 0), (3, 8, 20)):
            with self.subTest(version=version), mock.patch.object(GUARD.sys, "version_info", version):
                stderr = io.StringIO()
                with mock.patch.object(
                    GUARD, "_parser", side_effect=AssertionError("parser reached")
                ), mock.patch.object(
                    GUARD, "_validated_api_url", side_effect=AssertionError("URL validation reached")
                ), mock.patch.object(
                    GUARD, "_validate_repository", side_effect=AssertionError("repository validation reached")
                ), mock.patch.object(
                    GUARD, "_validate_item", side_effect=AssertionError("API reached")
                ), mock.patch.object(
                    GUARD.Path, "resolve", side_effect=AssertionError("cwd resolution reached")
                ):
                    with contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as raised:
                            GUARD.main(["--item", ITEM_ID])
                self.assertEqual(raised.exception.code, 1)
                verdict = json.loads(stderr.getvalue())
                self.assertEqual(set(verdict), {"status", "reason"})
                self.assertEqual(verdict["status"], "blocked")
                with mock.patch.object(
                    GUARD, "_validated_api_url", side_effect=AssertionError("URL validation reached")
                ), mock.patch.object(
                    GUARD, "_validate_repository", side_effect=AssertionError("repository validation reached")
                ), mock.patch.object(
                    GUARD, "_validate_item", side_effect=AssertionError("API reached")
                ), mock.patch.object(
                    GUARD.Path, "resolve", side_effect=AssertionError("cwd resolution reached")
                ):
                    with self.assertRaises(GUARD.UnsupportedInterpreter):
                        GUARD.check_launch(Path("/never"), ITEM_ID, "http://127.0.0.1:3001", ROOT_ID)

        for version in ((3, 9, 0), current_version):
            with self.subTest(version=version), mock.patch.object(GUARD.sys, "version_info", version):
                parser_seen = []
                original_parser = GUARD._parser
                expected_args = original_parser().parse_args(["--item", ITEM_ID])

                def parser() -> argparse.ArgumentParser:
                    parser_seen.append(True)
                    return original_parser()

                check_launch = mock.Mock(side_effect=GUARD.GuardFailure("main downstream sentinel"))
                stderr = io.StringIO()
                with mock.patch.object(GUARD, "_parser", side_effect=parser), mock.patch.object(
                    GUARD, "check_launch", check_launch
                ), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        GUARD.main(["--item", ITEM_ID])
                self.assertEqual(parser_seen, [True])
                check_launch.assert_called_once_with(
                    expected_args.cwd,
                    expected_args.item,
                    expected_args.api_url,
                    expected_args.project_root_id,
                )
                self.assertEqual(raised.exception.code, 1)
                self.assertEqual(
                    json.loads(stderr.getvalue()),
                    {"status": "blocked", "reason": "main downstream sentinel"},
                )
                with mock.patch.object(
                    GUARD, "_validated_api_url", side_effect=GUARD.GuardFailure("check downstream sentinel")
                ):
                    with self.assertRaisesRegex(GUARD.GuardFailure, "check downstream sentinel"):
                        GUARD.check_launch(Path("/never"), ITEM_ID, "http://127.0.0.1:3001", ROOT_ID)

    def test_supported_success_return_is_json_and_dispatches_once(self) -> None:
        """Exercise the successful path under both simulated and current Python."""
        expected_cwd = Path("/pr518-r4-success-cwd")
        expected_item = "44444444-4444-4444-8444-444444444444"
        expected_api = "http://127.0.0.1:4545"
        expected_root = "55555555-5555-4555-8555-555555555555"
        argv = [
            "--cwd",
            str(expected_cwd),
            "--item",
            expected_item,
            "--api-url",
            expected_api,
            "--project-root-id",
            expected_root,
        ]
        current_version = tuple(sys.version_info[:3])
        for version in ((3, 9, 0), current_version):
            with self.subTest(version=version):
                receipt = {
                    "status": "ok",
                    "cwd": str(expected_cwd),
                    "itemId": expected_item,
                    "api": expected_api,
                    "projectRootId": expected_root,
                }
                expected_stdout = json.dumps(receipt, sort_keys=True) + "\n"
                check_launch = mock.Mock(return_value=receipt)
                stdout = io.StringIO()
                stderr = io.StringIO()
                with mock.patch.object(GUARD.sys, "version_info", version), mock.patch.object(
                    GUARD, "check_launch", check_launch
                ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = GUARD.main(argv)
                self.assertEqual(result, 0)
                self.assertEqual(stdout.getvalue(), expected_stdout)
                self.assertEqual(stderr.getvalue(), "")
                check_launch.assert_called_once_with(
                    expected_cwd, expected_item, expected_api, expected_root
                )

    def test_empty_pre39_discovery_keeps_supported_control_green(self) -> None:
        """Ensure an empty old-runtime census cannot skip the modern control.

        The recording result is the forward regression guard for the split: one
        supported-runtime test must report one success and no skip even when
        pre-3.9 discovery is forced empty.
        """
        class RecordingResult(unittest.TestResult):
            def __init__(self) -> None:
                super().__init__()
                self.successes = 0

            def addSuccess(self, test: unittest.TestCase) -> None:
                self.successes += 1
                super().addSuccess(test)

        control = self.__class__("test_current_interpreter_passes_full_guard_and_reaches_instruments")
        result = RecordingResult()
        with mock.patch.object(sys.modules[__name__], "_pre39_interpreters", return_value=([], ["forced empty discovery"])):
            control.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.successes, 1)
        self.assertEqual(result.skipped, [])

    def test_interpreter_gate_has_both_directions_at_boundary(self) -> None:
        with self.assertRaises(GUARD.UnsupportedInterpreter):
            GUARD._require_supported_interpreter((3, 8, 20))
        for version_info in ((3, 9, 0), (3, 12, 13), sys.version_info):
            with self.subTest(version_info=version_info):
                GUARD._require_supported_interpreter(version_info)


if __name__ == "__main__":
    unittest.main()
