from __future__ import annotations

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
) -> Iterator[str]:
    payload = item or _item()
    if active_claim_ids is None:
        active_claim_ids = {ITEM_ID} if payload.get("isClaimed") is True else set()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
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
        with _api() as api_url:
            receipt = self.check(api_url)
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(receipt["itemId"], ITEM_ID)
        self.assertEqual(receipt["branch"], "codex/guard-test")
        self.assertEqual(receipt["head"], _run(self.repo, "git", "rev-parse", "HEAD"))

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

    def test_pre39_subprocess_refuses_and_current_interpreter_passes(self) -> None:
        with _api() as api_url:
            args = _guard_args(self.repo, api_url)
            control = subprocess.run(
                [sys.executable, str(MODULE_PATH), *args],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(control.returncode, 0, control.stderr)
            self.assertEqual(json.loads(control.stdout)["status"], "ok")

            interpreters, attempts = _pre39_interpreters()
            if not interpreters:
                self.skipTest("Arm A skipped; candidates tried: " + "; ".join(attempts))
            for interpreter, version in interpreters:
                with self.subTest(interpreter=str(interpreter), version=version):
                    result = subprocess.run(
                        [str(interpreter), str(MODULE_PATH), *args],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    verdict = json.loads(result.stderr)
                    self.assertEqual(verdict["status"], "blocked")
                    self.assertIn("3.9", verdict["reason"])
                    self.assertIn(version, verdict["reason"])
                    self.assertIn(
                        "uv run --no-project python tools/check_authoritative_edit.py",
                        verdict["reason"],
                    )
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertNotIn("AttributeError", result.stderr)
                    self.assertNotIn("removesuffix", result.stderr)

    def test_interpreter_gate_has_both_directions_at_boundary(self) -> None:
        with self.assertRaises(GUARD.UnsupportedInterpreter):
            GUARD._require_supported_interpreter((3, 8, 20))
        for version_info in ((3, 9, 0), (3, 12, 13), sys.version_info):
            with self.subTest(version_info=version_info):
                GUARD._require_supported_interpreter(version_info)


if __name__ == "__main__":
    unittest.main()
