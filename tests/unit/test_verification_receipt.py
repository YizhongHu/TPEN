from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "tools" / "verification_receipt.py"
SPEC = importlib.util.spec_from_file_location("verification_receipt", MODULE_PATH)
assert SPEC and SPEC.loader
RECEIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIPT)


def _run(cwd: Path, *args: str) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


class VerificationReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "repo"
        self.repo.mkdir()
        _run(self.repo, "git", "init", "-q", "-b", "codex/receipt-test")
        (self.repo / "tracked.txt").write_text("clean\n")
        _run(self.repo, "git", "add", "tracked.txt")
        _run(
            self.repo,
            "git",
            "-c",
            "user.name=Receipt Test",
            "-c",
            "user.email=receipt@example.test",
            "commit",
            "-qm",
            "fixture",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _pytest_script(self, body: str) -> Path:
        script = self.repo / "pytest"
        script.write_text("#!" + sys.executable + "\n" + body)
        script.chmod(script.stat().st_mode | 0o111)
        return script

    def _run_receipt(self, command: list[str], *extra: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        old_cwd = os.getcwd()
        os.chdir(self.repo)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = RECEIPT.main(["run", *extra, "--", *command])
        finally:
            os.chdir(old_cwd)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_classifier_table_includes_negative_controls_and_prefix_skip(self) -> None:
        cases = [
            (["pytest", "-k", "expr"], "SELECTED", ["-k", "expr"]),
            (["pytest", "-k=expr"], "SELECTED", ["-k=expr"]),
            (["pytest", "tests/unit/x.py"], "SELECTED", ["tests/unit/x.py"]),
            (["pytest", "tests/unit/x.py::test_y"], "SELECTED", ["tests/unit/x.py::test_y"]),
            (["pytest", "-m", "slow"], "SELECTED", ["-m", "slow"]),
            (["pytest", "--ignore", "tests/slow"], "SELECTED", ["--ignore", "tests/slow"]),
            (["pytest", "-x"], "SELECTED", ["-x"]),
            (["pytest", "--maxfail=1"], "SELECTED", ["--maxfail=1"]),
            (["pytest", "--co"], "SELECTED", ["--co"]),
            (["python", "-m", "pytest", "-q"], "UNSELECTED", []),
            (["python", "-m", "pytest", "-m", "slow"], "SELECTED", ["-m", "slow"]),
        ]
        for command, expected_selection, expected_tokens in cases:
            with self.subTest(command=command):
                result = RECEIPT.classify_command(command)
                self.assertEqual(result["selection"], expected_selection)
                self.assertEqual(result["selection_tokens"], expected_tokens)

    def test_classifier_has_both_polarities(self) -> None:
        self.assertEqual(RECEIPT.classify_command(["pytest", "-q"])["selection"], "UNSELECTED")
        self.assertEqual(RECEIPT.classify_command(["pytest", "tests/unit"])["selection"], "SELECTED")

    def test_claim_baseline_refuses_selected_command_before_pytest(self) -> None:
        script = self._pytest_script("open('ran', 'w').close()\n")
        code, stdout, stderr = self._run_receipt([str(script), "-k", "expr"], "--claim-baseline")
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("-k", stderr)
        self.assertFalse((self.repo / "ran").exists())

    def test_head_move_breaks_provenance_and_baseline(self) -> None:
        script = self._pytest_script(
            "import subprocess\n"
            "subprocess.run(['git', '-c', 'user.name=Receipt Test', '-c', "
            "'user.email=receipt@example.test', 'commit', '--allow-empty', '-qm', 'moved'], check=True)\n"
            "print('1 passed in 0.01s')\n"
        )
        code, stdout, _ = self._run_receipt([str(script)])
        self.assertEqual(code, 0)
        self.assertIn("head_stable=NO", stdout)
        self.assertIn("provenance=BROKEN", stdout)
        self.assertIn("baseline_eligible=NO", stdout)

    def test_dirty_tracked_tree_disqualifies_baseline(self) -> None:
        (self.repo / "tracked.txt").write_text("dirty\n")
        script = self._pytest_script("print('1 passed in 0.01s')\n")
        code, stdout, _ = self._run_receipt([str(script)])
        self.assertEqual(code, 0)
        self.assertIn("tracked_clean=NO", stdout)
        self.assertIn("baseline_eligible=NO[tracked_dirty]", stdout)

    def test_check_is_fail_closed_for_missing_and_ineligible_receipts(self) -> None:
        missing = self.repo / "missing.log"
        missing.write_text("pytest crashed before printing anything\n")
        self.assertNotEqual(RECEIPT.main(["check", str(missing)]), 0)

        ineligible = self.repo / "selected.log"
        ineligible.write_text(
            "TPEN-VERIFICATION-RECEIPT v1 provenance=OK head=" + "a" * 40
            + " head_after=" + "a" * 40
            + " head_stable=YES tracked_clean=YES branch=main selection=SELECTED[-k,expr]"
            + " baseline_eligible=NO[selection] pytest_exit=0 collected=1 passed=1 failed=0"
            + " errors=0 skipped=0 xfailed=0 xpassed=0 argv=pytest -k expr\n"
        )
        self.assertNotEqual(RECEIPT.main(["check", "--require-baseline", str(ineligible)]), 0)

    def test_count_parser_reads_summary_and_missing_summary_is_unknown(self) -> None:
        counts = RECEIPT.parse_pytest_summary(
            "collected 9 items\n"
            "2 passed, 1 failed, 1 error, 3 skipped, 1 xfailed, 1 xpassed in 0.2s\n"
        )
        self.assertEqual(counts["collected"], 9)
        self.assertEqual(counts["passed"], 2)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["errors"], 1)
        self.assertEqual(counts["skipped"], 3)
        self.assertEqual(counts["xfailed"], 1)
        self.assertEqual(counts["xpassed"], 1)
        missing = RECEIPT.parse_pytest_summary("internal error before pytest summary\n")
        self.assertIsNone(missing["passed"])
        self.assertIsNone(missing["collected"])


if __name__ == "__main__":
    unittest.main()
