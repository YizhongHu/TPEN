from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple


MODULE_PATH = Path(__file__).parents[2] / "tools" / "verification_receipt.py"
SPEC = importlib.util.spec_from_file_location("verification_receipt", MODULE_PATH)
assert SPEC and SPEC.loader
RECEIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIPT)
OBSERVER_PATH = Path(__file__).parents[2] / "tools" / "_verification_observer.py"
OBSERVER_SPEC = importlib.util.spec_from_file_location("_verification_observer", OBSERVER_PATH)
assert OBSERVER_SPEC and OBSERVER_SPEC.loader
OBSERVER = importlib.util.module_from_spec(OBSERVER_SPEC)
OBSERVER_SPEC.loader.exec_module(OBSERVER)


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
        (self.repo / "test_sample.py").write_text(
            "def test_one():\n    assert True\n\n"
            "def test_two():\n    assert True\n"
        )
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
        # Cannon occasionally races another cleanup worker under .git/hooks.
        # Retry this exact temporary root instead of turning teardown noise
        # into a false unit failure.
        root = Path(self._tmp.name)
        try:
            self._tmp.cleanup()
        except OSError:
            for _ in range(3):
                try:
                    shutil.rmtree(str(root))
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                shutil.rmtree(str(root), ignore_errors=True)

    def _run_receipt(
        self,
        args: Sequence[str] = ("-q",),
        *,
        claim: bool = False,
        environment: Optional[Dict[str, str]] = None,
        receipt_json: Optional[Path] = None,
    ) -> Tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        command = [sys.executable, "-m", "pytest"] + list(args)
        old_cwd = os.getcwd()
        old_environment = os.environ.copy()
        os.chdir(self.repo)
        if environment is None or "PYTEST_ADDOPTS" not in environment:
            os.environ.pop("PYTEST_ADDOPTS", None)
        if environment:
            os.environ.update(environment)
        try:
            wrapper_args: List[str] = ["run"]
            if claim:
                wrapper_args.append("--claim-baseline")
            if receipt_json is not None:
                wrapper_args.extend(["--receipt-json", str(receipt_json)])
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = RECEIPT.main(wrapper_args + ["--"] + command)
        finally:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_environment)
        return code, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def _receipt_line(output: str) -> str:
        for line in output.splitlines():
            if line.startswith(RECEIPT.SENTINEL + " "):
                return line
        raise AssertionError("receipt line missing")

    def _write_conftest(self, body: str) -> None:
        (self.repo / "conftest.py").write_text(body)

    def test_classifier_table_is_bidirectional(self) -> None:
        # This table is an instrument for both directions, not a hope that
        # flags vanish. It covers only unambiguous pre-flight selections that
        # can be refused before allocation cost; --maxfail and --co are
        # deliberately observer-only because pytest is the authority for
        # those truncation settings. Keeping -x as pre-flight refusal is a
        # deliberate asymmetry: it has no useful value to consume, whereas
        # --maxfail/--co must pass through so the observer can report the
        # resolved option source.
        cases = [
            (["pytest", "-k", "expr"], "SELECTED"),
            (["pytest", "-k=expr"], "SELECTED"),
            (["pytest", "tests/unit/x.py"], "SELECTED"),
            (["pytest", "tests/unit/x.py::test_y"], "SELECTED"),
            (["pytest", "-m", "slow"], "SELECTED"),
            (["pytest", "--ignore", "tests/slow"], "SELECTED"),
            (["pytest", "-x"], "SELECTED"),
            (["python", "-m", "pytest", "-q"], "UNSELECTED"),
            (["python", "-m", "pytest", "-m", "slow"], "SELECTED"),
            (["pytest", "-q", "-p", "no:cacheprovider"], "UNSELECTED"),
            (["pytest", "-q", "-n", "4"], "UNSELECTED"),
            (["pytest", "-q", "-c", "pytest.ini"], "UNSELECTED"),
            (["pytest", "-q", "-o", "cache_dir=/tmp/cache"], "UNSELECTED"),
            (["pytest", "-q", "--junitxml", "/tmp/r.xml"], "UNSELECTED"),
            (["pytest", "-q", "--rootdir", "."], "UNSELECTED"),
            (["pytest", "-q", "-W", "ignore::DeprecationWarning"], "UNSELECTED"),
            (["pytest", "-q", "--durations", "10"], "UNSELECTED"),
            (["pytest", "-q", "--basetemp", "/tmp/pytest-tmp"], "UNSELECTED"),
            (["pytest", "-q", "--log-file", "/tmp/pytest.log"], "UNSELECTED"),
            (["pytest", "--color", "no"], "UNSELECTED"),
            (["pytest", "--capture", "sys"], "UNSELECTED"),
            (["pytest", "--tb", "short"], "UNSELECTED"),
            (["pytest", "--junit-xml", "/tmp/r.xml"], "UNSELECTED"),
            (["pytest", "-Wignore::RuntimeWarning"], "UNSELECTED"),
            (["pytest", "-ocache_dir=/tmp/cache"], "UNSELECTED"),
            (["pytest", "--sw"], "UNSELECTED"),
        ]
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(RECEIPT.classify_command(command)["selection"], expected)

    def test_observer_refuses_observer_only_truncation_options(self) -> None:
        for args, source in (
            (("-q", "--maxfail=1"), "option:maxfail"),
            (("-q", "--co"), "option:collectonly"),
        ):
            with self.subTest(args=args):
                code, stdout, _ = self._run_receipt(args, claim=True)
                self.assertEqual(code, 0)
                receipt = self._receipt_line(stdout)
                self.assertIn("selection=SELECTED[", receipt)
                self.assertIn(source, receipt)
                self.assertIn("baseline_eligible=NO", receipt)

    def test_classifier_has_both_polarities(self) -> None:
        self.assertEqual(RECEIPT.classify_command(["pytest", "-q"])["selection"], "UNSELECTED")
        self.assertEqual(RECEIPT.classify_command(["pytest", "tests/unit"])["selection"], "SELECTED")

    def test_claim_baseline_still_refuses_explicit_argv_selection(self) -> None:
        self._write_conftest(
            "from pathlib import Path\n"
            "def pytest_sessionstart(session):\n"
            "    Path('ran').write_text('pytest ran\\n')\n"
        )
        code, stdout, stderr = self._run_receipt(("-q", "-k", "one"), claim=True)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("-k", stderr)
        self.assertFalse((self.repo / "ran").exists())

    def test_observer_positive_control_can_fail_selection(self) -> None:
        code, stdout, _ = self._run_receipt(("-q", "-k", "one"))
        self.assertEqual(code, 0)
        receipt = self._receipt_line(stdout)
        self.assertIn("selection=SELECTED[", receipt)
        self.assertIn("option:keyword", receipt)
        self.assertIn("baseline_eligible=NO", receipt)

    def test_observer_option_presence_is_fail_closed_and_polarized(self) -> None:
        def make_observation(options: Dict[str, object]) -> Dict[str, object]:
            return {
                "invocation_args": [],
                "ini_addopts": [],
                "ini_testpaths": [],
                "env_pytest_addopts": None,
                "options": options,
                "collected": 2,
                "deselected": 0,
                "stats": {field: 0 for field in RECEIPT._COUNT_FIELDS},
            }

        git = {
            "head": "a" * 40,
            "branch": "main",
            "tracked_clean": "YES",
            "git_ok": True,
        }
        # Build the PRESENT arm with every alias, then derive ABSENT by
        # subtracting every failed_first alias. Iterating the mapping's first
        # alias would let a future alias addition silently repopulate the arm.
        attributes = {
            alias: None
            for logical_name, aliases in OBSERVER._SELECTION_OPTION_ALIASES.items()
            if logical_name != "exitfirst"
            for alias in aliases
        }
        failed_first_aliases = set(OBSERVER._SELECTION_OPTION_ALIASES["failed_first"])
        missing_attributes = {
            name: value for name, value in attributes.items() if name not in failed_first_aliases
        }
        missing = OBSERVER._selection_options(
            SimpleNamespace(option=SimpleNamespace(**missing_attributes))
        )
        missing_facts = RECEIPT._receipt_facts(
            ["pytest"],
            RECEIPT.classify_command(["pytest"]),
            git,
            git,
            0,
            make_observation(missing),
            None,
        )
        self.assertEqual(missing["failed_first"]["state"], "ABSENT")
        self.assertEqual(missing_facts["provenance"], "BROKEN")
        self.assertEqual(missing_facts["selection"], "UNKNOWN")
        self.assertEqual(missing_facts["baseline_eligible"], "NO")

        present = OBSERVER._selection_options(SimpleNamespace(option=SimpleNamespace(**attributes)))
        present_facts = RECEIPT._receipt_facts(
            ["pytest"],
            RECEIPT.classify_command(["pytest"]),
            git,
            git,
            0,
            make_observation(present),
            None,
        )
        self.assertEqual(present["failed_first"]["state"], "FOUND")
        self.assertEqual(present_facts["selection"], "UNSELECTED")
        self.assertEqual(present_facts["provenance"], "OK")
        self.assertEqual(present_facts["baseline_eligible"], "YES")

    def test_real_pytest_option_map_and_failed_first_flag(self) -> None:
        receipt_json = self.repo / "receipt.json"
        code, stdout, _ = self._run_receipt(receipt_json=receipt_json)
        self.assertEqual(code, 0)
        facts = json.loads(receipt_json.read_text())
        options = facts["selection_options"]
        # This list is intentionally independent of the implementation map:
        # the real pytest config must empirically resolve every expected name.
        expected = {
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
        }
        self.assertEqual(set(options), expected)
        required = expected - {"exitfirst"}
        self.assertTrue(all(options[name]["state"] == "FOUND" for name in required))
        self.assertEqual(options["failed_first"]["attribute"], "failedfirst")
        self.assertIn(options["exitfirst"]["state"], {"FOUND", "ABSENT"})

        code, stdout, _ = self._run_receipt(("-q", "--ff"), claim=True)
        self.assertEqual(code, 0)
        receipt = self._receipt_line(stdout)
        self.assertIn("selection=SELECTED[", receipt)
        self.assertIn("option:failed_first", receipt)
        self.assertIn("baseline_eligible=NO", receipt)

    def test_env_addopts_is_observed_and_refused(self) -> None:
        code, stdout, _ = self._run_receipt(claim=True, environment={"PYTEST_ADDOPTS": "-k one"})
        self.assertEqual(code, 0)
        receipt = self._receipt_line(stdout)
        self.assertIn("PYTEST_ADDOPTS:-k", receipt)
        self.assertIn("baseline_eligible=NO", receipt)

    def test_ini_addopts_override_is_observed_and_refused(self) -> None:
        code, stdout, _ = self._run_receipt(("-q", "-o", "addopts=-k one"), claim=True)
        self.assertEqual(code, 0)
        receipt = self._receipt_line(stdout)
        self.assertIn("ini:addopts:-k", receipt)
        self.assertIn("baseline_eligible=NO", receipt)

    def test_stepwise_option_is_observed_and_refused(self) -> None:
        code, stdout, _ = self._run_receipt(("-q", "--sw"), claim=True)
        self.assertEqual(code, 0)
        receipt = self._receipt_line(stdout)
        self.assertIn("option:stepwise", receipt)
        self.assertIn("baseline_eligible=NO", receipt)

    def test_collection_hook_deselection_is_observed(self) -> None:
        self._write_conftest(
            "def pytest_collection_modifyitems(config, items):\n"
            "    removed = [items.pop()]\n"
            "    config.hook.pytest_deselected(items=removed)\n"
        )
        code, stdout, _ = self._run_receipt(claim=True)
        self.assertEqual(code, 0)
        receipt = self._receipt_line(stdout)
        self.assertIn("deselected:1", receipt)
        self.assertIn("collected=1", receipt)
        self.assertIn("baseline_eligible=NO", receipt)

    def test_stderr_noise_does_not_replace_structured_counts(self) -> None:
        self._write_conftest(
            "import sys\n"
            "def pytest_unconfigure(config):\n"
            "    sys.stderr.write('diagnostic: 99 passed in historical report\\n')\n"
        )
        code, stdout, _ = self._run_receipt()
        self.assertEqual(code, 0)
        receipt = self._receipt_line(stdout)
        self.assertIn("collected=2", receipt)
        self.assertIn("passed=2", receipt)
        self.assertNotIn("passed=99", receipt)
        self.assertIn("diagnostic: 99 passed", stdout)

    def test_missing_and_mismatched_observation_are_broken(self) -> None:
        self._write_conftest(
            "import os\n"
            "def pytest_unconfigure(config):\n"
            "    os.unlink(os.environ['TPEN_VERIFICATION_OBSERVATION_PATH'])\n"
        )
        code, stdout, _ = self._run_receipt()
        self.assertEqual(code, 0)
        receipt = self._receipt_line(stdout)
        self.assertIn("provenance=BROKEN", receipt)
        self.assertIn("selection=UNKNOWN[", receipt)
        self.assertIn("baseline_eligible=NO", receipt)

        self._write_conftest(
            "import json\n"
            "import os\n"
            "def pytest_unconfigure(config):\n"
            "    path = os.environ['TPEN_VERIFICATION_OBSERVATION_PATH']\n"
            "    with open(path) as handle:\n"
            "        observation = json.load(handle)\n"
            "    observation['nonce'] = 'stale'\n"
            "    with open(path, 'w') as handle:\n"
            "        json.dump(observation, handle)\n"
        )
        code, stdout, _ = self._run_receipt()
        self.assertEqual(code, 0)
        receipt = self._receipt_line(stdout)
        self.assertIn("provenance=BROKEN", receipt)
        self.assertIn("nonce_mismatch", receipt)

        classification = RECEIPT.classify_command(["pytest"])
        git = {
            "head": "a" * 40,
            "branch": "main",
            "tracked_clean": "YES",
            "git_ok": True,
        }
        missing = RECEIPT._receipt_facts(
            ["pytest"], classification, git, git, 0, None, "observation_missing"
        )
        self.assertEqual(missing["provenance"], "BROKEN")
        self.assertEqual(missing["selection"], "UNKNOWN")
        self.assertEqual(missing["baseline_eligible"], "NO")

        observation_path = self.repo / "observation.json"
        observation_path.write_text(json.dumps({"version": 1, "nonce": "stale"}))
        observation, error = RECEIPT._read_observation(str(observation_path), "current")
        self.assertIsNone(observation)
        self.assertEqual(error, "nonce_mismatch")

    def test_checker_rejects_each_contradictory_field(self) -> None:
        code, stdout, _ = self._run_receipt()
        self.assertEqual(code, 0)
        original = self._receipt_line(stdout)
        mutations = [
            ("head_stable=YES", "head_stable=NO"),
            ("tracked_clean=YES", "tracked_clean=NO"),
            ("pytest_exit=0", "pytest_exit=2"),
            ("selection=UNSELECTED", "selection=SELECTED[-k,one]"),
            ("passed=2", "passed=-2"),
        ]
        for old, new in mutations:
            with self.subTest(old=old):
                log = self.repo / "contradictory.log"
                log.write_text(original.replace(old, new, 1) + "\n")
                self.assertNotEqual(RECEIPT.main(["check", "--require-baseline", str(log)]), 0)

    def test_checker_accepts_identical_duplicate_and_rejects_disagreement(self) -> None:
        code, stdout, _ = self._run_receipt()
        self.assertEqual(code, 0)
        original = self._receipt_line(stdout)
        duplicate = self.repo / "duplicate.log"
        duplicate.write_text(original + "\n" + original + "\n")
        self.assertEqual(RECEIPT.main(["check", "--require-baseline", str(duplicate)]), 0)

        disagreement = self.repo / "disagreement.log"
        changed = original.replace("head_stable=YES", "head_stable=NO", 1)
        disagreement.write_text(original + "\n" + changed + "\n")
        self.assertNotEqual(RECEIPT.main(["check", str(disagreement)]), 0)

    def test_post_run_tracked_dirtiness_disqualifies_baseline(self) -> None:
        self._write_conftest(
            "from pathlib import Path\n"
            "def pytest_sessionfinish(session, exitstatus):\n"
            "    Path('tracked.txt').write_text('dirtied after pytest\\n')\n"
        )
        code, stdout, _ = self._run_receipt()
        self.assertEqual(code, 0)
        receipt = self._receipt_line(stdout)
        self.assertIn("tracked_clean_before=YES", receipt)
        self.assertIn("tracked_clean_after=NO", receipt)
        self.assertIn("baseline_eligible=NO", receipt)

    def test_checker_rejects_missing_receipt(self) -> None:
        log = self.repo / "missing.log"
        log.write_text("pytest crashed before receipt\n")
        self.assertNotEqual(RECEIPT.main(["check", str(log)]), 0)


if __name__ == "__main__":
    unittest.main()
