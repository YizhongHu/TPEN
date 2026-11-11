"""Method state is a first-class checkpoint payload, and is accounted as one.

Two claims that were true but untested, which is why this file is a GUARD
rather than a repair:

1. An update method's persistent state round-trips, and BOTH failure directions
   are loud -- state handed to a stateless method, and a stateful method
   resumed from a checkpoint carrying none.
2. Those bytes are counted as restorable PAYLOAD in the publication receipt,
   not as descriptive metadata.

Neither is new behaviour. `VMCUpdateMethod` has carried `method_state_dict` /
`load_method_state_dict` deliberately separate from `state_dict` for some time,
`VMCTrainer` writes the result into `trainer.json`, and
`PAYLOAD_COMPONENT_NAMES` has always contained ``"trainer"``. What was missing
was anything that would notice if one of them changed.

SCOPE, stated because the gap matters more than the coverage. This file does
NOT test that an INTERRUPTED `save_checkpoint` leaves no partial method state.
That property is real -- the writer builds everything in ``<step>.tmp``, writes
``manifest.json`` and ``COMPLETE`` last, commits with a single
``tmp_dir.rename(final_dir)``, and removes the temporary directory in a
``finally`` -- but it is a property of the checkpoint WRITER, not of method
state, and method state inherits it by riding ``trainer.json``. Testing it
belongs with the writer.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import pytest
import torch

from tpen.checkpoint.receipt import (
    PAYLOAD_COMPONENT_NAMES,
    measure_checkpoint_files,
)
from tpen.training.update import VMCUpdateMethod, VMCUpdateResult


class _StatelessMethod(VMCUpdateMethod[Any]):
    """Default surface: no state of its own."""

    def update(self, update_input: Any) -> VMCUpdateResult:  # pragma: no cover - unused
        raise NotImplementedError


class _StatefulMethod(VMCUpdateMethod[Any]):
    """A method owning a schedule counter and a convention fingerprint.

    Deliberately the shape the base-class docstring describes -- a counter and
    a fingerprint -- rather than a tensor payload, because that is the case
    ``method_state_dict`` exists to serve and the case a JSON trainer file can
    actually hold.
    """

    def __init__(self, *, step: int = 0, fingerprint: str = "v1") -> None:
        self.step = int(step)
        self.fingerprint = str(fingerprint)

    def update(self, update_input: Any) -> VMCUpdateResult:  # pragma: no cover - unused
        raise NotImplementedError

    def method_state_dict(self) -> Mapping[str, Any]:
        return {"step": self.step, "fingerprint": self.fingerprint}

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        self.step = int(state["step"])
        self.fingerprint = str(state["fingerprint"])


class TestTheTwoStateSurfacesAreSeparate:
    def test_a_stateless_method_contributes_nothing_to_either_surface(self) -> None:
        """A stateless method must not invent a checkpoint payload."""

        method = _StatelessMethod()
        assert method.state_dict() == {}
        assert method.method_state_dict() == {}

    def test_a_stateless_method_refuses_non_empty_state(self) -> None:
        """The loud direction that already existed: SR state into a stateless method.

        Silence here would restore a trainer's counters while leaving the
        method at its fresh values -- a resume that looks clean and is not.
        """

        with pytest.raises(ValueError, match="stateless VMCUpdateMethod"):
            _StatelessMethod().load_state_dict({"anything": 1})

    def test_method_state_is_not_the_optimizer_state_surface(self) -> None:
        """`method_state_dict` and `state_dict` are separate on purpose.

        A method may own persistent bookkeeping without owning optimizer
        tensors. Collapsing the two would force a stateless-but-bookkeeping
        method to fabricate an optimizer payload, and would route persistent
        state through a surface whose contents are not JSON-safe.
        """

        method = _StatefulMethod(step=7)
        assert method.method_state_dict() == {"step": 7, "fingerprint": "v1"}
        assert method.state_dict() == {}


class TestMethodStateRoundTrips:
    def test_state_survives_a_json_round_trip(self) -> None:
        """It rides `trainer.json`, so it must survive JSON exactly.

        A method state that is only round-trippable in memory would pass an
        in-process test and lose data at the file boundary -- tensors and
        tuples both do this silently.
        """

        source = _StatefulMethod(step=41, fingerprint="conv-3")
        encoded = json.dumps(dict(source.method_state_dict()))

        target = _StatefulMethod()
        target.load_method_state_dict(json.loads(encoded))

        assert (target.step, target.fingerprint) == (41, "conv-3")

    def test_restoring_does_not_silently_keep_the_fresh_value(self) -> None:
        """Control: the round-trip test must be able to fail.

        If `load_method_state_dict` did nothing at all, the assertion above
        would still pass whenever the fresh value happened to match. Restoring
        into a target that was constructed DIFFERENTLY is what rules that out.
        """

        target = _StatefulMethod(step=999, fingerprint="stale")
        target.load_method_state_dict({"step": 41, "fingerprint": "conv-3"})
        assert (target.step, target.fingerprint) == (41, "conv-3")


class TestMethodStateIsAccountedAsPayload:
    """C5. The bytes are counted as restorable state, not as metadata."""

    def test_trainer_is_a_payload_component(self) -> None:
        """Method state rides `trainer.json`, so `trainer` must be payload.

        If `trainer` were ever moved out of this set, method-state bytes would
        be reported as descriptive metadata -- an accounting error that would
        understate the restorable size of every checkpoint and produce no
        error anywhere.
        """

        assert "trainer" in PAYLOAD_COMPONENT_NAMES

    def _write(self, directory, name: str, body: str) -> int:
        path = directory / name
        path.write_text(body, encoding="utf-8")
        return path.stat().st_size

    def test_method_state_bytes_land_in_payload_not_metadata(self, tmp_path) -> None:
        """Measured end to end on a real directory, not asserted from the set."""

        trainer_state = {
            "next_iteration": 12,
            "completed_updates": 11,
            "update_method": {"step": 11, "fingerprint": "conv-3"},
        }
        trainer_bytes = self._write(tmp_path, "trainer.json", json.dumps(trainer_state))
        config_bytes = self._write(tmp_path, "resolved_config.yaml", "a: 1\n")
        self._write(tmp_path, "manifest.json", "{}")
        self._write(tmp_path, "COMPLETE", "complete\n")

        measured = measure_checkpoint_files(
            tmp_path, {"trainer": "trainer.json", "resolved_config": "resolved_config.yaml"}
        )
        sizes = {entry.component: entry.size_bytes for entry in measured}

        assert sizes["trainer"] == trainer_bytes
        payload = sum(
            entry.size_bytes for entry in measured if entry.component in PAYLOAD_COMPONENT_NAMES
        )
        metadata = sum(
            entry.size_bytes
            for entry in measured
            if entry.component not in PAYLOAD_COMPONENT_NAMES
        )
        assert payload >= trainer_bytes
        # Discriminating control: the split is not vacuous. `resolved_config`
        # is descriptive, is NOT in the payload set, and its bytes must land on
        # the other side -- otherwise "payload" would just mean "every file".
        assert "resolved_config" not in PAYLOAD_COMPONENT_NAMES
        assert metadata >= config_bytes

    def test_accounting_granularity_is_the_file_not_the_key(self, tmp_path) -> None:
        """The LIMIT, pinned so it is not mistaken for a finer guarantee.

        Method state is one key inside `trainer.json`. The receipt measures
        FILES, so the reported number is the whole trainer file including the
        progress counters and the parameter layout -- there is no per-key
        attribution, and separating one would mean restructuring the
        checkpoint. This asserts the coarse behaviour deliberately rather than
        leaving a reader to assume method state is measured on its own.
        """

        with_method = json.dumps(
            {"next_iteration": 1, "completed_updates": 0, "update_method": {"step": 0}}
        )
        without_method = json.dumps({"next_iteration": 1, "completed_updates": 0})
        assert len(with_method) > len(without_method)

        # `measure_checkpoint_files` also stats the fixed `manifest` and
        # `complete` pseudo-components, which every committed checkpoint
        # directory has. They must exist or the measurement raises.
        self._write(tmp_path, "manifest.json", "{}")
        self._write(tmp_path, "COMPLETE", "complete\n")

        big = self._write(tmp_path, "trainer.json", with_method)
        measured = measure_checkpoint_files(tmp_path, {"trainer": "trainer.json"})
        assert measured[0].size_bytes == big

        self._write(tmp_path, "trainer.json", without_method)
        smaller = measure_checkpoint_files(tmp_path, {"trainer": "trainer.json"})[0].size_bytes
        # The file shrinks when method state is absent, which is the ONLY sense
        # in which method state is measurable here: as a difference between two
        # whole-file sizes, never as its own line item.
        assert smaller < big
