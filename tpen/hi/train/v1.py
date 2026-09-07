"""Import façade for the HI v2 materialization boundary.

This makes the L2b API available from the namespace family used by HI train
artifacts.  It deliberately does not register a production caller.
"""

from importlib import import_module


_coordinate = import_module("experiments.atomistic.he-importance.stage_coordinate")

MaterializationError = _coordinate.MaterializationError
MaterializedCell = _coordinate.MaterializedCell
OptimizerCell = _coordinate.OptimizerCell
canonical_json = _coordinate.canonical_json
content_hash = _coordinate.content_hash
intended_configurations = _coordinate.intended_configurations
materialize_intended_configurations = _coordinate.materialize_intended_configurations
materialize_stage = _coordinate.materialize_stage
seed_labels = _coordinate.seed_labels
seed_namespace = _coordinate.seed_namespace
seed_namespaces = _coordinate.seed_namespaces

__all__ = [
    "MaterializationError",
    "MaterializedCell",
    "OptimizerCell",
    "canonical_json",
    "content_hash",
    "intended_configurations",
    "materialize_intended_configurations",
    "materialize_stage",
    "seed_labels",
    "seed_namespace",
    "seed_namespaces",
]
