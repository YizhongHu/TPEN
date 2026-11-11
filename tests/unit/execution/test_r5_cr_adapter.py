"""Formal review round 5 reviewer tests (CR): issue 12 (test adapter exported by the production package)."""


# Issue 13 export-removal probe retired by the binding R5 disposition.

# PREDICTED: GREEN at CR (issue 12 control)
def test_r5_production_package_exports_execution_gateways():
    import tpen.execution as ex

    expected = {"WorkerStartGateway", "BalsamAdapter", "select_recovery"}
    missing = expected - set(ex.__all__)
    assert not missing, f"production gateway exports missing: {missing!r}"
