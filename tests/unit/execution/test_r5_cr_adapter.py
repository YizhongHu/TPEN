"""Formal review round 5 reviewer tests (CR): issue 12 (test adapter exported by the production package)."""


# PREDICTED: RED at CR (issue 12)
def test_r5_chain_resume_test_adapter_is_not_exported_by_production_package():
    import tpen.execution as ex

    forbidden = {"ChainResumeReceipt", "compare_channel_parity", "e5_channel_map"}
    observed = forbidden & set(ex.__all__)
    assert not observed, f"test-only exports observed in production __all__: {observed!r}"


# PREDICTED: GREEN at CR (issue 12 control)
def test_r5_production_package_exports_execution_gateways():
    import tpen.execution as ex

    expected = {"WorkerStartGateway", "BalsamAdapter", "select_recovery"}
    missing = expected - set(ex.__all__)
    assert not missing, f"production gateway exports missing: {missing!r}"
