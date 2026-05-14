import pandas as pd
import pytest


def test_simple_delegators(lg_panel_mp):
    panel, _, _, _, _, _ = lg_panel_mp

    assert isinstance(panel.results(), pd.DataFrame)
    assert isinstance(panel.results(index=0), pd.DataFrame)
    assert isinstance(panel.results(ignore_nan=True), pd.DataFrame)
    assert isinstance(panel.CLL(), pd.DataFrame)
    assert isinstance(panel.CLL(average=True), pd.DataFrame)
    assert isinstance(panel.ESS(), pd.DataFrame)
    assert isinstance(panel.ESS(average=True), pd.DataFrame)
    assert isinstance(panel.time(), pd.DataFrame)
    assert isinstance(panel.traces(), pd.DataFrame)


def test_prune_and_mix_and_match(lg_panel_mp):
    panel, _, _, _, _, _ = lg_panel_mp
    panel.prune(n=1, refill=True)
    panel.mix_and_match()


def test_plot_traces_empty_history(lg_panel_setup_some_shared, capsys):
    panel, _, _ = lg_panel_setup_some_shared
    result = panel.plot_traces(which="shared", show=False)
    assert result is None
    captured = capsys.readouterr()
    assert "No trace data to plot." in captured.out


def test_plot_traces_shared(lg_panel_mp):
    pytest.importorskip("plotly")
    panel, _, _, _, _, _ = lg_panel_mp
    fig = panel.plot_traces(which="shared", show=False)
    assert fig is not None
    assert hasattr(fig, "show")


def test_plot_traces_unitLogLik(lg_panel_mp):
    pytest.importorskip("plotly")
    panel, _, _, _, _, _ = lg_panel_mp
    fig = panel.plot_traces(which="unitLogLik", show=False)
    assert fig is not None
    assert hasattr(fig, "show")


def test_plot_traces_unit_param(lg_panel_mp):
    pytest.importorskip("plotly")
    panel, _, _, _, _, _ = lg_panel_mp
    # Q1 is unit-specific (shared params are A1, C1 in the lg fixture).
    fig = panel.plot_traces(which="Q1", show=False)
    assert fig is not None
    assert hasattr(fig, "show")


    with pytest.raises(ValueError, match="not found among unit-specific parameters"):
        panel.plot_traces(which="nonexistent_param", show=False)


def test_plot_panel_simulations(lg_panel_mp):
    pytest.importorskip("plotly")
    import jax

    panel, _, _, _, _, _ = lg_panel_mp
    key = jax.random.key(0)

    # Test lines mode
    fig_lines = panel.plot_simulations(nsim=2, mode="lines", key=key, show=False)
    assert fig_lines is not None
    # 2 units, each with 2 sims + 1 actual data = 3 traces per subplot
    # (Though plotly might organize traces differently)
    assert len(fig_lines.data) >= 6

    # Test quantiles mode
    fig_q = panel.plot_simulations(nsim=5, mode="quantiles", key=key, show=False)
    assert fig_q is not None
    assert any(d.fill == "toself" for d in fig_q.data)
