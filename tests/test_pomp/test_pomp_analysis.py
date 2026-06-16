import jax
import pytest
import pypomp as pp


def test_plot_traces_happy_path():
    pytest.importorskip("plotly")

    LG = pp.models.LG()
    rw_sd = pp.RWSigma(
        sigmas={n: 0.02 for n in LG.canonical_param_names}, init_names=[]
    ).geometric_cooling(a=0.5)
    LG.mif(J=2, M=2, rw_sd=rw_sd, key=jax.random.key(0))
    LG.pfilter(J=2)

    fig = LG.plot_traces(show=False)
    assert fig is not None
    assert hasattr(fig, "show")


def test_plot_traces_empty_history():
    LG = pp.models.LG()
    with pytest.warns(UserWarning, match="No trace data to plot."):
        result = LG.plot_traces(show=False)
    assert result is None


def test_plot_simulations_happy_path():
    pytest.importorskip("plotly")

    LG = pp.models.LG()
    key = jax.random.key(1)

    # Test lines mode
    fig_lines = LG.plot_simulations(nsim=5, mode="lines", key=key, show=False)
    assert fig_lines is not None
    assert len(fig_lines.data) >= 6  # 5 sims + 1 actual data

    # Test quantiles mode
    fig_q = LG.plot_simulations(nsim=10, mode="quantiles", key=key, show=False)
    assert fig_q is not None
    assert any(d.fill == "toself" for d in fig_q.data)  # Should have a fill region
