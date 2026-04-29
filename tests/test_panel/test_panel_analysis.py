import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# Calling plt.show() under the Agg backend emits a benign UserWarning;
# suppress it for the show=True branches we exercise here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive:UserWarning"
)


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
    # show=True is safe under the Agg backend; covers the plt.show() branch.
    panel, _, _, _, _, _ = lg_panel_mp
    g = panel.plot_traces(which="shared", show=True)
    assert g is not None
    plt.close("all")


def test_plot_traces_unitLogLik(lg_panel_mp):
    panel, _, _, _, _, _ = lg_panel_mp
    g = panel.plot_traces(which="unitLogLik", show=True)
    assert g is not None
    plt.close("all")


def test_plot_traces_unit_param(lg_panel_mp):
    panel, _, _, _, _, _ = lg_panel_mp
    # Q1 is unit-specific (shared params are A1, C1 in the lg fixture).
    g = panel.plot_traces(which="Q1", show=True)
    assert g is not None
    plt.close("all")


def test_plot_traces_invalid_which(lg_panel_mp):
    panel, _, _, _, _, _ = lg_panel_mp
    with pytest.raises(
        ValueError, match="not found among unit-specific parameters"
    ):
        panel.plot_traces(which="nonexistent_param", show=False)


