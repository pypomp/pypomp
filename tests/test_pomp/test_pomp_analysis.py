import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import jax  # noqa: E402
import pytest  # noqa: E402
import pypomp as pp  # noqa: E402

# Calling plt.show() under the Agg backend emits a benign UserWarning.
pytestmark = pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive:UserWarning"
)


@pytest.fixture(scope="module")
def lg_with_history():
    LG = pp.models.LG()
    rw_sd = pp.RWSigma(
        sigmas={n: 0.02 for n in LG.canonical_param_names}, init_names=[]
    )
    LG.mif(J=2, M=2, a=0.5, rw_sd=rw_sd, key=jax.random.key(0))
    LG.pfilter(J=2)
    return LG


def test_plot_traces_happy_path(lg_with_history):
    g = lg_with_history.plot_traces(show=True)
    assert g is not None
    plt.close("all")


def test_plot_traces_empty_history(capsys):
    LG = pp.models.LG()
    result = LG.plot_traces(show=False)
    assert result is None
    assert "No trace data to plot." in capsys.readouterr().out
