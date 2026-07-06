import pickle
from typing import Any, cast
from dataclasses import dataclass
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import jax
import jax.numpy as jnp

from pypomp.core.results.base import (
    _merge_results,
    BaseResult,
    PompEstimationTracesMixin,
    PanelPompEstimationTracesMixin,
)
from pypomp.core.results.history import ResultsHistory
from pypomp.core.results.pomp import PompPFilterResult, PompMIFResult, PompTrainResult
from pypomp.core.results.panel import (
    PanelPompPFilterResult,
    PanelPompMIFResult,
    PanelPompTrainResult,
)
from pypomp.core.parameters import PompParameters, PanelParameters
from pypomp.core.rw_sigma import RWSigma
from pypomp.core.learning_rate import LearningRate
from pypomp.core.optimizer import Adam

# =====================================================================
# 1. Base Class & Merge Helper Test Cases
# =====================================================================


# Concrete subclass of BaseResult defined at module level for pickle compatibility
@dataclass(eq=False)
class DummyResult(BaseResult):
    theta: Any = None
    traces_da: Any = None
    custom_field: str = ""

    def to_dataframe(self, ignore_nan: bool = False):
        return pd.DataFrame([{"custom": self.custom_field, "logLik": 1.0}])

    @staticmethod
    def merge(*results):
        return results[0]

    @property
    def _summary_config(self):
        return [("Custom Field", "custom_field")]


@pytest.fixture
def dummy_result_cls():
    return DummyResult


def test_merge_results_helper_errors(dummy_result_cls):
    # Empty results list
    with pytest.raises(
        ValueError, match="At least one DummyResult object must be provided."
    ):
        _merge_results(dummy_result_cls, [], [], [])

    # Type mismatch in results list
    key = jax.random.key(0)
    r1 = dummy_result_cls(method="dummy", execution_time=1.0, key=key)
    with pytest.raises(
        TypeError, match="All merged objects must be of type DummyResult."
    ):
        _merge_results(dummy_result_cls, cast(Any, [r1, "not_a_result"]), [], [])

    # Scalar field mismatch
    r2 = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key, custom_field="val1"
    )
    r3 = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key, custom_field="val2"
    )
    with pytest.raises(
        ValueError, match="All DummyResult objects must have the same custom_field."
    ):
        _merge_results(dummy_result_cls, [r2, r3], ["custom_field"], [])


def test_base_result_equality_and_serialization(dummy_result_cls):
    key0 = jax.random.key(0)
    key1 = jax.random.key(1)
    t = pd.Timestamp("2026-06-15 00:00:00")

    r1 = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key0, timestamp=t, custom_field="a"
    )
    r2 = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key0, timestamp=t, custom_field="a"
    )
    r3 = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key1, timestamp=t, custom_field="a"
    )
    r4 = dummy_result_cls(
        method="dummy", execution_time=2.0, key=key0, timestamp=t, custom_field="a"
    )
    r5 = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key0, timestamp=t, custom_field="b"
    )

    # Structural equality checks
    assert r1 == r2
    assert r1 != r3  # key mismatch
    assert r1 != r4  # execution_time mismatch
    assert r1 != r5  # custom_field mismatch
    assert r1 != "not_dummy_result"

    # xr.DataArray comparison check
    da1 = xr.DataArray([1.0], dims=["dim1"])
    da2 = xr.DataArray([2.0], dims=["dim1"])
    r1_da = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key0, timestamp=t, traces_da=da1
    )
    r2_da = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key0, timestamp=t, traces_da=da1
    )
    r3_da = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key0, timestamp=t, traces_da=da2
    )
    r4_da = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key0, timestamp=t, traces_da=None
    )
    assert r1_da == r2_da
    assert r1_da != r3_da
    assert r1_da != r4_da

    # numpy array and jax Array comparison checks
    arr1 = np.array([1.0, np.nan])
    arr2 = np.array([1.0, np.nan])
    arr3 = np.array([2.0, np.nan])
    r1_arr = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key0, timestamp=t, theta=arr1
    )
    r2_arr = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key0, timestamp=t, theta=arr2
    )
    r3_arr = dummy_result_cls(
        method="dummy", execution_time=1.0, key=key0, timestamp=t, theta=arr3
    )
    assert r1_arr == r2_arr  # equal_nan=True path
    assert r1_arr != r3_arr

    # Custom serialization / pickling
    pickled = pickle.dumps(r1)
    unpickled = pickle.loads(pickled)
    assert unpickled == r1
    # Check JAX key restored successfully
    assert jnp.array_equal(
        jax.random.key_data(unpickled.key), jax.random.key_data(r1.key)
    )


def test_base_result_default_methods(dummy_result_cls):
    key = jax.random.key(0)
    res = dummy_result_cls(method="dummy", execution_time=1.2, key=key)

    assert res.CLL().empty
    assert res.ESS().empty
    assert res.traces().empty


def test_base_result_print_summary(dummy_result_cls, capsys):
    key = jax.random.key(0)
    res = dummy_result_cls(
        method="dummy", execution_time=1.2, key=key, custom_field="test_summary"
    )

    # Simple print_summary test
    res.print_summary()
    captured = capsys.readouterr().out
    assert "Method: dummy" in captured
    assert "Custom Field: test_summary" in captured
    assert "Execution time: 1.2 seconds" in captured

    # Test cooling function prints
    # RWSigma cooling configurations: geometric, hyperbolic, cosine, custom
    sig = {"a": 0.1}

    # Geometric
    rw_geom = RWSigma(sig)  # Defaults to geometric
    res_geom = dummy_result_cls(method="dummy", execution_time=1.0, key=key)
    res_geom.rw_sd = rw_geom
    res_geom.print_summary()
    captured = capsys.readouterr().out
    assert "Cooling fraction (a): 0.5" in captured

    # Hyperbolic
    rw_hyper = RWSigma(sig).hyperbolic_cooling(0.2)
    res_hyper = dummy_result_cls(method="dummy", execution_time=1.0, key=key)
    res_hyper.rw_sd = rw_hyper
    res_hyper.print_summary()
    captured = capsys.readouterr().out
    assert "Cooling rate (s): 0.2" in captured

    # Cosine
    rw_cos = RWSigma(sig).cosine_cooling(0.1, 100)
    res_cos = dummy_result_cls(method="dummy", execution_time=1.0, key=key)
    res_cos.rw_sd = rw_cos
    res_cos.print_summary()
    captured = capsys.readouterr().out
    assert "Cosine min cooling (c): 0.1" in captured
    assert "Cosine duration (M): 100" in captured

    # Custom
    def dummy_cool_fn(nt, m, ntimes):
        return 1.0

    rw_cust = RWSigma(sig).custom_cooling(dummy_cool_fn)
    res_cust = dummy_result_cls(method="dummy", execution_time=1.0, key=key)
    res_cust.rw_sd = rw_cust
    res_cust.print_summary()
    captured = capsys.readouterr().out
    assert "Cooling function: dummy_cool_fn" in captured


# =====================================================================
# 2. Mixins Test Cases
# =====================================================================


def test_pomp_estimation_traces_mixin():
    from dataclasses import dataclass

    @dataclass(eq=False)
    class PompMixinResult(PompEstimationTracesMixin, BaseResult):
        theta: Any = None
        traces_da: Any = None

        def to_dataframe(self, ignore_nan: bool = False):
            return super().to_dataframe(ignore_nan)

        def traces(self):
            return super().traces()

        @staticmethod
        def merge(*results):
            return results[0]

        @property
        def _summary_config(self):
            return []

    key = jax.random.key(0)
    # Test empty traces_da
    r_empty = PompMixinResult(method="test", execution_time=1.0, key=key)
    assert r_empty.to_dataframe().empty
    assert r_empty.traces().empty

    # Test populated traces_da
    theta = PompParameters({"param1": 10.0, "param2": 20.0})
    traces_da = xr.DataArray(
        [
            [[1.0, 10.0, 20.0], [2.0, 11.0, 21.0]]
        ],  # shape: (theta_idx=1, iteration=2, variable=3)
        dims=["theta_idx", "iteration", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0, 1],
            "variable": ["logLik", "param1", "param2"],
        },
    )
    r_pop = PompMixinResult(
        method="test_mixin",
        execution_time=1.0,
        key=key,
        theta=theta,
        traces_da=traces_da,
    )

    # Check to_dataframe formats correctly (uses iteration=-1, i.e., last iteration)
    df_to = r_pop.to_dataframe()
    assert len(df_to) == 1
    assert "logLik" in df_to.columns
    assert df_to.loc[0, "logLik"] == 2.0
    assert df_to.loc[0, "param1"] == 11.0
    assert df_to.loc[0, "param2"] == 21.0
    assert pd.isna(df_to.loc[0, "se"])

    # Check traces formats correctly
    df_tr = r_pop.traces()
    assert len(df_tr) == 2
    assert "iteration" in df_tr.columns
    assert "method" in df_tr.columns
    assert list(df_tr["iteration"]) == [0, 1]
    assert list(df_tr["method"]) == ["test_mixin", "test_mixin"]


def test_panel_pomp_estimation_traces_mixin():
    from dataclasses import dataclass

    @dataclass(eq=False)
    class PanelMixinResult(PanelPompEstimationTracesMixin, BaseResult):
        theta: Any = None
        shared_traces: Any = None
        unit_traces: Any = None

        def to_dataframe(self, ignore_nan: bool = False):
            return super().to_dataframe(ignore_nan)

        def traces(self):
            return super().traces()

        @staticmethod
        def merge(*results):
            return results[0]

        @property
        def _summary_config(self):
            return []

    key = jax.random.key(0)
    # Test empty mixes
    r_empty = PanelMixinResult(method="test", execution_time=1.0, key=key)
    assert r_empty.to_dataframe().empty
    assert r_empty.traces().empty

    # Test populated traces
    shared_traces = xr.DataArray(
        [[[1.5, 10.0]]],  # shape: (theta_idx=1, iteration=1, variable=2)
        dims=["theta_idx", "iteration", "variable"],
        coords={"theta_idx": [0], "iteration": [0], "variable": ["logLik", "sh1"]},
    )
    unit_traces = xr.DataArray(
        [[[[2.5, 20.0]]]],  # shape: (theta_idx=1, iteration=1, unit=1, variable=2)
        dims=["theta_idx", "iteration", "unit", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0],
            "unit": ["unit1"],
            "variable": ["unitLogLik", "un1"],
        },
    )

    r_pop = PanelMixinResult(
        method="test_panel_mixin",
        execution_time=1.0,
        key=key,
        shared_traces=shared_traces,
        unit_traces=unit_traces,
    )

    # Check to_dataframe
    df_to = r_pop.to_dataframe()
    assert len(df_to) == 1
    assert "shared logLik" in df_to.columns
    assert "unit logLik" in df_to.columns
    assert df_to.loc[0, "shared logLik"] == 1.5
    assert df_to.loc[0, "unit logLik"] == 2.5
    assert df_to.loc[0, "unit"] == "unit1"

    # Check traces
    df_tr = r_pop.traces()
    assert len(df_tr) == 2  # 1 for shared (unit="shared"), 1 for unit (unit="unit1")
    assert set(df_tr["unit"]) == {"shared", "unit1"}
    assert "method" in df_tr.columns
    assert df_tr.loc[df_tr["unit"] == "shared", "logLik"].values[0] == 1.5
    assert df_tr.loc[df_tr["unit"] == "unit1", "logLik"].values[0] == 2.5


# =====================================================================
# 3. ResultsHistory Test Cases
# =====================================================================


def test_results_history(capsys):
    key = jax.random.key(0)

    # Concrete dummy for history elements
    class MockResult(BaseResult):
        def __init__(self, method, execution_time, traces_val=None):
            super().__init__(method=method, execution_time=execution_time, key=key)
            self._traces_val = traces_val if traces_val is not None else pd.DataFrame()

        def to_dataframe(self, ignore_nan: bool = False):
            return pd.DataFrame([{"method": self.method, "logLik": 1.0}])

        def CLL(self, average=False):
            return pd.DataFrame([{"CLL": 1.0}])

        def ESS(self, average=False):
            return pd.DataFrame([{"ESS": 2.0}])

        def traces(self):
            return self._traces_val

        @staticmethod
        def merge(*results):
            # Sum up execution times for mock merge verification
            m_time = sum(r.execution_time for r in results)
            return MockResult(results[0].method, m_time)

        @property
        def _summary_config(self):
            return []

    # Init history
    hist = ResultsHistory()
    assert len(hist) == 0
    assert hist.time().empty
    assert hist.results().empty
    assert hist.CLL().empty
    assert hist.ESS().empty
    assert hist.traces().empty

    with pytest.raises(ValueError, match="History is empty"):
        hist.last()

    # Append & add
    r1 = MockResult(
        method="m1",
        execution_time=10.0,
        traces_val=pd.DataFrame({"iteration": [0], "theta_idx": [0], "logLik": [-1.0]}),
    )
    r2 = MockResult(
        method="m2",
        execution_time=20.0,
        traces_val=pd.DataFrame({"theta_idx": [0], "logLik": [-2.0]}),
    )

    hist.append(r1)
    hist.add(r2)
    assert len(hist) == 2
    assert hist.last() == r2

    # Iteration & Indexing
    assert list(hist) == [r1, r2]
    assert hist[0] == r1
    assert hist[-1] == r2

    # Slice Indexing
    sub_hist = hist[0:1]
    assert isinstance(sub_hist, ResultsHistory)
    assert len(sub_hist) == 1
    assert sub_hist[0] == r1

    # Time DF
    df_t = hist.time()
    assert len(df_t) == 2
    assert list(df_t["method"]) == ["m1", "m2"]
    assert list(df_t["time"]) == [10.0, 20.0]

    # Print summary
    hist.print_summary()
    captured = capsys.readouterr().out
    assert "Results History:" in captured
    assert "[0] M1 Result:" in captured
    assert "[1] M2 Result:" in captured

    # Equality
    hist2 = ResultsHistory([r1, r2])
    assert hist == hist2
    assert hist != ResultsHistory([r1])
    assert hist != "not_a_history"

    # Delegation methods
    assert not hist.results(0).empty
    assert not hist.CLL(0).empty
    assert not hist.ESS(0).empty

    # Traces
    df_tr = hist.traces()
    # r1 traces has 'iteration': [0]
    # r2 traces does not have 'iteration', so traces() shifts / assigns iteration
    assert len(df_tr) == 2
    assert list(df_tr["iteration"]) == [0, 1]
    assert list(df_tr["logLik"]) == [-1.0, -2.0]

    # Merge histories
    h_a = ResultsHistory([MockResult("m1", 5.0), MockResult("m2", 15.0)])
    h_b = ResultsHistory([MockResult("m1", 10.0), MockResult("m2", 25.0)])

    # Merge histories with length mismatch
    h_bad = ResultsHistory([MockResult("m1", 5.0)])
    with pytest.raises(
        ValueError, match="All histories must have the same number of entries"
    ):
        ResultsHistory.merge(h_a, h_bad)

    # Empty merge
    assert len(ResultsHistory.merge()) == 0

    # Normal merge
    h_merged = ResultsHistory.merge(h_a, h_b)
    assert len(h_merged) == 2
    assert h_merged[0].execution_time == 15.0  # 5.0 + 10.0
    assert h_merged[1].execution_time == 40.0  # 15.0 + 25.0

    # Clear
    hist.clear()
    assert len(hist) == 0


# =====================================================================
# 4. Pomp Result Classes Test Cases
# =====================================================================


def test_pomp_pfilter_result():
    key = jax.random.key(0)
    theta = PompParameters({"param1": 1.0, "param2": 2.0})
    logLiks = xr.DataArray(
        [[1.5, 2.5]],  # shape: (theta_idx=1, rep=2)
        dims=["theta_idx", "rep"],
    )
    cll_da = xr.DataArray(
        [[[0.5, 0.6], [0.7, 0.8]]],  # shape: (theta_idx=1, rep=2, time=2)
        dims=["theta_idx", "rep", "time"],
    )
    ess_da = xr.DataArray(
        [[[10.0, 20.0], [30.0, 40.0]]],  # shape: (theta_idx=1, rep=2, time=2)
        dims=["theta_idx", "rep", "time"],
    )

    res = PompPFilterResult(
        method="pfilter",
        execution_time=1.5,
        key=key,
        theta=theta,
        logLiks=logLiks,
        J=1000,
        reps=2,
        thresh=0.5,
        CLL_da=cll_da,
        ESS_da=ess_da,
    )

    assert res.method == "pfilter"
    assert "Number of particles (J)" in [x[0] for x in res._summary_config]

    # to_dataframe
    df = res.to_dataframe()
    assert len(df) == 1
    assert "logLik" in df.columns
    assert "se" in df.columns
    assert not pd.isna(df.loc[0, "se"])
    assert "param1" in df.columns

    # CLL
    # average=False
    df_cll = res.CLL(average=False)
    assert "CLL" in df_cll.columns
    assert len(df_cll) == 4  # 1 theta_idx * 2 reps * 2 times
    # average=True
    df_cll_avg = res.CLL(average=True)
    assert len(df_cll_avg) == 2  # 1 theta_idx * 2 times (averaged over rep)

    # ESS
    # average=False
    df_ess = res.ESS(average=False)
    assert "ESS" in df_ess.columns
    assert len(df_ess) == 4
    # average=True
    df_ess_avg = res.ESS(average=True)
    assert len(df_ess_avg) == 2

    # traces
    df_tr = res.traces()
    assert len(df_tr) == 1
    assert df_tr.loc[0, "iteration"] == 0
    assert df_tr.loc[0, "param1"] == 1.0
    assert "se" in df_tr.columns
    assert not pd.isna(df_tr.loc[0, "se"])

    # Merge
    res2 = PompPFilterResult(
        method="pfilter",
        execution_time=2.5,
        key=key,
        theta=theta,
        logLiks=logLiks,
        J=1000,
        reps=2,
        thresh=0.5,
        CLL_da=cll_da,
        ESS_da=ess_da,
    )
    res_merged = PompPFilterResult.merge(res, res2)
    assert res_merged.execution_time == 2.5
    assert res_merged.J == 1000
    assert res_merged.logLiks.sizes["theta_idx"] == 2  # Concat theta_idx


def test_pomp_mif_result():
    key = jax.random.key(0)
    theta = PompParameters({"param1": 1.0})
    traces_da = xr.DataArray(
        [[[1.5, 1.0], [2.5, 1.1]]],  # shape: (theta_idx=1, iteration=2, variable=2)
        dims=["theta_idx", "iteration", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0, 1],
            "variable": ["logLik", "param1"],
        },
    )

    res = PompMIFResult(
        method="mif",
        execution_time=1.0,
        key=key,
        theta=theta,
        traces_da=traces_da,
        J=100,
        M=2,
        rw_sd=None,
        thresh=0.8,
        n_monitors=5,
    )

    assert res.method == "mif"
    assert "Number of iterations (M)" in [x[0] for x in res._summary_config]

    df_to = res.to_dataframe()
    assert len(df_to) == 1
    assert df_to.loc[0, "logLik"] == 2.5
    assert "se" in df_to.columns
    assert pd.isna(df_to.loc[0, "se"])
    assert df_to.loc[0, "param1"] == 1.1

    df_tr = res.traces()
    assert "se" in df_tr.columns
    assert df_tr["se"].isna().all()

    # Merge
    res2 = PompMIFResult(
        method="mif",
        execution_time=1.5,
        key=key,
        theta=theta,
        traces_da=traces_da,
        J=100,
        M=2,
        rw_sd=None,
        thresh=0.8,
        n_monitors=5,
    )
    res_merged = PompMIFResult.merge(res, res2)
    assert res_merged.M == 2
    assert res_merged.traces_da.sizes["theta_idx"] == 2


def test_pomp_train_result():
    key = jax.random.key(0)
    theta = PompParameters({"param1": 1.0})
    traces_da = xr.DataArray(
        [[[1.5, 1.0]]],  # shape: (theta_idx=1, iteration=1, variable=2)
        dims=["theta_idx", "iteration", "variable"],
        coords={"theta_idx": [0], "iteration": [0], "variable": ["logLik", "param1"]},
    )
    opt = Adam()
    lr = LearningRate({"param1": 0.01})

    res = PompTrainResult(
        method="train",
        execution_time=1.0,
        key=key,
        theta=theta,
        traces_da=traces_da,
        optimizer=opt,
        J=500,
        M=1,
        eta=lr,
        alpha=0.95,
        thresh=0.0,
        alpha_cooling=0.99,
    )

    assert res.method == "train"
    assert "Optimizer" in [x[0] for x in res._summary_config]

    # Merge
    res2 = PompTrainResult(
        method="train",
        execution_time=2.0,
        key=key,
        theta=theta,
        traces_da=traces_da,
        optimizer=opt,
        J=500,
        M=1,
        eta=lr,
        alpha=0.95,
        thresh=0.0,
        alpha_cooling=0.99,
    )
    res_merged = PompTrainResult.merge(res, res2)
    assert res_merged.optimizer == opt
    assert res_merged.execution_time == 2.0


# =====================================================================
# 5. PanelPomp Result Classes Test Cases
# =====================================================================


def test_panel_pomp_pfilter_result():
    key = jax.random.key(0)

    # Setup PanelParameters
    shared_df = pd.DataFrame({"shared": [1.0]}, index=["s1"])
    unit_df = pd.DataFrame({"u1": [2.0], "u2": [3.0]}, index=["up1"])
    theta = PanelParameters(
        theta=cast(Any, {"shared": shared_df, "unit_specific": unit_df})
    )

    logLiks = xr.DataArray(
        [[[1.0, 2.0], [3.0, 4.0]]],  # shape: (theta_idx=1, unit=2, rep=2)
        dims=["theta_idx", "unit", "rep"],
        coords={"theta_idx": [0], "unit": ["u1", "u2"], "rep": [0, 1]},
    )
    cll_da = xr.DataArray(
        [
            [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
        ],  # (theta_idx=1, unit=2, rep=2, time=2)
        dims=["theta_idx", "unit", "rep", "time"],
        coords={"theta_idx": [0], "unit": ["u1", "u2"], "rep": [0, 1], "time": [0, 1]},
    )
    ess_da = xr.DataArray(
        [[[[10.0], [20.0]], [[30.0], [40.0]]]],  # (theta_idx=1, unit=2, rep=2, time=1)
        dims=["theta_idx", "unit", "rep", "time"],
        coords={"theta_idx": [0], "unit": ["u1", "u2"], "rep": [0, 1], "time": [0]},
    )

    res = PanelPompPFilterResult(
        method="pfilter",
        execution_time=1.5,
        key=key,
        theta=theta,
        logLiks=logLiks,
        J=200,
        reps=2,
        thresh=0.1,
        CLL_da=cll_da,
        ESS_da=ess_da,
    )

    assert res.method == "pfilter"
    assert "Number of replicates" in [x[0] for x in res._summary_config]

    # to_dataframe
    df = res.to_dataframe()
    assert len(df) == 2  # 2 units
    assert "shared logLik" in df.columns
    assert "shared logLik se" in df.columns
    assert "unit logLik" in df.columns
    assert "unit logLik se" in df.columns
    assert not pd.isna(df["shared logLik se"].iloc[0])
    assert not pd.isna(df["unit logLik se"].iloc[0])
    assert "s1" in df.columns
    assert "up1" in df.columns

    # CLL
    # average=False
    df_cll = res.CLL(average=False)
    assert len(df_cll) == 8  # 1 * 2 * 2 * 2
    # average=True
    df_cll_avg = res.CLL(average=True)
    assert len(df_cll_avg) == 4  # 1 * 2 * 2 (averaged rep)

    # ESS
    # average=False
    df_ess = res.ESS(average=False)
    assert len(df_ess) == 4
    # average=True
    df_ess_avg = res.ESS(average=True)
    assert len(df_ess_avg) == 2

    # traces
    df_tr = res.traces()
    # 1 shared trace, 2 unit-specific traces -> total 3 rows
    assert len(df_tr) == 3
    assert set(df_tr["unit"]) == {"shared", "u1", "u2"}
    assert "se" in df_tr.columns
    assert not df_tr["se"].isna().any()

    # Merge
    res2 = PanelPompPFilterResult(
        method="pfilter",
        execution_time=2.0,
        key=key,
        theta=theta,
        logLiks=logLiks,
        J=200,
        reps=2,
        thresh=0.1,
        CLL_da=cll_da,
        ESS_da=ess_da,
    )
    res_merged = PanelPompPFilterResult.merge(res, res2)
    assert res_merged.execution_time == 2.0
    assert res_merged.logLiks.sizes["theta_idx"] == 2


def test_panel_pomp_mif_result():
    key = jax.random.key(0)
    theta = PanelParameters(None)
    shared_traces = xr.DataArray(
        [[[1.0]]],  # (theta_idx=1, iteration=1, variable=1)
        dims=["theta_idx", "iteration", "variable"],
        coords={"theta_idx": [0], "iteration": [0], "variable": ["logLik"]},
    )
    unit_traces = xr.DataArray(
        [[[[2.0]]]],  # (theta_idx=1, iteration=1, unit=1, variable=1)
        dims=["theta_idx", "iteration", "unit", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0],
            "unit": ["u1"],
            "variable": ["unitLogLik"],
        },
    )

    res = PanelPompMIFResult(
        method="mif",
        execution_time=1.0,
        key=key,
        theta=theta,
        shared_traces=shared_traces,
        unit_traces=unit_traces,
        J=50,
        M=1,
        rw_sd=None,
        thresh=0.0,
        n_monitors=1,
        block=True,
    )

    assert res.method == "mif"
    assert "Block" in [x[0] for x in res._summary_config]

    df_to = res.to_dataframe()
    assert len(df_to) == 1
    assert df_to.loc[0, "shared logLik"] == 1.0
    assert df_to.loc[0, "unit logLik"] == 2.0
    assert "shared logLik se" in df_to.columns
    assert "unit logLik se" in df_to.columns
    assert pd.isna(df_to.loc[0, "shared logLik se"])
    assert pd.isna(df_to.loc[0, "unit logLik se"])

    df_tr = res.traces()
    assert "se" in df_tr.columns
    assert df_tr["se"].isna().all()

    # Merge
    res2 = PanelPompMIFResult(
        method="mif",
        execution_time=3.0,
        key=key,
        theta=theta,
        shared_traces=shared_traces,
        unit_traces=unit_traces,
        J=50,
        M=1,
        rw_sd=None,
        thresh=0.0,
        n_monitors=1,
        block=True,
    )
    res_merged = PanelPompMIFResult.merge(res, res2)
    assert res_merged.block is True
    assert res_merged.execution_time == 3.0


def test_panel_pomp_train_result():
    key = jax.random.key(0)
    theta = PanelParameters(None)
    shared_traces = xr.DataArray(
        [[[1.0]]],  # (theta_idx=1, iteration=1, variable=1)
        dims=["theta_idx", "iteration", "variable"],
        coords={"theta_idx": [0], "iteration": [0], "variable": ["logLik"]},
    )
    unit_traces = xr.DataArray(
        [[[[2.0]]]],  # (theta_idx=1, iteration=1, unit=1, variable=1)
        dims=["theta_idx", "iteration", "unit", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0],
            "unit": ["u1"],
            "variable": ["unitLogLik"],
        },
    )
    opt = Adam()

    res = PanelPompTrainResult(
        method="train",
        execution_time=1.0,
        key=key,
        theta=theta,
        shared_traces=shared_traces,
        unit_traces=unit_traces,
        optimizer=opt,
        J=100,
        M=1,
        eta=None,
        alpha=0.9,
        alpha_cooling=1.0,
    )

    assert res.method == "train"
    assert "Discount factor (alpha)" in [x[0] for x in res._summary_config]

    # Merge
    res2 = PanelPompTrainResult(
        method="train",
        execution_time=1.2,
        key=key,
        theta=theta,
        shared_traces=shared_traces,
        unit_traces=unit_traces,
        optimizer=opt,
        J=100,
        M=1,
        eta=None,
        alpha=0.9,
        alpha_cooling=1.0,
    )
    res_merged = PanelPompTrainResult.merge(res, res2)
    assert res_merged.alpha == 0.9
    assert res_merged.execution_time == 1.2


def test_pfilter_result_single_parameter_set_0d_array():
    key = jax.random.key(0)
    theta = PompParameters({"param1": 1.0, "param2": 2.0})
    logLiks = xr.DataArray(
        [1.5],  # 1D array of shape (1,)
        dims=["theta_idx"],
    )

    res = PompPFilterResult(
        method="pfilter",
        execution_time=1.5,
        key=key,
        theta=theta,
        logLiks=logLiks,
        J=1000,
        reps=1,
        thresh=0.5,
    )

    # to_dataframe should work and not raise length mismatch error
    df = res.to_dataframe()
    assert len(df) == 1
    assert "logLik" in df.columns
    assert "se" in df.columns
    assert df.loc[0, "logLik"] == 1.5
    assert pd.isna(df.loc[0, "se"])
    assert df.loc[0, "param1"] == 1.0

    # traces should also work
    df_tr = res.traces()
    assert len(df_tr) == 1
    assert "logLik" in df_tr.columns
    assert df_tr.loc[0, "logLik"] == 1.5
