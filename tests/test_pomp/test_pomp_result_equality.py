"""Equality tests for single-unit Pomp result class (Result)."""

from copy import deepcopy

import jax
import pytest

import pypomp as pp
from pypomp.core.results import Result


@pytest.fixture(scope="module")
def lg_with_mif_result():
    LG = pp.models.LG()
    rw_sd = pp.RWSigma(
        sigmas={n: 0.02 for n in LG.canonical_param_names}, init_names=[]
    ).geometric_cooling(0.5)
    LG.mif(J=2, M=2, rw_sd=rw_sd, key=jax.random.key(0))
    res = LG.results_history[-1]
    assert isinstance(res, Result)
    assert res.method == "mif"
    return res


@pytest.fixture(scope="module")
def lg_with_train_result():
    LG = pp.models.LG()
    eta = pp.LearningRate({n: 0.01 for n in LG.canonical_param_names})
    LG.train(J=2, M=2, eta=eta, optimizer=pp.SGD(), key=jax.random.key(0))
    res = LG.results_history[-1]
    assert isinstance(res, Result)
    assert res.method == "train"
    return res


def test_mif_result_equality(lg_with_mif_result):
    res = lg_with_mif_result
    assert res == deepcopy(res)


def test_train_result_equality(lg_with_train_result):
    res = lg_with_train_result
    assert res == deepcopy(res)
