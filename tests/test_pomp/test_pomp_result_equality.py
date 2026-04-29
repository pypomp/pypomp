"""Equality tests for single-unit Pomp result classes (PompMIFResult,
PompTrainResult). Cover the structural-equality methods that no other
test currently exercises."""

from copy import deepcopy

import jax
import pytest

import pypomp as pp
from pypomp.core.results import PompMIFResult, PompTrainResult


@pytest.fixture(scope="module")
def lg_with_mif_result():
    LG = pp.models.LG()
    rw_sd = pp.RWSigma(
        sigmas={n: 0.02 for n in LG.canonical_param_names}, init_names=[]
    )
    LG.mif(J=2, M=2, a=0.5, rw_sd=rw_sd, key=jax.random.key(0))
    res = LG.results_history[-1]
    assert isinstance(res, PompMIFResult)
    return res


@pytest.fixture(scope="module")
def lg_with_train_result():
    LG = pp.models.LG()
    eta = {n: 0.01 for n in LG.canonical_param_names}
    LG.train(J=2, M=2, eta=eta, optimizer="SGD", key=jax.random.key(0))
    res = LG.results_history[-1]
    assert isinstance(res, PompTrainResult)
    return res


def test_mif_result_equality(lg_with_mif_result):
    res = lg_with_mif_result
    assert res == deepcopy(res)


def test_train_result_equality(lg_with_train_result):
    res = lg_with_train_result
    assert res == deepcopy(res)
