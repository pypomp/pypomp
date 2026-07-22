import pytest
import numpy as np
import jax
import pypomp as pp
from pypomp.core.learning_rate import LearningRate


def test_init_valid_cases():
    """Test initialization with various valid input formats."""
    # 1. Scalar learning rates
    rates_scalar = {"beta": 0.1, "rho": 0.01}
    lr = pp.LearningRate(rates_scalar)
    assert lr.rates == {"beta": 0.1, "rho": 0.01}

    # 2. Integer/numpy numbers coerced to floats
    rates_coerced = {"beta": 1, "rho": np.float64(0.02)}
    lr = pp.LearningRate(rates_coerced)
    assert lr.rates == {"beta": 1.0, "rho": 0.02}
    assert isinstance(lr.rates["beta"], float)
    assert isinstance(lr.rates["rho"], float)

    # 3. List and ndarray schedules
    rates_seq = {"beta": [0.1, 0.2], "rho": np.array([0.01, 0.02])}
    lr = pp.LearningRate(rates_seq)
    assert np.array_equal(lr.rates["beta"], np.array([0.1, 0.2]))
    assert np.array_equal(lr.rates["rho"], np.array([0.01, 0.02]))
    assert isinstance(lr.rates["beta"], np.ndarray)
    assert isinstance(lr.rates["rho"], np.ndarray)


def test_init_invalid_cases():
    """Test validation error cases in __init__."""
    # Rates must be a Mapping
    with pytest.raises(ValueError, match="rates must be a Mapping"):
        pp.LearningRate([("beta", 0.1)])  # type: ignore

    # Keys must be strings
    with pytest.raises(ValueError, match="All keys in rates must be strings"):
        pp.LearningRate({123: 0.1})  # type: ignore

    # Schedules must be 1D
    with pytest.raises(
        ValueError, match="Learning rate schedule for 'beta' must be 1D"
    ):
        pp.LearningRate({"beta": [[0.1, 0.2]]})  # type: ignore

    # Unsupported type
    with pytest.raises(
        TypeError,
        match="Learning rate for 'beta' must be float or 1D sequence",
    ):
        pp.LearningRate({"beta": "not_a_number"})  # type: ignore


def test_to_array_valid():
    """Test conversion to JAX Array."""
    rates = {"beta": 0.1, "rho": np.array([0.01, 0.02, 0.03])}
    lr = pp.LearningRate(rates)

    # M = 3, matching schedule size
    arr = lr.to_array(["beta", "rho"], M=3)
    assert isinstance(arr, jax.Array)
    expected = np.array([[0.1, 0.01], [0.1, 0.02], [0.1, 0.03]])
    assert np.allclose(arr, expected)


def test_to_array_m_eff():
    """Test conversion to JAX Array when M <= 0."""
    # When M is 0 or negative, M_eff is max(M, 1) = 1
    rates = {"beta": 0.1, "rho": 0.05}
    lr = pp.LearningRate(rates)
    arr = lr.to_array(["beta", "rho"], M=0)
    assert arr.shape == (1, 2)
    assert np.allclose(arr, np.array([[0.1, 0.05]]))


def test_to_array_invalid():
    """Test to_array with mismatched or missing parameters."""
    rates = {"beta": 0.1, "rho": np.array([0.01, 0.02])}
    lr = pp.LearningRate(rates)

    # Parameter not in learning rates
    with pytest.raises(
        ValueError, match="Parameter 'gamma' not found in learning rates"
    ):
        lr.to_array(["beta", "gamma"], M=2)

    # Mismatched array size M
    with pytest.raises(
        ValueError,
        match="Learning rate schedule for 'rho' has length 2, expected M=3",
    ):
        lr.to_array(["beta", "rho"], M=3)


def test_cosine_decay():
    """Test cosine_decay schedule."""
    rates = {"beta": 0.1, "rho": np.array([0.2, 0.4])}
    lr = pp.LearningRate(rates)

    # Out of bounds final_factor
    with pytest.raises(ValueError, match="final_factor should be between 0 and 1"):
        lr.cosine_decay(-0.1, 2)
    with pytest.raises(ValueError, match="final_factor should be between 0 and 1"):
        lr.cosine_decay(1.1, 2)

    # Successful decay
    decayed = lr.cosine_decay(final_factor=0.5, M=2)
    assert isinstance(decayed, LearningRate)

    # Factor for M=2:
    # iterations = np.array([0, 1])
    # factor = 0.5 + 0.5 * 0.5 * (1.0 + cos(pi * iterations / 2))
    # t=0: cos(0) = 1 => factor = 0.5 + 0.25 * (2) = 1.0
    # t=1: cos(pi/2) = 0 => factor = 0.5 + 0.25 * (1) = 0.75
    expected_factor = np.array([1.0, 0.75])
    assert np.allclose(decayed.rates["beta"], 0.1 * expected_factor)
    assert np.allclose(decayed.rates["rho"], np.array([0.2, 0.4]) * expected_factor)

    # Mismatched array size for cosine decay
    with pytest.raises(
        ValueError,
        match="Cannot apply cosine decay of length 3 to schedule of length 2 for 'rho'",
    ):
        lr.cosine_decay(0.5, 3)


def test_geometric_decay():
    """Test geometric_decay schedule."""
    rates = {"beta": 0.1, "rho": np.array([0.2, 0.4])}
    lr = pp.LearningRate(rates)

    # Out of bounds decay_rate
    with pytest.raises(ValueError, match="decay_rate should be between 0 and 1"):
        lr.geometric_decay(-0.1, 2)
    with pytest.raises(ValueError, match="decay_rate should be between 0 and 1"):
        lr.geometric_decay(1.1, 2)

    # Successful decay
    decayed = lr.geometric_decay(decay_rate=0.5, M=2)
    assert isinstance(decayed, LearningRate)
    expected_factor = np.array([1.0, 0.5])
    assert np.allclose(decayed.rates["beta"], 0.1 * expected_factor)
    assert np.allclose(decayed.rates["rho"], np.array([0.2, 0.4]) * expected_factor)

    # Mismatched array size for geometric decay
    with pytest.raises(
        ValueError,
        match="Cannot apply geometric decay of length 3 to schedule of length 2 for 'rho'",
    ):
        lr.geometric_decay(0.5, 3)


def test_linear_decay():
    """Test linear_decay schedule."""
    rates = {"beta": 0.1, "rho": np.array([0.2, 0.4])}
    lr = pp.LearningRate(rates)

    # Out of bounds final_factor
    with pytest.raises(ValueError, match="final_factor should be between 0 and 1"):
        lr.linear_decay(-0.1, 2)
    with pytest.raises(ValueError, match="final_factor should be between 0 and 1"):
        lr.linear_decay(1.1, 2)

    # Successful decay
    decayed = lr.linear_decay(final_factor=0.5, M=2)
    assert isinstance(decayed, LearningRate)
    expected_factor = np.array([1.0, 0.5])
    assert np.allclose(decayed.rates["beta"], 0.1 * expected_factor)
    assert np.allclose(decayed.rates["rho"], np.array([0.2, 0.4]) * expected_factor)

    # Mismatched array size for linear decay
    with pytest.raises(
        ValueError,
        match="Cannot apply linear decay of length 3 to schedule of length 2 for 'rho'",
    ):
        lr.linear_decay(0.5, 3)


def test_equality():
    """Test __eq__ method."""
    lr1 = pp.LearningRate({"beta": 0.1, "rho": np.array([1.0, 2.0])})
    lr2 = pp.LearningRate({"beta": 0.1, "rho": np.array([1.0, 2.0])})
    lr_diff_keys = pp.LearningRate({"beta": 0.1, "gamma": np.array([1.0, 2.0])})
    lr_diff_vals = pp.LearningRate({"beta": 0.1, "rho": np.array([1.0, 3.0])})
    lr_diff_type = pp.LearningRate({"beta": 0.1, "rho": 1.0})

    assert lr1 == lr2
    assert lr1 != lr_diff_keys
    assert lr1 != lr_diff_vals
    assert lr1 != lr_diff_type
    assert lr1 != "not a LearningRate"


def test_pytree_registration():
    """Test JAX PyTree flattening and unflattening."""
    lr = pp.LearningRate({"beta": 0.1, "rho": np.array([0.01, 0.02])})
    leaves, treedef = jax.tree_util.tree_flatten(lr)
    assert len(leaves) == 1
    assert np.allclose(leaves[0], np.array([[0.1, 0.01], [0.1, 0.02]]))

    # Unflatten PyTree
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, LearningRate)
    assert rebuilt == lr


def test_canonicalize():
    """Test reordering parameters with _canonicalize."""
    lr = pp.LearningRate({"beta": 0.1, "rho": 0.02})
    canonicalized = lr._canonicalize(["rho", "beta"])
    assert canonicalized.param_names == ("rho", "beta")
    assert np.allclose(canonicalized.rates_all_arr, np.array([0.02, 0.1]))

    # Test error case with missing/extra names
    with pytest.raises(
        ValueError, match="Parameter 'gamma' not found in learning rates"
    ):
        lr._canonicalize(["beta", "gamma"])


def test_mismatched_schedule_lengths():
    """Test error when schedules have conflicting lengths."""
    with pytest.raises(
        ValueError, match="All 1D learning rate schedules must have the same length"
    ):
        pp.LearningRate({"beta": [0.1, 0.2], "rho": [0.01, 0.02, 0.03]})


def test_str_repr():
    """Test __str__ and __repr__ formatting."""
    # 1. Scalar values
    lr_scalar = pp.LearningRate({"beta": 0.123456, "rho": 2})
    expected_str_scalar = "LearningRate(\n    'beta': 0.1235\n    'rho': 2\n)"
    assert str(lr_scalar) == expected_str_scalar
    assert repr(lr_scalar) == expected_str_scalar

    # 2. Empty array
    lr_empty = pp.LearningRate({"beta": np.array([])})
    expected_str_empty = "LearningRate(\n    'beta': []\n)"
    assert str(lr_empty) == expected_str_empty

    # 3. Short array (<= 5 elements)
    lr_short = pp.LearningRate({"beta": np.array([1.0, 2.0, 3.00004])})
    expected_str_short = "LearningRate(\n    'beta': [1, 2, 3]\n)"
    assert str(lr_short) == expected_str_short

    # 4. Long array (> 5 elements)
    lr_long = pp.LearningRate(
        {"beta": np.array([1.11111, 2.0, 3.0, 4.0, 5.0, 9.99999])}
    )
    expected_str_long = "LearningRate(\n    'beta': [1.111 ... 10] (len=6)\n)"
    assert str(lr_long) == expected_str_long
