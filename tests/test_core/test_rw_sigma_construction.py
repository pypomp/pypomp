"""Construction, validation, and container behavior of RWSigma."""

import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp


@pytest.mark.parametrize(
    "sigmas,init_names,expected_all",
    [
        (
            {"param1": 0.1, "param2": 0.2, "param3": 0.3},
            ["param1"],
            ["param1", "param2", "param3"],
        ),
        (
            {"param1": 0.1, "param2": 0.2},
            [],
            ["param1", "param2"],
        ),
        (
            {"param1": 0.1, "param2": 0.2},
            ["param1", "param2"],
            ["param1", "param2"],
        ),
    ],
)
def test_init_valid_cases(sigmas, init_names, expected_all):
    """Test initialization with valid inputs."""
    rw_sigma = pp.RWSigma(sigmas, init_names)

    assert rw_sigma.sigmas == sigmas
    assert rw_sigma.init_names == tuple(init_names)
    assert rw_sigma.param_names == tuple(expected_all)


@pytest.mark.parametrize(
    "sigmas,init_names,expected_error",
    [
        ("not a dict", [], "sigmas must be a dictionary"),
        (
            {"param1": "not a float"},
            [],
            "must be a float",
        ),
        ({"param1": 0.1}, "not a list", "init_names must be a list"),
        ({"param1": 0.1}, [1, 2], "All values in init_names list must be strings"),
        (
            {"param1": 0.1},
            ["param2"],
            "All init_names names must be in sigmas dictionary",
        ),
        (
            {"param1": 0.1},
            ["param1", "param1"],
            "Duplicate names found in init_names",
        ),
        (
            {"param1": -0.1},
            [],
            "All values in sigmas dictionary must be non-negative",
        ),
        (
            {1: 0.1},
            [],
            "All keys in sigmas must be strings",
        ),
    ],
)
def test_init_invalid_cases(sigmas, init_names, expected_error):
    """Test initialization with invalid inputs."""
    with pytest.raises(ValueError, match=expected_error):
        pp.RWSigma(sigmas, init_names)


@pytest.mark.parametrize(
    "sigmas,init_names,param_names,expected_sigmas,expected_sigmas_init",
    [
        # No param_names specified - uses the object's own (insertion) order.
        (
            {"param1": 0.1, "param2": 0.2, "param3": 0.3},
            ["param1"],
            None,
            jnp.array([0.0, 0.2, 0.3]),
            jnp.array([0.1, 0.0, 0.0]),
        ),
        # With specific param_names order (canonicalized to that order).
        (
            {"param1": 0.1, "param2": 0.2, "param3": 0.3},
            ["param1"],
            ["param2", "param1", "param3"],
            jnp.array([0.2, 0.0, 0.3]),
            jnp.array([0.0, 0.1, 0.0]),
        ),
        # All init parameters
        (
            {"param1": 0.1, "param2": 0.2},
            ["param1", "param2"],
            None,
            jnp.array([0.0, 0.0]),
            jnp.array([0.1, 0.2]),
        ),
        # No init parameters
        (
            {"param1": 0.1, "param2": 0.2},
            [],
            None,
            jnp.array([0.1, 0.2]),
            jnp.array([0.0, 0.0]),
        ),
        # Single parameter
        ({"param1": 0.5}, ["param1"], None, jnp.array([0.0]), jnp.array([0.5])),
    ],
)
def test_sigmas_arrays_valid_cases(
    sigmas, init_names, param_names, expected_sigmas, expected_sigmas_init
):
    """Test sigmas_array and sigmas_init_array with valid inputs."""
    rw_sigma = pp.RWSigma(sigmas, init_names)
    if param_names is not None:
        rw_sigma = rw_sigma._canonicalize(param_names)
    sigmas_array, sigmas_init_array = (
        rw_sigma.sigmas_array,
        rw_sigma.sigmas_init_array,
    )

    assert jnp.allclose(sigmas_array, expected_sigmas)
    assert jnp.allclose(sigmas_init_array, expected_sigmas_init)


@pytest.mark.parametrize(
    "sigmas,init_names,expected_sigmas,expected_sigmas_init",
    [
        # Zero values
        (
            {"param1": 0.0, "param2": 0.0},
            ["param1"],
            jnp.array([0.0, 0.0]),
            jnp.array([0.0, 0.0]),
        ),
        # Large values (insertion order: param1 is init, param2 is not)
        (
            {"param1": 1e6, "param2": 1e-6},
            ["param1"],
            jnp.array([0.0, 1e-6]),
            jnp.array([1e6, 0.0]),
        ),
    ],
)
def test_sigmas_arrays_edge_cases(
    sigmas, init_names, expected_sigmas, expected_sigmas_init
):
    """Test sigmas_array and sigmas_init_array with edge case values."""
    rw_sigma = pp.RWSigma(sigmas, init_names)
    sigmas_array, sigmas_init_array = (
        rw_sigma.sigmas_array,
        rw_sigma.sigmas_init_array,
    )

    assert jnp.allclose(sigmas_array, expected_sigmas)
    assert jnp.allclose(sigmas_init_array, expected_sigmas_init)


def test_validation_via_constructor_valid():
    """Validation happens in the constructor; check resulting attributes."""
    rw_sigma = pp.RWSigma({"a": 1.0, "b": 2.0}, ["a"])
    assert rw_sigma.sigmas == {"a": 1.0, "b": 2.0}
    assert rw_sigma.init_names == ("a",)
    assert rw_sigma.param_names == ("a", "b")


@pytest.mark.parametrize(
    "sigmas, init_names, expected_error",
    [
        ("invalid", [], "sigmas must be a dictionary"),
        ({"a": "string"}, [], "must be a float"),
        ({"a": 1.0}, "invalid", "init_names must be a list"),
        ({"a": 1.0}, [1, 2], "All values in init_names list must be strings"),
        ({"a": 1.0}, ["b"], "All init_names names must be in sigmas dictionary"),
        ({"a": 1.0, "b": 2.0}, ["a", "a"], "Duplicate names found in init_names"),
        (
            {"a": -1.0, "b": 2.0},
            [],
            "All values in sigmas dictionary must be non-negative",
        ),
    ],
)
def test_validation_via_constructor_invalid(sigmas, init_names, expected_error):
    """The constructor rejects invalid inputs with a descriptive message."""
    with pytest.raises(ValueError, match=expected_error):
        pp.RWSigma(sigmas, init_names)


def test_immutable_setitem():
    """RWSigma is immutable: item assignment is not supported."""
    rw_sigma = pp.RWSigma({"param1": 0.1, "param2": 0.2}, ["param1"])
    with pytest.raises(TypeError, match="does not support item assignment"):
        rw_sigma["param1"] = 0.5  # type: ignore[index]


def test_cooled():
    rw_sigma = pp.RWSigma({"param1": 0.1, "param2": 0.2}, ["param1"])
    new_rw_sigma = rw_sigma.cooled(0.5)
    assert rw_sigma.sigmas == {"param1": 0.1, "param2": 0.2}
    assert new_rw_sigma.sigmas == {"param1": 0.05, "param2": 0.1}


def test_copy():
    rw = pp.RWSigma({"param1": 0.1, "param2": 0.2}, ["param1"]).geometric_cooling(0.3)
    rw_copy = rw.copy()
    assert rw_copy == rw
    assert rw_copy is not rw
    assert rw_copy.sigmas == rw.sigmas
    assert rw_copy.init_names == rw.init_names


def test_cooled_invalid_factor():
    rw_sigma = pp.RWSigma({"param1": 0.1}, [])
    with pytest.raises(ValueError, match="factor must be >= 0"):
        rw_sigma.cooled(-0.1)


def test_container_methods():
    """Test dictionary-like container operations (read-only, insertion order)."""
    rw = pp.RWSigma({"param1": 0.1, "param2": 0.2}, ["param1"])

    # __getitem__
    assert rw["param1"] == 0.1
    assert rw["param2"] == 0.2
    with pytest.raises(KeyError, match="not found in sigmas"):
        _ = rw["param3"]

    # __contains__
    assert "param1" in rw
    assert "param3" not in rw

    # __len__
    assert len(rw) == 2

    # __iter__ (insertion order)
    assert list(rw) == ["param1", "param2"]

    # keys, values, items
    assert list(rw.keys()) == ["param1", "param2"]
    assert list(rw.values()) == [0.1, 0.2]
    assert list(rw.items()) == [("param1", 0.1), ("param2", 0.2)]

    # get
    assert rw.get("param1") == 0.1
    assert rw.get("param3") is None
    assert rw.get("param3", 0.5) == 0.5


def test_string_representations():
    """Test __str__ and __repr__ representations."""
    rw = pp.RWSigma({"param1": 0.1, "param2": 0.2}, ["param1"])
    expected_str = (
        "RWSigma(sigmas={'param1': 0.1, 'param2': 0.2}, "
        "init_names=('param1',), cooling='geometric')"
    )
    assert str(rw) == expected_str
    assert repr(rw) == expected_str


def test_init_coercion():
    """Test coercion of numpy and JAX numeric types in sigmas."""
    sigmas = {
        "param1": np.float32(0.25),
        "param2": jnp.array(0.25),
        "param3": np.int32(2),
    }
    rw = pp.RWSigma(sigmas)
    assert isinstance(rw.sigmas["param1"], float)
    assert isinstance(rw.sigmas["param2"], float)
    assert isinstance(rw.sigmas["param3"], float)
    assert rw.sigmas["param1"] == 0.25
    assert rw.sigmas["param2"] == 0.25
    assert rw.sigmas["param3"] == 2.0


@pytest.mark.parametrize(
    "invalid_val",
    [
        "not_a_number",
        True,  # Bools are excluded
        [1.0],
        {"val": 1.0},
    ],
)
def test_init_invalid_coercion(invalid_val):
    """Test that invalid types in sigmas raise ValueError."""
    with pytest.raises(ValueError, match="must be a float"):
        pp.RWSigma({"param1": invalid_val})


def test_init_names_type():
    """Test that init_names must be a list or tuple."""
    with pytest.raises(ValueError, match="init_names must be a list or tuple"):
        pp.RWSigma({"param1": 0.1}, init_names={"param1"})  # type: ignore


def test_canonicalize_reorders():
    """_canonicalize reorders arrays to match the requested parameter order."""
    rw = pp.RWSigma({"p1": 0.1, "p2": 0.2, "p3": 0.3}, ["p1"])
    rw_c = rw._canonicalize(["p3", "p1", "p2"])
    assert rw_c.param_names == ("p3", "p1", "p2")
    assert np.allclose(np.asarray(rw_c.sigmas_all_arr), [0.3, 0.1, 0.2])
    # init flag follows the parameter through the reorder.
    assert np.allclose(np.asarray(rw_c.sigmas_init_array), [0.0, 0.1, 0.0])
    # Content is preserved (equal up to reordering back).
    assert rw_c._canonicalize(["p1", "p2", "p3"]) == rw._canonicalize(
        ["p1", "p2", "p3"]
    )


def test_canonicalize_mismatch():
    rw = pp.RWSigma({"p1": 0.1, "p2": 0.2})
    with pytest.raises(ValueError, match="must match canonical_param_names"):
        rw._canonicalize(["p1", "p3"])
