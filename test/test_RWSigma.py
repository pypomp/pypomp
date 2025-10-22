import pytest
import jax.numpy as jnp
from pypomp.rw_sd_class import RWSigma


class TestRWSigma:
    """Test suite for RWSigma class."""

    @pytest.mark.parametrize(
        "sigmas,init_names,expected_not_init,expected_all",
        [
            (
                {"param1": 0.1, "param2": 0.2, "param3": 0.3},
                ["param1"],
                ["param2", "param3"],
                ["param2", "param3", "param1"],
            ),
            (
                {"param1": 0.1, "param2": 0.2},
                [],
                ["param1", "param2"],
                ["param1", "param2"],
            ),
            (
                {"param1": 0.1, "param2": 0.2},
                ["param1", "param2"],
                [],
                ["param1", "param2"],
            ),
        ],
    )
    def test_init_valid_cases(
        self, sigmas, init_names, expected_not_init, expected_all
    ):
        """Test initialization with valid inputs."""
        rw_sigma = RWSigma(sigmas, init_names)

        assert rw_sigma.sigmas == sigmas
        assert rw_sigma.init_names == init_names
        assert rw_sigma.not_init_names == expected_not_init
        assert rw_sigma.all_names == expected_all

    @pytest.mark.parametrize(
        "sigmas,init_names,expected_error",
        [
            ("not a dict", [], "sigmas must be a dictionary"),
            (
                {"param1": "not a float"},
                [],
                "All values in sigmas dictionary must be floats",
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
        ],
    )
    def test_init_invalid_cases(self, sigmas, init_names, expected_error):
        """Test initialization with invalid inputs."""
        with pytest.raises(ValueError, match=expected_error):
            RWSigma(sigmas, init_names)

    @pytest.mark.parametrize(
        "sigmas,init_names,param_names,expected_sigmas,expected_sigmas_init",
        [
            # No param_names specified - uses all_names order
            (
                {"param1": 0.1, "param2": 0.2, "param3": 0.3},
                ["param1"],
                None,
                jnp.array([0.2, 0.3, 0.0]),
                jnp.array([0.0, 0.0, 0.1]),
            ),
            # With specific param_names order
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
    def test_return_arrays_valid_cases(
        self, sigmas, init_names, param_names, expected_sigmas, expected_sigmas_init
    ):
        """Test _return_arrays with valid inputs."""
        rw_sigma = RWSigma(sigmas, init_names)
        sigmas_array, sigmas_init_array = rw_sigma._return_arrays(param_names)

        assert jnp.allclose(sigmas_array, expected_sigmas)
        assert jnp.allclose(sigmas_init_array, expected_sigmas_init)

    @pytest.mark.parametrize(
        "sigmas,init_names,param_names,expected_error",
        [
            (
                {"param1": 0.1, "param2": 0.2},
                ["param1"],
                ["param1"],
                "All param_names must be in all_names and vice versa",
            ),
            (
                {"param1": 0.1, "param2": 0.2},
                ["param1"],
                ["param1", "param2", "param3"],
                "All param_names must be in all_names and vice versa",
            ),
        ],
    )
    def test_return_arrays_invalid_param_names(
        self, sigmas, init_names, param_names, expected_error
    ):
        """Test _return_arrays with invalid param_names."""
        rw_sigma = RWSigma(sigmas, init_names)
        with pytest.raises(ValueError, match=expected_error):
            rw_sigma._return_arrays(param_names)

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
            # Large values
            (
                {"param1": 1e6, "param2": 1e-6},
                ["param1"],
                jnp.array([1e-6, 0.0]),
                jnp.array([0.0, 1e6]),
            ),
        ],
    )
    def test_return_arrays_edge_cases(
        self, sigmas, init_names, expected_sigmas, expected_sigmas_init
    ):
        """Test _return_arrays with edge case values."""
        rw_sigma = RWSigma(sigmas, init_names)
        sigmas_array, sigmas_init_array = rw_sigma._return_arrays()

        assert jnp.allclose(sigmas_array, expected_sigmas)
        assert jnp.allclose(sigmas_init_array, expected_sigmas_init)

    def test_validate_attributes_private_method_valid(self):
        """Test the private _validate_attributes method directly for valid cases."""
        rw_sigma = RWSigma({"param1": 0.1}, ["param1"])
        sigmas, init_names, not_init_names, all_names = rw_sigma._validate_attributes(
            {"a": 1.0, "b": 2.0}, ["a"]
        )
        assert sigmas == {"a": 1.0, "b": 2.0}
        assert init_names == ["a"]
        assert not_init_names == ["b"]
        assert all_names == ["b", "a"]

    @pytest.mark.parametrize(
        "sigmas, init_names, expected_error",
        [
            ("invalid", [], "sigmas must be a dictionary"),
            ({"a": "string"}, [], "All values in sigmas dictionary must be floats"),
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
    def test_validate_attributes_method_invalid(
        self, sigmas, init_names, expected_error
    ):
        """Test the _validate_attributes method directly for invalid cases."""
        rw_sigma = RWSigma({"param1": 0.1}, ["param1"])
        with pytest.raises(ValueError, match=expected_error):
            rw_sigma._validate_attributes(sigmas, init_names)
