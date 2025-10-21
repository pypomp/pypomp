import jax
import jax.numpy as jnp
import pytest
import pypomp as pp


@pytest.fixture(scope="module")
def simple_setup():
    # Set default values for tests
    LG = pp.LG()
    sigmas = 0.02
    sigmas_init = 0.1
    sigmas_long = {param_name: 0.02 for param_name in LG.param_names}
    sigmas_long[LG.param_names[-1]] = 0.0
    J = 2
    key = jax.random.key(111)
    a = 0.5
    M = 2
    theta = LG.theta
    return LG, sigmas, sigmas_init, sigmas_long, J, key, a, M, theta


@pytest.fixture(scope="function")
def simple(simple_setup):
    LG, sigmas, sigmas_init, sigmas_long, J, key, a, M, theta = simple_setup
    LG.results_history.clear()
    LG.theta = theta
    return LG, sigmas, sigmas_init, sigmas_long, J, key, a, M


def test_class_mif_basic(simple):
    LG, sigmas, sigmas_init, sigmas_long, J, key, a, M = simple
    for J, M in [(J, 2), (100, 10)]:
        LG.mif(
            J=J,
            M=M,
            sigmas=sigmas,
            sigmas_init=1e-20,
            a=a,
            key=key,
        )
        mif_out1 = LG.results_history[-1]
        traces = mif_out1["traces"]
        # traces is an xarray.DataArray with dims: (replicate, iteration, variable)
        # Check shape for first replicate
        assert traces.sel(replicate=0).shape == (M + 1, len(LG.theta[0]) + 1)
        # +1 for logLik column
        # Check that "logLik" is in variable coordinate
        assert "logLik" in list(traces.coords["variable"].values)
        # Check that all parameter names are in variable coordinate
        for param in LG.theta[0].keys():
            assert param in list(traces.coords["variable"].values)

    # check that sigmas isn't modified by mif
    assert sigmas == 0.02


def test_class_mif_sigmas_array(simple):
    LG, sigmas, sigmas_init, sigmas_long, J, key, a, M = simple
    # check that sigmas array input works
    LG.mif(
        sigmas=sigmas_long,
        sigmas_init=1e-20,
        J=J,
        M=M,
        a=a,
        key=key,
    )
    mif_out2 = LG.results_history[-1]
    traces2 = mif_out2["traces"]
    # check that the last parameter is never perturbed (assuming it's the 16th parameter)
    param_names = list(LG.theta[0].keys())
    last_param = param_names[15] if len(param_names) > 15 else param_names[-1]
    last_param_trace = traces2.sel(replicate=0, variable=last_param).values
    assert (last_param_trace == last_param_trace[0]).all()
    # check that some other parameter is perturbed
    first_param = param_names[0]
    first_param_trace = traces2.sel(replicate=0, variable=first_param).values
    assert (first_param_trace != first_param_trace[0]).any()


def test_mif_order_of_sigmas_consistency(simple):
    LG, sigmas, sigmas_init, sigmas_long, J, key, a, M = simple
    theta = LG.theta

    param_names = LG.param_names

    base_sigma = 0.01
    unique_sigmas = [base_sigma + 0.001 * i for i in range(len(param_names))]
    sigmas_dict = {k: v for k, v in zip(param_names, unique_sigmas)}

    reversed_param_names = list(reversed(param_names))
    sigmas_dict_reversed = {k: sigmas_dict[k] for k in reversed_param_names}

    LG.mif(
        theta=theta,
        J=J,
        M=M,
        sigmas=sigmas_dict,
        sigmas_init=sigmas_init,
        a=a,
        key=key,
    )
    traces_ref = LG.results_history[-1]["traces"]

    LG.results_history.clear()

    LG.mif(
        theta=theta,
        J=J,
        M=M,
        sigmas=sigmas_dict_reversed,
        sigmas_init=sigmas_init,
        a=a,
        key=key,
    )
    traces_rev = LG.results_history[-1]["traces"]

    for param in param_names:
        arr1 = traces_ref.sel(replicate=0, variable=param).values
        arr2 = traces_rev.sel(replicate=0, variable=param).values
        assert jnp.allclose(arr1, arr2), (
            f"Traces for param {param} differed after reordering sigmas dict keys:\n"
            f"default: {arr1}\nreversed: {arr2}"
        )
    arr1 = traces_ref.sel(replicate=0, variable="logLik").values
    arr2 = traces_rev.sel(replicate=0, variable="logLik").values
    nan_mask1 = jnp.isnan(arr1)
    nan_mask2 = jnp.isnan(arr2)
    assert jnp.array_equal(nan_mask1, nan_mask2), (
        f"NaN positions for logLik differed after reordering sigmas dict keys:\n"
        f"default NaN mask: {nan_mask1}\nreversed NaN mask: {nan_mask2}"
    )


def test_order_of_parameters_consistency(simple):
    LG, sigmas, sigmas_init, sigmas_long, J, key, a, M = simple
    # check that the order of parameters in the theta dict does not affect the results
    theta_orig = LG.theta[0]

    keys = list(theta_orig.keys())
    reversed_keys = list(reversed(keys))
    theta_reordered = {k: theta_orig[k] for k in reversed_keys}

    LG.mif(
        J=J,
        M=M,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        a=a,
        key=key,
        theta=theta_orig,
    )
    traces_orig = LG.results_history[-1]["traces"]

    LG.results_history.clear()

    LG.mif(
        J=J,
        M=M,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        a=a,
        key=key,
        theta=theta_reordered,
    )
    traces_reordered = LG.results_history[-1]["traces"]

    for param in theta_orig.keys():
        arr1 = traces_orig.sel(replicate=0, variable=param).values
        arr2 = traces_reordered.sel(replicate=0, variable=param).values
        assert jnp.allclose(arr1, arr2), (
            f"Traces differed after reordering theta dict keys for param {param}:\n"
            f"original: {arr1}\nreordered: {arr2}"
        )
    arr1 = traces_orig.sel(replicate=0, variable="logLik").values
    arr2 = traces_reordered.sel(replicate=0, variable="logLik").values
    # Handle NaN values by checking if they're in the same positions
    nan_mask1 = jnp.isnan(arr1)
    nan_mask2 = jnp.isnan(arr2)
    assert jnp.array_equal(nan_mask1, nan_mask2), (
        f"NaN positions differed after reordering theta dict keys:\n"
        f"original NaN mask: {nan_mask1}\nreordered NaN mask: {nan_mask2}"
    )
    # Check non-NaN values
    if not jnp.all(nan_mask1):
        non_nan_mask = ~nan_mask1
        assert jnp.allclose(arr1[non_nan_mask], arr2[non_nan_mask]), (
            f"logLik traces differed after reordering theta dict keys:\n"
            f"original: {arr1}\nreordered: {arr2}"
        )


def test_invalid_mif_input(simple):
    LG, sigmas, sigmas_init, sigmas_long, J, key, a, M = simple
    with pytest.raises(ValueError):
        LG.mif(
            sigmas=sigmas,
            sigmas_init=sigmas_init,
            M=M,
            a=a,
            J=-1,
            key=key,
        )
