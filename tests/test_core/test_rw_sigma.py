import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pypomp as pp
import pickle


class CallableObj:
    """A helper class that is callable but does not have __code__."""

    def __call__(self, nt, m, ntimes):
        return 1.0


def dummy_cooling_global(nt, m, ntimes):
    return 1.0


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
    with pytest.raises(TypeError):
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


def test_cooling_schedules():
    sigmas = {"param1": 0.1}
    # 1. Default init has geometric cooling with a=0.5
    rw = pp.RWSigma(sigmas)
    assert rw.a == 0.5
    factor = 0.5 ** (1 / 50)
    assert np.isclose(rw.cooling_factor(10, 5, 20), factor ** (10 / 20 + 5))

    # 2. geometric_cooling
    rw_geom = rw.geometric_cooling(0.3)
    assert rw_geom.a == 0.3
    f_geom = rw_geom.cooling_factor(0, 0, 50)
    assert np.isclose(f_geom, 1.0)

    # 3. cosine_cooling
    rw_cos = rw.cosine_cooling(0.1, 10)
    assert rw_cos.a is None
    f_cos = rw_cos.cooling_factor(5, 2, 10)
    expected_cos = 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * 0.25))
    assert np.isclose(f_cos, expected_cos)

    # 4. hyperbolic_cooling
    rw_hyper = rw.hyperbolic_cooling(0.2)
    assert rw_hyper.a is None
    f_hyper = rw_hyper.cooling_factor(5, 2, 10)
    assert np.isclose(f_hyper, 1.0 / 1.5)

    # 5. custom_cooling
    def custom_fn(nt, m, ntimes):
        return float(nt + m + ntimes)

    rw_custom = rw.custom_cooling(custom_fn)
    assert rw_custom.a is None
    assert rw_custom.cooling_factor(5, 2, 10) == 17.0


def test_cooling_factor_under_jit():
    """cooling_factor must be evaluable inside jit (traced arguments)."""
    rw = pp.RWSigma({"param1": 0.1}).geometric_cooling(0.5)

    @jax.jit
    def f(nt, m, ntimes):
        return rw.cooling_factor(nt, m, ntimes)

    factor = 0.5 ** (1 / 50)
    assert np.isclose(float(f(10, 5, 20)), factor ** (10 / 20 + 5))


def test_pytree_roundtrip():
    """RWSigma flattens/unflattens as a PyTree with numpy-array leaves."""
    rw = pp.RWSigma({"a": 0.1, "b": 0.2}, ["a"]).geometric_cooling(0.3)
    leaves, treedef = jax.tree_util.tree_flatten(rw)
    # Leaves are the two sigma arrays (stored as numpy for cheap pickling).
    assert len(leaves) == 2
    assert all(isinstance(leaf, np.ndarray) for leaf in leaves)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt == rw


def test_cooling_fn_equality():
    sigmas = {"param1": 0.1}
    rw1 = pp.RWSigma(sigmas).geometric_cooling(0.5)
    rw2 = pp.RWSigma(sigmas).geometric_cooling(0.5)
    assert rw1 == rw2

    rw3 = pp.RWSigma(sigmas).geometric_cooling(0.3)
    assert rw1 != rw3

    # Cosine cooling
    rw_cos1 = rw1.cosine_cooling(0.1, 10)
    rw_cos2 = rw2.cosine_cooling(0.1, 10)
    assert rw_cos1 == rw_cos2

    rw_cos3 = rw1.cosine_cooling(0.2, 10)
    assert rw_cos1 != rw_cos3

    # Hyperbolic cooling
    rw_hyp1 = rw1.hyperbolic_cooling(0.5)
    rw_hyp2 = rw2.hyperbolic_cooling(0.5)
    assert rw_hyp1 == rw_hyp2

    rw_hyp3 = rw1.hyperbolic_cooling(0.3)
    assert rw_hyp1 != rw_hyp3

    # Custom cooling - same function instance
    def fn_same(nt, m, ntimes):
        return 2.0

    rw_cust1 = rw1.custom_cooling(fn_same)
    rw_cust2 = rw2.custom_cooling(fn_same)
    assert rw_cust1 == rw_cust2

    # Custom cooling - different function instances defined on different lines
    def fn_other(nt, m, ntimes):
        return 2.0

    rw_cust3 = rw1.custom_cooling(fn_other)
    assert rw_cust1 != rw_cust3

    # Closures (different instances created on the same line by a factory)
    def make_fn(x):
        def fn(nt, m, ntimes):
            return float(x)

        return fn

    fn_a = make_fn(5)
    fn_b = make_fn(5)
    fn_c = make_fn(6)

    rw_cust_a = rw1.custom_cooling(fn_a)
    rw_cust_b = rw2.custom_cooling(fn_b)
    rw_cust_c = rw1.custom_cooling(fn_c)

    assert rw_cust_a == rw_cust_b
    assert rw_cust_a != rw_cust_c


def test_pickle():
    sigmas = {"param1": 0.1}
    rw = pp.RWSigma(sigmas)

    for rw_obj in [
        rw,
        rw.geometric_cooling(0.3),
        rw.cosine_cooling(0.1, 10),
        rw.hyperbolic_cooling(0.5),
    ]:
        pickled = pickle.dumps(rw_obj)
        unpickled = pickle.loads(pickled)
        assert unpickled == rw_obj
        # Verify the reconstructed cooling schedule still works.
        assert np.isclose(
            unpickled.cooling_factor(5, 2, 10), rw_obj.cooling_factor(5, 2, 10)
        )


def test_container_methods():
    """Test dictionary-like container operations (read-only, insertion order)."""
    rw = pp.RWSigma({"param1": 0.1, "param2": 0.2}, ["param1"])

    # __getitem__
    assert rw["param1"] == 0.1
    assert rw["param2"] == 0.2
    with pytest.raises(KeyError):
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


def test_invalid_cooling_parameters():
    """Test that invalid cooling parameters raise ValueErrors."""
    rw = pp.RWSigma({"param1": 0.1})

    # Geometric cooling
    with pytest.raises(ValueError, match="a should be between 0 and 1"):
        rw.geometric_cooling(-0.1)
    with pytest.raises(ValueError, match="a should be between 0 and 1"):
        rw.geometric_cooling(1.1)

    # Cosine cooling
    with pytest.raises(ValueError, match="c should be between 0 and 1"):
        rw.cosine_cooling(-0.1, 10)
    with pytest.raises(ValueError, match="c should be between 0 and 1"):
        rw.cosine_cooling(1.1, 10)
    with pytest.raises(ValueError, match="M must be positive"):
        rw.cosine_cooling(0.5, 0)
    with pytest.raises(ValueError, match="M must be positive"):
        rw.cosine_cooling(0.5, -5)

    # Hyperbolic cooling
    with pytest.raises(ValueError, match="s must be non-negative"):
        rw.hyperbolic_cooling(-0.1)


def test_eq_non_rw_sigma():
    """Test equality comparison with a non-RWSigma object."""
    rw = pp.RWSigma({"param1": 0.1})
    assert rw != "string"
    assert rw != {"param1": 0.1}


def test_eq_different_parameters():
    """Test equality comparison with different sigmas, init_names, or values."""
    rw1 = pp.RWSigma({"p1": 0.1, "p2": 0.2}, ["p1"])
    rw2 = pp.RWSigma({"p1": 0.1, "p3": 0.2}, ["p1"])  # different keys
    assert rw1 != rw2

    rw3 = pp.RWSigma({"p1": 0.1, "p2": 0.3}, ["p1"])  # different values
    assert rw1 != rw3

    rw4 = pp.RWSigma({"p1": 0.1, "p2": 0.2}, ["p2"])  # different init_names
    assert rw1 != rw4

    rw5 = pp.RWSigma({"p1": 0.1, "p2": 0.2}, ["p1"]).geometric_cooling(0.99)
    assert rw1 != rw5  # different geometric `a`


def test_eq_different_cooling_type():
    """Test equality comparison with different cooling types."""
    rw = pp.RWSigma({"p1": 0.1})
    rw_geom = rw.geometric_cooling(0.5)
    rw_cos = rw.cosine_cooling(0.5, 10)
    assert rw_geom != rw_cos


def test_eq_unknown_cooling_type():
    """Equality when the cooling type is an unrecognized/flat schedule."""
    rw1 = pp.RWSigma({"p1": 0.1})
    rw2 = pp.RWSigma({"p1": 0.1})

    rw1._set_cooling("none")
    rw2._set_cooling("none")
    assert rw1 == rw2

    rw2._set_cooling("unknown")
    assert rw1 != rw2


def test_eq_custom_cooling_edge_cases():
    """Equality for custom cooling functions with different closures/codes."""
    rw = pp.RWSigma({"p1": 0.1})

    # 1. Custom function without __code__ (callable class instance)
    c1 = CallableObj()
    c2 = CallableObj()
    rw_c1 = rw.custom_cooling(c1)
    rw_c2 = rw.custom_cooling(c2)
    assert rw_c1 != rw_c2
    assert rw_c1 == rw_c1

    # 2. One function has a closure, one does not
    def no_closure(nt, m, ntimes):
        return 1.0

    x = 1.0

    def with_closure(nt, m, ntimes):
        return float(x)

    rw_no = rw.custom_cooling(no_closure)
    rw_with = rw.custom_cooling(with_closure)
    assert rw_no != rw_with

    # 3. Closures with different lengths
    y = 2.0

    def with_closure_2(nt, m, ntimes):
        return float(x + y)

    rw_with_2 = rw.custom_cooling(with_closure_2)
    assert rw_with != rw_with_2


def test_pickle_custom_cooling():
    """Custom cooling functions survive pickling via cloudpickle."""
    rw = pp.RWSigma({"p1": 0.1})
    rw_custom = rw.custom_cooling(dummy_cooling_global)
    pickled = pickle.dumps(rw_custom)
    unpickled = pickle.loads(pickled)
    assert unpickled == rw_custom
    assert unpickled.cooling_type == "custom"
    assert unpickled.cooling_factor(1, 2, 3) == 1.0


def test_pickle_custom_cooling_closure():
    """A closure-based custom schedule also round-trips via cloudpickle."""
    import cloudpickle

    def make(scale):
        def fn(nt, m, ntimes):
            return scale * (m + 1.0)

        return fn

    rw = pp.RWSigma({"p1": 0.1}).custom_cooling(make(0.5))
    unpickled = cloudpickle.loads(cloudpickle.dumps(rw))
    assert np.isclose(unpickled.cooling_factor(0, 3, 10), rw.cooling_factor(0, 3, 10))


def test_flat_cooling_factor():
    """An unrecognized/flat cooling type evaluates to a constant factor of 1.0."""
    rw = pp.RWSigma({"p1": 0.1})
    rw._set_cooling("none")
    assert rw.cooling_factor(10, 5, 20) == 1.0
    assert rw.a is None and rw.s is None and rw.c is None and rw.M is None
    # Round-trips through pickle.
    rw2 = pickle.loads(pickle.dumps(rw))
    assert rw2.cooling_type == "none"
    assert rw2.cooling_factor(10, 5, 20) == 1.0
