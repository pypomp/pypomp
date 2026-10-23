"""The numerical checks run before CuTe is chosen. They run on XLA, so need no GPU."""

from pypomp.cute.verify import _check

from .test_cute_backend import particle_mean, sde_model, vectorized_rproc


def test_check_accepts_a_kernel_that_matches_xla():
    model = sde_model()
    assert _check(model, model.rproc.mechanics_pf_interp, False) is None


def test_check_rejects_a_kernel_that_differs_from_xla():
    model = sde_model()

    def off(X_, *args):
        out, t_idx = model.rproc.mechanics_pf_interp(X_, *args)
        return out * 1.01 + 0.1, t_idx

    assert "differ" in (_check(model, off, False) or "")


def test_coupling_probe():
    coupled = sde_model(particle_mean)
    assert "other particles" in (
        _check(coupled, coupled.rproc.mechanics_pf_interp, True) or ""
    )
    alone = sde_model(vectorized_rproc)
    assert _check(alone, alone.rproc.mechanics_pf_interp, True) is None
