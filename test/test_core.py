import pypomp.internal_functions as ifunc
import numpy as np
import pandas as pd


def test_calc_ys_covars():
    t0 = -1.0
    times = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    ys = np.array(pd.DataFrame({"Y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}))
    ctimes = np.array([0, 1, 2, 3, 4, 5])
    covars = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    order = "linear"
    dt = 0.5
    nstep = 1

    ys_ext, ys_obs, interp_covars, dt_array_ext = ifunc._calc_ys_covars(
        t0, times, ys, ctimes, covars, dt, None, order
    )

    # Check that the first and last times in dt_array_ext correspond to t0 and times[-1]
    times0 = np.concatenate((np.array([t0]), np.array(times)))
    nstep_array, dt_array = ifunc._calc_steps(times0, dt, None)
    dt_array_ext_expected = np.repeat(dt_array, nstep_array)
    assert np.allclose(np.array(dt_array_ext), dt_array_ext_expected)

    # Check that ys_ext is the correct length
    assert ys_ext.shape[0] == np.sum(nstep_array)

    # Check shapes
    assert dt_array_ext.shape[0] == ys_ext.shape[0]
    assert ys_obs.shape[0] == ys_ext.shape[0]
    assert interp_covars is not None
    assert interp_covars.shape[0] == ys_ext.shape[0] + 1
    assert interp_covars.shape[1] == covars.shape[1]
