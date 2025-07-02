import jax
import unittest
import pandas as pd
import pypomp as pp
import jax.numpy as jnp
import numpy.testing as npt
import xarray as xr


class TestPompClass_LG(unittest.TestCase):
    def setUp(self):
        self.LG1 = pp.LG()
        self.LG2 = pp.LG()

        # LG model expects 16 parameters: A(4), C(4), Q(4), R(4)
        shared_params = pd.DataFrame(
            {
                "shared": [
                    float(jnp.cos(0.2)),
                    float(-jnp.sin(0.2)),
                    float(jnp.sin(0.2)),
                    float(jnp.cos(0.2)),
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            },
            index=pd.Index(["A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4"]),
        )
        unit_specific_params = pd.DataFrame(
            {
                "LG1": [
                    1 / 100,
                    1e-4 / 100,
                    1e-4 / 100,
                    1 / 100,
                    1 / 10,
                    0.1 / 10,
                    0.1 / 10,
                    1 / 10,
                ],
                "LG2": [
                    1 / 100,
                    1e-4 / 100,
                    1e-4 / 100,
                    1 / 100,
                    1 / 10,
                    0.1 / 10,
                    0.1 / 10,
                    1 / 10,
                ],
            },
            index=pd.Index(["Q1", "Q2", "Q3", "Q4", "R1", "R2", "R3", "R4"]),
        )

        self.panel = pp.PanelPomp(
            {"LG1": self.LG1, "LG2": self.LG2},
            shared=shared_params,
            unit_specific=unit_specific_params,
        )
        self.J = 5
        self.sigmas = 0.02
        self.sigmas_init = 0.1
        self.M = 2
        self.a = 0.9
        self.key = jax.random.key(111)

    def test_basic_initialization(self):
        self.assertIsInstance(self.panel, pp.PanelPomp)

        # Test invalid initialization with single Pomp object
        with self.assertRaises(TypeError):
            pp.PanelPomp(
                self.LG1,  # type: ignore
                shared=pd.DataFrame({"shared": [0.1]}, index=pd.Index(["param1"])),
                unit_specific=pd.DataFrame({"LG1": [0.2]}, index=pd.Index(["param2"])),
            )

        # Test invalid initialization with non-Pomp object
        with self.assertRaises(TypeError):
            pp.PanelPomp(
                {"LG1": self.LG1, "LG2": 2},
                shared=pd.DataFrame({"shared": [0.1]}, index=pd.Index(["param1"])),
                unit_specific=pd.DataFrame(
                    {"LG1": [0.2], "LG2": [0.3]}, index=pd.Index(["param2"])
                ),
            )

        # Test invalid shared parameters format
        with self.assertRaises(ValueError):
            pp.PanelPomp(
                {"LG1": self.LG1, "LG2": self.LG2},
                shared=pd.DataFrame({"wrong_col": [0.1]}, index=pd.Index(["param1"])),
                unit_specific=pd.DataFrame(
                    {"LG1": [0.2], "LG2": [0.3]}, index=pd.Index(["param2"])
                ),
            )

        # Test invalid unit-specific parameters format
        with self.assertRaises(ValueError):
            pp.PanelPomp(
                {"LG1": self.LG1, "LG2": self.LG2},
                shared=pd.DataFrame({"shared": [0.1]}, index=pd.Index(["param1"])),
                unit_specific=pd.DataFrame(
                    {"wrong_unit": [0.2]}, index=pd.Index(["param2"])
                ),
            )

    def test_simulate(self):
        # Test that simulate runs to completion
        sim = self.panel.simulate(key=self.key)
        self.assertIsInstance(sim, dict)
        self.assertEqual(set(sim.keys()), {"LG1", "LG2"})
        for unit in sim:
            self.assertIn("X_sims", sim[unit])
            self.assertIn("Y_sims", sim[unit])

    def test_pfilter(self):
        # Test that pfilter runs to completion
        self.panel.pfilter(J=self.J, key=self.key)
        pfilter_out = self.panel.results_history[-1]
        self.assertIsInstance(pfilter_out, dict)
        self.assertEqual(
            set(pfilter_out.keys()),
            {"logLik", "shared", "unit_specific", "J", "thresh"},
        )
        self.assertIsInstance(pfilter_out["logLik"], xr.DataArray)
        self.assertEqual(
            set(pfilter_out["logLik"].coords["unit"].values), {"LG1", "LG2"}
        )
        self.assertEqual(set(pfilter_out["logLik"].coords["replicate"].values), {0})

    def test_mif(self):
        # Test that mif runs to completion
        self.panel.mif(
            J=self.J,
            sigmas=self.sigmas,
            sigmas_init=self.sigmas_init,
            M=self.M,
            a=self.a,
            key=self.key,
        )
        mif_out = self.panel.results_history[-1]
        self.assertIsInstance(mif_out, dict)
        self.assertIn("logLiks", mif_out)
        self.assertIn("shared_thetas", mif_out)
        self.assertIn("unit_specific_thetas", mif_out)
        self.assertIn("unit_logLiks", mif_out)

        # Check dimensions of output arrays
        if mif_out["shared_thetas"] is not None:
            self.assertEqual(
                mif_out["shared_thetas"].shape[0], self.M + 1
            )  # iterations
            self.assertEqual(mif_out["shared_thetas"].shape[1], len(self.panel.shared))
            self.assertEqual(mif_out["shared_thetas"].shape[2], self.J)  # particles

        if mif_out["unit_specific_thetas"] is not None:
            self.assertEqual(
                mif_out["unit_specific_thetas"].shape[0], self.M + 1
            )  # iterations
            self.assertEqual(
                mif_out["unit_specific_thetas"].shape[1], len(self.panel.unit_specific)
            )
            self.assertEqual(
                mif_out["unit_specific_thetas"].shape[2], self.J
            )  # particles
            self.assertEqual(mif_out["unit_specific_thetas"].shape[3], 2)  # units

    def test_mif_zero_sigmas(self):
        """Test that parameters remain unchanged when sigmas and sigmas_init are 0."""
        # Run mif with zero sigmas
        self.panel.mif(
            J=self.J,
            sigmas=0.0,
            sigmas_init=0.0,
            M=self.M,
            a=self.a,
            key=self.key,
        )
        mif_out = self.panel.results_history[-1]

        # Check shared parameters
        shared_initial = jnp.array(self.panel.shared["shared"].values)
        shared_final = mif_out["shared_thetas"][-1].mean(
            axis=1
        )  # Average over particles
        npt.assert_allclose(shared_initial, shared_final, rtol=1e-5)

        # Check unit-specific parameters
        for i, unit in enumerate(self.panel.unit_objects.keys()):
            unit_initial = jnp.array(self.panel.unit_specific[unit].values)
            unit_final = mif_out["unit_specific_thetas"][-1, :, :, i].mean(
                axis=1
            )  # Average over particles
            npt.assert_allclose(unit_initial, unit_final, rtol=1e-5)

        # Verify that parameters are identical across iterations
        for i in range(1, self.M + 1):
            npt.assert_allclose(
                mif_out["shared_thetas"][i].mean(axis=1),
                mif_out["shared_thetas"][0].mean(axis=1),
                rtol=1e-5,
            )

        for i in range(1, self.M + 1):
            for j in range(len(self.panel.unit_objects)):
                npt.assert_allclose(
                    mif_out["unit_specific_thetas"][i, :, :, j].mean(axis=1),
                    mif_out["unit_specific_thetas"][0, :, :, j].mean(axis=1),
                    rtol=1e-5,
                )


class TestPompClass_LG_AllUnitSpecific(unittest.TestCase):
    def setUp(self):
        self.LG1 = pp.LG()
        self.LG2 = pp.LG()

        # LG model expects 16 parameters: A(4), C(4), Q(4), R(4)
        # All parameters are unit-specific in this test
        shared_params = pd.DataFrame(
            {"shared": []},
            index=pd.Index([]),
        )
        unit_specific_params = pd.DataFrame(
            {
                "LG1": [
                    float(jnp.cos(0.2)),
                    float(-jnp.sin(0.2)),
                    float(jnp.sin(0.2)),
                    float(jnp.cos(0.2)),
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1 / 100,
                    1e-4 / 100,
                    1e-4 / 100,
                    1 / 100,
                    1 / 10,
                    0.1 / 10,
                    0.1 / 10,
                    1 / 10,
                ],
                "LG2": [
                    float(jnp.cos(0.3)),  # Different A matrix
                    float(-jnp.sin(0.3)),
                    float(jnp.sin(0.3)),
                    float(jnp.cos(0.3)),
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1 / 100,
                    1e-4 / 100,
                    1e-4 / 100,
                    1 / 100,
                    1 / 10,
                    0.1 / 10,
                    0.1 / 10,
                    1 / 10,
                ],
            },
            index=pd.Index(
                [
                    "A1",
                    "A2",
                    "A3",
                    "A4",
                    "C1",
                    "C2",
                    "C3",
                    "C4",
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4",
                    "R1",
                    "R2",
                    "R3",
                    "R4",
                ]
            ),
        )

        self.panel = pp.PanelPomp(
            {"LG1": self.LG1, "LG2": self.LG2},
            shared=shared_params,
            unit_specific=unit_specific_params,
        )
        self.J = 5
        self.sigmas = 0.02
        self.sigmas_init = 0.1
        self.M = 2
        self.a = 0.9
        self.key = jax.random.key(111)

    def test_basic_initialization(self):
        self.assertIsInstance(self.panel, pp.PanelPomp)

    def test_simulate(self):
        # Test that simulate runs to completion
        sim = self.panel.simulate(key=self.key)
        self.assertIsInstance(sim, dict)
        self.assertEqual(set(sim.keys()), {"LG1", "LG2"})
        for unit in sim:
            self.assertIn("X_sims", sim[unit])
            self.assertIn("Y_sims", sim[unit])

    def test_pfilter(self):
        # Test that pfilter runs to completion
        self.panel.pfilter(J=self.J, key=self.key)
        pfilter_out = self.panel.results_history[-1]
        self.assertIsInstance(pfilter_out, dict)
        self.assertEqual(
            set(pfilter_out.keys()),
            {"logLik", "shared", "unit_specific", "J", "thresh"},
        )
        self.assertIsInstance(pfilter_out["logLik"], xr.DataArray)
        self.assertEqual(
            set(pfilter_out["logLik"].coords["unit"].values), {"LG1", "LG2"}
        )
        self.assertEqual(set(pfilter_out["logLik"].coords["replicate"].values), {0})

    def test_mif(self):
        # Test that mif runs to completion
        self.panel.mif(
            J=self.J,
            sigmas=self.sigmas,
            sigmas_init=self.sigmas_init,
            M=self.M,
            a=self.a,
            key=self.key,
        )
        mif_out = self.panel.results_history[-1]
        self.assertIsInstance(mif_out, dict)
        self.assertIn("logLiks", mif_out)
        self.assertNotIn("shared_thetas", mif_out)
        self.assertIn("unit_specific_thetas", mif_out)
        self.assertIn("unit_logLiks", mif_out)

        # Check dimensions of output arrays
        self.assertEqual(
            mif_out["unit_specific_thetas"].shape[0], self.M + 1
        )  # iterations
        self.assertEqual(
            mif_out["unit_specific_thetas"].shape[1], len(self.panel.unit_specific)
        )  # parameters
        self.assertEqual(mif_out["unit_specific_thetas"].shape[2], self.J)  # particles
        self.assertEqual(mif_out["unit_specific_thetas"].shape[3], 2)  # units
