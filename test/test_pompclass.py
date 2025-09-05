import jax
import unittest
import pickle
import pypomp as pp


class TestPompClass_LG(unittest.TestCase):
    def setUp(self):
        self.LG = pp.LG()
        self.J = 3
        self.ys = self.LG.ys
        self.theta = self.LG.theta
        self.covars = self.LG.covars
        self.sigmas = 0.02
        self.a = 0.5
        self.M = 2
        self.key = jax.random.key(111)

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.dmeas = self.LG.dmeas

    def test_basic_initialization(self):
        self.assertEqual(self.LG.covars, self.covars)

    def test_invalid_initialization(self):
        for arg in ["ys", "theta", "rinit", "rproc", "dmeas"]:
            with self.assertRaises(Exception):
                kwargs = {
                    "ys": self.ys,
                    "theta": self.theta,
                    "rinit": self.rinit,
                    "rproc": self.rproc,
                    "dmeas": self.dmeas,
                }
                kwargs[arg] = None
                pp.Pomp(**kwargs)

    def test_results(self):
        # Check that results() returns one row per parameter set and correct columns
        # pfilter: should be one row per parameter set (len(theta))
        n_paramsets = len(self.LG.theta)
        self.LG.pfilter(J=self.J, reps=1, key=self.key)
        res_pfilter = self.LG.results()
        self.assertEqual(res_pfilter.shape[0], n_paramsets)  # one row per parameter set
        expected_cols = {"logLik", "se", *self.LG.theta[0].keys()}
        self.assertEqual(set(res_pfilter.columns), expected_cols)

        # mif: should be one row per parameter set (len(theta))
        self.LG.mif(
            J=self.J,
            sigmas=self.sigmas,
            sigmas_init=self.sigmas,
            M=self.M,
            a=self.a,
            key=self.key,
        )
        res_mif = self.LG.results()
        n_paramsets = len(self.LG.theta)
        self.assertEqual(res_mif.shape[0], n_paramsets)  # one row per parameter set
        self.assertEqual(set(res_mif.columns), expected_cols)

        # train: should be one row per parameter set (len(theta))
        self.LG.train(J=self.J, M=1, key=self.key)
        res_train = self.LG.results()
        n_paramsets = len(self.LG.theta)
        self.assertEqual(res_train.shape[0], n_paramsets)  # one row per parameter set
        self.assertEqual(set(res_train.columns), expected_cols)

    def test_sample_params(self):
        param_bounds = {
            "R0": (0, 100),
            "sigma": (0, 100),
            "gamma": (0, 100),
        }
        n = 10
        key = jax.random.key(1)
        param_sets = pp.Pomp.sample_params(param_bounds, n, key)
        self.assertEqual(len(param_sets), n)
        for params in param_sets:
            param_names = list(params.keys())
            self.assertEqual(param_names, list(param_bounds.keys()))
            for param_name, value in params.items():
                self.assertIsInstance(value, float)

    def test_theta_carryover_mif(self):
        # Check that theta estimate from mif is correctly carried over to attribute and traces
        theta_order = list(self.LG.theta[0].keys())
        self.LG.mif(
            J=self.J,
            sigmas=self.sigmas,
            sigmas_init=self.sigmas,
            M=self.M,
            a=self.a,
            key=self.key,
        )
        self.assertEqual(theta_order, list(self.LG.theta[0].keys()))
        self.LG.pfilter(J=self.J, reps=2)
        self.assertEqual(
            list(self.LG.results_history[-1]["theta"][0].keys()), theta_order
        )
        traces_da = self.LG.results_history[-2]["traces"]
        param_names = traces_da.coords["variable"].values[1:]
        last_row = traces_da.sel(
            replicate=0, iteration=traces_da.sizes["iteration"] - 1
        )
        last_param_values = [
            float(last_row.sel(variable=param).values) for param in param_names
        ]
        self.assertEqual(
            list(self.LG.results_history[-1]["theta"][0].values()),
            last_param_values,
        )
        traces = self.LG.traces()
        # Only compare the parameter values
        self.assertEqual(
            traces.iloc[-1, 4:].values.tolist(), traces.iloc[-2, 4:].values.tolist()
        )

    # TODO: merge mif and train tests
    def test_theta_carryover_train(self):
        # Check that theta estimate from train is correctly carried over to attribute and traces
        theta_order = list(self.LG.theta[0].keys())
        self.LG.train(
            J=self.J,
            M=1,
            key=self.key,
        )
        self.assertEqual(theta_order, list(self.LG.theta[0].keys()))
        self.LG.pfilter(J=self.J, reps=2)
        self.assertEqual(
            list(self.LG.results_history[-1]["theta"][0].keys()), theta_order
        )
        traces_da = self.LG.results_history[-2]["traces"]
        param_names = traces_da.coords["variable"].values[1:]
        last_row = traces_da.sel(
            replicate=0, iteration=traces_da.sizes["iteration"] - 1
        )
        last_param_values = [
            float(last_row.sel(variable=param).values) for param in param_names
        ]
        self.assertEqual(
            list(self.LG.results_history[-1]["theta"][0].values()),
            last_param_values,
        )
        traces = self.LG.traces()
        # Only compare the parameter values
        self.assertEqual(
            traces.iloc[-1, 4:].values.tolist(), traces.iloc[-2, 4:].values.tolist()
        )

    def test_pickle(self):
        # Generate results to pickle
        self.LG.pfilter(J=self.J, reps=1, key=self.key)
        # Pickle the object
        pickled_data = pickle.dumps(self.LG)

        # Unpickle the object
        unpickled_obj = pickle.loads(pickled_data)

        # Check that the unpickled object has the same attributes
        self.assertEqual(self.LG.ys.values.tolist(), unpickled_obj.ys.values.tolist())
        self.assertEqual(self.LG.theta, unpickled_obj.theta)
        self.assertEqual(self.LG.covars, unpickled_obj.covars)
        self.assertEqual(self.LG.rinit, unpickled_obj.rinit)
        self.assertEqual(self.LG.rproc, unpickled_obj.rproc)
        self.assertEqual(self.LG.dmeas, unpickled_obj.dmeas)
        self.assertEqual(self.LG.rproc.dt, unpickled_obj.rproc.dt)
        self.assertEqual(self.LG.results_history, unpickled_obj.results_history)
        self.assertEqual(
            self.LG.traces().values.tolist(), unpickled_obj.traces().values.tolist()
        )

        # Check that the unpickled object can be pickled again if rmeas is None
        unpickled_obj.rmeas = None
        pickled_data = pickle.dumps(unpickled_obj)

        # Check that the unpickled object can still be used for filtering
        unpickled_obj.pfilter(J=self.J, reps=1)

    def test_prune(self):
        # Run pfilter with multiple replicates to generate results
        self.LG.pfilter(J=self.J, reps=5, key=self.key)
        # Save the original theta list length
        orig_theta = self.LG.theta.copy()
        orig_len = len(orig_theta)
        # Prune to top 2 thetas, refill to original length
        self.LG.prune(n=2, refill=True)
        self.assertIsInstance(self.LG.theta, list)
        self.assertEqual(len(self.LG.theta), orig_len)
        # The unique thetas should be at most 2
        unique_thetas = [tuple(sorted(d.items())) for d in self.LG.theta]
        self.assertLessEqual(len(set(unique_thetas)), 2)
        # Prune to top 1 theta, do not refill
        self.LG.prune(n=1, refill=False)
        self.assertIsInstance(self.LG.theta, list)
        self.assertEqual(len(self.LG.theta), 1)
        # The theta should be a dict
        self.assertIsInstance(self.LG.theta[0], dict)
        # Prune with n greater than available thetas (should not error, just return all)
        self.LG.theta = orig_theta.copy()
        self.LG.prune(n=10, refill=False)
        self.assertEqual(len(self.LG.theta), min(10, len(orig_theta)))
        # Test error if results are empty
        LG2 = self.LG.__class__(
            ys=self.LG.ys.copy(),
            theta=self.LG.theta[0].copy(),
            rinit=self.LG.rinit,
            rproc=self.LG.rproc,
            dmeas=self.LG.dmeas,
        )
        with self.assertRaises(IndexError):
            LG2.prune(n=1)
