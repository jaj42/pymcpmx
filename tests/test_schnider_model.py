"""
Tests for read_nonmem_dataset_padded and the Schnider 3-compartment model.

Covers:
  - Dataset shape consistency
  - DV/valid_idx alignment
  - n_subj computation (regression: np.unique(dict.values()) returns 1)
  - Model free-parameter count (catches n_subj=1 silently collapsing all etas)
  - Laplace MAP fit quality (sigma_prop, theta_CL)
"""

import unittest
from importlib import resources

import numpy as np
from pmxmc import assets
from pmxmc.io import read_nonmem_dataset_padded


def _load_schnider():
    with resources.open_text(assets, "schnider.csv") as fd:
        return read_nonmem_dataset_padded(fd, sep=",", dv_col="CP")


class TestDatasetDimensions(unittest.TestCase):
    """Fast structural checks on the padded dataset."""

    @classmethod
    def setUpClass(cls):
        cls.ds = _load_schnider()
        cls.n_occ = cls.ds["dt"].eval().shape[0]
        cls.n_subj = len(set(cls.ds["occid_map"].values()))

    def test_n_occasions(self):
        self.assertEqual(self.n_occ, 48)

    def test_n_subjects(self):
        self.assertEqual(self.n_subj, 24)

    def test_n_subjects_not_one(self):
        # Regression: np.unique(dict.values()) treats dict_values as a single
        # element, returning length 1.  set() must be used instead.
        self.assertGreater(self.n_subj, 1)

    def test_array_shapes_consistent(self):
        n = self.n_occ
        self.assertEqual(self.ds["rate"].eval().shape[0], n)
        self.assertEqual(self.ds["bolus"].eval().shape[0], n)
        self.assertEqual(self.ds["meas_idx"].eval().shape[0], n)
        self.assertEqual(len(self.ds["id"]), n)

    def test_dv_and_valid_idx_same_length(self):
        self.assertEqual(len(self.ds["dv"]), len(self.ds["valid_idx"]))

    def test_total_observations(self):
        self.assertEqual(len(self.ds["dv"]), 1006)

    def test_max_meas(self):
        self.assertEqual(self.ds["meas_idx"].eval().shape[1], 21)

    def test_bio_indices_in_range(self):
        ids = self.ds["id"]
        self.assertTrue(np.all(ids >= 0))
        self.assertTrue(np.all(ids < self.n_subj))

    def test_bio_indices_cover_all_subjects(self):
        self.assertEqual(len(set(self.ds["id"].tolist())), self.n_subj)

    def test_dts_nonnegative(self):
        self.assertTrue(np.all(self.ds["dt"].eval() >= 0))

    def test_valid_idx_in_flat_range(self):
        n_occ = self.n_occ
        max_meas = self.ds["meas_idx"].eval().shape[1]
        vidx = self.ds["valid_idx"]
        self.assertTrue(np.all(vidx >= 0))
        self.assertTrue(np.all(vidx < n_occ * max_meas))


class TestModelConstruction(unittest.TestCase):
    """Fast checks that build_model produces a correctly-shaped model."""

    @classmethod
    def setUpClass(cls):
        from pmxmc.examples.schnider_vectorized import build_model

        cls.ds = _load_schnider()
        cls.model = build_model(cls.ds)
        cls.n_subj = len(set(cls.ds["occid_map"].values()))

    def test_free_param_count(self):
        # 6 thetas + 4 sds + 1 sigma_prop + 4 * 24 etas = 107
        n_free = sum(v.size.eval() for v in self.model.free_RVs)
        self.assertEqual(n_free, 107)

    def test_eta_shape(self):
        # Each eta must have shape (n_subj,), not (1,)
        for name in ("eta_V1", "eta_V2", "eta_CL", "eta_Q2"):
            with self.subTest(eta=name):
                self.assertEqual(self.model[name].shape.eval()[0], self.n_subj)

    def test_c_obs_n_observations(self):
        n_obs = len(self.ds["dv"])
        self.assertEqual(self.model["C_obs"].shape.eval()[0], n_obs)


class TestModelFit(unittest.TestCase):
    """Integration test: Laplace MAP fit on the Schnider dataset."""

    @classmethod
    def setUpClass(cls):
        import pytensor

        pytensor.config.floatX = "float64"
        from pmxmc.examples.schnider_vectorized import build_model
        from pmxmc.utils import add_omegas
        from pymc_extras import inference

        cls.ds = _load_schnider()
        cls.model = build_model(cls.ds)
        add_omegas(cls.model)
        with cls.model:
            cls.idata = inference.fit_laplace(model=cls.model, gradient_backend="jax")

    def test_sigma_prop(self):
        sigma = float(self.idata.posterior["sigma_prop"].mean())
        self.assertAlmostEqual(
            sigma, 0.222, delta=0.015, msg="sigma_prop MAP should be ~0.22"
        )

    def test_theta_CL_plausible(self):
        theta_CL = float(self.idata.posterior["theta_CL"].mean())
        self.assertGreater(theta_CL, 1.0)
        self.assertLess(theta_CL, 4.0)

    def test_theta_V1_plausible(self):
        theta_V1 = float(self.idata.posterior["theta_V1"].mean())
        self.assertGreater(theta_V1, 2.0)
        self.assertLess(theta_V1, 8.0)


if __name__ == "__main__":
    unittest.main()
