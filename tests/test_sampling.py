"""
Unit tests for sampling code
"""
import os

import unittest

import numpy as np
import torch

from foldingdiff import sampling

class TestSamplingReproducible(unittest.TestCase):
    """
    Test that sampling is reproducible
    """
    def setUp(self) -> None:
        self.mini_model = os.path.join(
            os.path.dirname(__file__), "mini_model_for_testing", "results"
        )
        assert os.path.isdir(self.mini_model)
        self.full_model = "wukevin/foldingdiff"

    def test_repro_simple(self):
        torch.manual_seed(1234)
        samp_1 = sampling.sample_simple(self.full_model, n=1, sweep_lengths=[50, 51]).pop()
        torch.manual_seed(1234)
        samp_2 = sampling.sample_simple(self.full_model, n=1, sweep_lengths=[50, 51]).pop()
        self.assertTrue(np.allclose(samp_1.values, samp_2.values))
    
