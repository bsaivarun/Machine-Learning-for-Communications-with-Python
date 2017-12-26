
import sys
sys.path.append('../')
import unittest
import numpy as np
from mlcomm.tools import it

class MLcommITTESTS(unittest.TestCase):
    def test_discrete_entr(self):
        self.assertAlmostEqual(it.discrete_entr([0.3, 0.7]), 0.8813, places=3)
        self.assertAlmostEqual(it.discrete_entr((0.25, 0.25, 0.25, 0.25)), 2, places=3)
        self.assertAlmostEqual(it.discrete_entr([1, 0]), 0)
        self.assertRaises(ValueError, it.discrete_entr, (0.3, 0.8))
        self.assertRaises(ValueError, it.discrete_entr, 'str')

    def test_discrete_cross_entr(self):
        self.assertAlmostEqual(it.discrete_cross_entr([0.3, 0.7], [0.3, 0.7]), it.discrete_entr([0.3, 0.7]), places=3)
        self.assertAlmostEqual(it.discrete_cross_entr((0.3, 0.7), [0.3, 0.7]), it.discrete_entr([0.3, 0.7]), places=3)
        self.assertEqual(it.discrete_cross_entr((1, 0), (0, 1)), np.inf)
        self.assertEqual(it.discrete_cross_entr((0, 1), (0, 1)), 0)
        self.assertRaises(ValueError, it.discrete_cross_entr, (0.3, 0.8), (0.6, 0.4))
        self.assertRaises(ValueError, it.discrete_cross_entr, (0.6, 0.4), (0.3, 0.8))
        self.assertRaises(ValueError, it.discrete_cross_entr, 'str1', (0.3, 0.7))
        self.assertRaises(ValueError, it.discrete_cross_entr, (0.3, 0.7), 'str1')
        self.assertRaises(ValueError, it.discrete_cross_entr, (0.3, 0.6, 0.1), (0.6, 0.4))
    
    def test_discrete_kl_dis(self):
        self.assertAlmostEqual(it.discrete_kl_div([0.3, 0.7], [0.3, 0.7]), 0, places=3)
        self.assertAlmostEqual(it.discrete_kl_div([0.3, 0.7], [0.4, 0.6]), 0.0312, places=4)
        self.assertAlmostEqual(it.discrete_kl_div([0.4, 0.6], [0.3, 0.7]), 0.0326, places=4)

if __name__ == '__main__':
    unittest.main()
