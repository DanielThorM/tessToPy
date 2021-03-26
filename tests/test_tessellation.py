import unittest
import numpy as np
import os
import tessToPy.geometry as tg
import tessToPy.tessIO as tio
import tessToPy.tessellation as ts

class TestTessellation(unittest.TestCase):
    def setUp(self):
        tess_file_name = '../tests/n10-id1.tess'
        self.tess = ts.PeriodicTessellation(tess_file_name)

    def test_copy(self):
        self.tess_copy = self.tess.copy()
        edges = list((self.tess.edges.keys()))
        test_edges = list((self.tess_copy.edges.keys()))
        self.assertEqual(edges, test_edges)
        volume = self.tess.polyhedrons[1].volume()
        test_volume = self.tess_copy.polyhedrons[1].volume()
        self.assertIsNone(np.testing.assert_allclose(volume, test_volume))


if __name__ == '__main__':
    unittest.main()