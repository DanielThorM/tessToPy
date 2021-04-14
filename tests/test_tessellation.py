import unittest
import numpy as np
import os
import tessToPy.geometry as tg
import tessToPy.tessIO as tio
import tessToPy.tessellation as ts

class TestSimpleTessellation(unittest.TestCase):
    def setUp(self):
        tess_file_name = 'tess_files/n10-id1.tess'
        self.tess = ts.PeriodicTessellation(tess_file_name)

    def test_copy(self):
        self.tess_copy = self.tess.copy()
        edges = list((self.tess.edges.keys()))
        test_edges = list((self.tess_copy.edges.keys()))
        self.assertEqual(edges, test_edges)
        volume = self.tess.polyhedrons[1].volume()
        test_volume = self.tess_copy.polyhedrons[1].volume()
        self.assertIsNone(np.testing.assert_allclose(volume, test_volume))

    def test_edge_deletion(self):
        success = self.tess_copy.delete_edge(self.edges[232])
        self.assertTrue(success)

class TestComplexTessellation(unittest.TestCase):
    def setUp(self):
        tess_file_name = 'tess_files/n400_from_morpho-id1.tess'
        self.tess = ts.PeriodicTessellation(tess_file_name)

    def test_merged_vertices(self):
        success = self.tess.delete_edge(self.tess.edges[5624])
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()