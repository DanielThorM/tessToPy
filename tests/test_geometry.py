import unittest
import numpy as np
#import sys, os
#testdir = os.path.dirname(__file__)
#srcdir = '../tessToPy/tessToPy'
#sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import tessToPy.geometry as tg

class TestVertex(unittest.TestCase):
    def setUp(self):
        self.coords = np.array([[0, 0, 0], [1.0, 1.1, 1.2]])
        self.periodicity = np.array([-1, -1, -1])
        self.vertices = [tg.Vertex(0, self.coords[0]), tg.Vertex(1, list(self.coords[1]))]
        self.vertices[0].add_slave(self.vertices[1])

    def test_vector_to_master(self):
        offset = self.coords[0]-self.coords[1]
        vector_to_master = self.vertices[1].vector_to_master()
        self.assertIsNone(np.testing.assert_allclose(vector_to_master, offset))

    def test_periodicity_to_master(self):
        periodicity_to_master = self.vertices[1].periodicity_to_master()
        self.assertIsNone(np.testing.assert_equal(periodicity_to_master, self.periodicity))


    def test_no_master(self):
        self.assertRaises(Exception, self.vertices[0].vector_to_master)
        self.assertRaises(Exception, self.vertices[0].periodicity_to_master)


class TestEdge(unittest.TestCase):
    def setUp(self):
        self.coords = np.array([[0, 0, 0], [0.0, 1.1, 1.2], [0.0, 2.1, 2.2], [0.0, 1.0, 1.0]])
        self.vertices = [tg.Vertex(id_, coord) for id_, coord in enumerate(self.coords)]
        self.edges = [tg.Edge(0, self.vertices[:2]), tg.Edge(0, self.vertices[2:4])]

    def test_xn(self):
        edge = self.edges[0]
        x0 = self.coords[0]
        x1 = self.coords[1]
        xm = (x0+x1)/2
        self.assertIsNone(np.testing.assert_allclose(edge.x0(), x0))
        self.assertIsNone(np.testing.assert_allclose(edge.x1(), x1))
        self.assertIsNone(np.testing.assert_allclose(edge.xm(), xm))

    def test_vector(self):
        edge = self.edges[0]
        x0 = self.coords[0]
        x1 = self.coords[1]
        vector = x1-x0
        self.assertIsNone(np.testing.assert_allclose(edge.vector(), vector))

    def test_length(self):
        edge = self.edges[0]
        x0 = self.coords[0]
        x1 = self.coords[1]
        vector = x1-x0
        length = np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        self.assertEqual(edge.length(), length)

    def test_vector_to_master(self):
        edge_a = self.edges[0]
        edge_b = self.edges[1]
        edge_a.add_slave(edge_b)
        vector_to_master = edge_a.xm() - edge_b.xm()
        self.assertIsNone(np.testing.assert_allclose(edge_b.vector_to_master(), vector_to_master))

    def test_periodicity_to_master(self):
        edge_a = self.edges[0]
        edge_b = self.edges[1]
        edge_a.add_slave(edge_b)
        offset = edge_a.xm() - edge_b.xm()
        periodicity_to_master = np.sign(offset)
        self.assertIsNone(np.testing.assert_allclose(edge_b.periodicity_to_master(), periodicity_to_master))

    def test_direction_relative_to_master(self):
        edge_a = self.edges[0]
        edge_b = self.edges[1]
        edge_a.add_slave(edge_b)
        self.assertEqual(edge_b.direction_relative_to_master(), -1)
        self.assertEqual(edge_b.reverse().direction_relative_to_master(), 1)

    def test_edge_reverse(self):
        edge_a = self.edges[0]
        edge_b = self.edges[1]
        edge_a.add_slave(edge_b)
        edge_a_rev = edge_a.reverse()
        self.assertIsNone(np.testing.assert_allclose(edge_a.vector(), -1*edge_a_rev.vector()))

    def test_replace_vertex(self):
        edge_a = self.edges[0]
        vector_org = edge_a.vector()
        edge_a.replace_vertex(self.vertices[0],self.vertices[2])
        vector = self.coords[1]-self.coords[2]
        self.assertIsNone(np.testing.assert_allclose(edge_a.vector(), vector))
        edge_a.replace_vertex(self.vertices[2], self.vertices[0])
        self.assertIsNone(np.testing.assert_allclose(edge_a.vector(), vector_org))

class TestFace(unittest.TestCase):


    def test_xn(self):
        coords= np.array([[0,0,0],[1.0, 1.1, 1.2]])
        a = tg.Vertex(0, coords[0])
        b = tg.Vertex(1, coords[1])
        c = tg.Edge(0, [a, b])
        self.assertIsNone(np.testing.assert_allclose(c.x0(), coords[0]))
        self.assertIsNone(np.testing.assert_allclose(c.x1(), coords[1]))
        self.assertIsNone(np.testing.assert_allclose(c.xm(), (coords[0]+coords[1]/2)))

    def test_vector(self):
        coords = np.array([[0.0, 0, 0], [1.0, 1.1, 1.2]])
        a = tg.Vertex(0, coords[0])
        b = tg.Vertex(1, coords[1])
        c = tg.Edge(0, [a, b])
        self.assertIsNone(np.testing.assert_allclose(c.vector(), coords[1]))

    def test_length(self):
        coords = np.array([[0.0, 0, 0], [1.0, 1.1, 1.2]])
        a = tg.Vertex(0, coords[0])
        b = tg.Vertex(1, coords[1])
        c = tg.Edge(0, [a, b])
        length = np.sqrt(coords[1,0]**2+coords[1,1]**2+coords[1,2]**2)
        self.assertEqual(c.length(), length)

    def test_vector_to_master(self):
        coords = np.array([[0, 0, 0], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [1.0, 1.0, 1.0]])
        a, b, c, d = [tg.Vertex(id_, coord) for id_, coord in enumerate(coords)]
        ea = tg.Edge(0, [a, b])
        eb = tg.Edge(0, [c, d])
        ea.add_slave(eb)
        offset = np.array([-1.0, -1.0, -1.0])
        self.assertIsNone(np.testing.assert_allclose(eb.vector_to_master(), offset))

    def test_periodicity_to_master(self):
        coords = np.array([[0, 0, 0], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [1.0, 1.0, 1.0]])
        a, b, c, d = [tg.Vertex(id_, coord) for id_, coord in enumerate(coords)]
        ea = tg.Edge(0, [a, b])
        eb = tg.Edge(0, [c, d])
        ea.add_slave(eb)
        periodicity = np.array([-1.0, -1.0, -1.0])
        self.assertIsNone(np.testing.assert_allclose(eb.periodicity_to_master(), periodicity))

    def test_direction_relative_to_master(self):
        coords = np.array([[0, 0, 0], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [1.0, 1.0, 1.0]])
        a, b, c, d = [tg.Vertex(id_, coord) for id_, coord in enumerate(coords)]
        ea = tg.Edge(0, [a, b])
        eb = tg.Edge(0, [c, d])
        ea.add_slave(eb)
        self.assertEqual(eb.direction_relative_to_master(), -1)
        coords = np.array([[0, 0, 0], [1.0, 1.1, 1.2], [1.0, 1.0, 1.0], [2.0, 2.1, 2.2]])
        a, b, c, d = [tg.Vertex(id_, coord) for id_, coord in enumerate(coords)]
        ea = tg.Edge(0, [a, b])
        eb = tg.Edge(0, [c, d])
        ea.add_slave(eb)
        self.assertEqual(eb.direction_relative_to_master(), 1)

    def test_edge_reverse(self):
        coords = np.array([[0, 0, 0], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [1.0, 1.0, 1.0]])
        a, b, c, d = [tg.Vertex(id_, coord) for id_, coord in enumerate(coords)]
        ea = tg.Edge(0, [a, b])
        eb = tg.Edge(0, [c, d])
        ea.add_slave(eb)
        eb_rev = eb.reverse()
        ea_rev = ea.reverse()
        self.assertEqual(eb_rev.direction_relative_to_master(), 1)
        self.assertIsNone(np.testing.assert_allclose(ea.vector(), -1*ea_rev.vector()))

    def test_replace_vertex(self):
        coords = np.array([[0, 0, 0], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [1.0, 1.0, 1.0]])
        a, b, c, d = [tg.Vertex(id_, coord) for id_, coord in enumerate(coords)]
        ea = tg.Edge(0, [a, b])
        ea.replace_vertex(a,c)
        self.assertIsNone(np.testing.assert_allclose(ea.vector(), coords[1]-coords[2]))


if __name__ == '__main__':
    unittest.main()
