import unittest
import numpy as np
import tessToPy.geometry as tg

class TestVertex(unittest.TestCase):

    def test_vector_to_master(self):
        coords= np.array([[0,0,0],[1.0, 1.1, 1.2]])
        a = tg.Vertex(0, coords[0])
        b = tg.Vertex(1, list(coords[1]))
        a.add_slave(b)
        offset = coords[0]-coords[1]
        self.assertIsNone(np.testing.assert_allclose(b.vector_to_master(), offset))

    def test_periodicity_to_master(self):
        coords = np.array([[0, 0, 0], [1.0, 1.1, 1.2]])
        a = tg.Vertex(0, list(coords[0]))
        b = tg.Vertex(1, coords[1])
        a.add_slave(b)
        periodicity = np.array([-1, -1, -1])
        self.assertIsNone(np.testing.assert_equal(b.periodicity_to_master(), periodicity))


    def test_no_master(self):
        a = tg.Vertex(0, [0, 0, 0])
        self.assertRaises(Exception, a.vector_to_master)
        self.assertRaises(Exception, a.periodicity_to_master)


class TestEdge(unittest.TestCase):
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
