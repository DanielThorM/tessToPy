import matplotlib.pyplot as plt
import numpy as np
from absdict import *
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from mpl_toolkits.mplot3d.art3d import Line3DCollection
class Vertex(object):
    def __init__(self, id_, coord):
        self.id_ = id_
        self.coord = np.array(coord)
        self.slaves = []
        self.master = None
        self.part_of = []

    def __repr__(self):
        return f"Vertex({self.id_},{self.coord})"


    def add_slave(self, new_slave):
        new_slave.master=self
        self.slaves.append(new_slave)

    def add_slaves(self, new_slaves):
        for new_slave in new_slaves:
            self.add_slave(new_slave)

    def remove_slave(self, old_slave):
        old_slave.master=None
        self.slaves.remove(old_slave)

    def update_slave(self, new_slave, old_slave):
        self.remove_slave(old_slave)
        self.add_slave(new_slave)

    def vector_to_master(self):
        if self.master==None:
            raise Exception('No master vertex')
        return self.master.coord - self.coord

    def periodicity_to_master(self):
        if self.master == None:
            raise Exception('No master vertex')
        return np.sign(self.vector_to_master())

class Edge(object):
    def __init__(self, id_, verts):
        self.id_ = id_
        self.verts = verts
        self.slaves = []
        self.master = None
        self.part_of = []

    def __repr__(self):
        return f"Edge({self.id_})"

    def add_slave(self, new_slave):
        new_slave.master = self
        self.slaves.append(new_slave)

    def add_slaves(self, new_slaves):
        for new_slave in new_slaves:
            self.add_slave(new_slave)

    def remove_slave(self, old_slave):
        old_slave.master = None
        self.slaves.remove(old_slave)

    def update_slave(self, new_slave, old_slave):
        self.remove_slave(old_slave)
        self.add_slave(new_slave)

    def replace_vertex(self, old_vertex, new_vertex):
        if self.verts[0].id_ == old_vertex.id_:
            self.verts[0]= new_vertex
        elif self.verts[1].id_ == old_vertex.id_:
            self.verts[1] = new_vertex
        else:
            raise Exception('Could not find old vertex in edge')

    def vector_to_master(self):
        if self.master == None:
            raise Exception('No master vertex')
        return self.master.xm() - self.xm()

    def periodicity_to_master(self):
        if self.master == None:
            raise Exception('No master vertex')
        periodicity = np.sign(self.vector_to_master())
        return periodicity

    def direction_relative_to_master(self):
        return (self.vector()/self.master.vector())[0]

    def x0(self):
        return self.verts[0].coord

    def x1(self):
        return self.verts[1].coord

    def vector(self):
        return self.x1() - self.x0()

    def xm(self):
        return (self.x0()+self.x1())/2

    def length(self):
        return np.linalg.norm(self.vector())

    def reverse(self):
        temp = Edge(id_=-self.id_, verts=self.verts[::-1])
        temp.slaves= self.slaves
        temp.master= self.master
        temp.part_of = self.part_of
        return temp

class Face(object):
    def __init__(self, id_, edges):
        self.id_ = id_
        self.edges = edges
        self.master = None
        self.slaves = []
        self.part_of = []

    def __repr__(self):
        return f"Face({self.id_})"

    def add_slave(self, new_slave):
        new_slave.master = self
        self.slaves.append(new_slave)

    def add_slaves(self, new_slaves):
        for new_slave in new_slaves:
            self.add_slave(new_slave)

    def verts_in_face(self):
        vert_list = []
        for edge in self.edges:
            vert_list.extend(edge.verts)
        return list(set(vert_list))

    def find_barycenter(self):
        return np.array([vert.coord for vert in self.verts_in_face()]).mean(axis=0)

    def find_face_eq(self):
        barycenter = self.find_barycenter()
        vectors = []
        for edge in self.edges: #edgeID=self.edges[1]
            v1=edge.x0() - barycenter
            v2=edge.x1() - barycenter
            v3 = np.cross(v1, v2)
            nv3 = v3 / np.linalg.norm(v3)
            vectors.append(nv3)
        averaged_vector = np.array(vectors).mean(axis=0)
        face_eq_d = np.dot(averaged_vector, barycenter)
        return np.array([face_eq_d, averaged_vector[0], averaged_vector[1], averaged_vector[2]])

    def find_angle_deviation(self):
        vectors=[]
        barycenter=self.find_barycenter()
        for edge in self.edges:
            v1=edge.x0() - barycenter
            v2=edge.x1() - barycenter
            v3 = np.cross(v1, v2)
            nv3 = v3 / np.linalg.norm(v3)
            vectors.append(nv3)
        mean_vector=np.array(vectors).mean(axis=0)
        angles=[]
        for i in range(len(vectors)):
            j = i+1
            if j ==len(vectors):
                j=0
            angles.append(np.arccos(
                np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0)))

        baryangles = []
        for i in range(len(vectors)):
            baryangles.append(np.arccos(
                np.clip(np.dot(vectors[i], mean_vector), -1.0, 1.0)))
        max_angle=max(angles)
        max_angle_ind = angles.index(max_angle)
        max_bary_ind=baryangles[max_angle_ind:max_angle_ind+2].index(max(baryangles[max_angle_ind:max_angle_ind+2]))

        return [self.edges[max_angle_ind+max_bary_ind], max_angle]

    def remove_edge(self, old_edge):
        self.edges.remove(old_edge)

    def replace_edge(self, new_edge, old_edge):
        replace_ind = [abs(edge.id_) for edge in self.edges].index(abs(old_edge.id_))
        if (self.edges[replace_ind].vector/new_edge.vector())[0] == -1.0:
            self.edges[replace_ind] = new_edge.reverse()
        else:
            self.edges[replace_ind] = new_edge

    def reverse(self):
        temp = Face(id_=-self.id_, edges=[edge.reverse() for edge in self.edges[::-1]])
        temp.master = self.master
        temp.slaves  = self.slaves
        temp.part_of = self.part_of
        return temp

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for edge in self.edges:
            ax.plot(*np.array([edge.x0(), edge.x1()]).swapaxes(0, 1), color='k')
        ax.scatter(*np.array([self.find_barycenter()]).swapaxes(0, 1), color='r')
        ax.quiver(
            *self.find_barycenter(),  # <-- starting point of vector
            *self.find_face_eq()[1:]*edge.length(),  # <-- directions of vector
            color='red', alpha=.8, lw=3)

class FaceOld(object):
    def __init__(self, edge_dict, id_, edges, state=0):
        self.edge_dict = edge_dict
        self.id_ = id_
        self.edges = edges
        self.state = state
        self.master_to = []
        self.slave_to = []
        self.parents= []

    def verts_in_face(self):
        return list(set([self.edge_dict[edge].verts[0] for edge in self.edges]+[self.edge_dict[edge].verts[1] for edge in self.edges]))

    def find_barycenter(self):
        return np.array([self.edge_dict[self.edges[0]].vertex_dict[vert].coord for vert in self.verts_in_face()]).mean(axis=0)

    def find_face_eq(self):
        barycenter = self.find_barycenter()
        vectors = []
        for edge in self.edges: #edgeID=self.edges[1]
            v1=self.edge_dict[edge].x0() - barycenter
            v2=self.edge_dict[edge].x1() - barycenter
            v3 = np.cross(v1, v2)
            nv3 = v3 / np.linalg.norm(v3)
            vectors.append(nv3)
        averaged_vector = np.array(vectors).mean(axis=0)
        face_eq_d = np.dot(averaged_vector, barycenter)
        return [face_eq_d, averaged_vector[0], averaged_vector[1], averaged_vector[2]]

    def find_angle_deviation(self, plot_face=False):
        vectors=[]
        barycenter=self.find_barycenter()
        for edge in self.edges:
            v1=self.edge_dict[edge].x0() - barycenter
            v2=self.edge_dict[edge].x1() - barycenter
            v3 = np.cross(v1, v2)
            nv3 = v3 / np.linalg.norm(v3)
            vectors.append(nv3)

        mean_vector=np.array(vectors).mean(axis=0)
        angles=[]
        for i in range(len(vectors)):
            j = i+1
            if j ==len(vectors):
                j=0
            angles.append(np.arccos(
                np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0)))

        baryangles = []
        for i in range(len(vectors)):
            baryangles.append(np.arccos(
                np.clip(np.dot(vectors[i], mean_vector), -1.0, 1.0)))
        max_angle=max(angles)
        max_angle_ind = angles.index(max_angle)
        max_bary_ind=baryangles[max_angle_ind:max_angle_ind+2].index(max(baryangles[max_angle_ind:max_angle_ind+2]))

        return [self.edges[max_angle_ind+max_bary_ind], max_angle]

    def plot_face(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(self.edges)):
            ax.plot(*np.array([self.edge_dict[self.edges[i]].x0(),self.edge_dict[self.edges[i]].x1()]).swapaxes(0,1))
        ax.scatter(*np.array([self.edge_dict[self.edges[-1]].x0(),self.edge_dict[self.edges[-1]].x1()]).swapaxes(0,1))
        ax.scatter(*np.array([self.find_barycenter()]).swapaxes(0,1))

    def remove_edge(self, old_id):
        target_ind = [abs(edge) for edge in self.edges].index(abs(old_id))
        self.edges.pop(target_ind)

    def replace_edge(self, old_id, new_id):
        replaceInd = [abs(edge) for edge in self.edges].index(abs(old_id))
        sign = np.sign(self.edges[replaceInd])
        self.edges[replaceInd] = int(sign* new_id)

    def reverse(self):
        temp = FaceClass(edge_dict=self.edge_dict, id_=-self.id_, edges=[-1 * edge for edge in self.edges[::-1]], state = self.state)
        temp.master_to = self.master_to
        temp.slave_to  = self.slave_to
        temp.parents = self.parents
        return temp

class PolyhedronOld(object):
    def __init__(self, face_dict, id_, faces):
        self.face_dict = face_dict
        self.id_ = id_
        self.faces = faces

    def removeFace(self, old_id):
        target_ind = [abs(face) for face in self.faces].index(abs(old_id))
        self.faces.pop(target_ind)

    def replace_face(self, old_id, new_id):
        target_ind = [abs(face) for face in self.faces].index(abs(old_id))
        self.faces[target_ind] = new_id

if __name__ == "__main__":
    coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1],  [1,0,1], [1,1,1], [0,1,1]]
    verts = absDict()
    for id_, coord in enumerate(coords):
        verts[id_] = Vertex(id_, coord)
    verts[0].add_slaves(list(verts.values())[1:])
    edge_ids = [[0, 1], [1,2], [2, 3], [3, 0], [0, 4], [1, 5], [2,6], [3,7], [4, 5], [5, 6], [6,7], [7,4]]
    edges = absDict()
    for id_, edge in enumerate(edge_ids):
        edges[id_] = Edge(id_, [verts[i] for i in edge])
    edges[0].add_slaves([edges[2], edges[8], edges[10]])
    edges[1].add_slaves([edges[3],edges[9], edges[11]])
    edges[4].add_slaves([edges[5], edges[6], edges[7]])
    face_ids = [[0, 5, -8, -4],
             [1, 6, -9, -5],
             [2, 7, -10, -6],
             [3, 4, -11, -7],
             [0,1,2,3],
             [8,9,10,11]]
    faces = absDict()
    for id_, face in enumerate(face_ids):
        faces[id_] = Face(id_, [edges[i] for i in face])
    faces[0].add_slave(faces[2])
    faces[3].add_slave(faces[3])
    faces[4].add_slave(faces[5])
    self = faces[0]
