import numpy as np
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


class EdgeOld(object):
    def __init__(self, vertex_dict, id_, verts, state=0):
        self.vertex_dict=vertex_dict
        self.id_ = id_
        self.verts = verts
        self.state = state
        self.master_to = []
        self.slave_to = []
        self.parents = []

    def vector(self):
        return self.vertex_dict[self.verts[1]].coord - self.vertex_dict[self.verts[0]].coord

    def x0(self):
        return self.vertex_dict[self.verts[0]].coord

    def x1(self):
        return self.vertex_dict[self.verts[1]].coord

    def length(self):
        return np.linalg.norm(self.vector())

    def reverse(self):
        temp = EdgeClass(self.vertex_dict, id_=-self.id_, verts=self.verts[::-1], state=self.state)
        temp.master_to = self.master_to
        temp.slave_to = self.slave_to
        temp.parents = self.parents
        return temp

    def reyplace_vertex(self, old_id, new_id):
        if self.verts[0] == old_id:
            self.verts[0] = new_id
        elif self.verts[1] == old_id:
            self.verts[1] = new_id
        else:
            raise Exception('Could not find old vertex in edge')
class Face(object):
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
class Polyhedron(object):
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

