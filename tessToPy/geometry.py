import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
sys.path.insert(0, '../tessToPy/')
from absdict import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class PeriodicComponent(object):
    def __init__(self, id_):
        self.id_ = id_
        self.slaves = []
        self.master = None
        self.per_to_m = None
        self.vect_to_m = None
        self.part_of = []

    def calc_periodicity(self):
        self.per_to_m = self.periodicity_to_master()
        self.vect_to_m = self.vector_to_master()

    def add_slave(self, slave):
        if type(slave) == type([]):
            for slave_ in slave:
                self.add_slave(slave_)
        else:
            slave.master=self
            slave.calc_periodicity()
            self.slaves.append(slave)

    def remove_slave(self, slave):
        slave.master=None
        self.slaves.remove(slave)

    def update_slave(self, new_slave, old_slave):
        self.remove_slave(old_slave)
        self.add_slave(new_slave)

    def vector_to_master(self):
        if self.master == None:
            raise Exception('No master vertex')
        return self.master.xm() - self.xm()

    def periodicity_to_master(self):
        if self.master == None:
            raise Exception('No master vertex')
        periodicity = np.sign(self.vector_to_master())
        return periodicity

class PeriodicCompositComponent(PeriodicComponent):
    def __init__(self, id_, parts):
        super().__init__(id_)
        self.parts = []
        self.add_part(parts)

    def add_part(self, part):
        if type(part) == type([]): #check isinstance of classes
            for part_ in part:
                self.add_part(part_)
        else:
            part.part_of.append(self)
            self.parts.append(part)

    def replace_part(self, new_part, old_part):
        # does not work for reversed parts
        if np.sign(self.id_) == -1:
            raise Exception('Can not alter reversed part')
        replace_ind = [abs(part.id_) for part in self.parts].index(abs(old_part.id_))
        direction_relative_to_old = new_part.direction_relative_to_other(self.parts[replace_ind])
        if direction_relative_to_old == -1.0:
            new_part.part_of.append(self)
            old_part.part_of.remove(self)
            self.parts[replace_ind] = new_part.reverse()

        elif direction_relative_to_old == 1.0:
            new_part.part_of.append(self)
            old_part.part_of.remove(self)
            self.parts[replace_ind] = new_part
        else:
            raise Exception('Failed to replace part')

    def remove_part(self, old_part):
        del_ind = [abs(part.id_) for part in self.parts].index(abs(old_part.id_))
        old_part.part_of.remove(self)
        del self.parts[del_ind]

    def reverse(self):
        if np.sign(self.id_) == -1:
            return self.org_comp
        else:
            return reversedPCC(self)

    def direction_relative_to_master(self):
        if self.master == None:
            return 1
        else:
            return self.direction_relative_to_other(self.master)

class reversedPCC(PeriodicCompositComponent):
    '''Every time an attribute is called, the supplied instance is copied and reversed,
        returning only the reversed result of the attribute'''

    def __init__(self, org_component):
        self.org_comp = org_component

    def __repr__(self):
        temp = copy.copy(self.org_comp)
        temp.id_ = -temp.id_
        '''Returns representation of the object'''
        return temp.__repr__()

    def __getattr__(self, attr):
        temp = copy.copy(self.org_comp)
        temp.id_ = -temp.id_
        if hasattr(temp.parts[0], 'reverse'):
            temp.parts = [part.reverse() for part in temp.parts[::-1]]
        else:
            temp.parts = temp.parts[::-1]
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        return getattr(temp, attr)

class Vertex(PeriodicComponent):
    def __init__(self, id_, coord):
        self.coord = np.array(coord)
        super().__init__(id_)

    def __repr__(self):
        return f"Vertex({self.id_},{self.coord})"

    def direction_relative_to_other(self, vert):
        '''Does not have a direction relative to other'''
        return 1

    def plot(self, ax = None, color= 'k' , marker = 'o', **kwargs):
        if  ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*self.coord, color=color, marker=marker, **kwargs)

    def xm(self):
        '''Barycenter'''
        return self.coord

class Edge(PeriodicCompositComponent):
    def __init__(self, id_, parts):
        super().__init__(id_, parts)

    def __repr__(self):
        return f"Edge({self.id_})"

    def direction_relative_to_other(self, edge):
        vect_sum = self.vector() + edge.vector()
        if np.all(np.isclose(vect_sum, np.array([0,0,0]))):
            return -1
        elif np.all(np.isclose(vect_sum, 2*self.vector())):
            return 1
        else:
            return None

    def x0(self):
        return self.parts[0].coord

    def x1(self):
        return self.parts[1].coord

    def vector(self):
        return self.x1() - self.x0()

    def xm(self):
        '''Barycenter'''
        return (self.x0()+self.x1())/2

    def length(self):
        return np.linalg.norm(self.vector())

    def plot(self, ax=None, color='k'):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        coords = np.array([[self.x0(), self.x1()]])
        edge = Line3DCollection(coords)
        edge.set_edgecolor(color)
        ax.add_collection3d(edge)
        #ax.plot(*np.array([self.x0(), self.x1()]).swapaxes(0, 1), color=color, **kwargs)
        #coord = np.array([self.x0(), self.x1(), self.x0()])
        #edge = Line3DCollection(coord)
        #ax.add_collection3d(edge)

class Face(PeriodicCompositComponent):
    def __init__(self, id_, parts):
        super().__init__(id_, parts)

    def __repr__(self):
        return f"Face({self.id_})"

    def verts_in_face(self):
        vert_list = []
        for edge in self.parts:
            vert_list.extend(edge.parts)
        return list(set(vert_list))

    def xm(self):
        return np.array([vert.coord for vert in self.verts_in_face()]).mean(axis=0)

    def face_eq(self):
        barycenter = self.xm()
        vectors = []
        for edge in self.parts:
            v1=edge.x0() - barycenter
            v2=edge.x1() - barycenter
            v3 = np.cross(v1, v2)
            nv3 = v3 / np.linalg.norm(v3)
            vectors.append(nv3)
        averaged_vector = np.array(vectors).mean(axis=0)
        face_eq_d = np.dot(averaged_vector, barycenter)
        return np.array([face_eq_d, averaged_vector[0], averaged_vector[1], averaged_vector[2]])

    def angle_deviation(self):
        vectors=[]
        barycenter=self.xm()
        for edge in self.parts:
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

        return [self.parts[max_angle_ind+max_bary_ind], max_angle]

    def area(self):
        barycenter = self.xm()
        area = 0
        for edge in self.parts:
            coords = np.array([barycenter,
                               edge.x0(),
                               edge.x1()])
            area += area_tri(coords)
        return area

    def direction_relative_to_other(self, face):
        if np.all(np.isclose(face.face_eq()[1:], -1*self.face_eq()[1:])):
            return -1
        elif np.all(np.isclose(face.face_eq()[1:], self.face_eq()[1:])):
            return 1
        else:
            raise Exception('Faces not equal')


    def plot(self, ax = None, color = 'k', alpha=0.4, normal_vector = False, **kwargs):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        for edge in self.parts:
            coord = np.array([edge.x0(), edge.x1(), self.xm()])
            triangle = Poly3DCollection(coord, alpha=alpha, facecolors=color, **kwargs, zorder = 1)
            ax.add_collection3d(triangle)
        if normal_vector:
            #ax.scatter(*np.array([self.barycenter()]).swapaxes(0, 1), color='r')
            ax.quiver(
                *self.xm(),  # <-- starting point of vector
                *self.find_face_eq()[1:],  # <-- directions of vector
                color='red', alpha=.8, lw=3)

class Polyhedron(PeriodicCompositComponent):
    def __init__(self, id_, parts):
        super().__init__(id_, parts)

    def edges(self):
        edges = []
        for face in self.parts:
            edges.extend(face.parts)
        return list(set(edges))

    def remove_face(self, old_face):
        del_ind = [abs(face.id_) for face in self.parts].index(abs(old_face.id_))
        del self.parts[del_ind]

    def replace_face(self, new_face, old_face):
        replace_ind = [abs(face.id_) for face in self.parts].index(abs(old_face.id_))
        if new_face.direction_relative_to_other(self.parts[replace_ind]) == -1:
            self.parts[replace_ind] = new_face.reverse()
        elif new_face.direction_relative_to_other(self.parts[replace_ind]) == 1:
            self.parts[replace_ind] = new_face

    def barycenter(self):
        verts = []
        for face in self.parts:
            verts.extend(face.verts_in_face())
        unique_verts = set(verts)
        coords = np.array([vert.coord for vert in unique_verts])
        return np.mean(coords, axis=0)

    def volume(self):
        poly_volume = 0
        poly_barycenter = self.barycenter()
        for face in self.parts:
            face_barycenter = face.xm()
            for edge in face.parts:
                coords = np.array([poly_barycenter,
                                   face_barycenter,
                                   edge.x0(),
                                   edge.x1()])
                poly_volume += volume_tet(coords)
        return poly_volume

    def area(self):
        poly_area = 0
        for face in self.parts:
            poly_area += face.area()
        return poly_area

    def sphericity(self):
        return equivalent_surface_area(self.volume())/self.area()

    def plot(self, ax = None, facecolor = 'k', edgecolor = 'k', facealpha=0.8):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        for face in self.parts:
            face.plot(ax, color=facecolor, alpha = facealpha)
        for edge in self.edges():
            edge.plot(ax, color = edgecolor)

def volume_tet(coords):
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    v3 = coords[3] - coords[0]
    return np.abs(np.dot(np.cross(v1, v2), v3)) / 6.0

def area_tri(coords):
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    return np.abs(np.linalg.norm(np.cross(v1, v2))) / 2.0

def equivalent_surface_area(volume):
    return surface_area_sphere(equivalent_radius(volume))

def equivalent_radius(volume):
    return np.cbrt((3/4)*volume/np.pi)

def surface_area_sphere(radius):
    return 4*np.pi*radius**2

if __name__ == "__main__":
    coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1],  [1,0,1], [1,1,1], [0,1,1]]
    verts = absdict()
    for id_, coord in enumerate(coords):
        verts[id_] = Vertex(id_, coord)
    verts[0].add_slave(list(verts.values())[1:])
    edge_ids = [[0, 1], [1,2], [2, 3], [3, 0], [0, 4], [1, 5], [2,6], [3,7], [4, 5], [5, 6], [6,7], [7,4]]
    edges = absdict()
    for id_, edge in enumerate(edge_ids):
        edges[id_] = Edge(id_, [verts[i] for i in edge])
    edges[0].add_slave([edges[2], edges[8], edges[10]])
    edges[1].add_slave([edges[3],edges[9], edges[11]])
    edges[4].add_slave([edges[5], edges[6], edges[7]])
    face_ids = [[0, 5, -8, -4],
             [1, 6, -9, -5],
             [2, 7, -10, -6],
             [3, 4, -11, -7],
             [0,1,2,3],
             [8,9,10,11]]
    faces = absdict()
    for id_, face in enumerate(face_ids):
        faces[id_] = Face(id_, [edges[i] for i in face])
    faces[0].add_slave(faces[2])
    faces[3].add_slave(faces[3])
    faces[4].add_slave(faces[5])
    face = faces[1]
    self = Polyhedron(1, list(faces.values()))

