import numpy as np
from absdict import *
def read_tess(file_name):
    with open(file_name, 'r') as tess_raw:
        lines = tess_raw.readlines()
    return lines

def get_verts(lines):
    verts = {}
    start_ind = lines.index(' **vertex\n')
    for line in lines[start_ind + 2:start_ind + 2 + int(lines[start_ind + 1])]:
        id_ = int(line.split()[0])
        coord = np.array(list(map(float, line.split()[1:-1])))
        verts[id_] = Vertex(id_=id_, coord=coord)
    return verts


def get_edges(lines, verts):
    edges = absdict()
    start_ind = lines.index(' **edge\n')
    for line in lines[start_ind + 2:start_ind + 2 + int(lines[start_ind + 1])]:
        id_ = int(line.split()[0])
        edge_verts = [verts[vid_] for vid_ in map(int, line.split()[1:3])]  # Edge vertex 0 and 1
        edges[id_] = Edge(id_=id_, verts=edge_verts)
    return edges


def get_faces(self):
    faces = AbsDict()
    start_ind = self.lines.index(' **face\n')
    num_faces = int(self.lines[start_ind + 1])
    for i in range(num_faces):
        vertex_line_ind = start_ind + 2 + i * 4
        edge_line_ind = vertex_line_ind + 1
        face_edges = list(map(int, self.lines[edge_line_ind].split()[1:]))
        id_ = int(self.lines[vertex_line_ind].split()[0])
        faces[id_] = FaceClass(self.edges, id_=id_, edges=face_edges)
    return faces


def get_polyhedrons(self):
    polyhedrons = {}
    start_ind = self.lines.index(' **polyhedron\n')
    n_polyhedrons = int(self.lines[start_ind + 1])
    for i in range(n_polyhedrons):
        polyhedron_line_ind = start_ind + 2 + i
        id_ = int(self.lines[polyhedron_line_ind].split()[0])
        poly_faces = list(map(int, self.lines[polyhedron_line_ind].split()[2:]))
        polyhedrons[id_] = PolyhedronClass(self.faces, id_=id_, faces=poly_faces)
    return polyhedrons