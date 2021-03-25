import numpy as np
import sys
sys.path.insert(0, '../tessToPy/')
from absdict import *
from geometry import *

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
        edges[id_] = Edge(id_=id_, parts=edge_verts)
    return edges


def get_faces(lines, edges):
    faces = absdict()
    start_ind = lines.index(' **face\n')
    num_faces = int(lines[start_ind + 1])
    for i in range(num_faces):
        vertex_line_ind = start_ind + 2 + i * 4
        edge_line_ind = vertex_line_ind + 1
        face_edges = [edges[eid_] for eid_ in map(int, lines[edge_line_ind].split()[1:])]
        id_ = int(lines[vertex_line_ind].split()[0])
        faces[id_] = Face(id_=id_, parts=face_edges)
    return faces


def get_polyhedrons(self):
    polyhedrons = {}
    start_ind = self.lines.index(' **polyhedron\n')
    n_polyhedrons = int(self.lines[start_ind + 1])
    for i in range(n_polyhedrons):
        polyhedron_line_ind = start_ind + 2 + i
        id_ = int(self.lines[polyhedron_line_ind].split()[0])
        poly_faces = list(map(int, self.lines[polyhedron_line_ind].split()[2:]))
        polyhedrons[id_] = PolyhedronClass(self.faces, id_=id_, parts=poly_faces)
    return polyhedrons

def get_periodicity(lines, verts, edges, faces):
    periodicity_start_ind = lines.index(' **periodicity\n')
    vertex_start_ind = periodicity_start_ind + lines[periodicity_start_ind:].index('  *vertex\n')
    n_verts = int(lines[vertex_start_ind + 1])
    for line in lines[vertex_start_ind + 2: vertex_start_ind + 2 + n_verts]:
        id_0 = int(line.split()[0])
        id_1 = int(line.split()[1])
        verts[id_1].add_slave(verts[id_0])

    edge_start_ind = periodicity_start_ind + lines[periodicity_start_ind:].index('  *edge\n')
    n_edges = int(lines[edge_start_ind + 1])
    for line in lines[edge_start_ind + 2: edge_start_ind + 2 + n_edges]:
        id_0 = int(line.split()[0])
        id_1 = int(line.split()[1])
        edges[id_1].add_slave(edges[id_0])

    face_start_ind = periodicity_start_ind +lines[periodicity_start_ind:].index('  *face\n')
    n_faces = int(lines[face_start_ind + 1])
    for line in lines[face_start_ind + 2: face_start_ind + 2 + n_faces]:
        id_0 = int(line.split()[0])
        id_1 = int(line.split()[1])
        faces[id_1].add_slave(faces[id_0])

def get_domain_size(lines):
    start_ind = lines.index(' **domain\n')
    domain_start_ind = start_ind + 5
    n_verts = 8
    domain = {}
    for line in lines[domain_start_ind: domain_start_ind + n_verts*2:2]: #line=self.lines[domain_start_ind: domain_start_ind + n_verts*2:2] [0]
        id_ = int(line.split()[0])
        coord = np.array(list(map(float, line.split()[1:-1])))
        domain[id_] = coord
    return domain[7]-domain[1]

if __name__ == "__main__":
    lines = read_tess('../tests/n10-id1.tess')
    verts = get_verts(lines)
    edges = get_edges(lines, verts)
    faces = get_faces(lines, edges)
    get_periodicity(lines, verts, edges, faces)
    domain_size = get_domain_size
