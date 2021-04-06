import numpy as np
import os
from tessToPy.absdict import *
import tessToPy.tessIO as tio
import tessToPy.geometry as tg
import matplotlib.pyplot as plt
import copy
import time
import scipy.optimize

class Tessellation(object):
    def __init__(self, tess_file_name):
        self.tess_file_name = tess_file_name
        self.read_tess()
        self.periodic = False
        # For storing  rejected edges, such that they are not tried again.
        self.vertex_id_counter = max(self.vertices.keys())
        self.edge_id_counter = max(self.edges.keys())
        self.rejected_edge_del = []
        self.deleted_edges = []
        self.edge_lengths = self.get_edge_lengths()

    def read_tess(self):
        with open(self.tess_file_name, 'r') as tess_raw:
            self.lines=tess_raw.readlines()
        self.vertices = tio.get_verts(self.lines)
        self.edges = tio.get_edges(self.lines, self.vertices)
        self.faces = tio.get_faces(self.lines, self.edges)
        self.polyhedrons = tio.get_polyhedrons(self.lines, self.faces)
        self.domain_size = tio.get_domain_size(self.lines)

    def plot(self, alpha = 0.8, facecolor = 'gray', deleted_edges = []):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for poly in self.polyhedrons.values():
            poly.plot(ax, facealpha = alpha, facecolor = facecolor)
        for edge in deleted_edges:
            edge.plot(ax, color = 'r')

    def get_edge_lengths(self):
        lengths = np.array([[edge.length(), edge] for edge in self.edges.values()
                            if edge.id_ not in self.rejected_edge_del])
        if len(lengths) == 0:
            print('No more edges to find')
            return []
        else:
            #Sort edge lenghts
            lengths = lengths[lengths[:, 0].argsort()]
            return lengths

    def check_for_duplicate_vertices(self):
        main_vert_set = set(self.vertices.values())
        derived_verts = []
        for poly in self.polyhedrons.values():
            for face in poly.parts:
                for edge in face.parts:
                    derived_verts.extend(edge.parts)
        derived_vert_set = set(derived_verts)
        if main_vert_set != derived_vert_set:
            raise Exception('Duplicate vertices found')

class PeriodicTessellation(Tessellation):
    def __init__(self, tess_file_name):
        super().__init__(tess_file_name)
        tio.get_periodicity(self.lines, self.vertices, self.edges, self.faces)
        self.periodic = True

    def copy(self):
        tio.write_tess(self, 'temp')
        tess_copy = PeriodicTessellation('temp')
        os.remove('temp')
        tess_copy.tess_file_name = self.tess_file_name
        tess_copy.rejected_edge_del = self.rejected_edge_del
        tess_copy.deleted_edges = self.deleted_edges
        tess_copy.vertex_id_counter = self.vertex_id_counter
        tess_copy.edge_id_counter = self.edge_id_counter
        return tess_copy

    def regularize(self, n=1):
        '''Try to remove the n shortest edges'''
        self.edge_lengths = self.get_edge_lengths()
        for i in range(n):
            if len(self.edge_lengths) <= 1:
                print('No more edges to check!')
                break
            edge = self.edge_lengths[i, 1] #Edge232
            if edge in self.edges.values():
                print(f'Trying to delete edge {edge.id_}, i = {i}')
                _ = self.try_delete_edge(edge)

    def delete_edge(self, edge, del_layer=0):
        print_trigger = False
        if del_layer == 0: print_trigger = True
        self.new_verts = []
        self.del_verts = []
        ##################################################################################
        # Find edge dependencies and new vertex locations
        ##################################################################################
        t = time.time()
        # The dependent edges and vertices are found as offsets from the master edge.
        edges, edge_periodicities, vertices, vertex_periodicities = self.find_periodic_dependecies(edge)

        # The new vertex location for a edge collapse, and new vertex locations for moved slave vertices are found
        new_edge_vertex_locs, updated_vertex_locs = self.calc_new_vertices(edges, edge_periodicities, vertices,
                                                                        vertex_periodicities)
        elapsed = time.time() - t
        if del_layer == 0:
            print('Time to find dependencies and new vertex locations: {:.3f} s'.format(elapsed))


        ##################################################################################
        # Each edge is merged to the new vertex location. The dependent vertices are moved.
        ##################################################################################
        t = time.time()
        for affected_edge, new_edge_vertex in zip(edges, new_edge_vertex_locs):
            # affected_edge, new_edge_vertex = list(zip(edges, new_edge_vertex_locs))[0]
            new_vert_ = self.replace_edge_with_vertex(affected_edge, new_edge_vertex, print_trigger)
            self.new_verts.append(new_vert_)

        for affected_vertex, new_vertex_loc in zip(vertices, updated_vertex_locs):
            # affected_vertex, newVertexLoc = list(zip(vertices, new_vertex_vertices))[0]
            new_vert_ = self.update_vertex_loc(affected_vertex, new_vertex_loc)
            self.new_verts.append(new_vert_)

        elapsed = time.time() - t
        if del_layer == 0:
            print('Time to delete dependencies: {:.3f} s'.format(elapsed))

        ##################################################################################
        # Periodicity of the new vertices are assigned
        ##################################################################################
        self.resolve_vertex_periodicity(self.new_verts)

        # Update the vertex and edge periodicity of the affected edges
        self.update_periodicity_internal(self.new_verts)
        ####################################################################
        # Find all affected edges, by newVertexList, and check internal angles
        ########################################################################

        sorted_angles = self.face_deviation(self.new_verts)

        if sorted_angles[0, 1] < 20 * np.pi / 180.:
            return True
        elif del_layer == 0:
            checked_edges = []
            for edge, angle in zip(sorted_angles[:,0], sorted_angles[:,1]):

                # edge, angle =  sorted_angles[3,0], sorted_angles[3,1]
                if angle > 20 * np.pi / 180. and abs(edge.id_) not in checked_edges:
                    try:
                        dep_edges = self.find_periodic_dependecies(edge)[0]
                        checked_edges.extend(abs(dep_edge.id_) for dep_edge in dep_edges)
                        if self.try_delete_edge(edge, del_layer=del_layer+1) == True:
                            print('{} st/nd layer deletion of edge {} was successful'.format(
                                del_layer + 1, int(edge.id_)))
                            print('--------------------------------------------------------------')
                            return True
                        else:
                            pass
                    except:
                        print('Error encountered in {} st/nd layer deletion of edge {}'.format(
                            del_layer + 1, int(edge.id_)))
                else:
                    pass
            return False
        else:
            return False

    def face_deviation(self, new_verts):
        affected_edges = self.affected_parents(new_verts)
        affected_faces = self.affected_parents(affected_edges)
        angles = np.array([face.angle_deviation() for face in affected_faces])
        for i in range(len(angles)):
            if np.sign(angles[i,0].id_) == -1:
                angles[i,0] = angles[i,0].reverse()
        return angles[angles[:, 1].argsort()[::-1], :]

    def resolve_vertex_periodicity(self, verts):
        all_verts = []
        for vert in verts:
            if vert.master != None:
                all_verts.append(vert.master)
                vert.master = None
            if vert.slaves != []:
                all_verts.extend(vert.slaves)
                vert.slaves = []
        all_verts = all_verts + verts
        all_verts[0].add_slave(all_verts[1:])

    def try_delete_edge(self, edge, del_layer=0):
        self.tess_copy = self.copy()
        tess_copy_edge = self.tess_copy.edges[edge.id_]
        if self.tess_copy.delete_edge(tess_copy_edge, del_layer=del_layer) == True:
            if del_layer==0:
                print('Delete accepted, structure updated')
                print('----------------------------------------')
            self.reload(self.tess_copy)
            dep_edges, _, _, _ = self.find_periodic_dependecies(edge)
            self.deleted_edges.extend(dep_edges)
            self.edge_lengths = self.get_edge_lengths()
            self.tess_copy = []
            return True
        else:
            print('Delete of edge {} rejected'.format(edge.id_))
            print('----------------------------------------')
            dep_edges, _, _, _ = self.find_periodic_dependecies(edge)
            for dep_edge in dep_edges:
                self.rejected_edge_del.append(dep_edge)
            self.edge_lengths = self.get_edge_lengths()
            self.tess_copy = []
            return False

    def reload(self, tess_copy):
        self.vertices = tess_copy.vertices
        self.edges = tess_copy.edges
        self.faces = tess_copy.faces
        self.polyhedrons = tess_copy.polyhedrons
        self.edge_lengths = self.get_edge_lengths()
        self.vertex_id_counter = tess_copy.vertex_id_counter
        self.edge_id_counter = tess_copy.edge_id_counter

    def find_periodic_dependecies(self, edge):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        ############################################################3
        # First, edge dependecies should be found
        #########################################################
        if edge.slaves == [] and edge.master == None:  # Not slave or master
            master_edge = edge
        if edge.master != None:
            # If master is not none, this edge is a slave.
            # The only master for this slave edge is edge.master
            master_edge = edge.master

        elif edge.slaves != []:
            # If slaves is not empty, this edge can not be a slave and must be a master edge
            # The master edge is now assigned as itself.
            master_edge = edge
        # the dependency list is initiated with the master edge
        dep_edges = [master_edge]
        # Each slave edge is added.
        dep_edges.extend(master_edge.slaves)

        ############################################################3
        # Then, edges and corresponding periodicities should be found
        #########################################################
        # Initializing with master edge
        dep_edge_periodicities = [np.array([0, 0, 0])]
        for edge in dep_edges[1:]:
            #Per_to_m of the slaves need to be reversed when using master as reference
            dep_edge_periodicities.append(-1*edge.per_to_m)

        ############################################################3
        # Vertex dependecies should also be included. The dep_verts are initialized with the master verts.
        # Each vertex in the dependent edges might have a dependent vertex.
        # These vertexes might not have the same periodic dependencies as the edges, and need to be found and sorted.
        #########################################################
        dep_verts = [*dep_edges[0].parts]
        dep_vert_periodicities = [np.array([0, 0, 0]), np.array([0, 0, 0])]
        # The slave edge vertices are added to the list with their respective periodicities,
        # relative to the master edge vertices. This is ued
        for edge in dep_edges[1:]:
            dep_verts.extend(edge.parts)
            #This does not account for direction of edge. These will be filtered out later anyway.
            dep_vert_periodicities.extend([-1*edge.per_to_m, -1*edge.per_to_m])
        # The edge vertices are recorded, and will be removed later
        edge_verts = copy.copy(dep_verts)
        # The edge vertices are checked for periodicities outside the edge periodicities
        for vert, periodicity in zip(dep_verts,
                                    dep_vert_periodicities):  # vert, periodicity = dep_verts[2], dep_vert_periodicities[2]
            if vert.master != None:
                if vert.master not in dep_verts:
                    dep_verts.append(vert.master)
                    dep_vert_periodicities.append(periodicity + vert.per_to_m)
            elif vert.slaves != []:
                for slave in vert.slaves:
                    if slave not in dep_verts:
                        dep_verts.append(slave)
                        dep_vert_periodicities.append(periodicity - slave.per_to_m)
        # The collected vertices are then filtered to remove the edge vertices
        dep_vertices = [vertex for vertex in dep_verts if
                    vertex not in edge_verts]
        dep_vertex_periodicities = [periodicity for vertex, periodicity in zip(dep_verts, dep_vert_periodicities) if
                                vertex not in edge_verts]

        return dep_edges, dep_edge_periodicities, dep_vertices, dep_vertex_periodicities

    def calc_new_vertices(self, edges, edge_periodicities, vertices, vertex_periodicities):

        def distance_to_plane(point, plane_equation):
            # planeEquation[1:] should be unity
            return abs(np.dot(plane_equation[1:], point) - plane_equation[0])

        def lsq_distance(point, plane_equations):
            return np.sqrt(sum([distance_to_plane(point + plane_equation[1], np.array(plane_equation[0])) ** 2
                                for plane_equation in plane_equations]))

        #Plane equations is a list of [face_eq, offset from master]
        plane_equations = []
        starting_point = edges[0].xm()
        for edge, periodicity in zip(edges, edge_periodicities):
            connected_edges = [con_edge for vert in edge.parts for con_edge in vert.part_of]

            connected_faces = [con_face for edge in connected_edges for
                               con_face in edge.part_of]
            plane_equations.extend(
                [[face.face_eq()] + [periodicity * self.domain_size] for face in
                 set(connected_faces)])

        for vertex, periodicity in zip(vertices, vertex_periodicities):
            connected_edges = [con_edge for con_edge in
                            vertex.part_of]
            connected_faces = [con_face for edge in connected_edges for
                               con_face in edge.part_of]
            plane_equations.extend(
                [[face.face_eq()] + [periodicity * self.domain_size] for face in
                 set(connected_faces)])

        new_master_vertex = scipy.optimize.minimize(lsq_distance, starting_point, plane_equations, ).x
        new_edge_locs = [new_master_vertex + periodicity * self.domain_size for periodicity in
                             edge_periodicities]
        new_vertex_locs = [new_master_vertex + periodicity * self.domain_size for periodicity in
                               vertex_periodicities]
        return new_edge_locs, new_vertex_locs

    def replace_edge_with_vertex(self, edge, new_edge_vertex, print_trigger=False):
        #edge = affected_edge
        # The list of vertices  to be removed is found
        old_verts = edge.parts

        # Find new vertex_id from the maximum value in the list  +1
        self.vertex_id_counter += 1
        new_vertex_id = self.vertex_id_counter

        # Create the new vertex with the new coordinate
        self.vertices[new_vertex_id] = tg.Vertex(id_=new_vertex_id, coord=new_edge_vertex)
        new_vertex = self.vertices[new_vertex_id]

        # Initiate list of all edges about to be affected by the merging
        for old_vert in old_verts: #old_vert  = old_verts[1]
            affected_edges = self.affected_parents([old_vert])
            affected_edges.remove(edge)
            for affected_edge in affected_edges:
                affected_edge.replace_part(new_vertex, old_vert)

        # Remove deleted edge from affected faces
        affected_faces = copy.copy(edge.part_of)
        for face in affected_faces: #face = affected_faces[0]
            face.remove_part(edge)
            # Check if face has collapsed:
            # If face eliminated:
            if len(face.parts) <= 2:
                #raise Exception('Face needs to be deleted')
                self.collapse_face(face, print_trigger=print_trigger)

        if print_trigger == True:
            print(f'Suggested edge for deletion: edge {edge.id_}')
            print(f'New vertex ID: {new_vertex.id_}')

        #Delete edge and assiciated verts
        del self.edges[edge.id_]
        for vert in old_verts:
            self.del_verts.append(vert)
            del self.vertices[vert.id_]

        return new_vertex

    def update_vertex_loc(self, vertex, new_vertex_loc):
        '''Updates the location of an existing vertex'''
        #vertex = affected_vertex
        vertex.coord = new_vertex_loc
        vertex.master = None
        vertex.slaves = []
        return vertex

    def collapse_face(self, face, print_trigger=False):

        # if the collapsed face does not belong in a slave/master combo, but the deleted edge did,
        # the edge to be deleted face should not move the masterVertex.
        #[self.tess_copy.edges[edge].master_to for edge in self.tess_copy.faces[39].edges]
        #print (face)
        old_edges = face.parts
        # Remove face from poly parent
        for id_ in [poly.id_ for poly in face.part_of]:
            self.polyhedrons[id_].remove_part(face)

        # Merge the two edges for all remaining faces
        self.edge_id_counter += 1
        new_edge_id = self.edge_id_counter


        new_edge_vertices = old_edges[0].parts
        for vert in new_edge_vertices:
            vert.part_of.remove(self.edges[abs(old_edges[0].id_)])
            vert.part_of.remove(self.edges[abs(old_edges[1].id_)])

        self.edges[new_edge_id] = tg.Edge(id_=new_edge_id, parts=new_edge_vertices)
        new_edge = self.edges[new_edge_id]

        # for each old edge, remove and replace with new edge
        for old_edge in old_edges: #old_edge=old_edges[0]
            # Find all parent faces and replace
            for face_id in [face.id_ for face in old_edge.part_of] : #face_ = old_edge.part_of[1]
                if self.faces[face_id] != face:
                    self.faces[face_id].replace_part(new_edge, old_edge)

        if print_trigger == True:
            print('Suggested face for deletion: face {}'.format(face))
            print('Coalesced edges {},{} to edge: {}'.format(abs(old_edges[0].id_), abs(old_edges[1].id_), new_edge_id))

        collapsed_poly = []
        for poly in face.part_of: #poly = face.part_of[0]
            if len(poly.parts) <= 2:
                raise Exception('Polyhedron needs to be deleted')
                #collapsed_poly.append(poly)
                self.collapse_polyhedron(poly, print_trigger = print_trigger)

        #Delete all components
        del self.faces[face.id_]
        del self.edges[old_edges[0].id_]
        del self.edges[old_edges[1].id_]

        #return new_edge

    def collapse_polyhedron(self, poly, print_trigger = False):
        rem_face, del_face = poly.parts
        #Replace one of the faces with the remaining one
        for poly_ in del_face.part_of:
            poly_.replace_part(rem_face, del_face)

        #All edges of affected faces must be updated to the remaining edge_set.
        #Oriengation might differ, so they need to be sorted
        del_edges = del_face.parts
        rem_edges = [edge for del_edge in del_edges for edge in rem_face.parts
                     if edge.direction_relative_to_other(del_edge) != None]

        for old_edge, new_edge in zip(del_edges, rem_edges):
            if new_edge.direction_relative_to_other(old_edge) == -1:
                new_edge_oriented = new_edge.reverse()
            else:
                new_edge_oriented = new_edge
            for face in old_edge.part_of:
                if face != del_face:
                    face.replace_part(new_edge, old_edge)
            for old_vert, new_vert in zip(old_edge.parts, new_edge_oriented.parts):
                for affected_edge in old_vert.part_of:
                    if affected_edge != old_edge:
                        affected_edge.replace_part(new_vert, old_vert)

        for old_edge in del_edges:
            for old_vert in old_edge.verts:
                self.del_verts.append(old_vert)
                del self.vertices[old_vert.id_]
            del self.edges[old_edge.id_]
        del self.faces[del_face.id_]
        del self.polyhedrons[poly.id_]
        return rem_edges

    def affected_parents(self, affected_parts):
        affected_parents = []
        for part in affected_parts:
            for component in part.part_of:
                if component not in affected_parents:
                    affected_parents.append(component)
        return affected_parents

    def update_periodicity_internal(self, affected_vertices):
        all_affected_edges = self.affected_parents(affected_vertices)

        all_affected_faces = self.affected_parents(all_affected_edges)

        for all_affected in [all_affected_edges, all_affected_faces]:
            for item in all_affected:
                item.master = None
                item.slaves = []

        #self.update_periodicity_internal_verts(all_affected_vertices)
        self.update_periodicity_internal_edges(all_affected_edges)
        self.update_periodicity_internal_faces(all_affected_faces)

    def update_periodicity_internal_verts(self, all_affected_vertices):
        t = time.time()
        checked_vertices=[]
        for vertex in all_affected_vertices: #vertex = all_affected_vertices[0] 0->3
            for slave in all_affected_vertices:#slave = all_affected_vertices[3] #array([-1.,  0.,  0.])
                if vertex != slave and slave not in checked_vertices and vertex not in checked_vertices:
                    master_coord = vertex.coord
                    slave_coord = slave.coord
                    coord_offset = self.check_if_periodic(master_coord, slave_coord)
                    if coord_offset != None:
                        checked_vertices.append(slave)
            checked_vertices.append(vertex)

        elapsed = time.time() - t
        print('Time to find vertex periodicity: {:.3f} s'.format(elapsed))

    def update_periodicity_internal_edges(self, all_affected_edges):
        checked_edge_list = []
        for edge in all_affected_edges: #edge =  list(all_affected_edges)[1]
            if edge not in checked_edge_list:
                verts = edge.parts
                connected_verts = []
                for vert in verts:
                    if vert.slaves != []:
                        connected_verts.extend(vert.slaves)
                    elif vert.master != None:
                        connected_verts.extend(vert.master.slaves)
                        connected_verts.remove(vert)

                parent_edges = []
                for connected_vert in connected_verts:
                    for parent_edge in connected_vert.part_of:
                        if parent_edge not in parent_edges:
                            parent_edges.append(parent_edge)


                for slave in parent_edges: #slave = list(parent_edges)[0]
                    if slave.direction_relative_to_other(edge) != None:
                        edge.add_slave(slave)
                        checked_edge_list.append(slave)
            checked_edge_list.append(edge)

    def update_periodicity_internal_faces(self, all_affected_faces):
        checked_face_list = []
        for face in all_affected_faces:  #face =  list(all_affected_faces)[0]
            if face not in checked_face_list:
                edges = face.parts
                connected_edges = []
                for edge in edges:
                    if edge.slaves != []:
                        connected_edges.extend(edge.slaves)
                    elif edge.master != None:
                        connected_edges.extend(edge.master.slaves)

                parent_faces = self.affected_parents(connected_edges)
                master_vector = face.face_eq()[1:]
                for slave in parent_faces:
                    if self.compare_arrays(abs(slave.face_eq()[1:]), abs(master_vector), scaled_rtol=1e-09, atol=0.0):
                        if slave not in face.slaves:
                            face.add_slave(slave)
                            checked_face_list.append(slave)
            checked_face_list.append(face)

    def check_if_periodic(self, master_coord, slave_coord):
        coord_offset = slave_coord - master_coord
        def test_floatingpoint():
            base_size = 10000000
            val = 0.1
            val_floatingPoint = .1+ base_size
            rel_val = val_floatingPoint- base_size
            np.isclose(val, rel_val, rtol=1e-10, atol=0)
        rtol = 1e-9*max(self.domain_size)
        offset_is_zero = [np.isclose(offset, 0.0, rtol=rtol, atol=0.0) for offset in coord_offset]
        offset_as_unity =  np.array(list(map(int,[not i for i in offset_is_zero])))
        comping_coord = slave_coord + (offset_as_unity * self.domain_size * -1*np.sign(coord_offset))
        if self.compare_arrays(master_coord, comping_coord) == True:
            return coord_offset
        else:
            return None

    def compare_arrays(self, arr0, arr1, scaled_rtol=1e-09, atol=0.0):
        rtol = scaled_rtol * max(self.domain_size)
        return np.allclose(arr0, arr1, rtol=rtol, atol=atol)

if __name__ == '__main__':
    tess_file_name = 'tests/n10-id1.tess'
    self = PeriodicTessellation(tess_file_name)
    org_self = PeriodicTessellation(tess_file_name)
    self.regularize(n=50)
    del_layer = 0
    print_trigger = True
    #edge = self.edges[24]


