import numpy as np
import os
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
        self.edge_lengths = self.get_edge_lengths()

    def read_tess(self):
        with open(self.tess_file_name, 'r') as tess_raw:
            self.lines=tess_raw.readlines()
        self.vertices = tio.get_verts(self.lines)
        self.edges = tio.get_edges(self.lines, self.vertices)
        self.faces = tio.get_faces(self.lines, self.edges)
        self.polyhedrons = tio.get_polyhedrons(self.lines, self.faces)
        self.domain_size = tio.get_domain_size(self.lines)

    def plot(self, alpha = 0.8, facecolor = 'gray'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for poly in self.polyhedrons.values():
            poly.plot(ax, facealpha = alpha, facecolor = facecolor)

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
        tess_copy.vertex_id_counter = self.vertex_id_counter
        tess_copy.edge_id_counter = self.edge_id_counter
        return tess_copy

    def regularize(self, n):
        '''Try to remove the n shortest edges'''

        for i in range(n):
            if len(self.edge_lengths) <= 1:
                print('No more edges to check!')
                break
            edge = self.edge_lengths[0, 1] #Edge232
            print(f'Trying to delete edge {edge.id_}')
            self.try_delete_edge(edge)

    def delete_edge(self, edge, del_layer=0):
        if del_layer == 0: print_trigger = True
        new_verts = []
        collapsed_faces = []
        verts_for_deletion = []
        coalesced_edges = []

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
            # affected_edge, new_edge_vertex = list(zip(edges, new_edge_vertices))[0]
            new_verts_, collapsed_faces_, verts_for_deletion_ = self.replace_edge_with_vertex(affected_edge,
                                                                                            new_edge_vertex,
                                                                                             print_trigger)
            new_verts.append(new_verts_)
            collapsed_faces.append(collapsed_faces_)
            if collapsed_faces_ != []:
                print (f'Collapsed {collapsed_faces_[0]}, for edge {edge}')
            verts_for_deletion.extend(verts_for_deletion_)

        for affected_vertex, new_vertex_loc in zip(vertices, updated_vertex_locs):
            # affected_vertex, newVertexLoc = list(zip(vertices, new_vertex_vertices))[0]
            updated_verts_ = self.update_vertex_loc(affected_vertex, new_vertex_loc)
            new_verts.append(updated_verts_)
            collapsed_faces.append([])
            verts_for_deletion.extend([])
        elapsed = time.time() - t
        if del_layer == 0:
            print('Time to delete dependencies: {:.3f} s'.format(elapsed))

        ##################################################################################
        # Each edge is merged to the new vertex location. The dependent vertices are moved.
        ##################################################################################

        t = time.time()
        #Check if dependent vertices have merged:
        duplicate_vertex_sets = self.check_for_duplicate_vertices()
        elapsed = time.time() - t
        print('Time to check for duplicate vertices: {:.3f} s'.format(elapsed))

        collapsed_polyhedrons = []
        for collapsed_faces_pr_edge in collapsed_faces:
            if collapsed_faces_pr_edge != []:
                for collapsed_face in collapsed_faces_pr_edge:  # collapsedFace=820
                    temp_edge, col_poly = self.delete_face_to_edge(collapsed_face, print_trigger)
                    coalesced_edges.append(temp_edge)
                    if col_poly != []:
                        for poly in col_poly:
                            coalesced_edges.extend(self.collapse_polyhedron(poly))

        elapsed = time.time() - t
        print('Time to deal with collapsed polyhedrons: {:.3f} s'.format(elapsed))

        filtered_new_verts = list(set(self.vertices.values()).intersection(set(new_verts)))
        affected_vertices = copy.copy(filtered_new_verts)

        # Update the vertex and edge periodicity of the affected edges
        self.update_periodicity_internal(affected_vertices, coalesced_edges)
        # self.findParents()
        ####################################################################
        # Find all affected edges, by newVertexList, and check internal angles
        ########################################################################
        affected_edges = [edge_id for vert_id in new_vertex_list for edge_id in self.vertices[vert_id].parents]
        angles = np.array([self.faces[face_id].find_angle_deviation()
                           for edge_id in affected_edges for face_id in self.edges[edge_id].parents])
        sorted_angles = angles[angles[:, 1].argsort()[::-1], :]
        self.new_vertex_list = new_vertex_list
        self.deleted_verts_list = deleted_verts_list
        if sorted_angles[0, 1] < 20 * np.pi / 180.:
            return True
        elif del_layer == 0:
            checked_edges = []
            for edge_angle in sorted_angles:  # edge_angle =  sorted_angles[0]
                if edge_angle[1] > 20 * np.pi / 180. and int(abs(edge_angle[0])) not in checked_edges:
                    try:
                        layer_edge_id = int(abs(edge_angle[0]))
                        dep_edges = self.find_periodic_dependecies(self.edges[layer_edge_id])[0]
                        checked_edges.extend(dep_edge.id_ for dep_edge in dep_edges)
                        self.tess_copy = copy.deepcopy(self)
                        # new_vertex_list_layer = self.tess_copy.remove_edge(layer_edge_id, del_layer=del_layer + 1)
                        if self.tess_copy.remove_edge(layer_edge_id, del_layer=del_layer + 1):
                            new_vertex_list.extend(self.tess_copy.new_vertex_list)
                            for vert in self.tess_copy.deleted_verts_list:
                                if vert in new_vertex_list:
                                    new_vertex_list.remove(vert)
                            self.vertices = self.tess_copy.vertices
                            self.edges = self.tess_copy.edges
                            self.faces = self.tess_copy.faces
                            self.polyhedrons = self.tess_copy.polyhedrons
                            self.edge_lengths = self.tess_copy.edge_lengths
                            self.vertex_id_counter = self.tess_copy.vertex_id_counter
                            self.edge_id_counter = self.tess_copy.edge_id_counter
                            print('{} st/nd layer deletion of edge {} was successful'.format(
                                del_layer + 1, int(edge_angle[0])))
                            print('--------------------------------------------------------------')
                        else:
                            self.tess_copy = []
                            print('{} st/nd layer deletion of edge {} failed with new angle {}'.format(
                                del_layer + 1, int(edge_angle[0]), edge_angle[1]))
                    except:
                        print('Error encountered in {} st/nd layer deletion of edge {}'.format(
                            del_layer + 1, int(edge_angle[0])))
                else:
                    pass
            new_vertex_list_final = set(new_vertex_list)
            filtered_vertex_list = [vert_id for vert_id in new_vertex_list_final if vert_id in self.vertices.keys()]
            affected_edges = [edge_id for vert_id in filtered_vertex_list for edge_id in
                              self.vertices[vert_id].parents
                              if edge_id in self.edges.keys()]
            angles = np.array([self.faces[face_id].find_angle_deviation()
                               for edge_id in affected_edges for face_id in self.edges[edge_id].parents])
            sorted_angles = angles[angles[:, 1].argsort()[::-1], :]
            if sorted_angles[0, 1] < 20 * np.pi / 180.:
                return True
            else:
                return False
        else:
            return False

    def try_delete_edge(self, edge):
        self.tess_copy = self.copy()
        if self.tess_copy.delete_edge(edge):
            self.reload(self.tess_copy)
            print('Delete accepted, structure updated')
            print('----------------------------------------')
            self.tess_copy = []

        else:
            print('Delete of edge {} rejected'.format(edge.id_))
            print('----------------------------------------')
            dep_edges, _, _, _ = self.find_periodic_dependecies(edge)
            for dep_edge in dep_edges:
                self.rejected_edge_del.append(dep_edge)
            self.edge_lengths = self.find_edge_lengths()
            self.tess_copy = []

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
        affected_edges = []
        collapsed_faces = []
        # For each vertex about to be merged, update all affected edges with new vertex_id
        for vert in old_verts:  # vert = old_verts[0]
            # Find edges connected to each vertices about to be merges. Exlude the edge to be removed
            affected_edges_pr_vert = list(set(
                [edge_ for edge_ in vert.part_of if
                 edge_.id_ != edge.id_]))
            for affected_edge in affected_edges_pr_vert:
                # Find index in edge.verts where old vertex is located
                affected_edge.replace_part(new_vertex, vert)
            # Add edges from each vertex to the collection
            affected_edges.extend(affected_edges_pr_vert)

        # Remove deleted edge from affected faces
        affected_faces = copy.copy(edge.part_of)
        for face in affected_faces:
            face.remove_part(edge)
            # Check if face has collapsed:
            # If face eliminated:
            if len(face.parts) <= 2:
                collapsed_faces.append(face)

        if print_trigger == True:
            print(f'Suggested edge for deletion: edge {edge.id_}')
            print(f'New vertex ID: {new_vertex.id_}')

        #Delete edge and assiciated verts
        del self.edges[edge.id_]
        for vert in old_verts:
            del self.vertices[vert.id_]

        def test_vert_replace():
            for face in self.faces.values():
                for edge in face.parts:
                    if vert in edge.parts:
                        print (f'{face}, {edge}')
                        raise Exception('Old verts not cleanded from faces')

        return new_vertex, collapsed_faces, old_verts

    def update_vertex_loc(self, vertex, new_vertex_loc):
        '''Updates the location of an existing vertex'''
        #vertex = affected_vertex
        vertex.coord = new_vertex_loc
        return vertex

    def delete_face_to_edge(self, collapsed_face, print_trigger=False):

        # if the collapsed face does not belong in a slave/master combo, but the deleted edge did,
        # the edge to be deleted face should not move the masterVertex.
        #[self.tess_copy.edges[edge].master_to for edge in self.tess_copy.faces[39].edges]
        old_edges = collapsed_face.parts
        # Remove face from poly parent
        for polyhedron in collapsed_face.part_of:
            polyhedron.remove_part(collapsed_face)

        # Merge the two edges for all remaining faces
        self.edge_id_counter += 1
        new_edge_id = self.edge_id_counter


        new_edge_vertices = old_edges[0].parts
        self.edges[new_edge_id] = tg.Edge(id_=new_edge_id, parts=new_edge_vertices)
        new_edge = self.edges[new_edge_id]

        # for each old edge, remove and replace with new edge
        for old_edge in old_edges: #oldEdge=remEdges[0]
            # Find all parent faces and replace
            for face in old_edge.part_of:
                if face != collapsed_face:
                    face.replace_part(new_edge, old_edge)

        if print_trigger == True:
            print('Suggested face for deletion: face {}'.format(collapsed_face))
            print('Coalesced edges {},{} to edge: {}'.format(abs(old_edges[0].id_), abs(old_edges[1].id_), new_edge_id))

        collapsed_poly = []
        for poly in collapsed_face.part_of:
            if len(poly.parts) <= 2:
                collapsed_poly.append(poly)
                self.collapse_polyhedron(poly)

        #Delete all components
        del self.faces[collapsed_face.id_]
        del self.edges[old_edges[0].id_]
        del self.edges[old_edges[1].id_]

        return new_edge, collapsed_poly

    def collapse_polyhedron(self, poly):
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
                del self.vertices[old_vert.id_]
            del self.edges[old_edge.id_]
        del self.faces[del_face.id_]
        del self.polyhedrons[poly.id_]
        return rem_edges

    def update_periodicity_internal(self, affected_vertices, coaleced_edges=[]):
        all_affected_edges = coaleced_edges
        for vertex in affected_vertices:
            all_affected_edges.extend(vertex.part_of)
        all_affected_edges = set(all_affected_edges)

        all_affected_vertices = []
        all_affected_faces = []

        for edge in all_affected_edges:
            all_affected_vertices.extend(edge.parts)
            all_affected_faces.extend(edge.part_of)

        all_affected_vertices = set(all_affected_vertices)
        all_affected_faces = set(all_affected_faces)
        for all_affected in [all_affected_vertices, all_affected_edges, all_affected_faces]:
            for item in all_affected:
                item.master = None
                item.slaves = []

        self.update_periodicity_internal_verts(all_affected_vertices)
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
        for edge in all_affected_edges: #edge =  affected_edges[0]
            if edge not in checked_edge_list:
                verts = edge.parts
                connected_verts = []
                for vert in verts:
                    if vert.slaves != []:
                        connected_verts.extend(verts.slaves)
                    elif vert.master != None:
                        connected_verts.extend(vert.master.slaves)
                        connected_verts.remove(vert)
                parent_edges = set([parent_edge for connected_vert in connected_verts for parent_edge in
                                    connected_vert.part_of]).intersection(all_affected_edges)
                master_vector = edge.vector()
                for slave in parent_edges: #parentEdgeID = 51
                    if slave.direction_relative_to_other(edge.vector()) != None:
                        edge.add_slave(slave)
                        checked_edge_list.append(slave)
            checked_edge_list.append(edge)

    def update_periodicity_internal_faces(self, all_affected_faces):
        checked_face_list = []
        for face in all_affected_faces:  #edge =  affected_edges[0]
            if face not in checked_face_list:
                edges = face.parts
                connected_edges = []
                for edge in edges:
                    if edge.slaves != []:
                        connected_edges.extend(edge.slaves)
                    elif edge.master != None:
                        connected_edges.extend(edge.master.slaves)
                        connected_edges.remove(edge)
                parent_faces = set([parent_face for connected_edge in connected_edges for parent_face in
                                    connected_edge.part_of]).intersection(all_affected_faces)
                master_vector = face.face_eq()[1:]

                for slave in parent_faces:
                    if self.compare_arrays(abs(slave.face_eq()[1:]), abs(master_vector), rel_tol=1e-09, abs_tol=0.0):
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

    def compare_arrays(self, arr0, arr1, rel_tol=1e-09, abs_tol=0.0):
        rtol = rel_tol * max(self.domain_size)
        return np.allclose(arr0, arr1, rtol=rel_tol, atol=abs_tol)

if __name__ == '__main__':
    tess_file_name = '../tests/n10-id1.tess'
    self = PeriodicTessellation(tess_file_name)
    del_layer = 0



