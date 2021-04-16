import sys
sys.path.insert(0, '../tessToPy/')
import numpy as np
import math
import os
from tessToPy.absdict import *
import tessToPy.tessIO as tio
import tessToPy.geometry as tg
import matplotlib.pyplot as plt
import copy
import time
import scipy.optimize
import subprocess


class Tessellation(object):
    def __init__(self, tess_file_name, tess_dict=None):
        self.tess_file_name = tess_file_name
        self.mesh_file_name = None
        self.periodic = False
        if tess_dict == None:
            #If not provided a tessellation dict, read from .tess file
            self.read_tess()
        else:
            #If provided a tessellation dict
            self.load_tess_dict(tess_dict)

        # Counter for IDs when new instances are created.
        self.vertex_id_counter = max(self.vertices.keys())
        self.edge_id_counter = max(self.edges.keys())
        # For storing  rejected edges, such that they are not tried again.
        self.rejected_edge_del = []
        # For storing deleted edge ids, if edges are highlighted in the original
        self.deleted_edges = []
        # List of edges with corresponding edge lengths, sorted shortes to longest
        self.edge_lengths = self.get_edge_lengths()
        # List of affected vertices in first layer of deletion if multilayer deletion is attempted.
        # This ensures that the original affected edges are included in subsequent layers of deletion.
        self.org_new_vert_ids = []

    def read_tess(self):
        """Read and creates the tessellation from a .tess file"""
        with open(self.tess_file_name, 'r') as tess_raw:
            self.lines=tess_raw.readlines()
        self.vertices = tio.get_verts(self.lines)
        self.edges = tio.get_edges(self.lines, self.vertices)
        self.faces = tio.get_faces(self.lines, self.edges)
        self.polyhedrons = tio.get_polyhedrons(self.lines, self.faces)
        self.domain_size = tio.get_domain_size(self.lines)

    def load_tess_dict(self, tess_dict):
        """Loads the tessellation from a tessellation dict"""
        self.vertices, self.edges, self.faces, self.polyhedrons, self.domain_size, self.periodic = \
            tio.load_tess_dict(tess_dict)

    def plot(self, ax=None, alpha = 0.8, facecolor='gray', highlight_edges=[], highlight_faces=[]):
        """
        Visulizes the tessellation. Faces and edges can be highlighted.
        """
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        for poly in self.polyhedrons.values():
            poly.plot(ax, facealpha = alpha, facecolor=facecolor)
        for edge in highlight_edges:
            edge.plot(ax, color = 'r')
        for face in highlight_faces:
            face.plot(ax, color = 'r', alpha = alpha)

        ax.set_xlim(0, self.domain_size[0])
        ax.set_ylim(0, self.domain_size[1])
        ax.set_zlim(0,self.domain_size[2])
        return ax

    def get_edge_lengths(self):
        """Returns sorted list of edge lengts and edges, [[length(), edge],...]"""
        rej_ids = [edge.id_ for edge in self.rejected_edge_del]
        lengths = np.array([[edge.length(), edge] for edge in self.edges.values()
                            if edge.id_ not in rej_ids])
        if len(lengths) == 0:
            print('No more edges to find')
            return []
        else:
            #Sort edge lenghts
            lengths = lengths[lengths[:, 0].argsort()]
            return lengths

    def check_for_duplicate_vertices(self):
        """Checks for vertices in the model not present in the list of vertices. Primarily for debugging"""
        main_vert_set = set(self.vertices.values())
        derived_verts = []
        for poly in self.polyhedrons.values():
            for face in poly.parts:
                for edge in face.parts:
                    derived_verts.extend(edge.parts)
        derived_vert_set = set(derived_verts)
        if main_vert_set != derived_vert_set:
            raise Exception('Duplicate vertices found')

    def edge_length_distribution(self, ax=None, bins=50, fc = (0, 0, 0, 0.5)):
        """Plots the histogram of edge lengths. An ax can be provided for overlaying different histograms.
        Returns ax, bins"""
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        edge_lengths = [edge.length() for edge in self.edges.values()]
        _, bins, _ = ax.hist(edge_lengths, density=True, bins=bins, fc=fc,
                             lw=1, edgecolor='k', )
        ax.set_xlabel(r'Edge length')
        ax.set_ylabel(r'PDF')

        return ax, bins

    def volume_distribution(self, ax=None, bins=50, fc = (0, 0, 0, 0.5)):
        """Plots the histogram of polyhedron volumes. An ax can be provided for overlaying different histograms.
        Returns ax, bins"""
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        volumes = [poly.volume() for poly in self.polyhedrons.values()]
        _, bins, _ = ax.hist(volumes, density=True, bins=bins, fc=fc,
                             lw=1, edgecolor='k', )
        ax.set_xlabel(r'Volume')
        ax.set_ylabel(r'PDF')

        return ax, bins

    def sphericity_distribution(self, ax=None, bins=50, fc = (0, 0, 0, 0.5)):
        """Plots the histogram of polyhedron sphericity. An ax can be provided for overlaying different histograms.
        Returns ax, bins"""
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        sphericity = [poly.sphericity() for poly in self.polyhedrons.values()]
        _, bins, _ = ax.hist(sphericity, density=True, bins=bins, fc=fc,
                             lw=1, edgecolor='k', )
        ax.set_xlabel(r'Sphericity')
        ax.set_ylabel(r'PDF')

        return ax, bins

    def write(self, file_name=None):
        """Writes a .tess file for later use. Not yet compatible with reading back to NEPER"""
        tio.write_tess(self, file_name)

    def create_tess_dict(self):
        """Creates a dict representation of the tessellation. Primarily for copying the structure."""
        return tio.tess_dict(self)

class PeriodicTessellation(Tessellation):
    def __init__(self, tess_file_name,  tess_dict=None):
        super().__init__(tess_file_name, tess_dict)
        if tess_dict == None:
            tio.get_periodicity(self.lines, self.vertices, self.edges, self.faces)
        self.periodic = True

    def copy(self):
        """Returns a copy of the tessellation"""
        tess_dict = self.create_tess_dict()
        tess_copy = PeriodicTessellation('temp', tess_dict=tess_dict)
        tess_copy.tess_file_name = self.tess_file_name
        tess_copy.rejected_edge_del = self.rejected_edge_del
        tess_copy.deleted_edges = self.deleted_edges
        tess_copy.vertex_id_counter = self.vertex_id_counter
        tess_copy.edge_id_counter = self.edge_id_counter
        tess_copy.org_new_vert_ids = self.org_new_vert_ids
        return tess_copy

    def regularize(self, n=1):
        """Try to remove the n shortest edges. Returns True after completion."""
        self.edge_lengths = self.get_edge_lengths()
        for i in range(n):
            if len(self.edge_lengths) <= 1:
                print('No more edges to check!')
                break
            edge = self.edge_lengths[0, 1] #Takes the shortest edge from the list
            if edge in self.edges.values():
                print(f'Trying to delete edge {edge.id_}, i = {i}')
                _ = self.try_delete_edge(edge)
        return True

    def delete_edge(self, edge, del_layer=0):
        """Deletes the provided edge. Returns True is successful."""
        print_trigger = False
        if del_layer == 0: print_trigger = True #Action printing is suppressed for multilayer deletion

        #List of new/ affected vertices. This is used for assigning periodicity between the new, and affected, vertices.
        self.new_verts = []
        #List of deleted vertices
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
            # affected_edge, new_edge_vertex = list(zip(edges, new_edge_vertex_locs))[1]
            #The selected edge is replaced with a new vertex.
            new_vert_ = self.replace_edge_with_vertex(affected_edge, new_edge_vertex, print_trigger)
            self.new_verts.append(new_vert_)

        for affected_vertex, new_vertex_loc in zip(vertices, updated_vertex_locs):
            # affected_vertex, newVertexLoc = list(zip(vertices, new_vertex_vertices))[0]
            # The selected vertex is moved to the new, periodic location.
            new_vert_ = self.update_vertex_loc(affected_vertex, new_vertex_loc)
            self.new_verts.append(new_vert_)

        elapsed = time.time() - t
        if del_layer == 0:
            print('Time to delete dependencies: {:.3f} s'.format(elapsed))


        # Periodicity of the new/affected vertices are assigned
        self.new_verts = self.resolve_vertex_periodicity(self.new_verts)

        # Update the periodicity of the affected edges and faces.
        self.update_periodicity_internal(self.new_verts)

       #Affected vertices are identified. If multilayer deletion, original vertices need to be included.
        new_vert_ids = [vert.id_ for vert in self.new_verts]
        if del_layer == 0:
            self.org_new_vert_ids = [vert.id_ for vert in self.new_verts]

        if del_layer == 1:
            new_vert_ids.extend(self.org_new_vert_ids)
            new_vert_ids = set(new_vert_ids).intersection(list(self.vertices.keys()))
        affected_verts = [self.vertices[id_] for id_ in new_vert_ids]

        #The sorted angular deviation caused by non planar edges are found from the affected vertices.
        sorted_angles = self.face_deviation(affected_verts)

        #If the maximum angle caused by an edges is less than 20 degrees, the function returns True,
        # indicating a successful edge deletion. Should be made a parameter
        if sorted_angles[0, 1] < 20 * np.pi / 180.:
            return True
        #If the maximum angle is too large, the edges causing large angular deviation are tried for deletion.
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
        """Returns the list of affected edges sorted by the angular deviation of the affected edges.
        [[edge, float(angle)], ...]"""
        affected_edges = self.affected_parents(new_verts)
        affected_faces = self.affected_parents(affected_edges)
        angles = np.array([face.angle_deviation() for face in affected_faces])
        for i in range(len(angles)):
            if np.sign(angles[i,0].id_) == -1:
                angles[i,0] = angles[i,0].reverse()
        return angles[angles[:, 1].argsort()[::-1], :]

    def resolve_vertex_periodicity(self, verts):
        '''Resets periodicity of new/ affected vertices. Returns list of vertices.
         Unchanged if coincident vertices have not occured.'''
        all_verts = []
        #Check if previously dependent but unconnected verts have merged
        new_verts = []
        old_verts = []
        for vert_a in verts:
            for vert_b in verts:
                if vert_b != vert_a and vert_a not in old_verts:
                    if self.compare_arrays(vert_a.coord, vert_b.coord):
                        new_verts.append(self.resolve_coincident_vertices(vert_a, vert_b))
                        old_verts.extend([vert_a, vert_b])
        verts.extend(new_verts)
        for old_vert in old_verts:
            if old_vert in verts:
                verts.remove(old_vert)

        #Check if affected verts have any other periodicities, should be unnecessary
        for vert in verts:
            if vert.master != None:
                all_verts.append(vert.master)
                vert.master = None
            if vert.slaves != []:
                all_verts.extend(vert.slaves)
                vert.slaves = []
        all_verts = all_verts + verts
        all_verts[0].add_slave(all_verts[1:])
        return verts

    def try_delete_edge(self, edge, del_layer=0):
        """Tries to delete the selected edge. A copy of the tessellation is created, upon which the delete is attempted.
        A successful delete results in the main structure update.
        A rejected deletion erases the copy and the rejected edge is recorded to avoid further attempts."""
        self.tess_copy = self.copy()
        tess_copy_edge = self.tess_copy.edges[edge.id_]
        dep_edges, _, _, _ = self.find_periodic_dependecies(edge)
        if self.tess_copy.delete_edge(tess_copy_edge, del_layer=del_layer) == True:
            if del_layer==0:
                print('Delete accepted, structure updated')
                print('----------------------------------------')
            self.reload(self.tess_copy)
            self.deleted_edges.extend(dep_edges)
            self.edge_lengths = self.get_edge_lengths()
            self.tess_copy = []
            return True
        else:
            print('Delete of edge {} rejected'.format(edge.id_))
            print('----------------------------------------')
            for dep_edge in dep_edges:
                self.rejected_edge_del.append(dep_edge)
            self.edge_lengths = self.get_edge_lengths()
            self.tess_copy = []
            self.edge_lengths = self.get_edge_lengths()
            return False

    def reload(self, tess_copy):
        """Updates the main structure based on a supplied tessellation."""
        self.vertices = tess_copy.vertices
        self.edges = tess_copy.edges
        self.faces = tess_copy.faces
        self.polyhedrons = tess_copy.polyhedrons
        self.edge_lengths = self.get_edge_lengths()
        self.vertex_id_counter = tess_copy.vertex_id_counter
        self.edge_id_counter = tess_copy.edge_id_counter

    def find_periodic_dependecies(self, edge):
        """Finds all dependent edges and vertices, calculating the periodic offset of the edges and vertices
         relative to only one main edge."""
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
        """Calculates the location of the new vertices resulting from edge
        deletion and updated vertex location of dependent vertices.
        This function minimizes the distance of the new/updated vertices from the plane of the connected faces.
        The function returns [[new vertex loc],[updated vertex loc]]"""
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
        """Replaces an edge with a new vertex. The connecting edges are assigned the new vertex.
        If on edge in a triangular face is deleted,
        the face is removed and the two colinear edges are replaced by a new edge."""
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
        for face in affected_faces: #face = affected_faces[2]
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
        """Updates the location of an existing vertex"""
        #vertex = affected_vertex
        vertex.coord = new_vertex_loc
        vertex.master = None
        vertex.slaves = []
        return vertex

    def collapse_face(self, face, print_trigger=False):
        """Removes a face when an edge deletion causes a triangular face to collapse.
        If the face removal causes a polyhedron to collapse, this is also resolved."""
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

    def collapse_polyhedron(self, poly):
        """Deletes a collapsed polyhedron."""
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
        """Returns list of 'parents' from the supplied parts, e.g. edges connected to vertices or faces connected to edges."""
        affected_parents = []
        for part in affected_parts:
            for component in part.part_of:
                if component not in affected_parents:
                    affected_parents.append(component)
        return affected_parents

    def update_periodicity_internal(self, affected_vertices):
        """Updates the periodicity of the edges and faces connected to the affected vertices."""
        all_affected_edges = self.affected_parents(affected_vertices)

        all_affected_faces = self.affected_parents(all_affected_edges)

        for all_affected in [all_affected_edges, all_affected_faces]:
            for item in all_affected:
                item.master = None
                item.slaves = []


        self.update_periodicity_internal_edges(all_affected_edges)
        self.update_periodicity_internal_faces(all_affected_faces)

    def update_periodicity_internal_edges(self, all_affected_edges):
        """Updates the periodicity of the affected edges by checking if any combination of edges are periodic."""
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
        """Updates the periodicity of the affected faces by checking if any combination of edges are periodic."""
        checked_face_list = []
        for face in all_affected_faces:  #face =  list(all_affected_faces)[0]
            if face not in checked_face_list:
                master_vector = face.face_eq()[1:]
                for slave in all_affected_faces:
                    if self.compare_arrays(abs(slave.face_eq()[1:]), abs(master_vector))\
                            and slave != face:
                        if slave not in face.slaves:
                            face.add_slave(slave)
                            checked_face_list.append(slave)
            checked_face_list.append(face)

    def check_periodicity_face(self):
        """Checks if master/slave faces are periodic. Primarily used for debugging"""
        for face in self.faces.values():
            if face.slaves != []:
                for slave in face.slaves: #slave = face.slaves[0]
                    if self.compare_arrays(slave.periodicity_to_master()*self.domain_size, slave.vector_to_master()) == False:
                        print ('Master face {} and slave face {} no longer periodic'.format(face.id_, slave.id_))
                        raise Exception('Faces not periodic')
        print ('All faces still periodic')

    def outer_faces(self):
        """Returns list of faces on the perimeter of the tessellation."""
        #Outer faces are only part of one polyhedron
        #[face for face in self.faces.values() if len(face.part_of) ==1]
        face_list = []
        for face in self.faces.values():
            if face.slaves != []:
                face_list.append(face)
                face_list.extend(face.slaves)
        return face_list

    def compare_arrays(self, arr0, arr1, rel_tol=1e-09, abs_tol=1e-09):
        """Checks if pairwise elements of two arrays are close in value"""
        return np.all([math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a,b in zip(arr0, arr1)])

    def resolve_coincident_vertices(self, vert_a, vert_b):
        """Resolves vertices that have become coincident due to an edge deletion.
        This happens if two vertices are periodic to each of the vertices of an edge,
        but without an edge between them"""
        self.vertex_id_counter += 1
        new_vertex_id = self.vertex_id_counter

        # Create the new vertex with the new coordinate
        self.vertices[new_vertex_id] = tg.Vertex(id_=new_vertex_id, coord=vert_a.coord)
        new_vertex = self.vertices[new_vertex_id]

        for vert in [vert_a, vert_b]:
            for id_ in [part.id_ for part in vert.part_of]:
                self.edges[id_].replace_part(new_vertex, vert)

        #Check if any edges are coalleced
        coalesced_edge_pairs = []
        for edge_a in new_vertex.part_of:
            for edge_b in new_vertex.part_of:
                if edge_a != edge_b and edge_a not in [value for values in coalesced_edge_pairs for value in values]:
                    if self.compare_arrays(edge_a.xm(), edge_b.xm()):
                        coalesced_edge_pairs.append([edge_a, edge_b])

        for coalesced_edge_pair in coalesced_edge_pairs:
            # Merge the two edges for all remaining faces
            old_edges = coalesced_edge_pair
            self.edge_id_counter += 1
            new_edge_id = self.edge_id_counter

            new_edge_vertices = old_edges[0].parts
            for vert in new_edge_vertices:
                vert.part_of.remove(self.edges[abs(old_edges[0].id_)])
                vert.part_of.remove(self.edges[abs(old_edges[1].id_)])

            self.edges[new_edge_id] = tg.Edge(id_=new_edge_id, parts=new_edge_vertices)
            new_edge = self.edges[new_edge_id]

            # for each old edge, remove and replace with new edge
            for old_edge in old_edges:  # old_edge=old_edges[0]
                # Find all parent faces and replace
                for face_id in [face.id_ for face in old_edge.part_of]:  # face_ = old_edge.part_of[1]
                    self.faces[face_id].replace_part(new_edge, old_edge)

            #if print_trigger == True:
            print('Coincident vertices resolved: Edges {},{} to edge: {}'.format(abs(old_edges[0].id_), abs(old_edges[1].id_), new_edge_id))

            # Delete all components
            del self.edges[old_edges[0].id_]
            del self.edges[old_edges[1].id_]
        del self.vertices[vert_a.id_]
        del self.vertices[vert_b.id_]
        return new_vertex

    def write_geo(self, mesh_file_name = None):
        """Writes a .geo file for meshing with GMSH"""
        if self.mesh_file_name==None and mesh_file_name==None:
            self.mesh_file_name=self.tess_file_name
        elif self.mesh_file_name == None and mesh_file_name != None:
            self.mesh_file_name=mesh_file_name
        else:
            pass
        with open(self.mesh_file_name.rsplit('.')[0] + '.geo', 'w+') as geo_file:
            geo_file.write('SetFactory("OpenCASCADE");\n')
            for id, vert in zip(self.vertices.keys(), self.vertices.values()):
                geo_file.write('Point ({id}) = {{{:.10f}, {:.10f}, {:.10f}}};\n'.format(id=id, *vert.coord))

            for id, edge in zip(self.edges.keys(), self.edges.values()):
                geo_file.write('Line ({id}) = {{{}, {}}};\n'.format(id=id, *[vert.id_ for vert in edge.parts]))

            for id, face in zip(self.faces.keys(), self.faces.values()):
                geo_file.write('Curve Loop ({id}) = {{'.format(id=id*10)+', '.join(map(str,
                                                                                       [edge.id_ for edge in face.parts]))+'};\n')
                geo_file.write('Surface ({id}) = {{{id2}}};\n'.format(id=id*10, id2=id*10))

            #Writes settings from self.gmsh list of commands
            for line in self.gmsh:
                geo_file.write(line)
            return self.mesh_file_name

    def mesh2D(self, elem_size, mesh_type=None, recombine=True, mesh_file_name=None,
               corner_refine_factor=2., mesh_algo=8, recomb_algo=0, second_order=False):
        """Function for creating a meshed representation of the tessellation, using GMSH.
        mesh_type dictates refinment of the mesh closer to edges and corners of surfaces.
        Returns file name of mesh."""
        self.gmsh=[]
        if mesh_type==None:
            self.gmsh.append('Field[1] = MathEval;\n')
            self.gmsh.append('Field[1].F = "{}";\n'.format(elem_size))
            self.gmsh.append('Background Field = 1;\n')
        elif mesh_type== 'Distance':
            self.gmsh.append('Field[1] = Distance;\n')
            self.gmsh.append('Field[1].NodesList = {{{}}};\n'.format(str(list(self.vertices.keys())).replace(']', '').replace('[','')))
            self.gmsh.append('Field[2] = Threshold;\n')
            self.gmsh.append('Field[2].IField = 1;\n')
            self.gmsh.append('Field[2].LcMin = {};\n'.format(elem_size / corner_refine_factor))
            self.gmsh.append('Field[2].LcMax = {};\n'.format(elem_size))
            self.gmsh.append('Field[2].DistMin = {};\n'.format(elem_size * 4))
            self.gmsh.append('Field[2].DistMax = {};\n'.format(elem_size * 8))
            self.gmsh.append('Background Field = 2;\n')
            self.gmsh.append('Mesh.CharacteristicLengthExtendFromBoundary = 0;\n')
            self.gmsh.append('Mesh.CharacteristicLengthFromPoints = 0;\n')
            self.gmsh.append('Mesh.CharacteristicLengthFromCurvature = 0;\n')
        self.gmsh.append('Mesh.Algorithm = {};\n'.format(mesh_algo)) #6
        self.gmsh.append('Mesh.Smoothing = 3;\n')
        if recombine==True:
            self.gmsh.append('Mesh.RecombinationAlgorithm = {};\n'.format(recomb_algo))
            self.gmsh.append('Recombine Surface {:};\n')
            self.gmsh.append('Recombine Surface {:};\n')
            self.gmsh.append('Mesh.Smoothing = 3;\n')
        if second_order == True:
            self.gmsh.append('Mesh.ElementOrder = 2;\n')
            self.gmsh.append('Mesh.SecondOrderIncomplete = 1;\n')
        self.write_geo(mesh_file_name)
        subprocess.run('gmsh {}'.format(self.mesh_file_name.rsplit('.',1)[0])+'.geo -2 -nt 2 -format key')
        return self.mesh_file_name.rsplit('.')[0]+'.key'


if __name__ == '__main__':
    pass
    #tess_file_name = 'tests/tess_files/n10-id1.tess'
    #tess_file_name = 'tests/tess_files/n400_from_morpho-id1.tess'
    #self = []
    #self = PeriodicTessellation(tess_file_name)



