import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
import random
import sys
import networkx as nx
import math

## python varian of AdTree code based on:
# https://github.com/tudelft3d/AdTree/blob/main/AdTree/skeleton.cpp#L473

DBL_MAX = sys.float_info.max
FLT_MAX = sys.float_info.max

class SGraphVertexProp:
    def __init__(self, cVert=None, nParent=0, lengthOfSubtree=0.0, radius=0.0, visited=False):
        self.cVert = cVert if cVert is not None else [0.0, 0.0, 0.0]  # Assuming cVert is a 3D vector (list)
        self.nParent = nParent
        self.lengthOfSubtree = lengthOfSubtree
        self.radius = radius  # used only by the smoothed skeleton
        self.visited = visited

    # Getters
    def get_vertex(self):
        return self.cVert

    def get_parent(self):
        return self.nParent

    def get_length_of_subtree(self):
        return self.lengthOfSubtree

    def get_radius(self):
        return self.radius

    def is_visited(self):
        return self.visited

    # Setters
    def set_vertex(self, vert):
        self.cVert = vert

    def set_parent(self, parent):
        self.nParent = parent

    def set_length_of_subtree(self, length):
        self.lengthOfSubtree = length

    def set_radius(self, r):
        self.radius = r

    def set_visited(self, visit):
        self.visited = visit

class SGraphEdgeProp:
    def __init__(self, nWeight=0.0, nRadius=0.0, vecPoints=None):
        self.nWeight = nWeight
        self.nRadius = nRadius
        self.vecPoints = vecPoints if vecPoints is not None else []  # Default to an empty list

    # Getters
    def get_weight(self):
        return self.nWeight

    def get_radius(self):
        return self.nRadius

    def get_points(self):
        return self.vecPoints

    # Setters
    def set_weight(self, weight):
        self.nWeight = weight

    def set_radius(self, radius):
        self.nRadius = radius

    def set_points(self, points):
        self.vecPoints = points


class Skeleton:
    def __init__(self):
        self.Points_ = None
        self.KDtree_ = None
        self.quiet_ = True
        self.TrunkRadius_ = 0
        self.TreeHeight_ = 0
        self.BoundingDistance_ = 0
        self.VecLeaves_ = []
        # self.delaunay_ = {}
        # self.MST_ = {}
        self.simplified_skeleton_ = nx.Graph()
        self.smoothed_skeleton_ = nx.Graph()
        self.delaunay_ = nx.Graph()
        self.MST_ = nx.Graph()



    def __del__(self):
        if self.KDtree_:
            del self.KDtree_
        if self.Points_:
            del self.Points_
        if len(self.VecLeaves_) > 0:
            self.VecLeaves_.clear()

    def build_delaunay(self, cloud):
        self.delaunay_.clear()

        if not self.quiet_:
            print("read vertices into the delaunay...")

        nPoints = cloud.n_vertices()
        nPoints = cloud.n_vertices()

        points = cloud.get_vertex_property("v:point")
        new_vertices = self.centralize_main_points(cloud)
        # new_vertices = points

        for i in range(nPoints):
            pV = {
                "cVert": new_vertices[i],
                "nParent": 0,
                "lengthOfSubtree": 0.0
            }
            self.add_vertex(pV, self.delaunay_, i)

        if not self.quiet_:
            print("generate delaunay edges...")

        # Create a Delaunay triangulation using the points
        coords = np.array([points[v] for v in cloud.vertices()])
        delaunay_triangulation = Delaunay(coords)
        for simplex in delaunay_triangulation.simplices:
            for i in simplex:
                for j in simplex[1:]:
                    if i != j:
                        self.delaunay_.add_edge(i, j)

        # from skeletonisation_methods.plantscan3d.xu import k_closest_points_from_ann, connect_componenets
        # result = k_closest_points_from_ann(coords, k=5, connect_all_points=False)
        # result = connect_componenets(coords, result, k=5)
        # for i in result:
        #     for j in i[1:]:
        #         # edges_list = list(self.delaunay_.edges)
        #         # if (i[0], j) in edges_list or (j, i[0]) in edges_list:
        #         #     continue
        #         self.delaunay_.add_edge(i[0], j)

        
        # for simplex in delaunay_triangulation.simplices:
        #     for i in range(len(simplex)):
        #         for j in range(i + 1, len(simplex)):
        #             self.delaunay_.add_edge(simplex[i], simplex[j])
            # for i, j in zip(simplex, simplex[1:] + simplex[:1]):
                # self.add_edge(self.vertex(i, self.delaunay_), self.vertex(j, self.delaunay_), self.delaunay_)
                # self.delaunay_.add_edge(i, j)

        # from xu_method import visualize_examples
        # visualize_examples.vis(None, new_vertices, np.array(self.delaunay_.edges), None, None, None)


        if not self.quiet_:
            print("compute Delaunay graph edges weights...")

        self.compute_delaunay_weight()

        if not self.quiet_:
            print("finish the delaunay graph building!")

        return True

    def add_vertex(self, pV, graph, i):
        """
        Add a vertex to the graph with optional attributes.
        """
        graph.add_node(i, **pV)

    def vertex(self, index, graph):
        return index
        # Implement the vertex method
        # return graph.nbodes[index]

    def cVert(self, graph):
        temp =[graph._node[x]["cVert"] for x in graph.nodes]
        return np.array(temp)
    
    def vertices(self, graph):
        # temp =[graph._node[x]["cVert"] for x in graph.nodes]
        # return np.array(temp)
        return graph.nodes

    def add_edge(self, source, target, edge_props, graph):
        """
        Add an edge between two vertices in the graph with optional attributes.
        """
        graph.add_edge(source, target, **edge_props)

    def edges(self, graph):
        return np.array(graph.edges)
    
    def corrected_edges(self, graph):
        mapping = {}
        for i, x in enumerate(list(graph.nodes)):
            mapping[x] = i
        corrected_edges = []
        for e in graph.edges:
            corrected_edges.append([mapping[e[0]],mapping[e[1]]])
        return np.array(corrected_edges)



    def extract_mst(self):
        self.MST_.clear()

        if not self.quiet_:
            print("extracting MST...")

        vp = self.delaunay_.nodes
        for i, v in enumerate(vp):
            pV = {
                "cVert": self.delaunay_._node[v]["cVert"],
                "nParent": 0,
                "lengthOfSubtree": 0.0
            }
            self.add_vertex(pV, self.MST_, i)

        if not self.quiet_:
            print("get the root vertex...")

        self.compute_root_vertex(self.MST_)

        if not self.quiet_:
            print("compute the shortest spanning tree...")
        from skeletonisation_methods.plantscan3d.xu import points_dijkstra_shortest_path
        # from xu_method.xu_skeletonisation import points_dijkstra_shortest_path
        # self.get_edge_weight = lambda x, y: self.delaunay_[x][y]['weight']  
        remaniangraph = []
        for x, y in self.delaunay_._adj.items():
            remaniangraph.append(list(y.keys()))
        shortest_pathes = points_dijkstra_shortest_path(self.cVert(self.delaunay_), remaniangraph, self.RootV_)
        parents = shortest_pathes[0]
        distances = shortest_pathes[1]


        # distances, parents = self.dijkstra_shortest_paths(self.delaunay_, self.RootV_, self.get_edge_weight)

        for nP, parent in enumerate(parents):
            if self.vertex(nP, self.MST_) != parent:
                pEdge = {"nWeight": 0.0, "nRadius": 0.0, "vecPoints": []}
                self.add_edge(self.vertex(nP, self.MST_), parent, pEdge, self.MST_)
            self.MST_._node[self.vertex(nP, self.MST_)]["nParent"] = parent

        try:
            self.MST_.remove_node(9223372036854775807)
        except nx.exception.NetworkXError:
            pass


        if not self.quiet_:
            print("compute the subtree length for each vertex...")

        self.compute_length_of_subtree(self.MST_, self.RootV_)

        if not self.quiet_:
            print("finish the minimum spanning tree extraction!")

        return True

    def simplify_skeleton(self):
        if not self.quiet_:
            print("step 1: eliminate unimportant small edges")

        self.keep_main_skeleton(self.MST_, 0.019)

        if not self.quiet_:
            print("step 2: iteratively merge collapsed edges")

        self.merge_collapsed_edges()

        if not self.quiet_:
            print("finish the skeleton graph refining!")

        return True

    def smooth_skeleton(self):
        if len(self.simplified_skeleton_.edges) < 2:
            print("skeleton does not exist!")
            return False

        self.smoothed_skeleton_.clear()

        if not self.quiet_:
            print("smoothing skeleton...")

        pathList = self.get_graph_for_smooth()

        for currentPath in pathList:
            interpolatedPoints, interpolatedRadii = [], []
            numOfSlices = 20
            numOfSlicesCurrent = []

            for n_node in range(len(currentPath) - 1):
                sourceV = currentPath[n_node]
                targetV = currentPath[n_node + 1]
                pSource = self.simplified_skeleton_.nodes[sourceV]["cVert"]
                pTarget = self.simplified_skeleton_.nodes[targetV]["cVert"]
                branchLength = np.linalg.norm(pSource - pTarget)
                numOfSlicesCurrent.append(max(int(branchLength * numOfSlices), 2))

                tangentOfSource = (pTarget - pSource) / np.linalg.norm(pTarget - pSource)
                tangentOfSource *= branchLength

                A = 2 * (pSource - pTarget)
                B = 3 * (pTarget - pSource) - 2 * tangentOfSource
                C = tangentOfSource
                D = pSource

                for n in range(numOfSlicesCurrent[-1]):
                    t = n / numOfSlicesCurrent[-1]
                    point = A * t**3 + B * t**2 + C * t + D

                    if len(interpolatedPoints) == 0 or np.linalg.norm(interpolatedPoints[-1] - point) > 1e-5:
                        interpolatedPoints.append(point)
                        interpolatedRadii.append(0)

            if len(interpolatedPoints) > 1:
                vertices = []
                for point in interpolatedPoints:
                    v = {"cVert": point, "radius": 0.0}
                    self.add_vertex(v, self.smoothed_skeleton_, len(vertices))
                    vertices.append(len(vertices))

                for i in range(len(vertices) - 1):
                    self.add_edge(vertices[i], vertices[i + 1], {}, self.smoothed_skeleton_)

        return True

    def get_graph_for_smooth(self):
        path_list = []  # List to hold all paths
        current_path = [self.RootV_]  # Start with the root vertex
        path_list.append(current_path)  # Add the root path

        cursor = 0
        while cursor < len(path_list):
            current_path = path_list[cursor]
            end_v = current_path[-1]

            # Check if current path has reached a leaf
            if len(list(self.simplified_skeleton_.adj[end_v])) == 1 and end_v != self.simplified_skeleton_.nodes[end_v].get('nParent'):
                cursor += 1
            else:
                # Find the "fattest" child vertex (with the largest radius)
                max_radius = -1
                fatest_child = None
                not_fastest_children = []
                for neighbor in self.simplified_skeleton_.adj[end_v]:
                    if neighbor != self.simplified_skeleton_.nodes[end_v].get('nParent'):
                        edge = self.simplified_skeleton_.get_edge_data(end_v, neighbor)
                        radius = edge.get('nRadius', 0)  # Get radius from the edge attributes
                        if radius > max_radius:
                            if fatest_child is not None:
                                not_fastest_children.append(fatest_child)
                            fatest_child = neighbor
                            max_radius = radius
                        else:
                            not_fastest_children.append(neighbor)

                # Create new paths for the non-fastest children
                for child in not_fastest_children:
                    new_path = [end_v, child]
                    path_list.append(new_path)

                # Add the fattest child to the current path
                path_list[cursor].append(fatest_child)

        return path_list

    def compute_branch_radius(self):
        if not self.quiet_:
            print("step 1: assign points to corresponding branch edges")
        
        self.assign_points_to_edges()

        if not self.quiet_:
            print("step 2: fit accurate radius to the trunk")

        self.fit_trunk()

        if not self.quiet_:
            print("step 3: adjust the radius for all left branches")

        self.compute_all_edges_radius(self.TrunkRadius_)

        if not self.quiet_:
            print("finish the branches inflation!")

        return True

    def add_leaves(self):
        if not self.quiet_:
            print("step 1: find leaf vertices in the tree graph")

        leafVertices = self.find_end_vertices()

        if not self.quiet_:
            print("step 2: randomly generate leaves for each leaf vertex")

        if len(self.VecLeaves_) > 0:
            self.VecLeaves_.clear()

        for leafVertex in leafVertices:
            self.generate_leaves(leafVertex, 0.05)

        if not self.quiet_:
            print("finish adding the leaves!")

        return True

    def keep_main_skeleton(self, i_Graph, subtree_Threshold):
        self.simplified_skeleton_.clear()

        vp = self.vertices(i_Graph)
        for v in vp:
            if v==9223372036854775807:
                continue
            pV = {"cVert": i_Graph.nodes[v]["cVert"], "nParent": i_Graph.nodes[v]["nParent"], "lengthOfSubtree": i_Graph.nodes[v]["lengthOfSubtree"]}
            self.add_vertex(pV, self.simplified_skeleton_, v)

        stack = [self.RootV_]

        while stack:
            currentV = stack.pop()

            for adj in self.adjacent_vertices(currentV, i_Graph):
                if adj != i_Graph.nodes[currentV]["nParent"]:
                    child2Current = np.linalg.norm(i_Graph.nodes[currentV]["cVert"] - i_Graph.nodes[adj]["cVert"])
                    subtreeRatio = (i_Graph.nodes[adj]["lengthOfSubtree"] + child2Current) / i_Graph.nodes[currentV]["lengthOfSubtree"]
                    
                    if subtreeRatio >= subtree_Threshold:
                        # pEdge = {"nWeight": i_Graph[adj]["nWeight"], "nRadius": i_Graph[adj]["nRadius"], "vecPoints": i_Graph[adj]["vecPoints"]}
                        pEdge = i_Graph._adj[currentV][adj]
                        self.add_edge(currentV, adj, pEdge, self.simplified_skeleton_)
                        stack.append(adj)

        self.compute_length_of_subtree(self.simplified_skeleton_, self.RootV_)
        self.compute_graph_edges_weight(self.simplified_skeleton_)
        self.compute_all_edges_radius(self.TrunkRadius_)

        return
    def merge_collapsed_edges(self):
        vp = self.vertices(self.simplified_skeleton_)
        bChange = True
        numComplex = 0
        
        while bChange:
            bChange = False
            for dVertex in list(vp):# Create a copy of vp to avoid modification during iteration
                if dVertex not in vp: ## to deal with removed ponts
                    continue  
                if (self.out_degree(dVertex, self.simplified_skeleton_) > 2 or
                    (self.simplified_skeleton_.nodes[dVertex]["nParent"] == dVertex and self.out_degree(dVertex, self.simplified_skeleton_) > 1)):
                    if self.check_overlap_child_vertex(self.simplified_skeleton_, dVertex):
                        bChange = True
                        numComplex += 1
                elif self.out_degree(dVertex, self.simplified_skeleton_) == 2 and self.simplified_skeleton_.nodes[dVertex]["nParent"] != dVertex:
                    if self.check_single_child_vertex(self.simplified_skeleton_, dVertex):
                        bChange = True
                        numComplex += 1

        self.compute_length_of_subtree(self.simplified_skeleton_, self.RootV_)
        self.compute_graph_edges_weight(self.simplified_skeleton_)
        self.compute_all_edges_radius(self.TrunkRadius_)

    def out_edges(self, node, graph):
        return list(graph.edges(node))

    def out_degree(self, node, graph):
        return graph.degree(node)  # Use .out_degree(node) if it's a DiGraph

    def check_overlap_child_vertex(self, i_Graph, i_dVertex):
        nMinMergeValue = DBL_MAX
        vecChilds = []

        listAdj = self.out_edges(i_dVertex, i_Graph)
        for e in listAdj:
            # currentV = self.target(e, i_Graph) if self.source(e, i_Graph) == i_dVertex else self.source(e, i_Graph)
            currentV = e[1] if e[0] == i_dVertex else e[0]
            if i_Graph.nodes[currentV]["nParent"] == i_dVertex:
                vecChilds.append(currentV)


        sourceV, targetV = None, None
        for i in range(len(vecChilds) - 1):
            for j in range(i + 1, len(vecChilds)):
                vi = vecChilds[i]
                vj = vecChilds[j]
                merge_i2j = self.compute_merge_value(i_Graph, vi, vj)
                merge_j2i = self.compute_merge_value(i_Graph, vj, vi)

                if merge_i2j < merge_j2i and merge_i2j < nMinMergeValue:
                    nMinMergeValue = merge_i2j
                    sourceV = vi
                    targetV = vj
                elif merge_j2i < merge_i2j and merge_j2i < nMinMergeValue:
                    nMinMergeValue = merge_j2i
                    sourceV = vj
                    targetV = vi

        if nMinMergeValue > 1.0:
            return False
        else:
            return self.merge_vertices(i_Graph, sourceV, targetV, 0.5, 0.5)

    def check_single_child_vertex(self, i_Graph, i_dVertex):
        childV = None
        listAdj = self.out_edges(i_dVertex, i_Graph)
        for e in listAdj:
            # currentV = self.target(e, i_Graph) if self.source(e, i_Graph) == i_dVertex else self.source(e, i_Graph)
            currentV = e[1] if e[0] == i_dVertex else e[0]
            if i_Graph.nodes[currentV]["nParent"] == i_dVertex:
                childV = currentV

        parentV = i_Graph.nodes[i_dVertex]["nParent"]
        pParent = i_Graph.nodes[parentV]["cVert"]
        pCurrent = i_Graph.nodes[i_dVertex]["cVert"]
        pChild = i_Graph.nodes[childV]["cVert"]
        pCross = np.cross(pCurrent - pParent, pCurrent - pChild)
        distance = np.linalg.norm(pCross) / np.linalg.norm(pParent - pChild)

        # r = i_Graph[self.edge(i_dVertex, parentV, i_Graph)].nRadius
        r = i_Graph.edges[(i_dVertex, parentV)]["nRadius"]

        if distance >= 1.0 * r:
            return False
        else:
            i_Graph.nodes[childV]["nParent"] = parentV
            i_Graph.nodes[parentV]["lengthOfSubtree"] = i_Graph.nodes[childV]["lengthOfSubtree"] + np.linalg.norm(pParent - pChild)
            # self.clear_vertex(i_dVertex, i_Graph)
            i_Graph.remove_node(i_dVertex)

            pEdge = {"nWeight": (i_Graph.nodes[childV]["lengthOfSubtree"] + i_Graph.nodes[parentV]["lengthOfSubtree"]) / 2.0, "nRadius": r}
            self.add_edge(childV, parentV, pEdge, i_Graph)
            return True

    def merge_vertices(self, i_Graph, i_dSource, i_dTarget, i_wSource, i_wTarget):
        sGroupVertices = set(self.adjacent_vertices(i_dSource, i_Graph)) | set(self.adjacent_vertices(i_dTarget, i_Graph))
        mapAdjToRadius = {}

        for v in sGroupVertices:
            if i_Graph.edges.get((v, i_dTarget)) and i_Graph.edges.get((v, i_dSource)):
                sourceRadius = i_Graph.edges[(v, i_dSource)]["nRadius"]
                targetRadius = i_Graph.edges[(v, i_dTarget)]["nRadius"]
                mapAdjToRadius[v] = max(sourceRadius, targetRadius)
            elif i_Graph.edges.get((v, i_dTarget)):
                mapAdjToRadius[v] = i_Graph.edges[(v, i_dTarget)]["nRadius"]
            elif i_Graph.edges.get((v, i_dSource)):
                mapAdjToRadius[v] = i_Graph.edges[(v, i_dSource)]["nRadius"]

        pV = {"nParent": i_Graph.nodes[i_dTarget]["nParent"], "lengthOfSubtree": max(i_Graph.nodes[i_dSource]["lengthOfSubtree"], i_Graph.nodes[i_dTarget]["lengthOfSubtree"])}
        pSource = i_Graph.nodes[i_dSource]["cVert"]
        pTarget = i_Graph.nodes[i_dTarget]["cVert"]

        if pV["lengthOfSubtree"] == 0:
            pNew = (i_wSource * pSource + i_wTarget * pTarget) / (i_wSource + i_wTarget)
        else:
            pNew = (i_wSource * pSource * i_Graph.nodes[i_dSource]["lengthOfSubtree"] + i_wTarget * pTarget * i_Graph.nodes[i_dTarget]["lengthOfSubtree"]) / \
                   (i_wSource * i_Graph.nodes[i_dSource]["lengthOfSubtree"] + i_wTarget * i_Graph.nodes[i_dTarget]["lengthOfSubtree"])
        pV["cVert"] = pNew

        i_Graph.remove_node(i_dSource)
        i_Graph.remove_node(i_dTarget)

        # self.clear_vertex(i_dSource, i_Graph)
        # self.clear_vertex(i_dTarget, i_Graph)
        self.add_vertex(pV, i_Graph, i_dTarget)
        # i_Graph.add_node[i_dTarget] = pVs

        for v in sGroupVertices:
            if i_dSource != v and i_dTarget != v:
                pEdge = {"nRadius": mapAdjToRadius[v]}
                self.add_edge(i_dTarget, v, pEdge, i_Graph)
                if i_Graph.nodes[v]["nParent"] in [i_dSource, i_dTarget]:
                    i_Graph.nodes[v]["nParent"] = i_dTarget
                parentV = i_Graph.nodes[v]["nParent"]
                pEdge["nWeight"] = (pV["lengthOfSubtree"] + i_Graph.nodes[parentV]["lengthOfSubtree"]) / 2.0

        return True

    def compute_delaunay_weight(self):
        ep = self.edges(self.delaunay_)
        for e in ep:
            # dVertex1 = self.source(e, self.delaunay_)
            # dVertex2 = self.target(e, self.delaunay_)
            pVertex1 = self.delaunay_._node[e[0]]["cVert"]
            pVertex2 = self.delaunay_._node[e[1]]["cVert"]

            # pVertex1 = self.delaunay_[dVertex1]["cVert"]
            # pVertex2 = self.delaunay_[dVertex2]["cVert"]
            # self.delaunay_[e]["nWeight"] = np.linalg.norm(pVertex2 - pVertex1) ** 2
            self.delaunay_[e[0]][e[1]]['nWeight'] = np.linalg.norm(pVertex2 - pVertex1) ** 2 # Change weight from 3.0 to 4.5

    def compute_root_vertex(self, i_Graph):
        # vp = self.vertices(i_Graph)
        # initialVertex = next(iter(vp))
        # pCurrent = i_Graph[initialVertex]["cVert"]

        # for v in vp:
        #     pOther = i_Graph[v]["cVert"]
        #     if pOther[2] < pCurrent[2]:
        #         initialVertex = v

        # self.RootV_ = initialVertex
        # self.delaunay_._node[x]
        self.RootV_ = self.cVert(i_Graph)[:,2].argmin()


    def compute_length_of_subtree(self, i_Graph, i_dVertex):
        i_Graph.nodes[i_dVertex]["lengthOfSubtree"] = 0.0
        adjList = self.adjacent_vertices(i_dVertex, i_Graph)

        for child in adjList:
            if child != i_Graph.nodes[i_dVertex]["nParent"]:
                self.compute_length_of_subtree(i_Graph, child)
                pChild = i_Graph.nodes[child]["cVert"]
                pCurrent = i_Graph.nodes[i_dVertex]["cVert"]
                distance = np.linalg.norm(pCurrent - pChild)
                child_Length = i_Graph.nodes[child]["lengthOfSubtree"] + distance

                if i_Graph is self.MST_:
                    i_Graph.nodes[i_dVertex]["lengthOfSubtree"] += child_Length
                elif i_Graph is self.simplified_skeleton_:
                    if i_Graph.nodes[i_dVertex]["lengthOfSubtree"] < child_Length:
                        i_Graph.nodes[i_dVertex]["lengthOfSubtree"] = child_Length

    def compute_graph_edges_weight(self, i_Graph):
        ep = self.edges(i_Graph)
        for e in ep:
            # subtreeWeight = i_Graph[self.source(e, i_Graph)]["lengthOfSubtree"] + i_Graph[self.target(e, i_Graph)]["lengthOfSubtree"]
            subtreeWeight = i_Graph.nodes[e[0]]["lengthOfSubtree"] + i_Graph.nodes[e[1]]["lengthOfSubtree"]
            # i_Graph[e]["nWeight"] = subtreeWeight / 2.0
            i_Graph._adj[e[0]][e[1]]["nWeight"]= subtreeWeight / 2.0
            nx.set_edge_attributes(i_Graph, {(e[0], e[1]): {"nWeight": subtreeWeight / 2.0}})

    def compute_all_edges_radius(self, trunkRadius):
        trunkE = next(iter(self.out_edges(self.RootV_, self.simplified_skeleton_)))
        avrRadius = trunkRadius / (self.simplified_skeleton_.edges[trunkE]["nWeight"] ** 1.1)

        ep = self.edges(self.simplified_skeleton_)
        for e in ep:
            self.simplified_skeleton_.edges[e]["nRadius"] = (self.simplified_skeleton_.edges[e]["nWeight"] ** 1.1) * avrRadius

    def compute_merge_value(self, i_Graph, i_dSource, i_dTarget):
        assert i_Graph.nodes[i_dSource]["nParent"] == i_Graph.nodes[i_dTarget]["nParent"]

        parentV = i_Graph.nodes[i_dSource]["nParent"]
        dirSource = i_Graph.nodes[i_dSource]["cVert"] - i_Graph.nodes[parentV]["cVert"]
        dirTarget = i_Graph.nodes[i_dTarget]["cVert"] - i_Graph.nodes[parentV]["cVert"]
        nLengthSource = np.linalg.norm(dirSource)
        nLengthTarget = np.linalg.norm(dirTarget)
        dirSource /= nLengthSource
        dirTarget /= nLengthTarget
        alpha = np.dot(dirSource, dirTarget)
        # nRadiusTarget = i_Graph[self.edge(i_dTarget, parentV, i_Graph)]["nRadius"]
        nRadiusTarget = i_Graph.edges[(i_dTarget, parentV)]["nRadius"]

        if alpha > 0.9 and 0.5 <= nLengthSource / nLengthTarget <= 2:
            return nLengthSource * math.sin(math.acos(round(alpha,4))) / nRadiusTarget
        return DBL_MAX

    def centralize_main_points(self, cloud):
        if not self.quiet_:
            print("start centralizing the main-branch points")

        nPt = cloud.n_vertices()
        self.Points_ = np.array([cloud.get_vertex_property("v:point")[v] for v in cloud.vertices()])
        self.KDtree_ = KDTree(self.Points_)

        densityList = []
        vertices = []
        self.obtain_initial_radius(cloud)

        for pCurrent in self.Points_:
            distance = np.linalg.norm(pCurrent - self.RootPos_)
            threshold = self.TrunkRadius_ * (1 - distance / self.BoundingDistance_)
            neighbors = self.KDtree_.query_ball_point(pCurrent, threshold)
            density = len(neighbors) / threshold if threshold != 0 else 0
            densityList.append(density)

        epsilon = 0.5
        for pCurrent, density in zip(self.Points_, densityList):
            distance = np.linalg.norm(pCurrent - self.RootPos_)
            if distance != 0 and distance < epsilon * self.BoundingDistance_:
                neighbors = self.KDtree_.query_ball_point(pCurrent, self.TrunkRadius_ * (1 - distance / self.BoundingDistance_))
                dendiff, pSum = 0, np.zeros(3)
                for neighbor in neighbors:
                    currentDensity = densityList[neighbor]
                    pSum += self.Points_[neighbor]
                    dendiff += abs(currentDensity - density)
                dendiff /= len(neighbors)
                pSum /= len(neighbors)
                dendiff /= len(neighbors)

                if dendiff < 0.6:
                    vertices.append(pSum)
                    continue
            vertices.append(pCurrent)

        return vertices

    def obtain_initial_radius(self, cloud):
        points = cloud.get_vertex_property("v:point")
        pLowest = np.array([0.0, 0.0, FLT_MAX])

        # for v in cloud.vertices():
        #     pOther = points[v]
        #     if pOther[2] < pLowest[2]:
        #         pLowest = pOther

        pLowest = points[points[:,2].argmin()]
        pHighest = points[points[:,2].argmax()]
        self.TreeHeight_ = pHighest[2] - pLowest[2]
        self.BoundingDistance_ = np.linalg.norm(points - pLowest, axis=1).max()

        self.RootPos_ = pLowest

        if not self.quiet_:
            print(f"The root vertex coordinate is: {self.RootPos_}")

        # for v in cloud.vertices():
        #     pOther = points[v]
        #     if pOther[2] - pLowest[2] > self.TreeHeight_:
        #         self.TreeHeight_ = pOther[2] - pLowest[2]
        #     dist = np.linalg.norm(pOther - pLowest)
        #     if dist > self.BoundingDistance_:
        #         self.BoundingDistance_ = dist

        epsiony = 0.02
        trunkList = [points[v] for v in cloud.vertices() if (points[v][2] - pLowest[2]) <= epsiony * self.TreeHeight_]

        minX, maxX = np.min([pt[0] for pt in trunkList]), np.max([pt[0] for pt in trunkList])
        minY, maxY = np.min([pt[1] for pt in trunkList]), np.max([pt[1] for pt in trunkList])

        self.TrunkRadius_ = max(maxX - minX, maxY - minY) / 2.0
        if not self.quiet_:
            print(f"The initial radius is: {self.TrunkRadius_}")

    def assign_points_to_edges(self):
        if not self.KDtree_:
            if not self.quiet_:
                print("KD tree construction failed!")
            return

        for e in self.edges(self.simplified_skeleton_):
            sourceV, targetV = e[0], e[1]#self.get_edge_vertices(e)
            pSource = np.array(self.simplified_skeleton_.nodes[sourceV]["cVert"])
            pTarget = np.array(self.simplified_skeleton_.nodes[targetV]["cVert"])
            currentR = self.simplified_skeleton_.edges[e]["nRadius"]

            neighbors = points_in_cylinder(self.Points_, pSource, pTarget, 3.5*currentR)
            neighbors= list(np.argwhere(neighbors).reshape(-1))
            
            # self.KDtree_.query_line_intersection(pSource, pTarget, 3.5 * currentR)
            # neighbors = self.KDtree_.get_found_neighbors()

            for ptIndex in neighbors:
                pCurrent = self.Points_[ptIndex]
                cDirPoint = pCurrent - pSource
                cDirCylinder = pTarget - pSource
                cosAlpha = np.dot(cDirCylinder / np.linalg.norm(cDirCylinder), cDirPoint / np.linalg.norm(cDirPoint))
                if cosAlpha >= 0:
                    vecpoints =  self.simplified_skeleton_.edges[e].get("vecPoints", None)
                    if vecpoints is None:
                        self.simplified_skeleton_.edges[e]["vecPoints"] = [e[0], e[1]]
                    else:
                        self.simplified_skeleton_.edges[e]["vecPoints"].append(ptIndex)

    def fit_trunk(self):
        trunk_e = next(iter(self.out_edges(self.RootV_, self.simplified_skeleton_)))
        p_count = len(self.simplified_skeleton_.edges[trunk_e]["vecPoints"])
        if p_count <= 20:
            if not self.quiet_:
                print("Least squares fit failed due to insufficient points.")
            return

        # Construct the initial cylinder
        source_v, target_v = trunk_e#self.get_trunk_vertices(trunk_e)

        # Initialize the mean, the point cloud matrix
        p_top = np.array([0.0, 0.0, -np.inf])
        p_bottom = np.array([0.0, 0.0, np.inf])

        # PCA step (replace with an actual PCA method)
        pca = self.perform_pca(trunk_e, p_count)

        p_mean = pca['center']
        ev = pca['axis_0']  # Largest eigenvector (principal direction)

        # Adjust direction if needed
        if ev[2] < 0:
            ev = -ev

        c_dir_top = p_top - p_mean
        c_dir_bottom = p_bottom - p_mean
        n_length_top = np.linalg.norm(c_dir_top)
        n_length_bottom = np.linalg.norm(c_dir_bottom)

        cosine_top = np.dot(ev, c_dir_top)
        cosine_bottom = np.dot(ev, c_dir_bottom)

        p_source = p_mean + n_length_bottom * cosine_bottom * ev
        p_target = p_mean + n_length_top * cosine_top * ev

        # Initialize the cylinder object (placeholder)
        current_c = Cylinder(p_source, p_target, self.simplified_skeleton_.edges[trunk_e]['nRadius'])

        # Non-linear least squares adjustment
        pt_list = self.prepare_point_cloud(trunk_e, p_count)

        if current_c.LeastSquaresFit(pt_list):
            p_source_adjust = current_c.GetAxisPosition1()
            p_target_adjust = current_c.GetAxisPosition2()
            radius_adjust = current_c.GetRadius()

            # Update the weights
            dis_list = self.update_weights(pt_list, p_source_adjust, p_target_adjust, radius_adjust)

            # Conduct the second round of weighted least squares
            if current_c.LeastSquaresFit(pt_list):
                if not self.quiet_:
                    print("Successfully conducted the non-linear least squares!")

                # Adjust the positions of the trunk vertices
                p_source_adjust = current_c.GetAxisPosition1()
                p_target_adjust = current_c.GetAxisPosition2()
                radius_adjust = current_c.GetRadius()

                # Update the trunk edge and vertices
                self.simplified_skeleton_[source_v]['cVert'] = p_source_adjust
                self.simplified_skeleton_[target_v]['cVert'] = p_target_adjust
                self.simplified_skeleton_[trunk_e]['nRadius'] = radius_adjust
                self.TrunkRadius_ = radius_adjust
                return
        else:
            if not self.quiet_:
                print("The non-linear least squares is unsuccessful!")
            return

    def get_trunk_vertices(self, trunk_e):
        # Placeholder for extracting trunk vertices based on edge descriptor
        source_v = self.simplified_skeleton_[trunk_e]['source']
        target_v = self.simplified_skeleton_[trunk_e]['target']
        return source_v, target_v

    def perform_pca(self, trunk_e, p_count):
        # Placeholder for PCA logic, should return the center and principal axis of the points
        # points = [self.Points_[idx] for idx in self.simplified_skeleton_.edges[trunk_e]['vecPoints']]
        points_matrix = self.Points_[self.simplified_skeleton_.edges[trunk_e]['vecPoints']]

        # Perform PCA (using SVD or a PCA implementation from a library)
        mean = np.mean(points_matrix, axis=0)
        cov_matrix = np.cov(points_matrix, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        # The eigenvector with the largest eigenvalue is the principal axis
        principal_axis = eig_vecs[:, np.argmax(eig_vals)]

        return {'center': mean, 'axis_0': principal_axis}

    def prepare_point_cloud(self, trunk_e, p_count):
        # Prepare point cloud for least squares fitting
        # pt_list = []
        # for np_points in range(p_count):
        #     np_index = self.simplified_skeleton_.edges[trunk_e]['vecPoints'][np_points]
        #     pt = self.Points_[np_index]
        #     pt_list.append([pt[0], pt[1], pt[2], 1.0])  # Assuming weights are 1.0
        pt_array = np.ones((len(self.simplified_skeleton_.edges[trunk_e]['vecPoints']), 4))
        pt_array[:,:3] = self.Points_[self.simplified_skeleton_.edges[trunk_e]['vecPoints']]

        return pt_array.tolist()

    def update_weights(self, pt_list, p_source_adjust, p_target_adjust, radius_adjust):
        # Update weights based on distance from cylinder
        dis_list = []
        max_dis = -np.inf
        for pt in pt_list:
            pt_pos = np.array(pt[:3])
            dis = np.linalg.norm(np.cross(pt_pos - p_source_adjust, pt_pos - p_target_adjust)) / np.linalg.norm(p_target_adjust - p_source_adjust)
            dis = abs(dis - radius_adjust)
            max_dis = max(max_dis, dis)
            dis_list.append(dis)
        
        for i, dis in enumerate(dis_list):
            pt_list[i][3] = 1.0 - dis / max_dis

        return dis_list

    def generate_leaves(self, i_LeafVertex, leafsize_Factor):
        density = int(np.ceil(random.random() * 10))
        radius = 0.2 / np.log(len(self.edges(self.simplified_skeleton_)))

        pCurrent = np.array(self.simplified_skeleton_[i_LeafVertex]["cVert"])
        i_LeafParent = self.simplified_skeleton_[i_LeafVertex]["nParent"]
        pParent = np.array(self.simplified_skeleton_[i_LeafParent]["cVert"])
        pEnd = pCurrent - (random.random() / 2.0) * (pCurrent - pParent) / np.linalg.norm(pCurrent - pParent)

        for i in range(density):
            dirLeaf = np.random.uniform(-0.5, 0.5, 3)
            dirLeaf /= np.linalg.norm(dirLeaf)
            l = random.random() * radius
            pLeaf = pEnd + dirLeaf * l
            dirParent2Leaf = (pLeaf - pParent) / np.linalg.norm(pLeaf - pParent)
            normal = np.cross(dirParent2Leaf, dirLeaf) / np.linalg.norm(np.cross(dirParent2Leaf, dirLeaf))

            newLeaf = {
                "cPos": pLeaf,
                "cDir": dirLeaf,
                "cNormal": (normal + random.random() * np.random.uniform(-0.5, 0.5, 3) * 0.5) / np.linalg.norm(normal),
                "pSource": i_LeafVertex,
                "nLength": self.BoundingDistance_ * leafsize_Factor,
                "nRad": self.BoundingDistance_ * leafsize_Factor / 5
            }
            self.VecLeaves_.append(newLeaf)

    def reconstruct_branches(self, cloud, mesh):
        if not cloud:
            print("Point cloud does not exist.")
            return False

        if not self.build_delaunay(cloud):
            print("Failed Delaunay triangulation.")
            return False
        ve.vis(pc= cloud.get_vertex_property("v:point"), nodes = self.cVert(self.delaunay_), edges=np.array(self.delaunay_.edges))

        if not self.extract_mst():
            print("Failed extracting MST.")
            return False
        # ve.vis(pc= cloud.get_vertex_property("v:point"), nodes = self.cVert(self.MST_), edges=np.array(self.MST_.edges))


        if not self.simplify_skeleton():
            print("Failed skeleton simplification.")
            return False
        ve.vis(pc= cloud.get_vertex_property("v:point"), nodes = self.cVert(self.simplified_skeleton_), edges=self.corrected_edges(self.simplified_skeleton_))


        # if not self.compute_branch_radius():
        #     print("Failed computing branch radius.")
        #     return False

        if not self.smooth_skeleton():
            print("Failed smoothing branches.")
            return False
        ve.vis(pc= cloud.get_vertex_property("v:point"), nodes = self.cVert(self.smoothed_skeleton_), edges=self.corrected_edges(self.smoothed_skeleton_))


        # if not self.extract_branch_surfaces(mesh):
        #     print("Failed extracting branch surfaces.")
            # return False
        

        return True

    def reconstruct_leaves(self, mesh):
        if not self.add_leaves() or not self.VecLeaves_:
            return False

        for leaf in self.VecLeaves_:
            pCenter = leaf["cPos"] + 0.5 * leaf["cDir"] * leaf["nRad"]
            dirMajor = 0.5 * leaf["cDir"] * leaf["nLength"]
            dirMinor = 0.5 * np.cross(leaf["cDir"], leaf["cNormal"]) * leaf["nRad"]
            a = pCenter - dirMajor - dirMinor
            b = pCenter + dirMajor - dirMinor
            c = pCenter + dirMajor + dirMinor
            d = pCenter - dirMajor + dirMinor
            va, vb, vc, vd = map(mesh.add_vertex, [a, b, c, d])
            mesh.add_triangle(va, vb, vc)
            mesh.add_triangle(va, vc, vd)

        return True
    
    def adjacent_vertices(self, i, graph):
        return list(graph.adj[i])

class PointCloud:
    def __init__(self, points):
        # `points` is assumed to be a NumPy array of shape (n, 3) where n is the number of points
        self.points = points
        self.vertex_properties = {"v:point": points}

    def n_vertices(self):
        return self.points.shape[0]

    def vertices(self):
        return range(self.n_vertices())

    def get_vertex_property(self, prop_name):
        if prop_name in self.vertex_properties:
            return self.vertex_properties[prop_name]
        else:
            raise KeyError(f"Property {prop_name} not found in point cloud.")


# Placeholder class for Cylinder, you can use the actual implementation for your application
class Cylinder:
    def __init__(self, p_source, p_target, radius):
        self.p_source = p_source
        self.p_target = p_target
        self.radius = radius

    def LeastSquaresFit(self, pt_list):
        # Placeholder method for least squares fitting
        return True  # Return True if fitting is successful

    def GetAxisPosition1(self):
        return self.p_source

    def GetAxisPosition2(self):
        return self.p_target

    def GetRadius(self):
        return self.radius

    def SetAxisPosition1(self, p_source_adjust):
        self.p_source = p_source_adjust

    def SetAxisPosition2(self, p_target_adjust):
        self.p_target = p_target_adjust

    def SetRadius(self, radius_adjust):
        self.radius = radius_adjust

def points_in_cylinder(points, p1, p2, r):
    """
    points: (N, 3) array of points
    p1: (3,) array, one end of the cylinder axis
    p2: (3,) array, other end of the cylinder axis
    r: radius of the cylinder
    """
    # Cylinder axis vector
    d = p2 - p1
    d_norm = np.linalg.norm(d)
    d_unit = d / d_norm

    # Vectors from p1 to each point
    v = points - p1

    # Project v onto d to find the component along the cylinder axis
    t = np.dot(v, d_unit)

    # Check if within cylinder length
    inside_length = (t >= 0) & (t <= d_norm)

    # Closest point on the axis
    closest = p1 + np.outer(t, d_unit)

    # Distance from axis
    dist = np.linalg.norm(points - closest, axis=1)

    # Check if within radius
    inside_radius = dist <= r

    # Points inside both length and radius
    inside = inside_length & inside_radius

    return inside  # This is a boolean mask

if __name__=="__main__":
    sys.path.append("")
    import wurTomato
    from scripts import visualize_examples as ve
    import numpy as np

    obj = wurTomato.WurTomatoData()

    # points = obj.load_xyz_array(0)
    points, _ = obj.get_filtered_data(0)

    # Load points from an .xyz file
    def load_xyz_file(file_path):
        points = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(points)
    points = points[np.random.choice(points.shape[0], size=20000, replace=False)]

    # Replace with the path to your .xyz file
    # xyz_file_path = "skeletonisation_methods/adtree/Lille_11.xyz"
    # points = load_xyz_file(xyz_file_path)

    cloud = PointCloud(points)

    a = Skeleton()
    a.quiet_ = False
    a.reconstruct_branches(cloud, None)