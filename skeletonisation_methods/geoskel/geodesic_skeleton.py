################################################################
# Author     : Kyle Forgarty                                   
# Contact    :                         
# Date       :                                       
# Description: Code related to geodesic based nodes to graph.
# TODO implement in wurTomato.py
################################################################


import numpy as np
from scipy.spatial import KDTree
import potpourri3d as pp3d
import networkx as nx
from tqdm import tqdm


class GeodesicSkeleton:
    def __init__(self, pointcloud  : np.array, 
                       graph_nodes : np.array, ):
    
        self.plc    = pointcloud
        self.nodes  = graph_nodes

        # Precompute the KDTree for both sets of points 

        self.kdtree_plc = KDTree(self.plc)
        self.kdtree_nodes = KDTree(self.nodes)


        # Define the conversion between graph nodes and pointcloud 
        # points. 

        self.conversion_lookup  = self.construct_conversion_lookup()
        print("=====================================")
        print("Solving the heat equation on pointcloud.")
        self.solver = pp3d.PointCloudHeatSolver(pointcloud, t_coef=2)
        print('Completed the heat equation solver.')
        print("=====================================")

    def construct_conversion_lookup(self,):
        """
        Construct a lookup table that maps each graph node to the 
        closest point in the pointcloud and visa versa. 

        Note that we work with index position relative to self.plc
        and self.nodes.
        """

        conversion_lookup = []

        print("constructing conversion lookup:")
        for i in tqdm(range(len(self.nodes))):
            plc_index = self.nodes2pointcloud(i)
            conversion_lookup.append(plc_index)
        
        return conversion_lookup
    
    def nodes2pointcloud(self, node_index):
        '''
        returns the index of the nearest pointcloud point
        '''
        dists, index = self.kdtree_plc.query(self.nodes[node_index], k = 2)
        return index[1]
    
    def mask_connected_components(self, connected_components):
        dist_lookup = self.dist_lookup
        
        for component_list in connected_components:
            for component in component_list:
                for component_2 in component_list:
                    dist_lookup[component][component_2] = np.inf

        self.dist_lookup = dist_lookup
        return None 

    def find_connected_components(self):
        G = nx.Graph()
        edges = self.edge_list
        G.add_edges_from(edges)
        connected_components = list(nx.connected_components(G))
        return connected_components

    def update_edgelist(self, connected_components):
        dist_lookup = self.dist_lookup
        edge_list   = self.edge_list
        for component_list in connected_components:
            small_id = []
            small_val = []
            for component in component_list:
                smallest_ind = np.argmin(dist_lookup[component])
                smallest_val = dist_lookup[component][smallest_ind]
                small_id.append(smallest_ind)
                small_val.append(smallest_val)
        
            comp_add = np.argmin(small_val)
            comp_id = small_id[comp_add]
            init_id = list(component_list)[comp_add]
            edge_list.append([init_id, comp_id])
        self.edge_list = edge_list
        return None 
    
    def compute_geodesics(self,):
        """
        Construct the geodesic distances between every node 
        and every other node (via the pointcloud).

        NOTE: We define the distance between a node and itself
        to be inf to prevent the degenerate case.
        """
        
        print("Computing the geodesic distances between nodes:")
        dist_lookup = []
        for i, node in enumerate(tqdm(self.nodes)):
            
            # Map the ith node to its corresponding pointcloud index
            plc_id = self.conversion_lookup[i]

            # Compute the geodesic distances between the ith node and all other
            # pointcloud points. 
            dists = self.solver.compute_distance(plc_id)

            # if i==0:
            #     ve.vis_distance(self.plc, dists)

            # Convert distances back to graph node indices.
            dists = dists[self.conversion_lookup]

            # Set the current node to inf (to prevent degenerate cases)
            dists[i] = np.inf

            # Append the distances to the lookup table.
            dist_lookup.append(dists)
    
        self.dist_lookup = dist_lookup

        return None 

    def construct_graph(self,):
        """
        Given the distance lookup table we construct the graph.
        """
        print("Constructing the graph:")
        self.edge_list = []

        # (1) Construct the edge list for which every node has 
        # one connected component.

        for i in range(len(self.nodes)):
            sorted_list = np.argsort(self.dist_lookup[i])
            self.dist_lookup[i][sorted_list[0]] = np.inf
            self.edge_list.append([i, sorted_list[0]])

        # (2) Make the assumption that the graph is fully connected
        # and add the remaining edges.

        connected_components = self.find_connected_components()

        while len(connected_components) > 1:
            connected_components = self.find_connected_components()
            self.mask_connected_components(connected_components)
            self.update_edgelist(connected_components,)


        self.edge_list = np.unique([ sorted(e) for e in self.edge_list], axis=0)[1:] # [1: because of [0,0]]

        # ve.vis(self.plc, nodes=self.nodes, edges=self.edge_list)

        # (3) convert edge_list to directed graph
        next_edge_list = []
        stack = [0]
        while stack:
            node = stack.pop()
            temp_edge_list = self.edge_list.copy()
            self.edge_list = []
            for edge in temp_edge_list:
                if edge[0] == node:
                    next_edge_list.append(edge)
                    stack.append(edge[1])
                elif edge[1] == node:
                    next_edge_list.append(edge[::-1])
                    stack.append(edge[0])
                else:
                    self.edge_list.append(edge)

        self.edge_list = np.array(next_edge_list)


def run_geodesic(points, nodes):
    # raise NotImplementedError
    # visualize_examples.vis(points, nodes=nodes)
    gs = GeodesicSkeleton(points, nodes)
    gs.compute_geodesics()
    gs.construct_graph()
    # visualize_examples.vis(points, nodes=gs.nodes, edges=np.array(gs.edge_list))

    return gs.nodes, np.array(gs.edge_list), None