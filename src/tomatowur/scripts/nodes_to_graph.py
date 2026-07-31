import numpy as np

import networkx as nx
from tomatowur.scripts.utils_skeletonisation import undirected2directed


def nodes_to_graph(points=None, nodes=None, root_idx=None, method = "mst", **kwargs):
        """
        Converts a set of nodes (and optionally points) into a graph using the specified method.

        Parameters:
            points (np.ndarray, optional): An (N, 3) array representing the point cloud. Used for methods that require point cloud data.
            nodes (np.ndarray): An (M, 3) array with node positions.
            root_idx (int, optional): Index of the root node in the points array. If provided, the closest node to this point is set as root.
            method (str): Method to use for graph construction. Options are "mst", "geodesic", or "xu".
            **kwargs: Additional method-specific arguments.

        Returns:
            nodes (np.ndarray): Array of node positions (possibly reordered).
            edges (np.ndarray): Array of edges (pairs of node indices).
            edge_type (Any or None): Method-specific edge type information, or None.
        """
    
        ## if root_idx is not None add root node to nodes
        if root_idx is not None:
            root_nodes_idx = np.linalg.norm(nodes - points[root_idx], axis=1).argmin()
            nodes = np.array([nodes[root_nodes_idx]] + [node for idx, node in enumerate(nodes) if idx != root_nodes_idx])
        else:
            print("root_idx is None!!!")

        ## make sure that nodes are unique, MST failes for float32 nodes...
        _, indices = np.unique(nodes.astype(np.float16), axis=0, return_index=True)
        indices.sort()
        nodes = nodes[indices]

        if method == "geodesic":
            if points is None:
                 raise ValueError(f"In nodes_to_graph cannot run {method} method with points=None")
            if nodes is None:
                raise ValueError(f"In nodes_to_graph cannot run {method} method with nodes=None")
            from tomatowur.skeletonisation_methods.geoskel.geodesic_skeleton import run_geodesic
            nodes, edges, edge_type = run_geodesic(points, nodes)
            return nodes, edges, None


        elif method == "xu":
            if nodes is None:
                raise ValueError(f"In nodes_to_graph cannot run {method} method with nodes=None")
            
            from tomatowur.skeletonisation_methods.plantscan3d import xu
            nodes, edges, edge_type = xu.xu_method_connect_points(nodes, kwargs["parents"], kwargs["mtg"])
            return nodes, edges, edge_type
        
        elif method == "mst":
            if nodes is None:
                raise ValueError(f"In nodes_to_graph cannot run {method} method with nodes=None")
            import mistree as mist
            
            k_neighbours=10
            mst = mist.GetMST(x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2])
            degree, edge_length, branch_length, branch_shape, edge_index, branch_index = mst.get_stats(
                include_index=True, k_neighbours=k_neighbours)
            # Check connectivity
            
            G = nx.Graph()
            G.add_edges_from(edge_index.T)
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len)

            iteration = 1
            max_iterations = 10
            edge_index = edge_index ## shape 2, 387
            
            while not nx.is_connected(G) and iteration<max_iterations:
                k_neighbours += 10
                mst = mist.GetMST(x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2])
                degree, edge_length, branch_length, branch_shape, edge_index, branch_index = mst.get_stats(
                    include_index=True, k_neighbours=k_neighbours)
                # Check connectivity
                G = nx.Graph()
                G.add_edges_from(edge_index.T)

                components = list(nx.connected_components(G))
                new_largest_component = max(components, key=len)
                if len(new_largest_component)>len(largest_component):
                    largest_component = new_largest_component
                    edge_index = np.array(G.subgraph(largest_component).copy().edges()).T
                # import visualize_examples as ve
                # ve.vis(pc=np.unique(nodes.astype(np.float16),axis=0), nodes=nodes, edges = edge_index.T)

                iteration+=1
            if not nx.is_connected(G):
                # Check connectivity
                # G = nx.Graph()
                # G.add_edges_from(edge_index.T)


                # # Select the largest connected component
                # components = list(nx.connected_components(G))
                # largest_component = max(components, key=len)
                # G = G.subgraph(largest_component).copy()
                # edge_index = np.array(G.edges()).T
                print("Graph is not connected!, selected largest subgraph")
                # raise NotImplementedError

            edges = undirected2directed(edge_index.T, root=edge_index.min())

            return nodes, edges, None
        else:
            raise NotImplementedError
        
