import numpy as np
from collections import defaultdict, deque
import heapq
import sys
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

from scripts import visualize_examples
from skeletonisation_methods.plantscan3d.mtgmanip import initialize_mtg, pgltree2mtg, mtg2_nodes_edges_edge_types, saveNodeList, gaussian_filter, mtg2pgltree
from skeletonisation_methods.plantscan3d.io import write_mtg, read_mtg_file



"""This is a python implentation of the Xu method for skeletonisation of point clouds. Inspired by the Plantscan3D library
"""


def skeleton_from_distance_to_root_clusters(points, root, binsize, k, connect_all_points, verbose):
    if verbose:
        print("Compute Remanian graph.")
    
    # Assuming the presence of functions analogous to the C++ code
    if True:#'PGL_WITH_ANN' in globals():
        # remaniangraph = indices = [nxk], for each point return k closests point baed on 
        # nearest neighbour
        remaniangraph = k_closest_points_from_ann(points, k, connect_all_points)
    else:
        remaniangraph = k_closest_points_from_delaunay(points, k)
    
    visualise=False
    if visualise:
        edges = np.zeros((points.shape[0]*20, 2))
        edges[:,0] = np.repeat(np.array(remaniangraph)[:,0],20)
        edges[:,1] = np.array(remaniangraph).reshape(-1)
        visualize_examples.vis(nodes=points, edges=edges)

    # if connect_all_points:
    #     if verbose:
    #         print("Connect all components of Riemanian graph.")
    #     remaniangraph = connect_all_connex_components(points, remaniangraph, verbose)
    
    if verbose:
        print("Compute distance to root.")
    # find shortest path between root and pints using k closests points
    shortest_pathes = points_dijkstra_shortest_path(points, remaniangraph, root)
    parents = shortest_pathes[0]
    distances_to_root = shortest_pathes[1]

    visualise=False
    if visualise:
        edges = np.zeros((points.shape[0], 2))
        edges[:,0] = np.arange(points.shape[0])
        edges[:,1] = parents
        # visualize_examples.vis(pc=points, nodes=points, edges=edges[edges[:,1]!=UINT32_MAX], distances=distances_to_root)
        visualize_examples.vis(pc=points, distances=distances_to_root)

    
    if verbose:
        print("Compute cluster according to distance to root.")
    group_components = quotient_points_from_adjacency_graph(binsize, points, remaniangraph, distances_to_root)
    print("Nb of groups :", len(group_components))
    visualise = False
    if visualise:
        visualize_examples.vis_components(points, group_components)

    if verbose:
        print("Compute adjacency graph of groups.")
    group_adjacencies = quotient_adjacency_graph(remaniangraph, group_components)
    
    if verbose:
        print("Compute centroid of groups.")
    group_centroid = centroids_of_groups(points, group_components)
    
    if verbose:
        print("Compute spanning tree of groups.")
    shortest_pathes = points_dijkstra_shortest_path(group_centroid, group_adjacencies, 0)
    group_parents = shortest_pathes[0]
    
    visualise = False
    if visualise:
        edges = np.zeros((group_centroid.shape[0], 2))
        edges[:,0] = np.arange(group_centroid.shape[0])
        edges[:,1] = group_parents
        visualize_examples.vis(points, group_centroid, edges)


    return group_centroid, group_parents, group_components


def k_closest_points_from_ann(points, k, connect_all_points):
    
    kdtree = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = kdtree.kneighbors(points)
    
    result = indices.tolist()
    
    # if symmetric: ##connect_all_points
    #     result = symmetrize_connections(result)
    return result




REAL_MAX = float('inf')
UINT32_MAX = sys.maxsize
class PointDistance:
    def __init__(self, points):
        self.points = points

    def __call__(self, a, b):
        return np.linalg.norm(self.points[a] - self.points[b])

class PowerPointDistance:
    def __init__(self, points, powerdist):
        self.points = points
        self.powerdist = powerdist

    def __call__(self, a, b):
        return np.linalg.norm(self.points[a] - self.points[b]) ** self.powerdist

class PointDistanceXY:
    def __init__(self, points):
        self.points = points

    def __call__(self, a, b):
        temp = self.points[a] - self.points[b]
        temp[1:] = temp[1]*2
        return np.linalg.norm(temp)

def points_dijkstra_shortest_path(points, adjacencies, root, powerdist=1):
    if powerdist == 1:
        pdevaluator = PointDistance(points)
    elif powerdist == 100:
        pdevaluator = PointDistanceXY(points)
    else:
        pdevaluator = PowerPointDistance(points, powerdist)

    return dijkstra_shortest_path(adjacencies, root, pdevaluator)


def dijkstra_shortest_path(connections, root, distevaluator):
    """applies dijkstra algorithm, simplified connects each node to another node using the shortest distance from the root node.
    Returns:
    parents = each node with its parent node
    distance = distance with respect to root node"""
    nbnodes = len(connections)
    distances = [REAL_MAX] * nbnodes
    distances[root] = 0

    parents = [UINT32_MAX] * nbnodes
    parents[root] = root

    black, grey, white = 0, 1, 2
    colored = [black] * nbnodes

    class NodeCompare:
        def __init__(self, distances):
            self.distances = distances

        def __call__(self, node):
            return self.distances[node]

    Q = []
    heapq.heappush(Q, (0, root))

    while Q:
        current_distance, current = heapq.heappop(Q) #get first from Q
        
        if colored[current] == white:
            continue
        
        colored[current] = white
        for v in connections[current]:
            weight_uv = distevaluator(current, v)
            distance = weight_uv + distances[current]
            if distance < distances[v]: # if from all distances to V is faster, than current node is the fastest route to V
                distances[v] = distance
                parents[v] = current
                if colored[v] == black:
                    colored[v] = grey
                    heapq.heappush(Q, (distance, v))
                elif colored[v] == grey:
                    heapq.heappush(Q, (distance, v))

    return (parents, distances)


def get_sorted_element_order(array):
    return np.argsort(array)


def quotient_points_from_adjacency_graph(binsize: float, points: np.ndarray, adjacencies: list, distance_to_root: list):
    """
    Groups points into clusters based on their distances to a root point and their adjacency relationships. 
    add a new group if point is not yet in pointmap although it is within the same bin.s
    Args:
        binsize (float): The size of each bin used to group points based on their distance to the root.
        points (array): A Nx3 array of points to be grouped.
        adjacencies (list): A list of list with size N, and M number of adjacency points. [[1,4,2,4,5], ], so in this example point 0 is connected to 1,4 etc
        distance_to_root (list): A list of size [N] of distances from each point to the root point.
    Returns:
        list: A list of groups, where each group is a list of points that are clustered together.
    """


    nbpoints = len(points)
    sortedpoints = get_sorted_element_order(distance_to_root) ## indexes
    nextlimit = 0
    currentlimit = nextlimit
    currentbinlimit = 0
    nextbinlimit = binsize
    groups = []

    while nextlimit < nbpoints:
        ## get all points that are within nextbinlimit
        while nextlimit < nbpoints and distance_to_root[sortedpoints[nextlimit]] < nextbinlimit:
            nextlimit += 1

        if nextlimit == currentlimit:
            if nextlimit < nbpoints and distance_to_root[sortedpoints[nextlimit]] == REAL_MAX:
                break
            else:
                nextbinlimit += binsize
                continue

        pointmap = set()
        nbgroupincluster = 0

        # for every within a certain distance, get all connectoins to all points. Add a new group, if point is within sorted points but not yet added in pointmap
        for point in sortedpoints[currentlimit:nextlimit]:
            if point not in pointmap:
                pointmap.add(point)
                newgroup = [point]
                ## get all conections of current point
                stack = deque([adjacencies[point]])
                while stack:
                    curneighborhood = stack.pop()
                    for neighbor in curneighborhood:
                        pdist = distance_to_root[neighbor]
                        if currentbinlimit <= pdist < nextbinlimit and neighbor not in pointmap:
                            pointmap.add(neighbor)
                            newgroup.append(neighbor)
                            stack.append(adjacencies[neighbor])

                assert len(newgroup) > 0
                groups.append(newgroup)
                nbgroupincluster += 1
                # if len(groups)==82:
                #     print("debug")

        assert nbgroupincluster > 0
        currentlimit = nextlimit
        currentbinlimit = nextbinlimit
        nextbinlimit += binsize

        if currentlimit < nbpoints and distance_to_root[sortedpoints[currentlimit]] == REAL_MAX:
            break

    return groups


def quotient_adjacency_graph(adjacencies, groups):
    """Function to convert all adjacencies to a macroadjacencies by connecting groups.

    Args:
        adjacencies (list): list of k closest connection for every point list size=Nxk
        groups list): list with describes which groups a point belong [[group1_point_idx, group1_point_idx], [group2...]]

    Returns:
        _type_: _description_
    """
    nbpoints = len(adjacencies)
    group = [UINT32_MAX] * nbpoints  # Default group is no group
    ## assign points to group
    cgroup = 0
    for itgs in groups:
        for itg in itgs:
            group[itg] = cgroup
        cgroup += 1

    macroadjacencies = [[] for _ in range(len(groups))]

    cnode = 0
    for itn in adjacencies:
        cgroup = group[cnode] # get which group belongs to point itn
        if cgroup == UINT32_MAX:
            cnode += 1
            continue  # No group assigned to this point
        cmadjacency = macroadjacencies[cgroup] # get adjancies groyup if already assigned
        for itadgroup in itn:                   # for every connection of point itn
            adjacentgroup = group[itadgroup]    # get group of k-nearest point of itn
            assert adjacentgroup != UINT32_MAX
            if cgroup != adjacentgroup:         # if they are not the same means that points are close, but different gruop
                if adjacentgroup not in cmadjacency: # if not yet assigned make connection
                    cmadjacency.append(adjacentgroup) # this updates macroadjacencies
        cnode += 1

    return macroadjacencies


def connect_all_connex_components(points, adjacencies, verbose=False):
    nbtotalpoints = len(points)
    
    # Set of points not accessible from the root component
    nonconnected = set()

    # Connected points
    refpoints = []
    nbrefpoints = 0
    
    # Map of point ids from refpoints to original points structure
    pidmap = {}
    
    # Connections to add to connect all components
    addedconnections = []
    
    # Root to consider for next component
    next_root = 0
    
    while True:
        # Find all points accessible from next_root using Dijkstra
        parents, _ = points_dijkstra_shortest_path(points, adjacencies, next_root, powerdist=1)
        
        # Update the set of refpoints (connected points) and nonconnected points
        if next_root == 0:
            for pid, parent in enumerate(parents):
                if parent == UINT32_MAX:
                    nonconnected.add(pid)
                else:
                    refpoints.append(points[pid])
                    pidmap[nbrefpoints] = pid
                    nbrefpoints += 1
        else:
            toerase = []
            for pid in nonconnected:
                if parents[pid] != UINT32_MAX:
                    refpoints.append(points[pid])
                    pidmap[nbrefpoints] = pid
                    nbrefpoints += 1
                    toerase.append(pid)
            for pid in toerase:
                nonconnected.remove(pid)
        
        if verbose:
            print(f"\rNb points processed {nbrefpoints} ({100 * nbrefpoints / float(nbtotalpoints):.2f}%) [left: {len(nonconnected)} ({100 * len(nonconnected) / float(nbtotalpoints):.2f}%)].")
        
        if not nonconnected:
            break
        
        # Create a KDTree from the connected points
        kdtree = NearestNeighbors(n_neighbors=1).fit(refpoints)
        
        dist = REAL_MAX
        connection = None
        
        # Find the shortest connection between a nonconnected point and a connected one
        for pid in nonconnected:
            distances, indices = kdtree.kneighbors([points[pid]], 1)
            refpoint_index = indices[0][0]
            refpoint = pidmap[refpoint_index]
            newdist = np.linalg.norm(points[refpoint] - points[pid])
            if newdist < dist:
                connection = (refpoint, pid)
                dist = newdist
        
        assert connection is not None
        
        # Add the found connection into addedconnections and consider nonconnected point as next_root
        addedconnections.append(connection)
        next_root = connection[1]
    
    # Copy adjacencies and update it with addedconnections
    newadjacencies = [adj.copy() for adj in adjacencies]
    for first, second in addedconnections:
        newadjacencies[first].append(second)
        newadjacencies[second].append(first)
    
    return newadjacencies


def delaunay_point_connection(points):
    """
    Create connections between points using Delaunay triangulation.
    """
    # Perform Delaunay triangulation
    tri = Delaunay(points)
    
    # Initialize an adjacency list to store connections
    adjacency_list = defaultdict(set)
    
    # Iterate through the simplices (tetrahedrons in 3D)
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                # Add connections (both directions)
                adjacency_list[simplex[i]].add(simplex[j])
                adjacency_list[simplex[j]].add(simplex[i])
    
    # Convert adjacency list to a list of lists
    result = [list(adjacency_list[i]) for i in range(len(points))]
    
    return result


def k_closest_points_from_delaunay(points, k):
    """
    Computes the k-nearest neighbors for each point using Delaunay triangulation.
    """
    points = np.array(points)
    res = delaunay_point_connection(points)
    filteredres = []
    
    for pointid, neighbors in enumerate(res):
        if len(neighbors) <= k:
            filteredres.append(neighbors)
        else:
            neighbors.sort(key=lambda idx: np.linalg.norm(points[pointid] - points[idx]))
            filteredres.append(neighbors[:k])
    
    return filteredres

from sklearn.decomposition import PCA

def centroid_of_group(points, group):
    group_points = points[group]
    # return np.mean(group_points, axis=0)
    return np.median(group_points, axis=0)




def centroids_of_groups(points, groups):
    ## is this the reason why Xu method makes mistakes???
    result = np.zeros((len(groups), points.shape[1]))
    for cgroup, itgs in enumerate(groups):
        result[cgroup] = centroid_of_group(points, itgs)
    return result




def xu_method(points, root_idx=0, binratio=10, nearest_neighbour=20, colors=None, vis=False):
    np.random.seed(0)  # For reproducibility
    mtg = initialize_mtg(root=points[root_idx])
    
    # binratio = 20
    mini, maxi = np.min(points[:, 2]), np.max(points[:, 2])
    zdist = maxi - mini
    binlength = zdist / binratio
    # binsize = 0.1
    # k = 20
    connect_all_points = False
    verbose = True
    
    positions, parents, pointcomponents = skeleton_from_distance_to_root_clusters(points, root_idx, binlength, nearest_neighbour, connect_all_points, verbose)
    # xu_method_connect_points(positions, parents, mtg)
    return positions, parents, mtg




def xu_method_connect_points(positions, parents, mtg):

    startfrom=0
    filter_short_branch = False
    angle_between_trunk_and_lateral = 60
    pgltree2mtg(mtg, startfrom, parents, positions, None, filter_short_branch, angle_between_trunk_and_lateral)

    # mtg = mtgmanip.filter_mtg(mtg)
    # gaussian_filter(mtg, "position")

    # export_mtg(mtg, "mtg.mtg")
    # g = read_mtg_file('mtg.mtg')
    # Write the result into a file example.mtg

    nodes, edges, edge_types = mtg2_nodes_edges_edge_types(mtg)
 
    # edges = edges- [edges[0][0],edges[0][0]]
    # if vis:
        # visualize_examples.vis(points, nodes=nodes[filtered_nodes]) 
        # visualize_examples.vis(points, nodes=nodes, edges=np.asarray(edges))
        # visualize_examples.vis(points, nodes=nodes, edges=edges, edges_type=edge_type, root_idx=mtg.root)
        # visualize_examples.vis(points, nodes=nodes, edges=edges, edges_type=edge_types, root_idx=mtg.root)
        # Capture and save the frame as an image
        # frame_files.append(frame_filename)
    return nodes, edges, edge_types



def export_mtg(mtg, file_name):
    # Export all the properties defined in `g`.
    # We consider that all the properties are real numbers.

    # properties = [(p, 'FLOAT') for p in mtg.property_names() if p not in ['edge_type', 'index', 'label']]
    properties = [("positoin", "FLOAT")] #, ("edge_type", "STRING"), ("index", "INTEGER"), ("label", "STRING")]
    mtg_lines = write_mtg(mtg, properties)

    # Write the result into a file example.mtg
    f = open(file_name, 'w')
    f.write(mtg_lines)
    f.close()

def convert_skelet2tree(points):
    startfrom=0
    filter_short_branch = False
    angle_between_trunk_and_lateral = 60
    pgltree2mtg(mtg, startfrom, parents, positions, None, filter_short_branch, angle_between_trunk_and_lateral)

# # Example usage:
# points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
# # connections = [[1, 2], [0, 2], [0, 1]]
# root = 0
# # distance_evaluator = PointDistance(points)
# remaniangraph = k_closest_points_from_ann(points, k = 20, connect_all_points=False)

# shortest_pathes = points_dijkstra_shortest_path(points, remaniangraph, root, powerdist=0)
# parents = shortest_pathes[0]
# distances_to_root = shortest_pathes[1]
# print("Parents:", parents)
# print("Distances:", distances_to_root)



# # adjacencies = [[1, 2], [0, 2], [0, 1]]
# distance_to_root = [0, 1, 2]
# binsize = 1
# group_components = quotient_points_from_adjacency_graph(binsize, points, remaniangraph, distance_to_root)
# print("Groups:", group_components)
# group_adjacencies = quotient_adjacency_graph(remaniangraph, group_components)
# print("group_adjacencies:", group_adjacencies)

# group_centroid = centroids_of_groups(points, group_components)
# print("Centroid", group_centroid)

# shortest_pathes = points_dijkstra_shortest_path(group_centroid, group_adjacencies, powerdist=0)
# group_parents = shortest_pathes[0]



    

# Example Usage
if __name__ == "__main__":
    import pandas as pd
    xyz = pd.read_csv("aribido_subsampled.xyz", delimiter = " ", names=["x", "y", "z", "d"])
    xyz = pd.read_csv("tomato.csv", delimiter = ",", names=["x", "y", "z"])

    points = xyz[["x", "y", "z"]].values

    xu_method(points, binratio=40, vis=True)
    exit()
    import potpourri3d as pp3d
    import numpy as np
    import matplotlib.pyplot as plt
    point_cloud = points
    # Step 1: Create a point cloud processor
    solver = pp3d.PointCloudHeatSolver(point_cloud, 2)
    root_idx = findBottomCenterRoot(points)

    geodesic_distance = solver.compute_distance(root_idx)
    visualize_examples.vis_distance(points, geodesic_distance)



# https://github.com/openalea/plantgl/blob/d940c17942089f6f057abb6d73515a0808d26e6c/src/cpp/plantgl/algo/base/pointmanipulation.cpp#L92
# https://github.com/openalea/plantgl/blob/d940c17942089f6f057abb6d73515a0808d26e6c/src/cpp/plantgl/algo/base/pointmanipulation.cpp#L2119
# https://github.com/openalea/plantgl/blob/master/src/cpp/plantgl/algo/base/dijkstra.h#L321
# https://github.com/openalea/plantgl/blob/d940c17942089f6f057abb6d73515a0808d26e6c/src/cpp/plantgl/algo/base/pointmanipulation.cpp#L2382
