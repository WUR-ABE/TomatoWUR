from math import exp, sqrt, pi

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import networkx as nx

def gaussian_smoothing(nx_graph: nx, var0: float =.25, var1: float =.25, indices: list=[0,1], node_order_filtering: bool=True, num_children=1):
    """"Filter the graph based on node order and parent status, and updates nx_graph accordingly.
    works as follows: a current node is update with gw0, related parent and chilren are update with gw1.

    inputs:
        var0 (float): variance for node, decrease variance to increase important
        var1 (float): variance for parent and child nodes
        indices (list): indices to smooth. if [0, 1] only x and y will be smoothed
        node_order_filtering: if True only nodes with same node order will be smoothed
    returns:
        None
    """

    nprop = dict()
    gw0 = gaussian_weight(0, var0)
    gw1 = gaussian_weight(1, var1) # with x=1 lower value because 
    # self.visualise_graph()
    for node in nx_graph.nodes():
        # if node==5:
        #     print("debugging")
        value = nx_graph.nodes[node]["pos"]
        node_order = nx_graph.nodes[node]["node_order"]
        nvalues = [value * gw0]
        parent = list(nx_graph.predecessors(node))
        if parent!=[]:
            if not node_order_filtering: #(==False)
                nvalues.append(nx_graph.nodes[parent[0]]["pos"] * gw1)
            elif nx_graph.nodes[parent[0]]["node_order"] == node_order:
                nvalues.append(nx_graph.nodes[parent[0]]["pos"] * gw1)
        children = list(nx_graph.successors(node))
        if num_children is not None:
        # if all_children and node_order_filtering:
            # Get up to num_children parent nodes using ancestors and path
            parents = []
            current = node
            for _ in range(num_children):
                preds = list(nx_graph.predecessors(current))
                if preds:
                    current = preds[0]
                    parents.append(current)
                else:
                    break
            if node_order_filtering:
                parents = [child for child in parents if nx_graph.nodes[child]["node_order"] == node_order]

            parents+=list(nx.dfs_tree(nx_graph, node, len(parents)))[:len(parents)]
            children = parents
            # parents now contains up to num_children parent nodes (in order from closest to farthest)


        # children = [child for child in children if nx_graph.nodes[child]["edge_type"] == '<']
        if node_order_filtering:
            children = [child for child in children if nx_graph.nodes[child]["node_order"] == node_order]

        for child in children:
            nvalues.append(nx_graph.nodes[child]["pos"] * gw1)

        # nvalue = sum(nvalues[1:], nvalues[0]) / sum([gw0 + (len(nvalues) - 1) * gw1])
        nvalue = np.sum(nvalues, axis=0) / sum([gw0 + (len(nvalues) - 1) * gw1])

        nprop[node] = nvalue
    for node in nprop.keys():
        nx_graph.nodes[node]["pos"][indices] = nprop[node][indices]	
    
    return nx_graph
    

def gaussian_weight(x, var):
    return exp(-x ** 2 / (2 * var)) / sqrt(2 * pi * var) # corrected from Openalea


def find_closest_points(parent_points, child_points):
    # Find the closest points in the source to the target points
    tree = cKDTree(parent_points)
    distance = np.inf
    idx = 0
    for child_idx, child in enumerate(child_points):
        temp_distance, temp_idx = tree.query(child)
        if tree.query(child)[0]<distance:
            distance = temp_distance
            idx = temp_idx
    new_pose = parent_points[idx]

    return new_pose, idx, child_idx

def create_new_points(points, method = "spline", **kwargs):
    step_size = 0.005
    num_points = int(np.linalg.norm(points[-1] - points[0]) / step_size)
    extra_points_m = 0.05
    num_extra_points  = extra_points_m / step_size

    if len(points)<5 or num_points==0:
        return points, None
    
    elif method == "spline":
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Use splprep to fit a parametric spline
        tck, u = splprep([x, y, z], s=0.00001)  # `s=0` means no smoothing, use `s>0` for smoothing

        # Generate new parameter values for extrapolation
        u_new = np.linspace(0, 1.01, num_points)  # Extrapolate beyond [0, 1] range of u
        # Evaluate the spline at new parameter values
        x_new, y_new, z_new = splev(u_new, tck)
        xyz_new = np.column_stack([x_new, y_new, z_new])

        # Generate new parameter values for extrapolation
        u_new = np.linspace(-.15, -0.01, 5)  # Extrapolate beyond [0, 1] range of u
        # Evaluate the spline at new parameter values
        x_new, y_new, z_new = splev(u_new, tck)
        xyz_extrapolated = np.column_stack([x_new, y_new, z_new])

        return xyz_new, xyz_extrapolated

    elif method=="poly1d":
        raise NotImplementedError
        xyz_new = create_points_poly(points, num_points, degree=1, num_extra_points=num_extra_points)
    return xyz_new

def create_points_poly(points, num_points=100, degree=3, num_extra_points=10):
    # Extract x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Define parameter t (e.g., cumulative arc length or just indices)
    t = np.arange(len(points))
    # t = np.linspace(-0.01, 1.01, num_points)  # Extrapolate beyond [0, 1] range of u

    # t  = np.linspace(-0.1, 1.1, 100)  # Extrapolate beyond [0, 1] range of u

    # Fit polynomials for x(t), y(t), z(t)
    px = np.polyfit(t, x, degree)
    py = np.polyfit(t, y, degree)
    pz = np.polyfit(t, z, degree)

    # Create polynomial functions
    fx = np.poly1d(px)
    fy = np.poly1d(py)
    fz = np.poly1d(pz)


    # t_new = np.linspace(-.01 * (t[1]-t[0])*num_extra_points, -0.01, num_extra_points)  #
    t_new = np.linspace(t[0]-int(num_extra_points), 0, 100)  # Extrapolate beyond original range

    # Predict new points
    x_new = fx(t_new)
    y_new = fy(t_new)
    z_new = fz(t_new)
    xyz_new = np.column_stack([x_new, y_new, z_new])
    return xyz_new




# if __name__=="__main__":
# 