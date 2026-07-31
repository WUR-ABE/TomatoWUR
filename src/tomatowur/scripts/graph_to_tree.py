import numpy as np
import math
import networkx as nx


def direction(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# def get_single_edge_type_angle_based(graph: nx, node_id=0, position_key="pos", edge_type_key="edge_type",
#     angle_between_trunk_and_lateral=60):

#     node_id_pos = graph.nodes[node_id][position_key]
#     children = list(graph.successors(node_id))

#     parent_node_id = list(graph.predecessors(node_id))
#     if len(parent_node_id)==0: ## root, because it does not have any parents
#         parent_node_id_pos = node_id_pos - np.array([0, 0, .1])
#     else:
#         parent_node_id_pos = graph.nodes[parent_node_id[0]][position_key]

#     edge_types = {}
#     langles = []
#     langles_child = []
#     for child in children:
#         child_pos = graph.nodes[child][position_key]

#         first_edge_type = '<'
#         langle = math.degrees(math.acos(
#             round(np.dot(direction(node_id_pos - parent_node_id_pos), direction(child_pos - node_id_pos)),1))) # round for bug fix

#         if langle > angle_between_trunk_and_lateral: 
#             first_edge_type = '+'
#         else:
#             langles.append(langle)
#             langles_child.append(child)

#         edge_types[child] = first_edge_type
#         # stack.append(child)

#     # if multiple angles are smaller than 60, then all other larger angles are a branch
#     if len(langles)>1:
#         index_min_langles = langles.index(min(langles))
#         del langles_child[index_min_langles]
#         for child in langles_child:
#             edge_types[child] = "+"
#         # index_min_langles = langles.index(min(langles))
#         # for child in children:
#         # edge_types[children[langles.index(max(langles))]] = "+"
    
#     for child, edge_type in edge_types.items():
#         graph.nodes[child]['edge_type'] = edge_type #edge_types[child]
#         graph.edges[node_id, child]['edge_type'] = edge_types[child]

#     return graph


def get_single_edge_type_angle_based(graph: nx, node_id=0, position_key="pos", edge_type_key="edge_type",
    angle_between_trunk_and_lateral=60, n_parents=0):

    node_id_pos = graph.nodes[node_id][position_key]
    children = list(graph.successors(node_id))

    parent_node_id = list(graph.predecessors(node_id))
    parent_poses = []
    if len(parent_node_id)==0: ## root, because it does not have any parents
        parent_node_id_pos = node_id_pos - np.array([0, 0, .1])
    else:
        parent_node_id_pos = graph.nodes[parent_node_id[0]][position_key]
        new_parent_node_id = parent_node_id[0]
        for i in range(n_parents):
            parent_parent_node_id =list(graph.predecessors(new_parent_node_id))
            if len(parent_parent_node_id)>0:
                parent_poses.append(graph.nodes[parent_parent_node_id[0]][position_key])
                new_parent_node_id = parent_parent_node_id[0]
            else:
                break

    edge_types = {}
    langles = []
    langles_child = []
    for child in children:
        child_pos = graph.nodes[child][position_key]

        first_edge_type = '<'
        langles_parents = []
        langle = math.degrees(math.acos(
            round(np.dot(direction(node_id_pos - parent_node_id_pos), direction(child_pos - node_id_pos)),1))) # round for bug fix
        langles_parents.append(langle)

        for parent_parent_pos in parent_poses:
            langle2 = math.degrees(math.acos(
            round(np.dot(direction(node_id_pos - parent_parent_pos), direction(child_pos - node_id_pos)),1))) # round for bug fix 
            langles_parents.append(langle2)
        langle = min(langles_parents)

        if langle > angle_between_trunk_and_lateral: 
            first_edge_type = '+'
        else:
            langles.append(langle)
            langles_child.append(child)

        edge_types[child] = first_edge_type
        # stack.append(child)

    # if multiple angles are smaller than 60, then all other larger angles are a branch
    if len(langles)>1:
        index_min_langles = langles.index(min(langles))
        del langles_child[index_min_langles]
        for child in langles_child:
            edge_types[child] = "+"
        # index_min_langles = langles.index(min(langles))
        # for child in children:
        # edge_types[children[langles.index(max(langles))]] = "+"
    
    for child, edge_type in edge_types.items():
        graph.nodes[child]['edge_type'] = edge_type #edge_types[child]
        graph.edges[node_id, child]['edge_type'] = edge_types[child]

    return graph

def graph_edges_to_tree(graph, root_id = 0, **kwargs):
    stack = [root_id]
    while stack:
        parent_id = stack.pop()
        graph = get_single_edge_type(graph=graph, node_id=parent_id, **kwargs)
        stack+=list(graph.successors(parent_id))
    return graph


def get_single_edge_type(graph, method="angle_based", **kwargs):
    if method=="angle_based":
        graph = get_single_edge_type_angle_based(graph, **kwargs)
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented. Supported methods: 'angle_based'.")
    return graph
