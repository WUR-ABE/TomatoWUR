# from openalea.plantgl.all import *
import numpy as np
from collections import defaultdict
# from mtg import MTG
from skeletonisation_methods.plantscan3d.mtg import MTG



def determine_children(parents):
    children = defaultdict(list)
    root = None
    for pid, parent in enumerate(parents):
        if pid == parent:
            root = pid
        else:
            children[parent].append(pid)
    return children, root


def subtrees_size(children, root):
    size = {}

    def compute_size(node):
        if node in size:
            return size[node]
        total_size = 1
        for child in children[node]:
            total_size += compute_size(child)
        size[node] = total_size
        return total_size

    compute_size(root)
    return size


def direction(v):
    return v / np.linalg.norm(v)


def initialize_mtg(root, nodelabel='N'):
    mtg = MTG()
    # plantroot = mtg.root
    # branchroot = mtg.add_component(plantroot, label='P')
    # noderoot = mtg.add_component(branchroot, label=nodelabel)
    noderoot=0 # added code Bart
    mtg.property('position')[noderoot] = root
    mtg.property('radius')[noderoot] = None
    assert len(mtg.property('position')) == 1
    return mtg


def mtg2pgltree(mtg):
    vertices = mtg.vertices(scale=mtg.max_scale())
    vertex2node = dict([(vid, i) for i, vid in enumerate(vertices)])
    positions = mtg.property('position')
    nodes = [positions[vid] for vid in vertices]
    parents = [vertex2node[mtg.parent(vid)] if mtg.parent(vid) else vertex2node[vid] for vid in vertices]
    return nodes, parents, vertex2node

def pgltree2mtg(mtg, startfrom, parents, positions, radii=None, filter_short_branch=False, angle_between_trunk_and_lateral=60, nodelabel='N'):
    from math import degrees, acos
    # Creates a mtg by determining edge_type between parents and their children

    ## edge_type = "+", means a new child,
    ## edge_type = "<" just a new node


    rootpos = np.array(mtg.property('position')[startfrom])
    if np.linalg.norm(positions[0] - rootpos) > 1e-3:
        if len(mtg.children(startfrom)) > 0:
            edge_type = '+'
        else:
            edge_type = '<'
        startfrom = mtg.add_child(parent=startfrom, position=positions[0], label=nodelabel, edge_type=edge_type)

    children, root = determine_children(parents)
    clength = subtrees_size(children, root)

    mchildren = list(children[root])
    npositions = mtg.property('position')
    removed = []
    if len(mchildren) >= 2 and filter_short_branch:
        mchildren = [c for c in mchildren if len(children[c]) > 0]
        if len(mchildren) != len(children[root]):
            removed = list(set(children[root]) - set(mchildren))

    mchildren.sort(key=lambda x: -clength[x])
    # create list of tuples with (new_id, parent, edgetype)
    toprocess = [(c, startfrom, '<' if i == 0 else '+') for i, c in enumerate(mchildren)]
    while len(toprocess) > 0:
        nid, parent, edge_type = toprocess.pop(0)
        pos = positions[nid]
        parameters = dict(parent=parent, label=nodelabel, edge_type=edge_type, position=pos)
        if radii:
            parameters['radius'] = radii[nid]
        mtgnode = mtg.add_child(**parameters)
        mchildren = list(children[nid])
        if len(mchildren) > 0:
            if len(mchildren) >= 2 and filter_short_branch:
                mchildren = [c for c in mchildren if len(children[c]) > 0]
                if len(mchildren) != len(children[nid]):
                    removed = list(set(children[nid]) - set(mchildren))
            if len(mchildren) > 0:
                mchildren.sort(key=lambda x: -clength[x])
                first_edge_type = '<'
                # langle = degrees(acos(np.dot(direction(pos - npositions[parent]), direction(positions[mchildren[0]] - pos))))
                # round to bug fux fix for np.dot==1.00000002
                langle = degrees(acos(round(np.dot(direction(pos - npositions[parent]), direction(positions[mchildren[0]] - pos)),3)))

                if langle > angle_between_trunk_and_lateral: 
                    first_edge_type = '+'
                edges_types = [first_edge_type] + ['+' for i in range(len(mchildren) - 1)]
                toprocess += [(c, mtgnode, e) for c, e in zip(mchildren, edges_types)]
    print('Remove short nodes ', ','.join(map(str, removed)))
    return mtg


def nodelist2mtg(nodes, edges, edge_types=None, radius=None):
    mtg = MTG()

    counter= 0
    mtg.root = 0
    ii = mtg.add_component(mtg.root, label=str(counter), XX=nodes[0][0], YY=nodes[0][1], ZZ=nodes[0][2], position=nodes[0])
    # ii = mtg.add_child(0, child=1, label=str(counter)
    counter+=1
    if edges.min() == 0:
        edges = edges+[1,1]
    for x, (node, edge) in enumerate(zip(nodes[1:], edges)):
        parent = edge[0]
        child = edge[1]
        XX, YY, ZZ = node
        if edge_types is not None:
            edge_type = str(edge_types[x])
        else:
            edge_type = '<'
        # edge_type = '<'
        
        if radius:
            ii = mtg.add_child(parent, child=child, label=str(counter), edge_type=edge_type, XX=XX, YY=YY, ZZ=ZZ, position=node, radius=radius[x])
        else:
            ii = mtg.add_child(parent, child=child, label=str(counter), edge_type=edge_type, XX=XX, YY=YY, ZZ=ZZ, position=node)
        counter+=1

    return mtg


def filter_mtg(mtg):
    """
    Custom made function to filter the MTG based on the node orders and edge types.
    """


    node_orders = determine_node_order(mtg)
    a = mtg.copy()

    for x in mtg.vertices():
        if x==0:
            continue
        # if mtg[x].get("edge_type", "+")=="<":
        remove = True
        for child in mtg.children(x):
            if node_orders.get(child) in {0, 1, 2} and mtg[child]["edge_type"]=="+":
                remove = False
            # elif mtg[child]["edge_type"]=="+":
            #     remove = False
        if remove:
            # mtg._remove_vertex_properties(x)
            mtg.remove_vertex(x, reparent_child=True)
    mtg2 = mtg.sub_tree(0, copy=True)
    nodes, edges, edge_types = mtg2_nodes_edges_edge_types(mtg2)

    # nodes, edges, edge_types = mtg2_nodes_edges_edge_types(mtg)
    mtg = nodelist2mtg(nodes, edges, edge_types)
    parents = np.zeros(nodes.shape[0], dtype=int)
    parents[1:] = np.array(mtg.edges())[1:,0]
    pgltree2mtg(mtg, 1, parents, nodes, None, filter_short_branch=False, angle_between_trunk_and_lateral=60)


    # nodes, parents, vertex2node = mtg2pgltree(mtg2)
    # mtg3 = initialize_mtg(root=root)
    # pgltree2mtg(mtg3, startfrom, parents, nodes, None, filter_short_branch, angle_between_trunk_and_lateral)
    # mtg = mtg3.copy()

    return mtg


def mtg2_nodes_edges_edge_types(mtg):
    nodes = np.array(list(mtg.property("position").values()))
    edges = np.array(mtg.edges()[1:])
    # edges = edges - edges.min()
    edge_types = np.array(list(mtg.property("edge_type").values()))
    return nodes, edges, edge_types


def determine_node_order(mtg):
    node_orders = {}
    counter = 0
    # Initialize with the root node(s)
    for root in mtg.roots():
        node_orders[root] = 0  # Root node is order 0

        # Traverse the MTG to calculate orders
        stack = [root]
        while stack:
            parent = stack.pop()
            parent_order = node_orders[parent]

            for child in mtg.children(parent):
                edge_type = mtg.edge_type(child)
                # Determine the child's order based on the edge type
                if edge_type == "<":
                    node_orders[child] = parent_order  # Same order as parent
                elif edge_type == "+":
                    node_orders[child] = parent_order + 1  # Increase order by 1 for branches
                print(counter:=counter+1)
                stack.append(child)
    return node_orders


def saveNodeList(mtg, fname="nodelist.txt"):
    stream = open(fname, 'w')
    position = mtg.property('position')
    radius = mtg.property('radius')
    stream.write("# automatically exported mtg\n")
    stream.write("# vid parentid edgetype XX YY ZZ Radius\n")
    stream.write(str(mtg.nb_vertices(scale=mtg.max_scale())) + '\n')
    for vid in mtg.vertices(scale=mtg.max_scale()):
        p = position[vid]
        if isinstance(p, np.ndarray):
            stream.write(str(vid) + '\t' + ('' if mtg.parent(vid) is None else str(mtg.parent(vid))) + '\t' + str(mtg.edge_type(vid)) + '\t' + str(p[0]) + '\t' + str(p[1]) + '\t' + str(p[2]) + '\t' + str(radius.get(vid, '')) + '\n')
        else:
            stream.write(str(vid) + '\t' + ('' if mtg.parent(vid) is None else str(mtg.parent(vid))) + '\t' + str(mtg.edge_type(vid)) + '\t' + str(p.x) + '\t' + str(p.y) + '\t' + str(p.z) + '\t' + str(radius.get(vid, '')) + '\n')
    stream.close()

def gaussian_weight(x, var):
    from math import exp, sqrt, pi
    return exp(-x ** 2 / (2 * var)) / sqrt(2 * pi * var * var)


def gaussian_filter(mtg, propname, considerapicalonly=True):
    prop = mtg.property(propname)
    nprop = dict()
    gw0 = gaussian_weight(0, 1)
    gw1 = gaussian_weight(1, 1)
    for vid, value in list(prop.items()):
        nvalues = [value * gw0]
        parent = mtg.parent(vid)
        if parent and parent in prop:
            nvalues.append(prop[parent] * gw1)
        children = mtg.children(vid)
        if considerapicalonly: children = [child for child in children if mtg.edge_type(child) == '<']
        for child in children:
            if child in prop:
                nvalues.append(prop[child] * gw1)

        nvalue = sum(nvalues[1:], nvalues[0]) / sum([gw0 + (len(nvalues) - 1) * gw1])
        nprop[vid] = nvalue

    prop.update(nprop)


# def threshold_filter(mtg, propname):
#     from openalea.mtg.traversal import iter_mtg2

#     prop = mtg.property(propname)
#     nprop = dict()
#     for vid in iter_mtg2(mtg, mtg.root):
#         if vid in prop:
#             parent = mtg.parent(vid)
#             if parent and parent in prop:
#                 pvalue = nprop.get(parent, prop[parent])
#                 if pvalue < prop[vid]:
#                     nprop[vid] = pvalue

#     prop.update(nprop)


# def get_first_param_value(mtg, propname):
#     from openalea.mtg.traversal import iter_mtg2
#     scale = mtg.max_scale()

#     prop = mtg.property(propname)
#     for vid in iter_mtg2(mtg, mtg.root):
#         if vid in prop and mtg.scale(vid) == scale and not prop[vid] is None:
#             return prop[vid]


# def pipemodel(mtg, rootradius, leafradius, root=None):
#     from math import log
#     from openalea.mtg.traversal import post_order2
#     if root is None:
#         roots = mtg.roots(scale=mtg.max_scale())
#         assert len(roots) == 1
#         root = roots[0]

#     vertices = list(post_order2(mtg, root))

#     leaves = [vid for vid in vertices if len(mtg.children(vid)) == 0]
#     # pipeexponent = log(len(leaves)) / (log(rootradius) - log(leafradius))
#     # print pipeexponent
#     # invpipeexponent = 1./ pipeexponent

#     radiusprop = dict()
#     for vid in leaves:  radiusprop[vid] = leafradius

#     nbelems = dict()
#     for vid in leaves:  nbelems[vid] = 1
#     for vid in vertices:
#         if not vid in nbelems:
#             nbelems[vid] = sum([nbelems[child] for child in mtg.children(vid)]) + 1

#     print(root, nbelems[root])

#     # pipeexponent = log(nbelems[root]) / (log(rootradius) - log(leafradius))
#     pipeexponent = (log(rootradius) - log(leafradius)) / log(nbelems[root])
#     print(pipeexponent)
#     invpipeexponent = 1. / pipeexponent

#     for vid in vertices:
#         if not vid in radiusprop:
#             radiusprop[vid] = leafradius * (nbelems[vid] ** pipeexponent)

#     # for vid in post_order2(mtg, root):
#     #    if not vid in radiusprop:
#     #        rad = pow(sum([pow(radiusprop[child], pipeexponent) for child in mtg.children(vid)]), invpipeexponent)
#     #        radiusprop[vid] = rad

#     return radiusprop
