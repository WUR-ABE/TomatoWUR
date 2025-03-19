# -*- python -*-
# -*- coding: utf-8 -*-
#
#       OpenAlea.mtg
#
#       Copyright 2008-2009 INRIA - CIRAD - INRA  
#
#       File author(s): Christophe Pradal <christophe.pradal.at.cirad.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
# 
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
#
################################################################################
"""Implementation of a set of algorithms for the MTG datastructure"""



__docformat__ = "restructuredtext"
from skeletonisation_methods.plantscan3d import traversal
# import traversal

# try:
# from tree import InvalidVertex
from skeletonisation_methods.plantscan3d.tree import InvalidVertex

# except ImportError:
#     from openalea.container.tree import InvalidVertex


def ancestors(g, vid, **kwds):
    """ Return the vertices from vid to the root. 

    :Parameters:
        - `g`: a tree or an MTG
        - `vid`: a vertex id which belongs to `g` 

    :Returns: 
        an iterator from `vid` to the root of the tree.

    .. seealso: :func:`aml.Ancestors`
    """
    et = kwds.get('EdgeType','*')
    rt = kwds.get('RestrictedTo', 'NoRestriction')
    ci = kwds.get('ContainedIn')

    if ci is not None:
        c_scale = g.scale(ci)

    v = vid

    while v is not None:

        if rt == 'SameComplex':
            if g.complex(v) != g.complex(vid):
                break
        if ci and g.complex_at_scale(v, scale=c_scale) != ci:
            break

        yield v
        v = g.parent(v)

def path(g, vid1, vid2=None):
    """
    Compute the vertices between v1 and v2.
    If v2 is None, return the path between v1 and the root.
    Otherelse, return the path between v1 and v2.
    If the graph is oriented from v1 to v2, sign is positive.
    Else, sign is negative.
    """
    sign = 1
    v1, v2 = vid1, vid2
    if v2 is None:
        return ancestors(g,v1), sign

    l= list(ancestors(g,v2))
    try: 
        index = l.index(v1)
    except ValueError:
        l = list(ancestors(g,v1))
        try:
            index = l.index(v2)
            sign = -1
        except ValueError:
            return iter([]), 0

    return reversed(l[:index+1]), sign

def edge_type(g,v):
    return g.property('edge_type').get(v)

def topological_path(g,v1, v2=None, edge=None):
    p, sign = path(g,v1,v2)
    if sign == 0:
        return None

    if edge is None:
        return sum(1 for v in p), sign
    else: 
        return sum(1 for v in p if g.edge_type(v)==edge), sign
        
def order(g, v1, v2=None):
    return topological_path(g, v1, v2, '+')[0]

def alg_rank(g, v1, v2=None):
    p, sign = path(g,v1,v2)
    count = 0
    for v in p:
        if edge_type(g,v) == '<':
            count+=1
        else:
            break
    return count*sign

def rank(g, v1, v2=None):
    return abs(alg_rank(g,v1,v2))

def height(g, v1, v2=None):
    return topological_path(g, v1, v2 )[0]-1

def alg_order(g, v1, v2=None):
    p, s = topological_path(g, v1, v2, '+')
    if p is not None:
        return p*s


def alg_height(g, v1, v2=None):
    p, s = topological_path(g, v1, v2)
    if p is not None:
        return p*s

def father(g, vid, scale=-1, **kwds):
    """
    See aml.Father function.
    """
    edge_type = g.property('edge_type')

    et = kwds.get('EdgeType','*')
    rt = kwds.get('RestrictedTo', 'NoRestriction')
    ci = kwds.get('ContainedIn')

    current_scale = g.scale(vid)
    if scale <= 0 or scale == current_scale:
        p = g.parent(vid)
    elif scale < current_scale:
        vid = g.complex_at_scale(vid)
        p = g.parent(vid)
    else:
        vid = next(g.component_roots_at_scale_iter(vid, scale=scale))
        p = g.parent(vid)

    if et != '*':
        if edge_type.get(vid) != et:
            return None


    if rt != 'NoRestriction':
        if rt == 'SameComplex':
            if g.complex(p) != g.complex(vid):
                return None
        elif rt == 'SameAxis':
            if edge_type[vid] == '+':
                return None

    if ci is not None:
        c_scale = g.scale(ci)
        if g.complex_at_scale(vid, scale=c_scale) != g.complex_at_scale(p, scale=c_scale) != ci:
            return None
        
    return p

def successor(g, vid, **kwds):
    """
    TODO: see aml.Successor doc string.
    """
    edge_type = g.property('edge_type')

    rt = kwds.get('RestrictedTo', 'NoRestriction')
    ci = kwds.get('ContainedIn')

    son = None
    for v in g.children_iter(vid):
        if edge_type.get(v) == '<':
            son = v
            break
    else:
        return None

    if rt == 'SameComplex':
        if g.complex(son) != g.complex(vid):
            return None

    if ci is not None:
        c_scale = g.scale(ci)
        if g.complex_at_scale(vid, scale=c_scale) != g.complex_at_scale(son, scale=c_scale) != ci:
            return None

    return son

    
def predecessor(g, vid, **kwds):
    return father(g, vid, **kwds)

def root(g, vid, RestrictedTo='NoRestriction', ContainedIn=None):
    """
    TODO: see aml.Root doc string.
    """
    rt = RestrictedTo
    ci = ContainedIn

    v_current = vid
    for v in ancestors(g, vid):
        if rt == 'SameComplex':
            if g.complex(v) != g.complex(vid):
                break
        v_current = v

    return v_current


def location(g, vid, **kwds):
    """TODO: see doc aml.Location.
    """
    scale = kwds.get('Scale')
    ci = kwds.get('ContainedIn')

    if not scale or scale < 0:
        scale = g.max_scale()

    current_scale = g.scale(vid)
    return father(g, vid, scale=scale, ContainedIn=ci)


def sons(g, vid, **kwds):
    """TODO: see doc aml.sons.
    """
    et = kwds.get('EdgeType','*')
    rt = kwds.get('RestrictedTo', 'NoRestriction')
    ci = kwds.get('ContainedIn')
    scale = kwds.get('Scale')

    edge_type = g.property('edge_type')

    current_scale = g.scale(vid)
    if not scale or scale < 0:
        scale = current_scale
    
    if scale < current_scale:
        vid = g.complex_at_scale(vid, scale = scale)
    elif scale > current_scale:
        vid = next(g.component_roots_at_scale_iter(vid, scale=scale))
    children = g.children_iter(vid)
    
    if et != '*':
        children = (v for v in children if edge_type[v] == et)
    if ci is not None:
        c_scale = g.scale(ci)
        children = (v for v in children if g.complex_at_scale(v) == ci)

    return list(children)

def full_ancestors(g, v1, **kwds):
    " Return the vertices from v1 to the root. "
    edge_type = g.property('edge_type')
    et = kwds.get('EdgeType','*')
    rt = kwds.get('RestrictedTo', 'NoRestriction')
    ci = kwds.get('ContainedIn')

    v = v1
    if ci is not None:
        c_scale = g.scale(ci)

    while v is not None:
        if et != '*' and edge_type.get(v) != et:
            break

        if rt == 'SameComplex':
            if g.complex(v) != g.complex(v1):
                break
        elif rt == 'SameAxis':
            if edge_type.get(v) == '+':
                break
        if ci and g.complex_at_scale(v, scale=c_scale) != ci:
            break
        yield v
        v = g.parent(v)

def axis(g, vtx_id, scale=-1, **kwds):
    """TODO: see aml doc
    """
    edge_type = g.property('edge_type')

    rt = kwds.get('RestrictedTo', 'NoRestriction')
    ci = kwds.get('ContainedIn')
    kwds['EdgeType'] = '<'

    if ci is not None:
        c_scale = g.scale(ci)

    for v in ancestors(g, vtx_id, **kwds):
        if rt == 'SameComplex':
            if g.complex(v) != g.complex(vtx_id):
                break
        if ci and g.complex_at_scale(v, scale=c_scale) != ci:
            break

        if edge_type.get(v) == '+':
            break
    return local_axis(g, v, scale=scale, **kwds)

                
def descendants(g, vtx_id, scale=-1, **kwds):
    """TODO: see aml doc
    """
    edge_type = g.property('edge_type')

    et = kwds.get('EdgeType','*')
    rt = kwds.get('RestrictedTo', 'NoRestriction')
    ci = kwds.get('ContainedIn')

    if ci is not None:
        c_scale = g.scale(ci)

    vtx_id = vertex_at_scale(g, vtx_id, scale)

    v = vtx_id

    if rt == 'SameComplex':
        complex = g.complex(vtx_id)

    def visitor(v):
        if et in ['<', '+'] and et != edge_type.get(v, et):
            return False
        if rt == 'SameComplex':
            if g.complex(v) != complex:
                return False
        elif rt == 'SameAxis' and edge_type.get(v) == '+':
                return False

        if ci and g.complex_at_scale(v, scale=c_scale) != ci:
            return False
        return True


    return traversal.pre_order2_with_filter(g, vtx_id, pre_order_filter=visitor)
        
def extremities(g, vid, **kwds):
    """ TODO see aml doc
    Implement the method more efficiently...
    """
    vertices = set(descendants(g,vid, **kwds))
    for v in vertices:
        if g.is_leaf(v):
            yield v
        else:
            for vtx in g.children_iter(v):
                if vtx in vertices:
                    break
            else:
                yield v

def local_axis(g, vtx_id, scale=-1, **kwds):
    """ 
    Return a sequence of vertices connected by '<' edges. 
    The first element of the sequence is vtx_id.
    """

    edge_type = g.property('edge_type')

    et = kwds.get('EdgeType','*')
    rt = kwds.get('RestrictedTo', 'NoRestriction')
    ci = kwds.get('ContainedIn')

    vtx_id = vertex_at_scale(g, vtx_id, scale)

    if ci is not None:
        c_scale = g.scale(ci)

    
    v = vtx_id
    while v is not None:
        yield v
        vtx = v; v = None
        for vid in g.children_iter(vtx):
            if edge_type.get(vid) == '<':
                v = vid

                if rt == 'SameComplex':
                    if g.complex(v) != g.complex(vtx_id):
                        v = None
                if ci and g.complex_at_scale(v, scale=c_scale) != ci:
                    v = None

def vertex_at_scale(g, vtx_id, scale):
    if scale <= 0:
        return vtx_id

    current_scale = g.scale(vtx_id)
    if scale < current_scale:
        vtx_id = g.complex_at_scale(vtx_id, scale=scale)
    elif scale > current_scale:
        vtx_id = next(g.component_roots_at_scale_iter(vtx_id, scale=scale))
    return vtx_id

def trunk(g, vtx_id, scale=-1, **kwds):

    rt = kwds.get('RestrictedTo', 'NoRestriction')
    ci = kwds.get('ContainedIn')
    kwds['EdgeType'] = '<'

    vtx_id = vertex_at_scale(g, vtx_id, scale)

    v = root(g, vtx_id, RestrictedTo=rt, ContainedIn=ci)

    return local_axis(g, v, scale, **kwds)

def union(g1, g2, vid1=None, vid2=None, edge_type='<'):
    """ Return the union of the MTGs g1 and g2.

    :Parameters:

        - g1, g2 (MTG) : An MTG graph
        - vid1 : the anchor vertex identid=fier that belong to `g1`
        - vid2 : the root of the sub_mtg that belong to `g2` which will be added to g1.
        - edge_type (str) : the type of the edge which will connect vid1 to vid2
    """

    v1 = vid1 if vid1 is not None else g1.root
    if v1 not in g1:
        raise InvalidVertex(v1)
    v2 = vid2 if vid2 is not None else g2.root
    if v2 not in g2:
        raise InvalidVertex(v2)

    
    g = g1.sub_mtg(g1.root)

    #n2 = g._id+1

    treeid_id = {}
    subtree = traversal.iter_mtg2(g2, v2)
    if v1 is g1.root and v2 is g2.root:
        treeid_id[v2] = v1
        next(subtree)
    else:
        v2 = next(subtree)
        v = g.add_child(v1)
        treeid_id[v2] = v
        g._add_vertex_properties(v,g2.get_vertex_property(v2))
        g.node(v).edge_type = edge_type

    for vid in subtree:
        complex_id = treeid_id[g2.complex(vid)]
        v = g.add_component(complex_id)
        treeid_id[vid] = v
        
        pid = g2.parent(vid)
        if pid is not None:
            parent = treeid_id[pid]
            v = g.add_child(parent, child=v)

        # Copy the properties
        g._add_vertex_properties(v, g2.get_vertex_property(vid))


    return g
        
def split(g, scale=1):
    """ Split at scale.
    """
    return [g.sub_mtg(vid) for vid in g.component_roots_at_scale(g.root,scale=scale)]

def orders(g, scale=-1):
    """ Compute the order of all vertices at scale `scale`.
    
    If scale == -1, the compute the order for vertices at the finer scale.
    """
    orders = {}
    if scale <= 0:
        for vid in traversal.iter_mtg2(g, g.root):
            pid = g.parent(vid)
            p_order = 0 if pid is None else orders[pid]
            orders[vid] = p_order+1 if g.edge_type(vid) == '+' else p_order
    else:
        for rid in g.roots_iter(scale=scale):
            for vid in traversal.pre_order2(g, rid):
                pid = g.parent(vid)
                p_order = 0 if pid is None else orders[pid]
                orders[vid] = p_order+1 if g.edge_type(vid) == '+' else p_order

    return orders

def heights(g, scale=-1):
    """ Compute the order of all vertices at scale `scale`.
    
    If scale == -1, the compute the order for vertices at the finer scale.
    """
    heights = {}
    if scale <= 0:
        for vid in traversal.iter_mtg2(g, g.root):
            pid = g.parent(vid)
            p_height = -1 if pid is None else heights[pid]
            heights[vid] = p_height+1
    else:
        for rid in g.roots_iter(scale=scale):
            for vid in traversal.pre_order2(g, rid):
                pid = g.parent(vid)
                p_height = -1 if pid is None else heights[pid]
                heights[vid] = p_height+1

    return heights

def lookForCommonAncestor(g, commonAncestors, currentNode):       
    while not(currentNode is None):
        for i in range(len(commonAncestors)):
            node = commonAncestors[i]
            if node == currentNode:
                for j in range(i):
                    commonAncestors.popleft()
                return 
            i += 1
        currentNode = g.parent(currentNode)


def lowestCommonAncestor(g, nodes):
    """LCA algorithm"""     
    from collections import deque
    lca = None
    if (len(nodes) > 1):
        
        firstNode = nodes[0];
        firstAncestors = deque(g.Ancestors(firstNode))
        i = 1
        while i < len(nodes) and len(firstAncestors) > 0:
            currentNode = nodes[i]
            lookForCommonAncestor(g, firstAncestors, currentNode)
            i += 1

        if len(firstAncestors) > 0:
            lca = firstAncestors[0]
    
    return lca
   