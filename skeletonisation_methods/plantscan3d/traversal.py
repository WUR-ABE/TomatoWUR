# -*- coding: utf-8 -*-
# -*- python -*-
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
"""Tree  and MTG Traversals"""

from collections import deque

def pre_order(tree, vtx_id, complex=None, visitor_filter=None):
    ''' 
    Traverse a tree in a prefix way.
    (root then children)

    This is a non recursive implementation.
    '''
    if complex is not None and tree.complex(vtx_id) != complex:
        return
    if visitor_filter and not visitor_filter.pre_order(vtx_id):
        return

    edge_type = tree.property('edge_type')

    # 1. select first '+' edges
    successor = []
    yield vtx_id
    for vid in tree.children_iter(vtx_id):

        if edge_type.get(vid) == '<':
            successor.append(vid)
            continue


        for node in pre_order(tree, vid, complex, visitor_filter):
            yield node



    # 2. select then '<' edges
    for vid in successor:
        for node in pre_order(tree, vid, complex, visitor_filter):
            yield node

    if visitor_filter:
        visitor_filter.post_order(vtx_id)

    
def pre_order2_with_filter(tree, vtx_id, complex=None, pre_order_filter=None, post_order_visitor=None):
    ''' 
    Same algorithm than pre_order2.
    The goal is to replace the pre_order2 implementation.

    The problem is for the pre_order filter when it is also a visitor
    '''

    edge_type = tree.property('edge_type')
    
    def order_children(vid):
        ''' Internal function to retrieve the children in a correct order:
            - Branch before successor.
        '''
        plus = []
        successor = []
        for v in tree.children_iter(vid):
            if complex is not None and tree.complex(v) != complex:
                continue
            if pre_order_filter and not pre_order_filter(v):
                continue
            
            if edge_type.get(v) == '<':
                successor.append(v)
            else:
                plus.append(v)
        
        plus.extend(successor)
        child = plus
        return list(reversed(child))


    queue = deque()
    queue.append( (vtx_id, order_children(vtx_id)) )

    yield vtx_id
    # 1. select first '+' edges

    while queue:
        vtx_id, children = queue[-1]
        if children:
            vid = children.pop()
            yield vid
            queue.append((vid,order_children(vid)))
        else:
            vtx_id, children = queue.pop()
            if post_order_visitor:
                post_order_visitor(vtx_id)


def pre_order2(tree, vtx_id, complex=None, visitor_filter=None):
    ''' 
    Traverse a tree in a prefix way.
    (root then children)

    This is an iterative implementation.
    '''
    if complex is not None and tree.complex(vtx_id) != complex:
        return

    edge_type = tree.property('edge_type')
    
    queue = deque()
    queue.append(vtx_id)

    
    # 1. select first '+' edges

    while queue:
        plus = []
        successor = []
        vtx_id = queue.pop()
        yield vtx_id

        for vid in tree.children_iter(vtx_id):
            if complex is not None and tree.complex(vid) != complex:
                continue
            if visitor_filter and not visitor_filter.pre_order(tree, vid):
                continue

            if edge_type.get(vid) == '<':
                successor.append(vid)
            else:
                plus.append(vid)



        plus.extend(successor)
        child = plus
        queue.extend(reversed(child))


def pre_order_in_scale(tree, vtx_id, visitor_filter=None):
    ''' 
    Traverse a tree in a prefix way.
    (root then children)

    This is a non recursive implementation.
    '''
    
    queue = deque()
    queue.append(vtx_id)

    
    while queue:
        vtx_id = queue.pop()
        yield vtx_id

        for vid in tree.components_iter(vtx_id):
            if visitor_filter and not visitor_filter.pre_order(vid):
                continue
            queue.append(vid)


def post_order(tree, vtx_id, complex=None, visitor_filter=None):
    ''' 
    Traverse a tree in a postfix way.
    (from leaves to root)
    This is a recursive implementation
    '''
    if complex is not None and tree.complex(vtx_id) != complex:
        return
    if visitor_filter and not visitor_filter.pre_order(vtx_id):
        return
    
    for vid in tree.children_iter(vtx_id):

        for node in post_order(tree, vid, complex, visitor_filter):
            yield node


    if visitor_filter:
        visitor_filter.post_order(vtx_id)
    yield vtx_id


def post_order2(tree, vtx_id, complex=None, pre_order_filter=None, post_order_visitor=None):
    ''' 
    Traverse a tree in a postfix way.
    (from leaves to root)

    Same algorithm than post_order.
    The goal is to replace the post_order implementation.

        
    '''

    edge_type = tree.property('edge_type')
    if pre_order_filter is None:
        pre_order_filter = lambda v: True
    if post_order_visitor is None:
        post_order_visitor = lambda x: None
    
    def order_children(vid):
        ''' Internal function to retrieve the children in a correct order:
            - Branch before successor.
        '''
        plus = []
        successor = []
        for v in tree.children(vid):
            if complex is not None and tree.complex(v) != complex:
                continue
            if not pre_order_filter(v):
                continue
            
            if edge_type.get(v) == '<':
                successor.append(v)
            else:
                plus.append(v)
        
        plus.extend(successor)
        child = plus
        return reversed(child)

    visited = set([])
    
    queue = [vtx_id]

    # 1. select first '+' edges

    while queue:

        vtx_id = queue[-1]
        for vid in order_children(vtx_id):
            if vid not in visited:
                queue.append(vid)
                break
        else: # no child or all have been visited
            post_order_visitor(vtx_id)
            yield vtx_id
            visited.add(vtx_id)
            queue.pop()
            


def traverse_tree(tree, vtx_id, visitor):
  ''' 
  Traverse a tree in a prefix or postfix way.
  
  We call a visitor for each vertex.
  This is usefull for printing, computing or storing vertices 
  in a specific order. 
  
  See boost.graph.
  '''

  yield visitor.pre_order(vtx_id)

  for v in tree.children_iter(vtx_id):
     for res in traverse_tree(tree, v, visitor):
        yield res

  yield visitor.post_order(vtx_id)


class Visitor(object):
  ''' Used during a tree traversal. '''

  def pre_order(self, vtx_id): 
     pass

  def post_order(self, vtx_id): 
     pass


# old implementation
# Problem is the traversal traverse complex before components.
# def iter_mtg(mtg, vtx_id):
#     yield vtx_id
#     for vid in mtg.components(vtx_id):
#         for node in iter_mtg(mtg, vid):
#             yield node

def iter_scale(g, vtx_id, visited):
    """ Internal method used by :func:`iter_mtg` and :func:`iter_mtg_with_visitor`.

    .. warning:: Do not use. This function may be removed in other version.
    """
    if vtx_id is not None and vtx_id not in visited:
        for v in iter_scale(g, g._complex.get(vtx_id), visited):
            yield v
        visited[vtx_id] = True
        yield vtx_id

def iter_mtg(mtg, vtx_id):
    """Iterate on an MTG by traversiong `vtx_id` and all its components.
    
    This function traverse a complex before its components 
    and a parent before its children.

    :Usage: 

    ::
        
        for vid in iter_mtg(g,g.root):
            print vid


    :Parameters:

        - `mtg`: the multi-scale graph
        - `vtx_id`: the root of the sub-mtg which is traversed.

    :Returns: iter of vid.

        Traverse all the vertices contained in the sub_mtg defined by `vtx_id`.

    .. seealso:: :func:`iter_mtg2`, :func:`iter_mtg_with_filter`, :func:`iter_mtg2_with_filter`.

    .. note:: 

        Do not use this function. Use :func:`iter_mtg2` instead. 
        If several trees belong to `vtx_id`, only the first one will be traversed.

    .. note:: 
        
        This is a recursive implementation. It can be problematic for large MTG 
        with lots of scales (e.g. >40).
    """
    visited = {vtx_id:True}
    loc = vtx_id
    yield vtx_id
    while mtg._components.get(loc):
        loc = mtg._components[loc][0]
    vtx_id = loc

    for vid in pre_order2(mtg, vtx_id):
        for node in iter_scale(mtg, vid, visited):
            yield node


def iter_mtg2(mtg, vtx_id):
    """Iterate on an MTG by traversiong `vtx_id` and all its components.
    
    This function traverse a complex before its components 
    and a parent before its children.

    :Usage: 

    ::
        
        for vid in iter_mtg2(g,g.root):
            print vid


    :Parameters:

        - `mtg`: the multi-scale graph
        - `vtx_id`: the root of the sub-mtg which is traversed.

    :Returns: iter of vid.

        Traverse all the vertices contained in the sub_mtg defined by `vtx_id`.

    .. seealso:: :func:`iter_mtg`, :func:`iter_mtg_with_filter`, :func:`iter_mtg2_with_filter`
        
    .. note:: Use this function instead of :func:`iter_mtg`
    """
    visited = {vtx_id:True}
    complex_id = vtx_id

    max_scale = mtg.max_scale()

    yield vtx_id
    for vtx_id in mtg.component_roots_at_scale_iter(complex_id, max_scale):
        for vid in pre_order2(mtg, vtx_id):
            for node in iter_scale2(mtg, vid, complex_id, visited):
                yield node

def iter_scale2(g, vtx_id, complex_id, visited):
    """ Internal method used by :func:`iter_mtg` and :func:`iter_mtg_with_visitor`.

    .. warning:: Do not use. This function may be removed in other version.
    """
    if vtx_id is not None and \
       vtx_id not in visited and \
       g.complex_at_scale(vtx_id, g.scale(complex_id)) == complex_id:
        for v in iter_scale2(g, g._complex.get(vtx_id), complex_id, visited):
            yield v
        visited[vtx_id] = True
        yield vtx_id

    
def topological_sort(g, vtx_id, visited = None):
    ''' 
    Topological sort of a directed acyclic graph.

    This is not a fully recursive implementation.
    '''
    if visited is None:
        visited = {}

    yield vtx_id
    visited[vtx_id] = True
    for vid in g.out_neighbors(vtx_id):
        if vid in visited:
            continue
        for node in topological_sort(g, vid, visited):
            yield node

def pre_order_with_filter(tree, vtx_id, pre_order_filter=None, post_order_visitor=None):
    ''' 
    Traverse a tree in a prefix way.
    (root then children)

    This is an iterative implementation.
    
    TODO: make the naming and the arguments more consistent and user friendly.
    pre_order_filter is a functor which has to return a boolean.
    If the return value is False, the vertex is not visited.
    Otherelse, some computation can be done.

    The post_order_visitor is used to execute, store, compute a function when the 
    tree rooted on the vertex has been visited.
    
    '''
    if pre_order_filter and not pre_order_filter(vtx_id):
        return

    edge_type = tree.property('edge_type')

    # 1. select first '+' edges
    successor = []
    yield vtx_id
    for vid in tree.children_iter(vtx_id):
        if edge_type.get(vid) == '<':
            successor.append(vid)
            continue

        for node in pre_order_with_filter(tree, vid, pre_order_filter, post_order_visitor):
            yield node


    # 2. select then '<' edges
    for vid in successor:
        for node in pre_order_with_filter(tree, vid, pre_order_filter, post_order_visitor):
            yield node

    if post_order_visitor:
        post_order_visitor(vtx_id)

def iter_mtg_with_filter(mtg, vtx_id, pre_order_filter= None, post_order_visitor=None):
    """Iterate on an MTG by traversiong `vtx_id` and all its components.
    
    If defined, apply the two visitor functions before and after 
    having visited all the successor of a vertex.
    
    This function traverse a complex before its components 
    and a parent before its children.

    :Usage: 

    .. code-block:: python

        def pre_order_visitor(vid): 
            print vid
            return True
        def post_order_visitor(vid):
            print vid
        for vid in iter_mtg_with_filter(g,g.root, pre_order_visitor, post_order_visitor):
            pass


    :Parameters:

        - `mtg`: the multi-scale graph
        - `vtx_id`: the root of the sub-mtg which is traversed.

    :Optional Parameters:

        - `pre_order_visitor`: function called before traversing the children or components.
            This function returns a boolean. If False, the sub-mtg rooted on the vertex is skipped.
        - `post_order_visitor` : function called after the traversal of all the children and components.

    :Returns: iter of vid.

        Traverse all the vertices contained in the sub_mtg defined by `vtx_id`.

    .. seealso:: :func:`iter_mtg`, :func:`iter_mtg2`, :func:`iter_mtg2_with_filter`
        
    .. note:: Do not use this function. Instead use :func:`iter_mtg2_with_filter`

    """

    visited = {}


    cid = mtg.complex(vtx_id)
    if cid is not None:
        visited[cid] = True

    loc = vtx_id
    while mtg._components.get(loc):

        loc = mtg._components[loc][0]

        #if pre_order_filter and not pre_order_filter(loc):
        #    return


    vtx_id = loc

    for vid in pre_order_with_filter(mtg, vtx_id, pre_order_filter, post_order_visitor):
        for node in iter_scale(mtg, vid, visited):
            yield node

def iter_mtg2_with_filter(mtg, vtx_id, pre_order_filter=None, post_order_visitor=None):
    """Iterate on an MTG by traversiong `vtx_id` and all its components.
    
    If defined, apply the two visitor functions before and after 
    having visited all the successor of a vertex.
    
    This function traverse a complex before its components 
    and a parent before its children.

    :Usage: 

    .. code-block:: python

        def pre_order_visitor(vid): 
            print vid
            return True
        def post_order_visitor(vid):
            print vid
        for vid in iter_mtg_with_filter(g,g.root, pre_order_visitor, post_order_visitor):
            pass


    :Parameters:

        - `mtg`: the multi-scale graph
        - `vtx_id`: the root of the sub-mtg which is traversed.

    :Optional Parameters:

        - `pre_order_visitor`: function called before traversing the children or components.
            This function returns a boolean. If False, the sub-mtg rooted on the vertex is skipped.
        - `post_order_visitor` : function called after the traversal of all the children and components.

    :Returns: iter of vid.

        Traverse all the vertices contained in the sub_mtg defined by `vtx_id`.

    .. seealso:: :func:`iter_mtg`, :func:`iter_mtg2`, :func:`iter_mtg2_with_filter`
        
    .. note:: Use this function instead of :func:`iter_mtg_with_filter`

    """
    if pre_order_filter is None:
        pre_order_filter = lambda v: True
    if post_order_visitor is None:
        post_order_visitor = lambda x: None

    visited = {vtx_id:True}
    complex_id = vtx_id

    max_scale = mtg.max_scale()

    queue = [] #cpl

    pre_order_filter(vtx_id)#cpl
    yield vtx_id

    queue.append(vtx_id) #cpl

    for vtx_id in mtg.component_roots_at_scale_iter(complex_id, max_scale):
        for vid in pre_order2_with_filter(mtg, vtx_id, 
                                          pre_order_filter=None, 
                                          post_order_visitor=post_order_visitor,
                                          complex=None):
            for node in iter_scale2(mtg, vid, complex_id, visited):

                scale = mtg.scale(node)
                if scale != max_scale:
                    # we need to manage the right order for the post order visitor
                    prev_node = queue[-1]
                    prev_scale = mtg.scale(prev_node)

                    if prev_scale < scale:
                        # Nothing to do
                        queue.append(node)
                    elif prev_scale == scale:
                        # Test if the two vertices are connected
                        if mtg.parent(node) != prev_node:
                            post_order_visitor(prev_node)
                            queue[-1] = node
                        else:
                            queue.append(node)
                    else: # prev_scale > scale
                        prev_node = queue.pop()
                        while mtg.scale(prev_node) > scale:
                            post_order_visitor(prev_node)
                            prev_node = queue.pop()

                        if mtg.parent(node) != prev_node:
                            post_order_visitor(prev_node)
                            queue.append(node)
                        else:
                            queue.append(node)

                    
                pre_order_filter(node)
                yield node

        while len(queue) > 1:
            node = queue.pop()
            post_order_visitor(node)



