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
###############################################################################

'''
This module provides an implementation of Multiscale Tree Graph.
For interface definition, see openalea.container.interface package.

:todo: MTG definition and usage.
'''

__docformat__ = "restructuredtext"

import re
#import itertools
import warnings
import random
import copy

# import traversal
# import algo
from skeletonisation_methods.plantscan3d import traversal
from skeletonisation_methods.plantscan3d import algo

# from tree import PropertyTree, InvalidVertex
from skeletonisation_methods.plantscan3d.tree import PropertyTree, InvalidVertex



class MTG(PropertyTree):
    ''' A Multiscale Tree Graph (MTG) class.

    MTGs describe tree structures at different levels
    of details, named scales.
    For example, a botanist can described plants at different scales :

        - at scale 0, the whole scene.
        - at scale 1, the individual plants.
        - at scale 2, the axes of each plants.
        - at scale 3, the growth units of each axis, and so on.

    Each scale can have a label, e.g. :

     - scale 1 : P(lant)
     - scale 2 : A(xis)
     - sclae 3 : U(nit of growth)

    Compared to a classical tree, :func:`complex` can be seen as :func:`parent`
    and :func:`components` as :func:`children`.
    An element at :func:`scale` N belongs to a :func:`complex` at :func:`scale` N-1 and has :func:`components` at scale N+1:

     - /P/A/U (decomposition is noted using "/")

    Each scale is itself described as a tree or a forest (i.e. set of trees), e.g.:

     - /P1/P2/P3
     - A1+A2<A3
     - ...

    '''

    def __init__(self, filename='', has_date=False):
        ''' Create a new MTG object.

        :Usage:

            >>> g = MTG()
            >>> g = MTG('my_mtg.mtg')
        '''

        super(MTG, self).__init__()

        # Map a vid to its scale
        self._scale = {0:0}

        # Multiscale tree:
        # complex <=> parent : vid -> vid
        # components <=> children : vid -> [vid]
        self._complex = {}
        self._components = {}

        # add default properties
        self.add_property('edge_type')
        self.add_property('label')

        if filename:
            from .io import read_mtg_file
            self = read_mtg_file(filename, mtg=self, has_date=has_date)

    def __getitem__(self, vtx_id):
        """A simple getitem to extract relevant information on a vertex.

        """
        if self.has_vertex(vtx_id):
            d = self.get_vertex_property(vtx_id)
            d.update({"vid":vtx_id,
                      "index":self.index(vtx_id),
                      "complex":self.complex(vtx_id),
                      "parent":self.parent(vtx_id),
                      "scale":self._scale.get(vtx_id)})
        else:
            raise IndexError('MTG index out of range')
        return d




    #########################################################################
    # Querying scale infos
    #########################################################################

    def scale(self, vid):
        ''' Returns the scale of a vertex.

        All vertices should belong to a given scale.

        :Usage:

        .. code-block:: python

            g.scale(vid)

        :Parameters:

            - `vid` (int) - vertex identifier.

        :Returns:
            The scale of the vertex.
            It is a positive int in [0,g.max_scale()].
        '''
        try:
            return self._scale[vid]
        except:
            pass

    def nb_scales(self):
        '''
        :Returns:
            The number of scales defined in the mtg..
        :Returns Type:
            int

        .. note:: The complexity is :math:`O(n)`.
        '''
        return len(set(self._scale.values()))

    def scales_iter(self):
        '''Return the different scales of the mtg.

        :Returns:
            Iterator on scale identifiers (ints).

        .. note:: The complexity is :math:`O(n)`.
        '''
        return iter(set(self._scale.values()))

    def scales(self):
        '''Return the different scales of the mtg.

        :Returns:
            Iterator on scale identifiers (ints).

        .. note:: The complexity is :math:`O(n)`.
        '''
        return list(self.scales_iter())

    def max_scale(self):
        '''Return the max scale identifier.

        By convention, the mtg contains scales in :math:`[0,max\_scale]`.

        :Usage:
            >>> print g.max_scale()

        :Returns:
            S, the maximum scale identifier.

        .. note:: The complexity is :math:`O(n)`.
        .. seealso:: :func:`scale`, :func:`scales`
        '''
        return max(self.scales_iter())


    #########################################################################
    # Some Vertex List Graph Concept methods.
    #########################################################################
    def nb_vertices(self, scale = -1):
        '''
        Returns the number of vertices.

        :Usage:
            >>> g.nb_vertices()
            100
            >>> g.nb_vertices(scale=3)
            68

        :Parameters:
         - `scale` (int) - Id of scale for which to count
           vertices.

        :Returns:
            Number of vertices at `scale` or total
            number of vertices if scale < 0.
        '''
        if scale < 0:
            return len(self._scale)
        else:
            return len(self.vertices(scale=scale))

    def vertices(self, scale = -1):
        '''
        Return a list of the vertices contained in an MTG.

        The set of all vertices in the MTG is returned.
        Vertices from all scales are returned if no scale is given.
        Otherwise, it returns only the vertices of the given scale.
        The order of the elements in this array is not significant.

        :Usage:

        .. code-block:: python

            g = MTG()
            len(g) == len(list(g.vertices()))
            for vid in g.vertices(scale=2):
                print g.class_name(vid)

        :Optional Parameters:
            - `scale` (int): used to select vertices at a given scale.

        :Returns:
            Iterator on vertices at "scale" or on all
            vertices if scale < 0.

        :Returns Type:
            list of vid

        :Background:

        .. seealso:: :meth:`children`, :meth:`components`, :meth:`vertices_iter`..
        '''
        return list(self.vertices_iter(scale=scale))

    def vertices_iter(self, scale = -1):
        '''
        Return an iterator of the vertices contained in an MTG.

        The set of all vertices in the MTG is returned.
        Vertices from all scales are returned if no scale is given.
        Otherwise, it returns only the vertices of the given scale.
        The order of the elements in this array is not significant.

        :Usage:

        .. code-block:: python

            g = MTG()
            len(g) == len(list(g.vertices()))
            for vid in g.vertices(scale=2):
                print g.class_name(vid)

        :Optional Parameters:
            - `scale` (int): used to select vertices at a given scale.

        :Returns:
            Iterator on vertices at "scale" or on all
            vertices if scale < 0.

        :Returns Type:
            iter of vid

        :Background:

        .. seealso:: :meth:`children`, :meth:`components`.
        '''
        if scale < 0:
            return iter(self._scale.keys())
        else:
            return (vid for vid, sid in self._scale.items() if sid == scale)


    #########################################################################
    # Python Iterator and Container interfaces
    #########################################################################
    def __iter__(self):
        '''
        Iterable interface.

        :Usage:

        .. code-block:: python

            for v in g:
                print g.class_name(v)

        '''
        return self.vertices_iter()

    def __contains__(self, vid):
        '''
        Container interface

        :Usage:

        .. code-block:: python

            if v in g:
                print v, " is in the mtg."
        '''
        return self.has_vertex(vid)


    #########################################################################
    # GraphConcept methods.
    #########################################################################
    def has_vertex(self, vid):
        """
        Tests whether a vertex belongs to the graph.

        :Parameters:
         - `vid` (int) - vertex id to test
        :Returns Type:
            bool
        """
        return vid in self._scale


    def is_valid(self):
        """
        Tests the validity of the graph. Currently
        always returns True.

        :Returns Type:
            bool
        :todo: Implement this function.
        """
        return True

    def iteredges(self, scale=-1):
        warnings.warn("Deprecated, use iter_edges instead", DeprecationWarning)
        return self.iter_edges(scale)

    def iter_edges(self, scale=-1):
        """
        :Parameters:
         - `scale` (int) - Scale at which to iterate.
        :Returns:
            Iterator on the edges of the MTG at a given scale
            or on all edges if scale < 0.
        :Returns Type:
            iter
        """
        if scale < 0:
            return ((v,k) for k,v in self._parent.items())
        else:
            return ((parent, child) for child, parent in self._parent.items() if self.scale(parent) == scale)

    def edges(self, scale=-1):
        """
        :Parameters:
         - `scale` (int) - Scale at which to iterate.
        :Returns:
            Iterator on the edges of the MTG at a given scale
            or on all edges if scale < 0.
        :Returns Type:
            iter
        """
        return list(self.iter_edges(scale=scale))

    #########################################################################
    # MutableVertexGraphConcept methods.
    # TODO: Add methods from MutableVertexGraph concept.
    #########################################################################

    def add_element(self, parent_id, edge_type = '/', scale_id=None):
        """
        Add an element to the graph, if vid is not provided create a new vid ???
        .. warning: Not Implemented.

        :Parameters:
        - `parent_id` (int) - The id of the parent vertex
        - `edge_type` (str) - The type of relation:
            - "/" : component (default)
            - "+" : branch
            - "<" : successor.
        - `scale_id` (int)  - The id of the scale in which to
            add the vertex.

        :Returns:
            The vid of the created vertex

        """
        raise NotImplementedError

    def remove_vertex(self, vid, reparent_child=False):
        """
        Remove a specified vertex of the graph and
        remove all the edges attached to it.

        :Parameters:
            - `vid` (int) : the id of the vertex to remove
            - `reparent_child` (bool) : reparent the children of `vid` to its parent.
        :Returns: None
        """

        if reparent_child:
            new_parent_id = self.parent(vid)
            children = list(self.children_iter(vid)) # should do a copy because the list will be modified by replace_parent
            for cid in children:
                self.replace_parent(cid, new_parent_id)


        if self.nb_components(vid) == 0:
            super(MTG, self).remove_vertex(vid, reparent_child=reparent_child)
            if vid in self._components:
                del self._components[vid]
            if vid in self._scale:
                del self._scale[vid]
            if vid in self._complex:
                cid = self._complex[vid]
                l = self._components[cid]
                try:
                    i = l.index(vid)
                    del l[i]
                except ValueError:
                    pass
                del self._complex[vid]
        else:
            raise InvalidVertex('Can not remove vertex %d with components.'
            'Use remove_tree instead.'%vid)

    def clear(self):
        """Remove all vertices and edges from the MTG.

        This also removes all vertex properties.
        Don't change references to object such as internal dictionaries.

        :Example:
            .. code-block:: python

                >>> g.clear()
                >>> g.nb_vertices()
                0
                >>> len(g)
                0


        """
        super(MTG, self).clear()

        self._scale.clear()
        self._scale[0] = 0

        self._complex.clear()
        self._components.clear()

    def clear_properties(self, exclude=[]):
        """Remove all the properties of the MTG.

        :Example:
            .. code-block:: python

                >>> g.clear_properties()
        """
        super(MTG, self).clear()

        exclude = set(exclude)
        for p in ['edge_type', 'label']:
            exclude.add(p)

        props = set(self._properties.keys()) - exclude
        for k in props:
            del self._properties[k]

    def copy(self):
        """ Return a copy of the graph.

        :Returns:
            - `g` (MTG) - A copy of the MTG

        """
        return copy.deepcopy(self)

    def roots_iter(self, scale=0):
        ''' Returns an iterator of the roots of the tree graphs at a given scale.

        In an MTG, the MTG root vertex, namely the vertex `g.root`,
        can be decomposed into several, non-connected, tree graphs at a given scale.
        This is for example the case of an MTG containing the description of several plants.

        :Usage: ::

            roots = list(g.roots(scale=g.max_scale())

        :Returns:
            iterator on vertex identifiers of root vertices at a given `scale`.
        :Returns Type:
            iter

        .. image:: ../user/mtg_componentroots.png
        '''
        return (vid for vid in self.vertices_iter(scale=scale) if self.parent(vid) is None)

    def roots(self, scale=0):
        ''' Returns a list of the roots of the tree graphs at a given scale.

        In an MTG, the MTG root vertex, namely the vertex `g.root`,
        can be decomposed into several, non-connected, tree graphs at a given scale.
        This is for example the case of an MTG containing the description of several plants.

        :Usage: ::

            roots = g.roots(scale=g.max_scale()

        :Returns:
            list on vertex identifiers of root vertices at a given `scale`.
        :Returns Type:
            list of vid

        .. image:: ../user/mtg_componentroots.png
        '''
        return list(self.roots_iter(scale=scale))


    #########################################################################
    # MutableTreeConcept methods.
    #########################################################################
    def add_child(self, parent, child=None, **properties):
        ''' Add a child to a parent. Child is appended to the parent's child list.

        :Parameters:
         - `parent` (int) - The parent identifier.
         - `child`  (int or None) - The child identifier. If None,
                    an ID is generated.

        :Returns:
            Identifier of the inserted vertex (child)
        :Returns Type:
            int
        '''

        child = super(MTG, self).add_child(parent, child, **properties)
        self._scale[child] = self._scale[parent]
        return child

    def insert_sibling(self, vtx_id1, vtx_id2=None, **properties):
        '''
        Insert a sibling of vtk_id1. The vertex in inserted before vtx_id1.

        :Parameters:
         - `vtx_id1` (int) : a vertex identifier
         - `vtx_id2` (int) : the vertex to insert

        :Returns:
            Identifier of the inserted vertex (vtx_id2)
        :Returns Type:
            int
        '''
        vtx_id2 = super(MTG, self).insert_sibling(vtx_id1, vtx_id2, **properties)
        self._scale[vtx_id2] = self._scale[vtx_id1]
        return vtx_id2

    def insert_parent(self, vtx_id, parent_id=None, **properties):
        '''
        Insert parent_id between vtx_id and its actual parent.
        Inherit of the complex of the parent of vtx_id.

        :Parameters:
         - `vtx_id` (int): a vertex identifier
         - `parent_id` (int): a vertex identifier

        :Returns:
            Identifier of the inserted vertex (parent_id).
        :Returns Type:
            int
        '''
        if parent_id is None:
            self._id += 1
            parent_id = self._id

        self._scale[parent_id] = self.scale(vtx_id)

        parent_id = super(MTG, self).insert_parent(vtx_id, parent_id, **properties)

        return parent_id

    def replace_parent(self, vtx_id, new_parent_id, **properties):
        '''
        Change the parent of vtx_id to new_parent_id.
        The new parent of vtx_id is new_parent_id.
        vtx_id and new_parent_id must have the same scale.

        This function do not change the edge_type between vtx_id and its parent.

        Inherit of the complex of the parent of vtx_id.

        :Parameters:
         - `vtx_id` (int): a vertex identifier
         - `new_parent_id` (int): a vertex identifier

        :Returns:
            None
        '''
        #if new_parent_id not in self:
            #    raise ""
        if self.scale(vtx_id) != self.scale(new_parent_id):
            raise InvalidVertex("Can not replace vertex {} by vertex {} from a different scale".format(vtx_id, new_parent_id))

        old_complex = self._complex.get(vtx_id)

        super(MTG, self).replace_parent(vtx_id, new_parent_id, **properties)

        if old_complex is not None:
            self.replace_parent(old_complex, self.complex(new_parent_id))


    #########################################################################
    # Mutable Multiscale Tree Concept methods.
    #########################################################################

    def complex(self, vtx_id):
        '''
        Returns the complex of `vtx_id`.

        :Parameters:
         - `vtx_id` (int) - The vertex identifier.

        :Returns:
            complex identifier or None if vtx_id has no parent.
        :Return Type:
            int
        '''
        complex_id = self._complex.get(vtx_id)
        while complex_id is None:
            vtx_id = self.parent(vtx_id)
            if vtx_id is None:
                break
            complex_id = self._complex.get(vtx_id)
        return complex_id

    def complex_at_scale(self, vtx_id, scale):
        '''
        Returns the complex of `vtx_id` at scale `scale`.

        :Parameters:
         - `vtx_id`: The vertex identifier.
         - `scale`: The scale identifier.

        :returns:
            vertex identifier
        :Returns Type:
            int
        '''
        complex_id = vtx_id
        current_scale = self.scale(complex_id)
        for i in range(scale, current_scale):
            complex_id = self.complex(complex_id)
        return complex_id

    def components_iter(self, vid):
        '''
        returns a vertex iterator

        :param vid: The vertex identifier.

        :returns: iter of vertex identifier
        '''

        if vid in self._components:
            for v in self.component_roots_iter(vid):
                for vtx in traversal.pre_order(self, v, complex=vid):
                    yield vtx

    def components(self, vid):
        '''
        returns the components of a vertex

        :param vid: The vertex identifier.

        :returns: list of vertex identifier
        '''
        return list(self.components_iter(vid))

    def components_at_scale_iter(self, vid, scale):
        '''
        returns a vertex iterator

        :Parameters:
         - `vid`: The vertex identifier.

        :returns: iter of vertex identifier
        '''
        # oops: search in the tree all the nodes which do not have another
        # explicit complex.

        cur_scale = self.scale(vid)

        gen = (vid, )
        for i in range(cur_scale, scale):
            gen = (vid for vtx in gen for vid in self.components_iter(vtx) )

        return gen

    def components_at_scale(self, vid, scale):
        '''
        returns a vertex iterator

        :Parameters:
         - `vid`: The vertex identifier.

        :returns: iter of vertex identifier
        '''
        return list(self.components_at_scale_iter(vid, scale))

    def component_roots_iter(self, vtx_id):
        '''Return an iterator of the roots of the tree graphs that compose a vertex.
        '''
        components = self._components.get(vtx_id,[])

        for ci in components:
            p = self.parent(ci)
            if p is None or self.complex(p) != vtx_id:
                yield ci

    def component_roots(self, vtx_id):
        '''Return the set of roots of the tree graphs that compose a vertex.
        '''
        return list(self.component_roots_iter(vtx_id))

    def component_roots_at_scale_iter(self, vtx_id, scale):
        '''Return the set of roots of the tree graphs that compose a vertex.
        '''
        cur_scale = self.scale(vtx_id)
        if scale == -1 or scale == cur_scale+1:
           return self.component_roots_iter(vtx_id)
        elif scale > cur_scale+1:
            gen = (vtx_id,)
            for i in range(cur_scale, scale):
                gen = (vid for vtx in gen for vid in self.component_roots_iter(vtx))
            return gen
        else:
            return iter([])

    def component_roots_at_scale(self, vtx_id, scale):
        '''Return the list of roots of the tree graphs that compose a vertex.
        '''
        return list(self.component_roots_at_scale_iter(vtx_id, scale))

    def nb_components(self, vid):
        '''
        returns the number of components

        :Parameters:
         - `vid`: The vertex identifier.

        :returns: int
        '''
        return len(self.components(vid))

    # mutable
    def add_component(self, complex_id, component_id=None, **properties):
        '''
        Add a component at the end of the components

        :Parameters:
         - `complex_id`: The complex identifier.
         - `component_id`: Set the component identifier to this value if defined.

	:Returns: The id of the new component or the component_id if given.
        '''
        if component_id is None:
            self._id += 1
            component_id = self._id

        self._add_vertex_properties(component_id, properties)

        self._components.setdefault(complex_id,[]).append(component_id)
        self._complex[component_id] = complex_id
        self._scale[component_id] = self._scale[complex_id]+1

        return component_id

    def add_child_and_complex(self, parent, child=None, complex=None, **properties):
        '''
        Add a child at the end of children that belong to an other complex.

        :Parameters:
         - `parent`: The parent identifier.
         - `child`: Set the child identifier to this value if defined.
         - `complex`: Set the complex identifier to this value if defined.
        :returns: (vid, vid): child and complex ids.
        '''

        if complex is None:
            self._id += 1
            complex = self._id

        if child in self._children.get(parent, []):
            # add only the properties
            self._add_vertex_properties(child, properties)
        else:
            child = self.add_child(parent, child, **properties)
        self._scale[child] = self._scale[parent]


        parent_complex = self.complex(parent)

        if complex not in self._children.get(parent_complex, []):
            self.add_child(parent_complex, complex)
        self._scale[complex] = self._scale[parent_complex]

        self._components.setdefault(complex,[]).append(child)
        self._complex[child] = complex

        return child, complex

    def __str__(self):
        l = ["MTG : nb_vertices=%d, nb_scales=%d"%(self.nb_vertices(), self.nb_scales())]

        """
        from . import io
        s = io.display(self, display_id=True)
        l.append(s)
        """
        return '\n'.join(l)

    def display(self, max_scale=0, display_id=True, display_scale=False,
                nb_tab=12, **kwds):
        """ Print an MTG on the console.

        :Optional Parameters:
            - `max_scale`: do not print vertices of scale greater than max_scale
            - `display_id`: display the vid of the vertices
            - `display_scale`: display the scale of the vertices
            - `nb_tab`: display the MTG using nb_tab columns

        """
        from . import io
        print(("MTG : nb_vertices=%d, nb_scales=%d"%(self.nb_vertices(), self.nb_scales())))
        print(io.display(self, max_scale=max_scale, display_id=display_id,
                         display_scale=display_scale, nb_tab=nb_tab, **kwds))

    def plot_property(self, prop, **kwds):
        """
        Plot properties of MTG using matplotlib

        :Example:

            >>> g.plot_property('length')
        """
        import matplotlib
        import matplotlib.pyplot
        import numpy as np
        props = self.property(prop)
        pylab_colors = list(matplotlib.colors.cnames.keys())
        color = {}
        orders = algo.orders(self)
        for k in props:
            color[k]=pylab_colors[orders[k]]
        #color = {k:pylab_colors[orders[k]] for k in props} # not in Py2.6

        heights = algo.heights(self)
        h = np.array([heights[v] for v in props])
        _prop = np.array(list(props.values()))
        for v in props:
            matplotlib.pyplot.plot(heights[v], props[v], 'o', color = color[v])


    #########################################################################
    # Algorithms to copy extract and extend sub_mtg
    #########################################################################
    def sub_mtg(self, vtx_id, copy=True):
        """Return the submtg rooted on `vtx_id`.

        The induced sub mtg of the mtg are all the vertices which have vtx_id
        has a complex plus vtx_id.

        :Parameters:
          - `vtx_id`: A vertex of the original tree.
          - `copy`:
            If True, return a new tree holding the subtree. If False, the subtree is
            created using the original tree by deleting all vertices not in the subtree.

        :returns: A sub mtg of the mtg. If copy=True, a new MTG is returned.
            Else the sub mtg is created inplace by modifying the original tree.
        """

        if not copy:
            # remove all vertices not in the sub_tree

            bunch = set(traversal.pre_order_in_scale(self, vtx_id))
            remove_bunch = set(self) - bunch

            self.root = vtx_id

            # remove vertices by removing the element and deleting all the deges.
            # We do not use standard methods because the graph will not be functional
            # until the removal of all vertices.

            # force remove
            for vid in remove_bunch:

                # TODO: Build specific methods (_force_remove) to edit a MTG without
                # any verification. The MTG/Tree/whatever will be temporary invalid.

                # remove properties
                self._remove_vertex_properties(vid)
                del self._scale[vid]

                # remove parent edge
                pid = self.parent(vid)
                if pid is not None:
                    self._children[pid].remove(vid)
                    del self._parent[vid]
                # remove children edges
                for cid in self.children_iter(vid):
                    self._parent[cid] = None
                if vid in self._children:
                    del self._children[vid]

                # remove complex edges
                complex_id = self._complex.get(vid)
                if complex_id is not None:
                    self._components[complex_id].remove(vid)
                    del self._complex[vid]
                # remove components edges
                for cid in self.components_iter(vid):
                    del self._complex[cid]
                if vid in self._components:
                    del self._components[vid]

            # Update the scale of the nodes
            scale = self._scale
            root_scale = self.scale(vtx_id)
            for vid in scale:
                scale[vid] = scale[vid]-root_scale

            self._scale[self.root] = 0

            return self
        else:
            treeid_id = {}
            g = MTG()
            g.root = 0

            for name in self.properties():
                g.add_property(name)

            treeid_id[vtx_id] = g.root
            subtree = traversal.iter_mtg2(self, vtx_id)


            # Skip the first vertex vtx_id
            next(subtree)
            # Traverse all the sub_mtg.
            # Every vertex has a complex in this sub_mtg.
            # Complex vertices are traversed before there components and
            # parent before the children.

            for vid in subtree:
                complex_id = treeid_id[self.complex(vid)]
                v = g.add_component(complex_id)
                treeid_id[vid] = v

                pid = self.parent(vid)
                if pid in treeid_id:
                    parent = treeid_id[pid]
                    v = g.add_child(parent, child=v)

                # Copy the properties
                g._add_vertex_properties(v, self.get_vertex_property(vid))

            return g


    def reindex(self, mapping=None, copy=False):
        """ Assign a new identifier to each vertex.

        This method assigns a new identifier to each vertex of the MTG. The mapping can be user defined or
        is implicit (`mapping`). This method modify the MTG in place or return a new MTG (`copy`).

        :Usage:

        .. code-block:: python

            >>> g.reindex()
            >>> g1 = g.reindex(copy=True)
            >>> mymap = dict(zip(list(traversal.iter_mtg2(g,g.root)), range(len(g))))
            >>> g2 = g.reindex(mapping=mymap, copy=True)

        :Optional Parameters:

            - `mapping` (dict): define a mapping between old and new vertex identifiers.
            - `copy` (bool) : modify the object in place or return a new MTG.

        :Returns:

            - a MTG

        :Background:

            :func:`MTGs`

        .. seealso:: :meth:`sub_mtg`

        """
        # Manage also mapping as a function
        if not mapping:
            mapping = dict(list(zip(traversal.iter_mtg2(self, self.root), list(range(len(self))))))

        if copy:
            g = MTG()
            g.root = mapping.setdefault(self.root,0)

            for name in self.properties():
                g.add_property(name)

            subtree = traversal.iter_mtg2(self, self.root)

            # Skip the first vertex vtx_id
            next(subtree)

            # Traverse all the sub_mtg.
            # Every vertex has a complex in this sub_mtg.
            # Complex vertices are traversed before there components and
            # parent before the children.

            for vid in subtree:
                complex_id = mapping[self.complex(vid)]
                v_id = mapping[vid]
                v_id = g.add_component(complex_id, component_id=v_id)

                pid = self.parent(vid)
                if pid:
                    parent = mapping[pid]
                    v_id = g.add_child(parent, child=v_id)

                # Copy the properties
                g._add_vertex_properties(v_id, self.get_vertex_property(vid))

            return g
        else:
            # recreate new dict for _parent, _children, _complex, _components
            self._parent = dict((mapping[k], mapping.get(v)) for k, v in self._parent.items())
            self._children = dict((mapping[k], [mapping[v] for v in l]) for k, l in self._children.items())
            self._complex = dict((mapping[k], mapping.get(v)) for k, v in self._complex.items())
            self._components = dict((mapping[k], [mapping[v] for v in l]) for k, l in self._components.items())
            self._scale = dict((mapping[k], s) for k, s in self._scale.items())
            for name in self._properties:
                d = self._properties[name]
                self._properties[name] = dict((mapping[k], s) for k, s in d.items())

            return self

    def insert_scale(self, inf_scale=None, partition=None, default_label=None, preserve_order=True):
        """
        Add a scale to MTG

        :Parameters:
        - `inf_scale` (int) - New scale is inserted between inf_scale and inf_scale-1
        - `partition` (lambda v: bool) - Function defining new scale by quotienting vertices at inf_scale
        - `default_label` (str) - default label of inserted vertices
        - `preserve_order` (bool) - True iif children at new scale are ordered consistently
            with children at inf_scale

        :Returns:
            MTG with inserted scale

        :Remark:
            - New scale is inserted in self as well.
            - function partition should return True at roots of subtrees where partition changes
            and False elsewhere.

        """
        g = self
        if g.nb_scales() < 2:
            raise Exception('Can not insert a scale on a tree which is not an MTG')

        if inf_scale is None:
            inf_scale = g.max_scale()

        if partition is None:
            partition = lambda v: g.edge_type(v) != '<'

        inf_vertices = g.components_at_scale(g.root, scale=inf_scale)
        colors = [v  for v in inf_vertices if partition(v)]

        sup_vertices = g.components_at_scale(g.root, scale=inf_scale-1)

        # compute the component_roots
        sup_components = dict((v, next(g.component_roots_iter(v))) for v in sup_vertices)
        complexes = list(sup_components.values())

        if (set(complexes) - set(colors)):
            raise Exception("Error: the scale you want to insert is not included in the upper scale. Please modify the partition.")

        # Modification in_place:
        # Algo:
        # - Modify the _scale of the vertices at scale >= inf_scale
        # - create new ids
        # - Modify the _components of sup_vertices
        # - Modify the _complex of vertices
        # - add a tree at scale inf_scale with _complex in sup_vertices and _components in colors

        nb_scales = g.max_scale()

        new_scale = dict((vid,sid+1) for vid, sid in g._scale.items() if sid >= inf_scale)
        g._scale.update(new_scale)

        max_id = max(g._scale)
        complex_inf = dict((colors[i], max_id+i+1) for i in range(len(colors)))
        components_sup = dict((vid, complex_inf[cid]) for vid ,cid in sup_components.items())

        # add the new vertices
        g._scale.update( (vid, inf_scale) for vid in complex_inf.values())

        # remove all the components of sup_vertices
        for v in sup_vertices:
            if v in g._components:
                del g._components[v]

        # remove all the complex of inf_vertices
        for v in inf_vertices:
            if v in g._complex:
                del g._complex[v]

        # And new components
        for vid, component_id in components_sup.items():
            g.add_component(vid,component_id)

        # Add new complex
        for component_id, complex_id in complex_inf.items():
            g.add_component(complex_id, component_id)
            if not(default_label is None):
                g._add_vertex_properties(complex_id, dict(label=default_label))

        return fat_mtg(g, preserve_order)

    def remove_scale(self, scale):
        """ Remove all the vertices at a given scale.

        The upper and lower scale are then connected.

        Parameters:
        ===========
            - scale : the scale that have to be removed

        Returns
        =======
            - g : the input MTG modified in place.
            - results : a list of dict
                all the vertices that have been removed
        """

        g = self
        nb_scales = g.max_scale()
        if scale > nb_scales or scale < 1:
            return
        # 2 cases to consider: scale == 1 or max_scale

        vertices = g.components_at_scale(g.root, scale=scale)
        sup_vertices = g.components_at_scale(g.root, scale=scale-1)
        inf_vertices = g.components_at_scale(g.root, scale=scale+1)

        results = dict((v, g[v]) for v in vertices)

        new_complex = dict((cid, g.complex(v)) for v in vertices for cid in  g.component_roots_iter(v))

        # Algo:
        # Update scale of vertices
        # Remove vertices at scale=scale
        # Remove components at scale -1
        # Remove complex at scale +1
        # Add component between sup and inf scale

        new_scale = dict((vid,sid-1) for vid, sid in g._scale.items() if sid > scale)
        g._scale.update(new_scale)

        for v in vertices:
            del g._scale[v]
            if v in g._children:
                del g._children[v]
            if v in g._parent:
                del g._parent[v]
            if v in g._components:
                del g._components[v]
            if v in g._complex:
                del g._complex[v]

        for v in sup_vertices:
            if v in g._components:
                del g._components[v]

        for v in inf_vertices:
            if v in g._complex:
                del g._complex[v]

        # And new components
        for component_id, complex_id in new_complex.items():
            g.add_component(complex_id,component_id)

        # Update components
        g._components = dict((v, [cid for cid in components if g.parent(cid) is None or g.complex(g.parent(cid)) !=v])
            for v, components in g._components.items())

        g = fat_mtg(g)
        return g, results

    #########################################################################
    # Specialised algorithms for aml compatibility.
    #########################################################################
    def order(self, v1):
        """
        Order of a vertex in a graph.

        The order of a vertex in a graph is the number of '+' edges crossed
        when going from `v1`to `v2`.

        If v2 is None, the order of v1 correspond to the order of v1 with
        respect to the root.
        """
        _order = 0
        edge_type = self.property('edge_type')
        if not edge_type:
            return 0

        vid = v1
        while vid is not None:
            if edge_type.get(vid) == '+':
                _order += 1
            vid = self.parent(vid)

        return _order

    def edge_type(self, vid):
        """
        Type of the edge between a vertex and its parent.

        The different values are '<' for successor, and '+' for ramification.
        """
        return self.property('edge_type').get(vid,'')

    def label(self, vid):
        """Label of a vertex.

        :Usage:

            >>> g.label(v)

        :Parameters:
            - `vid` (int) : vertex of the MTG

        :Returns:
            The class and Index of the vertex (str).

        .. seealso:: :func:`MTG`, :func:`index`, :func:`class_name`
        """
        return self.property('label').get(vid, '')

    def class_name(self, vid):
        """Class of a vertex.

        The Class of a vertex are the first characters of the label.
        The label of a vertex is the string defined by the concatenation
        of the class and its index.

        The label thus provides general information about a vertex and
        enable to encode the plant components.

        The class_name may be not defined. Then, an empty string is returned.

        :Usage:

            >>> g.class_name(1)

        :Parameters:

            - `vid` (int)

        :Returns:
            The class name of the vertex (str).

        .. seealso:: :func:`MTG`, :func:`openalea.mtg.aml.Index`, :func:`openalea.mtg.aml.Class`
        """
        pattern = r'[a-zA-Z]+'
        label = self.property('label').get(vid)
        if not label:
            return ''
        else:
            m=re.match(pattern, label)
            if m:
                return m.group(0)
            else:
                return ''

    def index(self, vid):
        """
        Index of a vertex

        The Index of a vertex is a feature always defined and independent of time
        (like the index).
        It is represented by a non negative integer.
        The label of a vertex is the string defined by the concatenation
        of its class and its index.
        The label thus provides general information about a vertex and
        enables us to encode the plant components.
        """
        pattern = r'[0-9]+$'
        label = self.property('label').get(vid)
        if not label:
            return vid
        else:
            m=re.search(pattern, label)
            if m:
                return m.group(0)
            else:
                return vid

    #########################################################################
    # Proxy node interface
    #########################################################################
    def node(self, vid, klass=None):
        """
        Return a node associated to the vertex `vid`.

        It allows to access to the properties with an object oriented interface.

        :Example:

        .. code-block:: python

            node = g.node(1)
            print node.edge_type
            print node.label
            node.label = 'B'
            print g.label(1)

            print node.parent
            print list(node.children)
        """
        if klass is None:
            klass = _ProxyNode
        if vid in self:
            return klass(self,vid)
        else:
            # TODO: retunr an error
            return None

    #########################################################################
    # Compatibility with AML
    # Add deprecated decorator
    #########################################################################

    def VtxList(self, Scale=-1):
        """
        Array of vertices contained in a MTG

        The set of all vertices in the :func:`MTG` is returned as an array.
        Vertices from all scales are returned if no option is used.
        The order of the elements in this array is not significant.

        :Usage:

        .. code-block:: python

            >>> VtxList()
            >>> VtxList(Scale=2)

        :Optional Parameters:

            - `Scale` (int): used to select components at a particular scale.

        :Returns:

            - list of vid

        :Background:

            :func:`MTGs`

        .. seealso:: :meth:`MTG`, :meth:`scale`, :meth:`Class`, :meth:`index`.
        """
        return self.vertices(scale=Scale)

    Label = label
    Class = class_name
    Index = index
    Scale = scale

    def ClassScale(self, c):
        """
        Scale at which appears a given class of vertex

        Every vertex is associated with a unique class.
        Vertices from a given class only appear at a given scale
        which can be retrieved using this function.

        :Usage:

        .. code-block:: python

            ClassScale(c)

        :Parameters:

            - `c` (str) : symbol of the considered class

        :Returns:

            int

        .. seealso:: :func:`MTG`, :func:`Class`, :func:`Scale`, :func:`Index`.

        """
        for x in self.vertices_iter():
            if self.Class(x) == c:
                return self.scale(x)

    def EdgeType(self, v1, v2):
        """
        Type of connection between two vertices.

        Returns the symbol of the type of connection between two vertices (either `<` or `+`).
        If the vertices are not connected, None is returned.

        :Usage:

        .. code-block:: python

            EdgeType(v1, v2)

        :Parameters:

            - v1 (int) : vertex of the active MTG
            - v2 (int) : vertex of the active MTG

        :Returns:

            '<' (successor), '+' (branching) or `None`

        .. image:: ../user/mtg_edgetype.png

        .. seealso:: :func:`MTG`, :func:`Sons`, :func:`Father`.

        """
        if self.parent(v1) == v2:
            v1, v2 = v2, v1

        return self.property('edge_type').get(v2)

    def Defined(self, vid):
        """
        Test whether a given vertex belongs to the active MTG.

        :Usage:

        .. code-block:: python

            Defined(v)

        :Parameters:

            - v (int) : vertex of the active MTG

        :Returns:

            True or False

        .. seealso:: :func:`MTG`.
        """
        return vid in self

    def Rank(self, v1, v2=None):
        """
        Rank of one vertex with respect to another one.

        This function returns the number of consecutive '<'-type edges between two components,
        at the same scale, and does not take into account the order of vertices v1 and v2.
        The result is a non negative integer.

        :Usage:

        .. code-block:: python

            Rank(v1)
            Rank(v1, v2)

        :Parameters:

            - v1 (int) : vertex of the active MTG
            - v2 (int) : vertex of the active MTG

        :Returns:

            `int`

            If v1 is not an ancestor of v2 (or vise versa) within the same botanical axis,
            or if v1 and v2 are not defined at the same scale, an error value Undef id returned.

        .. seealso:: :func:`MTG`, :func:`Order`, :func:`Height`, :func:`EdgeType`, :func:`AlgRank`, :func:`AlgHeight`, :func:`AlgOrder`.

        """
        return algo.rank(self,v1,v2)

    def Height(self, v1, v2=None):
        """
        Number of components existing between two components in a tree graph.

        The height of a vertex (`v2`) with respect to another vertex (`v1`)
        is the number of edges (of either type '+' or '<') that must be crossed
        when going from `v1` to `v2` in the graph.

        This is a non-negative integer. When the function has only one argument `v1`,
        the height of `v1` correspond to the height of `v1`with respect
        to the root of the branching system containing `v1`.

        :Usage:

        .. code-block:: python

            Height(v1)
            Height(v1, v2)

        :Parameters:

            - v1 (int) : vertex of the active MTG
            - v2 (int) : vertex of the active MTG

        :Returns:

            int

        .. note::

            When the function takes two arguments, the order of the arguments is not important
            provided that one is an ancestor of the other. When the order is relevant, use
            function `AlgHeight`.

        .. seealso:: :func:`MTG`, :func:`Order`, :func:`Rank`, :func:`EdgeType`, :func:`AlgHeight`, :func:`AlgHeight`, :func:`AlgOrder`.

        """
        return algo.height(self, v1, v2)

    def AlgOrder(self, v1, v2):
        """
        Algebraic value defining the relative order of one vertex with respect to another one.

        This function is similar to function `Order(v1, v2)` : it returns the number of `+`-type edges
        between two components, at the same scale, but takes into account the order of vertices
        `v1` and `v2`.

        The result is positive if `v1` is an ancestor of `v2`,
        and negative if `v2` is an ancestor of `v1`.

        :Usage:

        .. code-block:: python

            AlgOrder(v1, v2)

        :Parameters:

            - v1 (int) : vertex of the active MTG.
            - v2 (int) : vertex of the active MTG.

        :Returns:

            int

            If `v1` is not an ancestor of `v2` (or vise versa), or if `v1` and `v2` are not defined
            at the same scale, an error value None is returned.


        .. seealso:: :func:`MTG`, :func:`Rank`, :func:`Order`, :func:`Height`, :func:`EdgeType`, :func:`AlgHeight`, :func:`AlgRank`.
        """
        return algo.alg_order(self, v1, v2)

    def AlgRank(self, v1, v2):
        """
        Algebraic value defining the relative rank of one vertex with respect to another one.

        This function is similar to function `Rank(v1, v2)` : it returns the number of `<`-type edges
        between two components, at the same scale, but takes into account the order of vertices
        `v1` and `v2`.

        The result is positive if `v1` is an ancestor of `v2`,
        and negative if `v2` is an ancestor of `v1`.

        :Usage:

        .. code-block:: python

            AlgRank(v1, v2)

        :Parameters:

            - v1 (int) : vertex of the active MTG.
            - v2 (int) : vertex of the active MTG.

        :Returns:

            int

            If `v1` is not an ancestor of `v2` (or vise versa), or if `v1` and `v2` are not defined
            at the same scale, an error value None is returned.

        .. seealso:: :func:`MTG`, :func:`Rank`, :func:`Order`, :func:`Height`, :func:`EdgeType`, :func:`AlgHeight`, :func:`AlgOrder`.

        """
        return algo.alg_rank(self, v1, v2)

    def AlgHeight(self, v1, v2):
        """
        Algebraic value defining the number of components between two components.

        This function is similar to function `Height(v1, v2)` : it returns the number of components
        between two components, at the same scale, but takes into account the order of vertices
        `v1` and `v2`.

        The result is positive if `v1` is an ancestor of `v2`,
        and negative if `v2` is an ancestor of `v1`.

        :Usage:

        .. code-block:: python

            AlgHeight(v1, v2)

        :Parameters:

            - v1 (int) : vertex of the active MTG.
            - v2 (int) : vertex of the active MTG.

        :Returns:

            int

            If `v1` is not an ancestor of `v2` (or vise versa), or if `v1` and `v2` are not defined
            at the same scale, an error value None is returned.

        .. seealso:: :func:`MTG`, :func:`Rank`, :func:`Order`, :func:`Height`, :func:`EdgeType`, :func:`AlgOrder`, :func:`AlgRank`.

        """
        return algo.alg_height(self, v1, v2)

    def Father(self, v, EdgeType='*', RestrictedTo='NoRestriction', ContainedIn=None, Scale = -1):
        """
        Topological father of a given vertex.

        Returns the topological father of a given vertex. And `None` if the father does not exist.
        If the argument is not a valid vertex, `None` is returned.

        :Usage:

        .. code-block:: python

            g.Father(v)
            g.Father(v, EdgeType='<')
            g.Father(v, RestrictedTo='SameComplex')
            g.Father(v, ContainedIn=complex_id)
            g.Father(v, Scale=s)

        :Parameters:

            v (int) : vertex of the active MTG

        :Optional Parameters:

            If no optional argument is specified,  the function returns the topological father
            of the argument (vertex that bears or precedes to the vertex passed as an argument).

            It may be usefull in some cases to consider that the function only applies to a
            subpart of the MTG (e.g. an axis).

            The following options enables us to specify such restrictions:

            - EdgeType (str) : filter on the type of edge that connect the vertex to its father.

              Values can be '<', '+', and '*'. Values '*' means both '<' and '+'.
              Only the vertex connected with the specified type of edge will be considered.

            - RestrictedTo (str) : filter defining a subpart of the MTG where the father
              must be considered. If the father is actually outside this subpart,
              the result is `None`. Possible subparts are defined using keywords in
              ['SameComplex', 'SameAxis', 'NoRestriction'].

              For instance, if `RestrictedTo` is set to 'SameComplex', :func:`Father(v)` returns a
              defined vertex only if the father `f` of `v` existsin the MTG and if `v` and `f`
              have the same complex.

            - ContainedIn (int) : filter defining a subpart of the MTG where the father
              must be considered. If the father is actually outside this subpart,
              the result is `None`.

              In this case, the subpart of the MTG is made of the vertices
              that composed `composite_id` (at any scale).

            - Scale (int) : the scale of the considered father. Returns the vertex from scale `s`
              which either bears and precedes the argument.

              The scale `s` can be lower than the argument's (corresponding to a question such as
              'which axis bears the internode?') or greater
              (e.g. 'which internodes bears this annual shoot?').

        :Returns:

            the vertex id of the Father (int)

        .. seealso:: :func:`MTG`, :func:`Defined`, :func:`Sons`, :func:`EdgeType`, :func:`Complex`, :func:`Components`.

        """
        if EdgeType not in ['+', '<', '*']:
            raise Exception('Invalid argument %s. Value of EdgeType is "<", "+" or "*".'%EdgeType)

        if RestrictedTo not in ['SameComplex', 'SameAxis', 'NoRestriction']:
            raise Exception('Invalid argument %s. Value of RestrictedTo is SameComplex, SameAxis, NoRestriction .'%RestrictedTo)

        return algo.father(self, v, scale=Scale, EdgeType=EdgeType, RestrictedTo=RestrictedTo, ContainedIn=ContainedIn)

    def Successor(self, v, RestrictedTo='NoRestriction', ContainedIn=None):
        """
        Vertex that is connected to a given vertex by a '<' edge type (i.e. in the same botanical axis).

        This function is equivalent to Sons(v, EdgeType='<')[0].
        It returns the vertex that is connected to a given vertex by a '<' edge type
        (i.e. in the same botanical axis).
        If many such vertices exist, an arbitrary one is returned by the function.
        If no such vertex exists, None is returned.

        :Usage:

        .. code-block:: python

            g.Successor(v)

        :Parameters:

            - v1 (int) : vertex of the active MTG

        :Optional Parameters:

            - RestrictedTo (str): cf. Father
            - ContainedIn (int): cf. Father

        :Returns:

            Returns vertex's id (int)


        :Examples:

        .. code-block:: python

            >>> g.Sons(v)
            [3, 45, 47, 78, 102]
            >>> g.Sons(v, EdgeType='+') # set of vertices borne by v
            [3, 45, 47, 102]
            >>> g.Sons(v, EdgeType-> '<') # set of successors of v
            [78]
            >>> g.Successor(v)
            78

        .. seealso:: :func:`MTG`, :func:`Sons`, :func:`Predecessor`.
        """
        _g = self
        return algo.successor(_g, v, RestrictedTo=RestrictedTo, ContainedIn=ContainedIn)

    def Predecessor(self, v, **kwds):
        """
        Father of a vertex connected to it by a '<' edge

        This function is equivalent to Father(v, EdgeType-> '<').
        It thus returns the father (at the same scale) of the argument
        if it is located in the same botanical.
        If it does not exist, None is returned.

        :Usage:

        .. code-block:: python

            Predecessor(v)

        :Parameters:

            - v (int) : vertex of the active MTG

        :Optional Parameters:

            - RestrictedTo (str): cf. `Father`
            - ContainedIn (int): cf. `Father`

        :Returns:

            return the vertex id (int)


        :Examples:

        .. code-block:: python

            >>> Predecessor(v)
            7
            >>> Father(v, EdgeType='+')
            >>> Father(v, EdgeType-> '<')
            7

        .. seealso:: :func:`MTG`, :func:`Father`, :func:`Successor`.
        """
        return self.Father(v, EdgeType='<', **kwds)

    def Root(self, v, RestrictedTo='*', ContainedIn=None):
        """
        Root of the branching system containing a vertex

        This function is equivalent to Ancestors(v, EdgeType='<')[-1].
        It thus returns the root of the branching system containing the argument.
        This function never returns None.

        :Usage:

        .. code-block:: python

            g.Root(v)

        :Parameters:

            - v (int) : vertex of the active MTG

        :Optional Parameters:

            - RestrictedTo (str): cf. Father
            - ContainedIn (int): cf. Father

        :Returns:

           return vertex's id (int)


        :Examples:

        .. code-block:: python

            >>> g.Ancestors(v) # set of ancestors of v
            [102,78,35,33,24,12]
            >>> g.Root(v) # root of the branching system containing v
            12

        .. image:: ../user/mtg_root.png

        .. seealso:: :func:`MTG`, :func:`Extremities`.
        """
        return algo.root(self, v, RestrictedTo=RestrictedTo,
                         ContainedIn=ContainedIn)

    def Complex(self, v, Scale=-1):
        """
        Complex of a vertex.

        Returns the complex of `v`. The complex of a vertex `v` has a scale lower than `v` :
        `Scale(v)` - 1. In a MTG, every vertex except for the MTG root (cf. `MTGRoot`),
        has a uniq complex. None is returned if the argument is the MTG Root
        or if the vertex is undefined.

        :Usage:

        .. code-block:: python

            g.Complex(v)
            g.Complex(v, Scale=2)

        :Parameters:

            - `v` (int) : vertex of the active MTG

        :Optional Parameters:

            - `Scale` (int) : scale of the complex

        :Returns:

            Returns vertex's id (int)

        :Details:

            When a scale different form Scale(v)-1 is specified using the optional parameter
            `Scale`, this scale must be lower than that of the vertex argument.

        .. todo:: Complex(v, Scale=10) returns v why ? is this expected

        .. seealso:: :func:`MTG`, :func:`Components`.
        """
        _g = self
        if Scale == -1 or Scale == _g.scale(v)-1:
            return _g.complex(v)
        else:
            return _g.complex_at_scale(v, scale=Scale)

    def Sons(self, v, RestrictedTo='NoRestriction', EdgeType='*', Scale=-1, ContainedIn= None):
        """
        Set of vertices born or preceded by a vertex

        The set of sons of a given vertex is returned as an array of vertices.
        The order of the vertices in the array is not significant.
        The array can be empty if there are no son vertices.

        :Usage:

        .. code-block:: python

            g.Sons(v)
            g.Sons(v, EdgeType= '+')
            g.Sons(v, Scale= 3)

        :Parameters:

            - v (int) : vertex of the active MTG

        :Optional Parameters:

            - RestrictedTo (str) : cf. :meth:`Father`
            - ContainedIn (int) : cf. :meth:`Father`
            - EdgeType (str) : filter on the type of sons.
            - Scale (int) : set the scale at which sons are considered.

        :Returns:

            list(vid)

        :Details:

            When the option EdgeType is applied, the function returns the set of sons
            that are connected to the argument with the specified type of relation.

        .. note:: `Sons(v, EdgeType= '<')` is not equivalent to `Successor(v)`.
            The first function returns an array of vertices while the second function
            returns a vertex.

            The returned vertices have the same scale as the argument.
            However, coarser or finer vertices can be obtained by specifying
            the optional argument `Scale` at which the sons are considered.


        :Examples:

        .. code-block:: python

            >>> g.Sons(v)
            [3,45,47,78,102]
            >>>  g.Sons(v, EdgeType= '+') # set of vertices borne by v
            [3,45,47,102]
            >>>  g.Sons(v, EdgeType= '<') # set of successors of v on the same axis
            [78]

        .. seealso:: :func:`MTG`, :func:`Father`, :func:`Successor`, :func:`Descendants`.
        """
        return algo.sons(self, v, EdgeType=EdgeType,
                         RestrictedTo=RestrictedTo,
                         Scale=Scale,
                         ContainedIn=ContainedIn)

    def Ancestors(self, v, EdgeType='*', RestrictedTo='NoRestriction', ContainedIn=None):
        """
        Array of all vertices which are ancestors of a given vertex

        This function returns the array of vertices which are located
        before the vertex passed as an argument.
        These vertices are defined at the same scale as `v`. The array starts by `v`,
        then contains the vertices on the path from `v` back to the root (in this order)
        and finishes by the tree root.

        .. note:: The anscestor array always contains at least the argument vertex `v`.

        :Usage:

        .. code-block:: python

            g.Ancestors(v)

        :Parameters:

            - v (int) : vertex of the active MTG

        :Optional Parameters:

            - RestrictedTo (str): cf. `Father`
            - ContainedIn (int): cf. `Father`
            - EdgeType (str): cf. `Father`

        :Returns:

            list of vertices's id (int)


        :Examples:

        .. code-block:: python

            >>> v # prints vertex v
            78
            >>> g.Ancestors(v) # set of ancestors of v at the same scale
            [78,45,32,10,4]
            >>> list(reversed(g.Ancestors(v))) # To get the vertices in the order from the root to the vertex v
            [4,10,32,45,78]


        .. seealso:: :func:`MTG`, :func:`Descendants`.
        """
        return list(algo.full_ancestors(self, v, RestrictedTo=RestrictedTo,
                                              EdgeType=EdgeType,
                                              ContainedIn=ContainedIn))

    def Descendants(self, v, EdgeType='*', RestrictedTo='NoRestriction', ContainedIn=None):
        """
        Set of vertices in the branching system borne by a vertex.

        This function returns the set of descendants of its argument as an array of vertices.
        The array thus consists of all the vertices, at the same scale as `v`,
        that belong to the branching system starting at `v`.
        The order of the vertices in the array is not significant.

        .. note:: The argument always belongs to the set of its descendants.

        :Usage:

        .. code-block:: python

            g.Descendants(v)

        :Parameters:

            - v (int) : vertex of the active MTG

        :Optional Parameters:

            - RestrictedTo (str): cf. `Father`
            - ContainedIn (int): cf. `Father`
            - EdgeType (str): cf. `Father`

        :Returns:

            list of int.


        :Examples:

        .. code-block:: python

            >>> v
            78
            >>> g.Sons(v) # set of sons of v
            [78,99,101]
            >>> g.Descendants(v) # set of descendants of v
            [78,99,101,121,133,135,156,171,190]

        .. image:: ../user/mtg_descendants.png

        .. seealso:: :func:`MTG`, :func:`Ancestors`.
        """
        return list(algo.descendants(self, v,
                                     RestrictedTo=RestrictedTo,
                                     ContainedIn=ContainedIn))

    def Extremities(self, v, RestrictedTo='NoRestriction', ContainedIn=None):
        """
        Set of vertices that are the extremities of the branching system
        born by a given vertex.

        This function returns the extremities of the branching system defined by the argument
        as a list of vertices. These vertices have the same scale as `v` and their order in
        the list is not signifiant. The result is always a non empty array.

        :Usage:

        .. code-block:: python

            Extremities(v)

        :Properties:

            -  v (int) : vertex of the active MTG

        :Optional Parameters:

            - RestrictedTo (str): cf. :func:`Father`
            - ContainedIn (int): cf. :func:`Father`

        :Returns:

            list of vertices's id (int)


        :Examples:

        .. code-block:: python

            >>> g.Descendants(v)
            [3, 45, 47, 78, 102]
            >>> g.Extremities(v)
            [47, 102]

        .. seealso:: :func:`MTG`, :func:`Descendants`, :func:`Root`, :func:`MTGRoot`.
        """
        return list(algo.extremities(self, v, RestrictedTo=RestrictedTo, ContainedIn=ContainedIn))

    def Components(self, v, Scale=-1):
        """
        Set of components of a vertex.

        The set of components of a vertex is returned as a list of vertices.
        If **s** defines the scale of **v**, components are defined at scale **s** + 1.
        The array is empty if the vertex has no components.
        The order of the components in the array is not significant.

        When a scale is specified using optional argument :arg:Scale,
        it must be necessarily greater than the scale of the argument.

        :Usage:

        .. code-block:: python

            Components(v)
            Components(v, Scale=2)

        :Parameters:

            - v (int) : vertex of the active MTG

        :Optional Parameters:

            - Scale (int) : scale of the components.

        :Returns:

            list of int

        .. image:: ../user/mtg_components.png

        .. seealso:: :func:`MTG`, :func:`Complex`.
        """
        _g = self
        scale = _g.scale(v)
        components = []
        if Scale == -1 or scale == Scale:
            components = _g.components(v)
        elif scale < Scale:
            components = _g.components_at_scale(v, scale=Scale)
        return components

    def ComponentRoots(self, v, Scale=-1):
        """
        Set of roots of the tree graphs that compose a vertex

        In a MTG, a vertex may have be decomposed into components.
        Some of these components are connected to each other, while other are not.
        In the most general case, the components of a vertex are organized into several tree-graphs.
        This is for example the case of a MTG containing the description of several plants:
        the MTG root vertex can be decomposed into tree graphs (not connected)
        that represent the different plants.
        This function returns the set of roots of these tree graphs at scale *Scale(v)+1*.
        The order of these roots is not significant.

        When a scale different from *Scale(v)+1* is specified using the optional argument :func:`Scale`,
        this scale must be greater than that of the vertex argument.

        :Usage:

        .. code-block:: python

            g.ComponentRoots(v)
            g.ComponentRoots(v, Scale=s)

        :Parameters:

            - v (int) : vertex of the active MTG

        :Optional Parameters:

            - Scale (str): scale of the component roots.

        :Returns:

            list of vertices's id (int)


        :Examples:

        .. code-block:: python

            >>> v=g.MTGRoot() # global MTG root
            0
            >>> g.ComponentRoots(v) # set of first vertices at scale 1
            [1,34,76,100,199,255]
            >>> g.ComponentRoots(v, Scale=2) # set of first vertices at scale 2
            [2,35,77,101,200,256]

        .. image:: ../user/mtg_componentroots.png

        .. seealso:: :func:`MTG`, :func:`Components`, :func:`Trunk`.
        """
        return self.component_roots_at_scale(v, scale=Scale)

    def Path(self, v1, v2):
        """
        List of vertices defining the path between two vertices

        This function returns the list of vertices defining the path
        between two vertices that are in an ancestor relationship.
        The vertex `v1` must be an ancestor of vertex `v2`.
        Otherwise, if both vertices are valid, then the empty array is returned
        and if at least one vertex is undefined, None is returned.


        :Usage:

        .. code-block:: python

            g.Path(v1, v2)

        :Parameters:

            - `v1` (int) : vertex of the active MTG
            - `v2` (int) : vertex of the active MTG

        :Returns:

            list of vertices's id (int)


        :Examples:

        .. code-block:: python

            >>> v # print the value of v
            78
            >>> g.Ancestors(v)
            [78,45,32,10,4]
            >>> g.Path(10,v)
            [10,32,45,78]
            >>> g.Path(9,v) # 9 is not an ancestor of 78
            []

        .. note:: `v1` can be equal to `v2`.

        .. image:: ../user/mtg_path.png

        .. seealso:: :func:`MTG`, :func:`Axis`, :func:`Ancestors`.
        """
        return list(algo.path(self, v1, v2)[0])

    def Axis(self, v, Scale=-1):
        """
        Array of vertices constituting a botanical axis

        An axis is a maximal sequence of vertices connected by '<'-type edges.
        Axis return the array of vertices representing the botanical axis which the argument v belongs to.
        The optional argument enables the user to choose the scale at which the axis decomposition is required.

        :Usage:

        .. code-block:: python

            Axis(v)
            Axis(v, Scale=s)

        :Parameters:

            - v (int) : Vertex of the active MTG

        :Optional Parameters:

            - Scale (str): scale at which the axis components are required.

        :Returns:

            list of vertices ids

        .. image:: ../user/mtg_axis.png

        .. seealso:: :func:`MTG`, :func:`Path`, :func:`Ancestors`.
        """
        return list(algo.axis(self, v, scale=Scale))

    def Trunk(self, v, Scale=-1):
        """
        List of vertices constituting the bearing botanical axis of a branching system.

        Trunk returns the list of vertices representing the botanical axis defined as
        the bearing axis of the whole branching system defined by `v`.
        The optional argument enables the user to choose the scale at which the trunk should be detailed.

        :Usage:

        .. code-block:: python

            Trunk(v)
            Trunk(v, Scale= s)

        :Parameters:

            - `v` (int) : Vertex of the active MTG.

        :Optional Parameters:

            - `Scale` (str): scale at which the axis components are required.

        :Returns:

            list of vertices ids

        .. todo:: check the usage of the optional argument Scale

        .. seealso:: :func:`MTG`, :func:`Path`, :func:`Ancestors`, :func:`Axis`.
        """
        return list(algo.trunk(self, v, scale=Scale))

################################################################################
# Graph generators
################################################################################
def fat_mtg(slim_mtg, preserve_order=False):
    """
    Compute missing edges at each scales based on the explicit edges
    defines at finer scales and decomposition relationship.
    If preserve_order is True, the order of children at coarsest scales
    is deduced from the order of children at finest scale
    """
    from .algo import lowestCommonAncestor
    max_scale = slim_mtg.max_scale()
    #print 'max_scale %d'%max_scale
    #roots = slim_mtg.roots(scale=max_scale)
    #assert len(list(roots)) == 1
    edge_type_property = slim_mtg.property('edge_type')

    for scale in range(max_scale-1,0,-1):
        _compute_missing_edges(slim_mtg, scale, edge_type_property)
        if preserve_order:
            # deduce the order of children
            # from the order of their components
            # Do not use traversal.pre_order:
            # may switch < and + children
            from .traversal import post_order
            for v in slim_mtg.vertices(scale=scale):
                # children at current scale
                cref = slim_mtg.children(v)
                if len(cref) > 1:
                    # children at lowest scale
                    cmp = [slim_mtg.component_roots(x)[0] for x in cref]
                    cmp_dic = dict(list(zip(cmp, cref)))
                    visitor = lambda x, g=slim_mtg, l=cmp: not(g.parent(x) in l)
                    visitor.post_order = visitor
                    visitor.pre_order = visitor
                    # descendants of v at lowest scale, minus the descendants of cmp
                    descendants = list(post_order(slim_mtg, slim_mtg.component_roots(v)[0],
                                                  visitor_filter=visitor))
                    ordered_children = []
                    # read the children wrt order defined by "descendants"
                    for x in descendants:
                        if x in cmp:
                            ordered_children += [x]
                    ch = [cmp_dic[c] for c in ordered_children if c in cmp_dic]
                    msg = "Bad children at vertex " + str(v)
                    msg += str(cref) + " / " + str(ch)
                    assert set(ch) == set(cref), msg
                    slim_mtg._children[v] = ch
    return slim_mtg


def _compute_missing_edges(mtg, scale, edge_type_property=None):
    """ Compute missing edges on an incomplete MTG at a given scale.

    This is often usefull to create a minimal MTG with missing edges.
    The missing edges can be computing by using the tree at the finer scale,
    and by adding edges between the complexes of these nodes.

    For all the non connected nodes (root nodes)
        - extract the components
        - compute parent's components and the edge type between

    """
    roots = mtg.roots(scale=scale)
    #print 'roots: ', list(roots), scale
    for vid in roots:
        components = mtg._components.get(vid)
        if components is None:
            print('ERROR: Missing component for vertex %d'%vid)
            continue
        #assert len(components) == 1
        cid = components[0]
        if mtg.parent(cid) is None:
            continue
        parent_id = mtg.complex(mtg.parent(cid))
        if parent_id is None:
            #roots.append(vid)
            print('ERROR: Missing parent for vertex %d'%cid)
            continue
        if edge_type_property:
            edge_type = edge_type_property.get(cid)
            mtg.add_child(parent_id, child=vid, edge_type=edge_type)
        else:
            mtg.add_child(parent_id, child=vid)

    return True



################################################################################
# Graph generators
################################################################################

def simple_tree(tree, vtx_id, nb_children=3, nb_vertices=20):
    """ Generate and add a regular tree to an existing one at a given vertex.

    Add a regular tree at a given vertex id position `vtx_id`.
    The length of the sub_tree is `nb_vertices`. Each new vertex has
    at most `nb_children` children.

    :Parameters:
        - `tree`: the tree thaat will be modified
        - `vtx_id` (id): vertex on which the sub tree will be added.
        - `nb_children` (int) : number of children that are added to each vertex
        - `nb_vertices` (int) : number of vertices to add

    :Returns:

        The modified `tree`

    :Examples:

        .. code-block:: python

            g = MTG()
            vid = g.add_component(g.root)
            simple_tree(g, vid, nb_children=2, nb_vertices=20)
            print len(g) # 22

    .. seealso:: :func:`random_tree`, :func:`random_mtg`

    """

    vid = vtx_id
    l=[vid]
    while nb_vertices > 0:
        n = min(nb_children, nb_vertices)
        vid = l.pop(0)
        for i in range(n):
            v = tree.add_child(vid)
            nb_vertices -= 1
            l.append(v)
    return tree

def random_tree(mtg, root, nb_children=3, nb_vertices=20):
    """ Generate and add a random tree to an existing one.

    Add a random sub tree at a given vertex id position `root`.
    The length of the sub_tree is `nb_vertices`.
    The number of children for each vertex is sampled according to `nb_children` distribution.
    If nb_children is an interger, the random distribution is uniform between [1, nb_children].
    Otherwise, you can give your own discrete distribution sampling function.

    :Parameters:
        - `mtg`: the mtg to modified
        - `root` (id): vertex id on which the sub tree will be added.

    :Optional Parameters:
        - `nb_vertices`
        - `nb_children` : an int or a discrete distribution sampling function.

    :Returns:

        The last added vid.

    :Examples:

        .. code-block:: python

            g = MTG()
            vid = g.add_component(g.root)
            random_tree(g, vid, nb_children=2, nb_vertices=20)
            print len(g) # 22

        .. code-block:: python
            from scipy.stats import poisson, binom

            g = MTG()
            vid = g.add_component(g.root)
            dist = poisson(1., loc=1).rvs
            random_tree(g, vid, nb_children=dist)
            print len(g) # 22

            dist = binom(5, 0.5, loc=1).rvs
            random_tree(g, vid, nb_children=dist)


    .. seealso:: :func:`simple_tree`, :func:`random_mtg`

    """
    from random import randint
    vid = root
    l=[vid]
    while nb_vertices > 0:
        if callable(nb_children):
            n = min(nb_children(), nb_vertices)
        else:
            n = min(randint(1,nb_children), nb_vertices)

        vid = l.pop(randint(0,len(l)-1))
        for i in range(n):
            edge_type = '+'
            if i == n/2:
                edge_type='<'

            v=mtg.add_child(vid, edge_type=edge_type)
            nb_vertices -= 1
            l.append(v)
    return l[-1]

def random_mtg(tree, nb_scales):
    """ Convert a tree into an MTG of `nb_scales`.

    Add a random sub tree at a given vertex id position `root`.
    The length of the sub_tree is `nb_vertices`. Each new vertex has
    at most `nb_children` children.

    :Parameters:
        - `mtg`: the mtg to modified
        - `root` (id): vertex id on which the sub tree will be added.

    :Returns:

        The last added vid.

    :Examples:

        .. code-block:: python

            g = MTG()
            random_tree(g, g.root, nb_children=2, nb_vertices=20)
            print len(g) # 21

    .. seealso:: :func:`simple_tree`, :func:`random_tree`

    """

    n = len(tree)
    # colors contained the colored vertices at each scale
    # based on the vertex of the initial tree
    colors = {}
    colors[nb_scales-1] = tree.vertices()
    for s in range(nb_scales-2, 0, -1):
        n = random.randint(1, n)
        l = random.sample(colors[s+1], n)
        l.sort()
        if tree.root not in l:
            l.insert(0, tree.root)
        colors[s] = l


    return colored_tree(tree, colors)[0]


def colored_tree(tree, colors):
    """
    Compute a mtg from a tree and the list of vertices to be quotiented.

    .. note:: The tree has to be a real tree and not an MTG

    :Example:

    .. code-block:: python

        from random import randint, sample
        g = MTG()
        random_tree(g, g.root, nb_vertices=200)

        # At each scale, define the vertices which will define a complex
        nb_scales=4
        colors = {}
        colors[3] = g.vertices()
        colors[2] = random.sample(colors[3], randint(1,len(g)))
        colors[2].sort()
        if g.root not in colors[2]:
            colors[2].insert(0, g.root)
        colors[1] = [g.root]

        g, mapping = colored_tree(g, colors)

    """

    nb_scales = max(colors.keys())+1

    map_index = {}
    g = MTG()
    # scale 0: 1 vertex
    count = 1

    for scale in range(1, nb_scales):
        map_index[scale] = {}
        for id in colors[scale]:
            map_index[scale][id] = count
            count += 1

    # build the mtg
    # 1. Add multiscale info
    index_scale = map_index[1]
    for id in colors[1]:
        g.add_component(g.root,index_scale[id])
    # Edit the graph with multiscale info
    for scale in range(2, nb_scales):
        prev_index_scale = index_scale
        index_scale = map_index[scale]
        for id in colors[scale]:
            complex_id = prev_index_scale.get(id)
            component_id = index_scale.get(id)

            if complex_id:
                g.add_component(complex_id, component_id)
            elif component_id:
                g._scale[component_id] = scale

    # copy the tree information in the MTG
    if isinstance(tree, MTG):
        max_scale = tree.max_scale()
        g._parent.update(dict(((index_scale[k], index_scale[v])
                                for k, v in tree._parent.items() if v is not None and tree.scale(v) == max_scale)))
        for parent, children in tree._children.items():

            if tree.scale(parent) == max_scale:
                g._children[index_scale[parent]] = [index_scale[id] for id in children]
    else:
        g._parent.update(dict(((index_scale[k], index_scale[v])
                                for k, v in tree._parent.items() )))
        for parent, children in tree._children.items():
            g._children[index_scale[parent]] = [index_scale[id] for id in children]

    # Copy the properties of the tree
    for pname, prop in tree.properties().items():
        property = g._properties.setdefault(pname, {})
        for id, v in prop.items():
            if tree.scale(id) == max_scale:
                property[index_scale[id]] = v

    return fat_mtg(g), dict(list(zip(list(index_scale.values()),list(index_scale.keys()))))



################################################################################
# Utilities
################################################################################

def display_tree(tree, vid, tab = "", labels = {}, edge_type = {}):
    '''
    Display a tree structure.
    '''
    if not labels:
        labels = tree.property('label')
    if not edge_type:
        edge_type = tree.property('edge_type')

    et = '%s'%(edge_type.get(vid,'/'))
    assert vid in tree
    label = labels.get(vid, str(vid))
    yield tab+et+label
    for v in tree.children_iter(vid):
        if edge_type.get(v) == '+':
            tab +='\t'
        for s in display_tree(tree, v, tab, edge_type=edge_type, labels=labels):
            yield s
        if edge_type.get(v) == '+':
            tab=tab[:-1]

def display_mtg(mtg, vid):
    """ Display an MTG

    ..todo:: Write doc.
    """
    label = mtg.property('label')
    edge_type = mtg.property('edge_type')
    current_vertex = vid
    tab = 0
    for vtx in traversal.iter_mtg2(mtg, vid):
        et = '/'
        if vtx != current_vertex:
            scale1 = mtg.scale(current_vertex)
            scale2 = mtg.scale(vtx)
            if scale1 >= scale2:
                et = edge_type[vtx]
                if scale1 == scale2:
                    if mtg.parent(vtx) != current_vertex:
                        tab = -1
                        et = '^'+et
                    else:
                        et = '^'+et
                elif scale1 > scale2:
                    v = current_vertex
                    for i in range(scale1-scale2):
                        v = mtg.complex(v)
                    if mtg.parent(vtx) == v:
                        et = '^'+et
                    else:
                        tab -= 1
                        et = '^'+et

            else:
                assert scale2 - scale1 == 1
                tab += 1

            yield tab* '\t' + et + label.get(vtx, str(vtx))
        current_vertex = vtx



def return_proxy(f):
    mtg_f = getattr(MTG, f.__name__)

    def new_f(self, *args, **kwds):
        id = mtg_f(self._g,self._vid,*args,**kwds)
        if id is not None:
            return self.__class__(self._g, id)
        else:
            return
    new_f.__name__ = f.__name__
    new_f.__doc__ = mtg_f.__doc__
    return new_f

def proxy(f):
    mtg_f = getattr(MTG, f.__name__)
    def new_f(self, *args, **kwds):
        return mtg_f(self._g,self._vid,*args,**kwds)
    new_f.__name__ = f.__name__
    new_f.__doc__ = mtg_f.__doc__
    return new_f

def return_iter_proxy(f):
    mtg_f = getattr(MTG, f.__name__)
    def new_f(self, *args, **kwds):
        return [self.__class__(self._g, id) for id in mtg_f(self._g,self._vid,*args,**kwds)]
    new_f.__name__ = f.__name__
    new_f.__doc__ = mtg_f.__doc__
    return new_f

def return_tuple_proxy(f):
    mtg_f = getattr(MTG, f.__name__)
    def new_f(self, *args, **kwds):
        return tuple(self.__class__(self._g, id) for id in mtg_f(self._g,self._vid,*args,**kwds))
    new_f.__name__ = f.__name__
    new_f.__doc__ = mtg_f.__doc__
    return new_f

class _ProxyNode(object):
    def __init__(self, g, vid):
        self.__dict__['_g'] = g
        self.__dict__['_vid'] = vid

    def __setattr__(self,name, value):
        g = self._g; vid = self._vid
        if name not in g.property_names():
            g.add_property(name)
        g.property(name)[vid] = value

    def __getattr__(self, name):
        g = self._g; vid = self._vid

        if name in g.property_names():
            return g.property(name).get(vid)
        else:
            raise AttributeError(name)

    def __eq__(self, other):
        return self._vid == other._vid
    def __hash__(self):
        return hash(self._vid)

    def __str__(self):
        return '_ProxyNode(%d)'%self._vid
    def __repr__(self):
        return '_ProxyNode(%d)'%self._vid

    # Wrappers for MTG methods.
    @proxy
    def scale(): pass
    @proxy
    def edge_type(): pass
    @proxy
    def index(): pass
    @return_proxy
    def parent(): pass
    @return_iter_proxy
    def children(): pass
    @proxy
    def nb_children(): pass
    @return_iter_proxy
    def siblings(): pass
    @proxy
    def nb_siblings(): pass
    @return_proxy
    def complex(): pass
    @return_iter_proxy
    def components(): pass
    @proxy
    def nb_components(): pass
    @return_proxy
    def complex_at_scale(): pass
    @return_iter_proxy
    def components_at_scale(): pass
    @return_iter_proxy
    def component_roots(): pass
    @return_iter_proxy
    def component_roots_at_scale(): pass
    @return_proxy
    def add_child(): pass
    @return_tuple_proxy
    def add_child_and_complex(): pass
    @return_proxy
    def insert_parent(): pass
    @return_proxy
    def insert_sibling(): pass
    @return_proxy
    def add_component(): pass
    @proxy
    def remove_vertex(): pass
    @return_iter_proxy
    def remove_tree(): pass
    def properties(self):
        return self._g.get_vertex_property(self._vid)

