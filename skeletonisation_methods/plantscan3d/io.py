# -*- python -*-
#
#       OpenAlea.mtg
#
#       Copyright 2008-2016 INRIA - CIRAD - INRA
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
"""This module provides functions to read / write mtg data structure."""

import re
from string import Template
from warnings import warn


from .mtg import *
from .traversal import iter_mtg, iter_mtg_with_filter

# try:
#     from openalea.core.logger import get_logger, logging

#     logger = get_logger('openalea.mtg')
#     _ch = logging.StreamHandler()
#     logger.addHandler(_ch)
# except:
logger = None

debug = 0


def log(*args):
    if debug:
        if logger:
            logger.debug('  '.join(map(str, args)))
        else:
            print('  '.join(map(str, args)))

################## UTILS

def get_expr(s, expr):
    res = re.search(expr, s)
    _str = ''
    if res:
        _str = s[res.start():res.end()]
    return _str

def get_label(s):
    name = r'[a-zA-Z0-9]+'
    return get_expr(s, name)

def get_name(s):
    name = r'[a-zA-Z]+'
    return get_expr(s, name)

def get_index(s):
    name = r'[0-9]+'
    return get_expr(s, name)

def get_args(s):
    args = r'\([0-9,-\.\+]+\)'
    return get_expr(s, args)

def get_float(s):
    args = r'[0-9-\+]+'
    num = get_expr(s, args)
    return float(num)

def replace_date(s, format):
    """
    Replace the date / by -
    """
    import re
    if format == 'DD/MM/YY':
        rawstr = r"""(?P<day>3[01]|[1-9]|[0-2][0-9])/(?P<month>1[012]|0[1-9]|[1-9])/(?P<year>\d\d)"""
    else:
        rawstr = r"""(?P<day>3[01]|[0-2]{0,1}\d)/(?P<month>1[012]|0\d)/(?P<year>19\d\d|20\d\d)"""

    def change_date(match_obj):
        day, month, year = match_obj.group('day'), match_obj.group('month'), match_obj.group('year')
        return '-'.join((day, month, year))

    return re.sub(rawstr, change_date, s)

def multiscale_edit(s, symbol_at_scale = {}, class_type={}, has_date = False, mtg=None):
    """Construction of an MTG from a string.

    :Parameters:

    - `s`: The string representing the MTG.
    - `symbol_at_scale`: A dict containing the scale for each symbol name.

    :Optional parameters:

    - `class_type`: A dict containing the type of the properties.
	- `has_date`: Is the MTG is a Dynamic MTG?
	- `mtg`: An existing MTG



    :Return:

        MTG object

    """
    def get_properties(name,vid=None, time=False):
        _type = dict([('INT', int), ('REAL', float), ('ALPHA', str), ('DD/MM/YY', str), ('DD/MM/YYYY', str), ('STRING', str)])
        args = {}
        l = name.strip().split('(')
        if not time:
            label = get_label(name)
            index = get_index(label)
            if index.isdigit():
                args['index'] = int(index)
            args['label'] = label
        if len(l) > 1:
            arg_string = l[1].strip()[:-1]
            if arg_string:
                ln = arg_string.split(',')
                for arg in ln:
                    k, v = arg.split('=')
                    klass = _type[class_type[k]]
                    try:
                        args[k] = klass(v)
                    except:
                        if vid is not None:
                            print('Args ', v, 'of vertex ', vid, 'of type ', k, 'is not of type ', str(klass))
                        else:
                            print('Args ', v, 'of type ', k, 'is not of type ', str(klass))
        return args

    def add_dynamic_properties(mtg, vid, args):
        print("Existing properties at ", vid, " ", mtg.get_vertex_property(vid))
        print("New property: ", args)

        # a property can be a list but not a timeserie.
        # Create a real timeserie object...
        static_properties = ['index', 'label']
        is_time_series= lambda x: isinstance(x,list)
        date = 'Date'

        new_date = args.get(date)

        if new_date is None:
            return

        old_date = mtg.property(date)[vid]
        for prop in args:
            if prop == date:
                continue
            old = mtg.property(prop).get(vid)
            if is_time_series(old):
                old.append((new_date, args[prop]))
            elif old:
                mtg.property(prop)[vid] = [(old_date, old), (new_date, args[prop])]
            else:
                mtg.property(prop)[vid] = [(new_date, args[prop])]



    implicit_scale = bool(symbol_at_scale)

    if debug:
        print(list(symbol_at_scale.keys()))

    mtg = mtg if mtg else MTG()

    vid = mtg.root # vid of the support tree, i.e. at the finest scale
    current_vertex = mtg.root
    branching_stack = []

    if not implicit_scale:
        symbols = ['/', '\\', '[', ']', '+', '<', '<<']
    else:
        symbols = ['/', '[', ']', '+', '<', '<<']

    pending_edge = '' # edge type for the next edge to be created
    scale = 0

    # 2. add some properties to the MTG
    mtg.add_property('index')
    for k in class_type:
        mtg.add_property(k)

    # remove from the date format the /
    if has_date:
        # print('replace all the date format by -')
        if 'DD/MM/YY' in list(class_type.values()):
            date_format = 'DD/MM/YY'
        else:
            date_format = 'DD/MM/YYYY'
        s = replace_date(s, date_format)


    for edge_type in symbols:
        if edge_type != '/' or not symbol_at_scale:
            s = s.replace(edge_type, '\n%s'%edge_type)
        else:
            # do not consider the date format
            for klass in list(symbol_at_scale.keys()):
                s = s.replace('/%s'%klass, '\n/%s'%klass)
    s = s.replace('<\n<', '<<')
    # TODO: Write a regular expression to allow several spaces
    s = s.replace(')(', ')\n(')
    s = s.replace(') (', ')\n(')
    s = s.replace('*(', '\n(')


    l = list(filter( None, s.split('\n')))

    for node in l:
        if node.startswith('<<'):
            tag = '<<'
            name = node[2:]
        elif node.startswith('(') and has_date:
            tag = '*'
            name = node[:]
        else:
            tag = node[0]
            name = node[1:]
            assert tag in symbols, tag

        if tag == '[':
            branching_stack.append(vid)
        elif tag == ']':
            vid = branching_stack.pop()
            current_vertex = vid
            scale = mtg.scale(vid)
        elif tag == '*':
            args = get_properties(name, vid=vid, time=True)
            print(vid, '*(', args, ')')
            # CPL Manage Dynamic_MTG
            add_dynamic_properties(mtg, vid, args)
        else:
            if class_type:
                args = get_properties(name, vid=vid)
            else:
                label = get_label(name)
                index = get_index(name)
                args = {'label':label}
                if index.isdigit():
                    args['index'] = int(index)

            if implicit_scale:
                symbol_class = get_name(name)
                try:
                    new_scale = symbol_at_scale[symbol_class]
                except:
                    print('NODE ',node, bool(tag=='*'))
                if tag == '/' and new_scale <= scale:
                    new_scale -= 1
                    pending_edge = '/'
                while new_scale < scale:
                    scale -= 1
                    current_vertex = mtg.complex(current_vertex)

            if tag in ['+', '<']:
                if mtg.scale(vid) == scale:
                    vid = mtg.add_child(vid, edge_type=tag, **args)
                    current_vertex = vid
                    pending_edge = ''
                else:
                    complex = mtg.complex(current_vertex)
                    current_vertex = mtg.add_component(complex, **args)
                    pending_edge = tag
            elif tag == '<<':
                index = args['index']
                label = args['label']
                previous_index = mtg.property('index')[current_vertex]
                pending_edge = ''
                _args = {}
                for i in range(previous_index+1, index+1):
                    if i == index:
                        _args = args
                    _args['index'] = i
                    _args['label'] = label.replace(str(index), str(i))
                    vid = mtg.add_child(vid, edge_type='<', **_args)
                    current_vertex = vid
            elif tag == '/':
                if mtg.scale(vid) == scale:
                    vid = mtg.add_component(vid, **args)
                    current_vertex = vid
                    scale += 1
                elif mtg.scale(vid) > scale:
                    scale += 1
                    component = mtg.add_component(current_vertex, **args)
                    if mtg.scale(vid) == scale and pending_edge != '/':
                        vid = mtg.add_child(vid,
                                            child=component,
                                            edge_type=pending_edge)
                        assert vid == component
                        current_vertex = vid
                    else:
                        current_vertex = component
                        # two case :
                        # 1. up and down in scales E+A/U/E
                        # 2. /P/P
                        if pending_edge == '/':
                            vid = current_vertex
                            scale = mtg.scale(vid)
                else:
                    vid = mtg.add_component(current_vertex, **args)
                    current_vertex = vid
            elif tag == '\\':
                scale -= 1
                current_vertex = mtg.complex(current_vertex)

    mtg = fat_mtg(mtg)
    return mtg

def read_lsystem_string( string,
                         symbol_at_scale,
                         functional_symbol={},
                         mtg=None ):
    """Read a string generated by a lsystem.

    :Parameters:

    - `string`: The lsystem string representing the axial tree.
    - `symbol_at_scale`: A dict containing the scale for each symbol name.

    :Optional parameters:

    - `functional_symbol`: A dict containing a function for specific symbols.
        The args of the function have to be coherent with those in the string.
        The return type of the functions have to be a dictionary of properties: dict(name, value)

    :Return:

        MTG object
    """

    import openalea.plantgl.all as pgl
    s = string

    def transform(turtle, mesh):
        x = turtle.getUp()
        z = turtle.getHeading()

        bo = pgl.BaseOrientation(x, z^x)
        matrix = pgl.Transform4(bo.getMatrix())
        matrix.translate(turtle.getPosition())
        mesh = mesh.transform(matrix)
        return mesh


    # 1. Create the mtg structure.
    if mtg is None:
        mtg = MTG()

    # 2. add some properties to the MTG
    mtg.add_property('index')
    mtg.add_property('can_label')
    mtg.add_property('geometry')


    vid = mtg.root # vid of the support tree, i.e. at the finest scale
    current_vertex = mtg.root
    branching_stack = []

    pending_edge = '' # edge type for the next edge to be created
    scale = 0

    lsys_symbols = ['[', ']', '/', '+', '^', 'f']
    modules = list(symbol_at_scale.keys())
    symbols = lsys_symbols + modules

    index = dict(list(zip(list(symbol_at_scale.keys()), [0]*len(symbol_at_scale))))

    is_ramif = False

    # 2. Create a PlantGL Turtle...
    turtle = pgl.Turtle()

    max_scale = max(symbol_at_scale.values())

    for edge_type in symbols:
        if edge_type != 'f':
            s = s.replace(edge_type, '\n%s'%edge_type)
        else:
            s = s.replace('f(', '\nf(')
    l = s.split()

    try:
        plant_name = [s for s in list(symbol_at_scale.keys()) if 'plant' in s.lower()][0]
    except:
        ValueError("""Incorrect plant name (should be plant)""")

    for node in l:
        # Check if node is a module

        tag = node[0]

        if tag == '[':
            branching_stack.append(vid)
            turtle.push()
            is_ramif = True
        elif tag == ']':
            vid = branching_stack.pop()
            current_vertex = vid
            scale = mtg.scale(vid)
            turtle.pop()
            is_ramif = False
        elif tag == '/':
            args = get_args(node[1:])
            if args:
                angle = get_float(args[1:-1])
                turtle.rollR(angle)
            else:
                turtle.rollR()
        elif tag == '+':
            args = get_args(node[1:])
            if args:
                angle = get_float(args[1:-1])
                turtle.left(angle)
            else:
                turtle.left()
        elif tag == '^':
            args = get_args(node[1:])
            if args:
                angle = get_float(args[1:-1])
                turtle.up(angle)
            else:
                turtle.up()
        elif tag == 'f' and node[1] == '(':
            args = get_args(node[1:])
            if args:
                length = get_float(args[1:-1])
                if length > 0:
                    turtle.f(length)
            else:
                turtle.f()
        else:
            # add new modules to the mtg (i.e. add nodes)
            name = get_name(node)
            if name not in modules:
                print('Unknow element %s'% name)
                continue

            module_scale = symbol_at_scale[name]
            if is_ramif:
                edge_type = '+'
            else:
                edge_type = '<'

            log(node, module_scale, edge_type )

            if module_scale == scale:
                if mtg.scale(vid) == scale:
                    vid = mtg.add_child(vid, edge_type=edge_type, label=name)
                    current_vertex = vid
                    pending_edge = ''

                    log('','Cas 1.1', scale,
                        'mtg.scale(vid)', mtg.scale(vid),
                        'generated vertex', vid)

                    assert mtg.scale(vid) == module_scale
                else:
                    # add the edge to the current vertex
                    current_vertex = mtg.add_child(current_vertex,
                                                   edge_type=edge_type,
                                                   label=name)
                    log('', 'Cas 1.2', scale,
                        'mtg.scale(vid)', mtg.scale(vid),
                        'generated vertex', current_vertex)
                    assert mtg.scale(current_vertex) == module_scale
                is_ramif = False
            elif module_scale > scale:
                log('', 'Cas 2', scale, 'mtg.scale(vid)', mtg.scale(vid))

                old_current_vertex = current_vertex
                while module_scale > scale:
                    if mtg.scale(vid) == scale:
                        assert vid == current_vertex
                        vid = mtg.add_component(vid)
                        current_vertex = vid
                        log('', '', 'Cas 2.1', scale, 'generate new component', current_vertex)
                        scale += 1
                        if module_scale == scale:
                            assert mtg.scale(current_vertex) == module_scale
                            mtg.property('label')[current_vertex] = name
                            break
                    else:
                        scale += 1
                        current_vertex = mtg.add_component(current_vertex)
                else:
                    log(node, 'add_child(%d, child=%d)'%(old_current_vertex, current_vertex))
                    mtg.property('label')[current_vertex] = name
                    if mtg.scale(vid) == scale:
                        vid = mtg.add_child(vid, child=current_vertex, edge_type=edge_type)
                        is_ramif = False
            else:
                assert module_scale < scale
                while module_scale < scale:
                    scale -= 1
                    current_vertex = mtg.complex(current_vertex)
                else:
                    current_vertex = mtg.add_child(current_vertex, edge_type=edge_type, label=name)
                    assert mtg.scale(current_vertex) == module_scale

            # MANAGE the properties, the geometry and the indices!!!
            index[name] += 1
            if name == plant_name:
                for k in list(index.keys()):
                    if k != name:
                        index[k] = 0

            mtg.property('index')[current_vertex] = index[name]
            if name in functional_symbol:
                features = eval(node, functional_symbol)
                geom = features.get('geometry')
                canlabel = features.get('label')
                if geom:
                    # get the transformation from the turtle
                    geom = transform(turtle, geom)
                    mtg.property('geometry')[current_vertex] = geom

                    if name == 'StemElement':
                        # parse args to know how the turtle has to move .
                        args = get_args(node)[1:-1]
                        list_args= args.split(',')
                        length = float(list_args[1]) # 2nd arg
                        if length > 0:
                            turtle.f(length)

                if canlabel:
                    canlabel.elt_id = index[name]
                    plant_id = mtg.complex_at_scale(current_vertex, scale=1)
                    canlabel.plant_id = mtg.property('index')[plant_id]
                    mtg.property('can_label')[current_vertex] = canlabel

    mtg = fat_mtg(mtg)
    return mtg

def axialtree2mtg(tree, scale, scene, parameters = None):
    """Create an MTG from an AxialTree.

    Tha axial tree has been generated by LPy. It contains both modules with parameters.
    The geometry is provided by the scene.
    The shape ids are the same that the module ids in the axial tree.
    For each module name in the axial tree, a `scale` and a list of parameters should be defined.
    The `scale` dict allow to add a module at a given scale in the MTG.
    The `parameters` dict map for each module name a list of parameter name that are added to the MTG.


    :Parameters:

      - `tree`: The axial tree generated by the L-system
      - `scale`: A dict containing the scale for each symbol name.
      - `scene`: The scene containing the geometry.
      - `parameters`: list of parameter names for each module.

    :Return: mtg

    :Example:

    .. code-block:: python

        tree # axial tree
        scales = {}
        scales['P'] = 1
        scales['A'] = 2
        scales['GU'] = 3

        params ={}
        params['P'] = []
        params['A'] = ['length', 'radius']
        params['GU'] = ['nb_flower']

        g = axialtree2mtg(tree, scales, scene, params)

    .. seealso:: :func:`mtg2axialtree`, :func:`lpy2mtg`, :func:`mtg2lpy`
    """
    def scene_id(scene):
        d = {}
        if scene:
            for sh in scene:
                d.setdefault(sh.id,[]).append(sh)
        return d

    def change_id(axial_id, mtg_id):
        """
        Change the id of the shape in the scene by the id of the mtg element.
        """
        mtg.property('_axial_id')[mtg_id] = axial_id
        if geoms:
            if axial_id in geoms:
                for shape in geoms[axial_id]:
                    shape.id = mtg_id
                mtg.property('geometry')[mtg_id]=geoms[axial_id]
            else:
                #print 'Be careful : no id ', axial_id
                pass

    # The string represented by the axial tree...

    geoms = scene.todict() if scene else dict() # scene_id(scene)
    mtg = MTG()
    if scene:
        mtg.add_property('geometry')

    mtg.add_property('_axial_id')

    if parameters is None:
        parameters = {}
    for label in parameters:
        for p in parameters[label]:
            if p not in mtg.property_names():
                mtg.add_property(p)

    vid = mtg.root
    current_vertex = vid
    branching_stack = [vid]

    pending_edge = '' # edge type for the next edge to be created

    max_scale = max(scale.values())
    for aid, module in enumerate(tree):
        label = module.name
        if label == '[':
            branching_stack.append(vid)
            pending_edge = '+'
        elif label == ']':
            vid = branching_stack.pop()
            current_vertex = vid
            pending_edge = ''
        elif (label not in scale) and (label not in parameters):
            continue
        else:
            _scale = scale[label]
            _params = parameters.get(label, [])

            params = {}
            params['label'] = label
            for p in _params:
                if module.hasParameter(p):
                    params[p] = module.getParameter(p)
                # otherwise check in parameterset
                elif module.argSize() == 1:
                    try:
                        pset = module.args[0]
                        if p in list(pset.__dict__.keys()):
                            params[p] = pset.__dict__[p]
                    except:
                        pass

            if mtg.scale(vid) == mtg.scale(current_vertex) == _scale:
                # Add a vertex at the finer scale
                if pending_edge == '+':
                    edge_type = '+'
                else:
                    edge_type = '<'
                #check if the edge_type is a good one:
                if edge_type == '+' and current_vertex != branching_stack[-1]:
                    edge_type = '<'

                params['edge_type'] = edge_type
                vid = mtg.add_child(vid, **params)
                current_vertex = vid
                pending_edge = ''
            elif mtg.scale(vid) < max_scale:
                assert mtg.scale(vid) == mtg.scale(current_vertex)
                # Descend in scale for the first time
                vid = mtg.add_component(vid, **params)
                current_vertex = vid
            elif mtg.scale(current_vertex) < _scale:
                assert mtg.scale(current_vertex) == _scale - 1
                current_vertex = mtg.add_component(current_vertex, **params)
                if mtg.scale(vid) == _scale:
                    if pending_edge == '+':
                        edge_type = '+'
                    else:
                        edge_type = '<'
                    params['edge_type'] = edge_type

                    vid = mtg.add_child(vid,
                                        child=current_vertex,
                                        **params)
                    assert vid == current_vertex
                    pending_edge = ''
            else:
                while mtg.scale(current_vertex) >= _scale:
                    current_vertex = mtg.complex(current_vertex)
                assert mtg.scale(current_vertex) == _scale - 1
                current_vertex = mtg.add_component(current_vertex, **params)
                pending_edge = ''

            #assert mtg.scale(current_vertex) == _scale

            #if max_scale == _scale:
            change_id(aid,current_vertex)

    mtg = fat_mtg(mtg)
    return mtg

def mtg2axialtree(g, parameters=None, axial_tree=None):
    """
    Create a MTG from an AxialTree with scales.

    :Parameters:

      - `axial_tree`: The axial tree managed by the L-system.
        Use an empty AxialTree if you do not want to concatenate this axial_tree with previous results.
      - `parameters`: list of parameter names for each module.

    :Return: mtg

    :Example:

    .. code-block:: python

        params = dict()
        params ['P'] = []
        params['A'] = ['length', radius']
        params['GU']=['nb_flower']
        tree = mtg2axialtree(g, params)

    .. seealso:: :func:`axialtree2mtg`, :func:`mtg2lpy`
    """

    edge_type = g.properties().get('edge_type', {})

    if parameters is None:
        parameters = {}

    tree = axial_tree
    if tree is None:
        import openalea.lpy as lpy
        tree = lpy.AxialTree()

    # Root of the MTG at scale 0
    vtx_id = next(g.roots_iter(scale=0))

    def axialtree_pre_order_visitor(vid, tree=tree):
        if vid == g.root:
            return True

        et = edge_type.get(vid, '/')
        if et in ('+', '/'):
            tree += '['

        name = g.class_name(vid)
        if not name:
            return False

        l = [name]

        params = parameters.get(name, [])
        if 'parameter_set' in params and len(params) == 1:
            from openalea.lpy.parameterset import ParameterSet
            exclude = ['geometry','label','edge_type','_axial_id']
            pset = {}
            for p in g.property_names():
                if not p in exclude:
                    arg = g.property(p).get(vid)
                    if arg is None:
                        continue
                    pset[p] = arg
            l.append(ParameterSet(**pset))
        else:
            for p in params:
                arg = g.property(p).get(vid)
                if arg is None:
                 # Be Careful, the argument is skipped if not defined.
                    continue
                l.append(arg)

        tree += tuple(l)
        return True

    def axialtree_post_order_visitor(vid, tree=tree):
        et = edge_type.get(vid, '/')
        if et in ('+', '/'):
            tree += ']'


    for v in traversal.iter_mtg2_with_filter(g, vtx_id,
                    axialtree_pre_order_visitor,
                    axialtree_post_order_visitor):

        pass

    return tree


def lpy2mtg(axial_tree, lsystem, scene = None):
    l = lsystem
    l.makeCurrent()
    context = l.context()
    modules = context.declaredModules()
    parameters = {}
    scales = {}
    for m in modules:
        label = m.name
        parameters[label] = m.parameterNames
        if 'parameter_set' in m.parameterNames and len(m.parameterNames) == 1:
            if axial_tree.count(label) > 0:
                index = axial_tree.find(label + '(p)')
                pset = axial_tree[index].args[0]
                parameters[label] = list(pset.__dict__.keys())
        scales[label] = m.scale

    tree = axial_tree
    if scene is None:
        scene = l.sceneInterpretation(tree)
    l.done()
    mtg = axialtree2mtg(tree, scales, scene, parameters)

    return mtg

def mtg2lpy(g, lsystem, axial_tree=None):
    """
    Create an AxialTree from a MTG with scales.

    :Usage:

    .. code-block:: python

        tree = mtg2lpy(g,lsystem)

    :Parameters:

        - `g`: The mtg which have been generated by an LSystem.
        - `lsystem`: A lsystem object containing various information.
            The `lsystem` is only used to retrieve the context and
            the parameters associated with each module name.

    :Optional Parameters:

        - `axial_tree`: an empty axial tree.
            It is used to avoid complex import in the code.

    :Return: axial tree

    .. seealso:: :func:`mtg2axialtree`
    """
    # Retrieve the set of modules, their label, scale and proerty names.

    l = lsystem
    l.makeCurrent()
    context = l.context()
    modules = context.declaredModules()
    parameters = {}
    for m in modules:
        parameters[m.name] = m.parameterNames
    l.done()

    return mtg2axialtree(g, parameters, axial_tree)





def mtg2mss(name, mtg, scene, envelop_type = 'CvxHull'):
    """ Convert an MTG into the multi-scale structure implemented by fractalysis.

    :Parameters:
        - `name`: name of the structure
        - `mtg`: the mtg to convert
        - `scene`: the scene containing the geometry
        - `envelop_type`: algorithm used to fit the geometry.between scales.

    :Returns: mss data structure.
    """
    from openalea.fractalysis.light import ssFromDict

    l = []
    for scale in range(1, mtg.nb_scales()-1):
        d = {}
        for vid in mtg.vertices_iter(scale=scale):
            d[vid] = mtg.components(vid)
        l.append(d)
    return ssFromDict(name, scene, l, envelop_type)


###############################################################################
# Class and methods to read the famous MTG file format.
###############################################################################

class Reader(object):
    """
    Parse a MTG string from a classic MTG file format.

    The mtg format is composed of a header and the mtg code.
    The header is used to construct and validate the mtg.
    The code contains topology relations and properties.
    """

    def __init__(self, string, has_line_as_param=True, mtg=None, has_date=False):
        self.mtg = mtg

        # First implementation.
        # Do not store 3 time the structure (mtg, txt and lines)
        self.txt = string
        self.lines = string.splitlines()

        # header information
        self._code = ""
        self._symbols = {}
        self._description = None
        self._features = {}
        self.has_date = has_date

        # debug
        self._no_line = 0
        self.warnings = []
        self.has_line_as_param = has_line_as_param

    def parse(self):
        """
        """
        self.header()
        self.code()

        self.errors()
        return self.mtg

    def header(self):
        """
        Parse an MTG header and create the mtg datastructure.

        An mtg header contains different parts:
            - code:  definition
            - classes: symbol name and scale
            - description: allowed relationship between symbols
            - features: property name and type
        """

        # 1. Read the file from the begining
        self._no_line = -1
        self.code_form()
        self.classes()
        self.description()
        self.features()


    def check(self):
        """
        Check the validity of the MTG without building it.
        """
        return True

    #### internal methods ####
    def code_form(self):
        """
        CODE: FORM-A / FORM-B
        """
        l = self._next_line()
        l = l.split('#')[0]
        code = l.split(':')
        if len(code) == 2 and 'CODE' in code[0] and 'FORM-' in code[1]:
            self._code = code[1]
        else:
            # error
            self.warnings.append((self._no_line, "Code form error"))

    def classes(self):
        """
        CLASSES:
        SYMBOL  SCALE   DECOMPOSITION   INDEXATION  DEFINITION
        ...
        """
        decomp = ['NONE', 'FREE', 'CONNECTED', 'NOTCONNECTED', 'LINEAR', 'PURELINEAR', '<-LINEAR', '+-LINEAR']
        l = self._next_line()
        if not l.startswith('CLASSES'):
            self.warnings.append((self._no_line, "CLASSES section not found."))

        l = self._next_line()
        l = l.split('#')[0]
        class_header = l.split()
        if class_header != ['SYMBOL', 'SCALE', 'DECOMPOSITION', 'INDEXATION', 'DEFINITION']:
            self.warnings.append((self._no_line, "CLASS header error."))

        while l:
            l = self._next_line()
            if l.startswith('DESCRIPTION'):
                break
            l = l.split('#')[0]
            line = l.split()
            if len(line) != 5:
                self.warnings.append((self._no_line, "CLASS error."))
                break
            else:
                symbol, scale, decomposition, indexation, definition = line
                # validation
                if not symbol.isalpha() and symbol != '$':
                    self.warnings.append((self._no_line, "Bad symbol %s."%symbol))
                if not scale.isdigit():
                    self.warnings.append((self._no_line, "Bad scale %s."%scale))
                if decomposition not in decomp:
                    self.warnings.append((self._no_line, "Bad decomposition id %s."%decomposition))
                # TODO: validate indexation
                if definition not in ['IMPLICIT', 'EXPLICIT']:
                    self.warnings.append((self._no_line, "Bad definition %s."%definition))

                if symbol != '$':
                    self._symbols[symbol] = int(scale)

        if l.startswith('DESCRIPTION'):
            self._no_line -= 1

    def description(self):
        """
        DESCRIPTION:
        LEFT RIGHT RELTYPE MAX
        U U,I + ?
        ...
        """
        l = self._next_line()
        if not l.startswith('DESCRIPTION'):
            self.warnings.append((self._no_line, "DESCRIPTION section not found."))

        l = self._next_line()
        l = l.split('#')[0]
        desc_header = l.split()
        if desc_header != ['LEFT', 'RIGHT', 'RELTYPE', 'MAX']:
            self.warnings.append((self._no_line, "DESCRIPTION header error."))

        while l :
            l = self._next_line()
            if l.startswith('FEATURES'):
                break
            l = l.split('#')[0]
            line = l.split()
            if len(line) < 2:
                self.warnings.append((self._no_line, "Class description error."))
                continue

            left = line[0]
            if left not in self._symbols:
                self.warnings.append((self._no_line, "Unknown left symbol %s."%left))
            right = ''.join(line[1:-2])
            rights = [symbol.strip() for symbol in right.split(',')]
            
            bad_right=  [x for x in rights if x not in self._symbols]

            if bad_right:
                self.warnings.append((self._no_line, "Unknown right symbols %s."%bad_right))

            reltype, _max = line[-2:]
            if reltype not in ['+', '<']:
                self.warnings.append((self._no_line, "Unknown relation type %s."%reltype))

            if _max != '?' and not _max.isdigit():
                msg = "Error in the maximum number of relationships (%s)."%_max
                msg += "Give a number or ?"
                self.warnings.append((self._no_line, msg))

        if l.startswith('FEATURES'):
            self._no_line -= 1

    def features(self):
        """
        FEATURES:
        NAME TYPE
        nb_plant INT
        """
        l = self._next_line()
        if not l.startswith('FEATURES'):
            self.warnings.append((self._no_line, "FEATURES section not found."))

        l = self._next_line()
        l = l.split('#')[0]
        f_header = l.split()
        if f_header != ['NAME', 'TYPE']:
            self.warnings.append((self._no_line, "FEATURES header error."))

        while l:
            l = self._next_line()
            if not l or l.startswith('MTG'):
                break
            l = l.split('#')[0]
            line = l.split()

            if len(line) != 2:
                self.warnings.append((self._no_line, "FEATURE description error."))
                continue
            name, _type = line
            if '/' in _type and 'date' in name.lower():
                self.has_date = True
            self._features[name] = _type

        if l.startswith('MTG'):
            self._no_line -= 1
        # add _line feature as int
        if self.has_line_as_param:
            self._features['_line'] = 'INT'

    def _next_line(self):
        self._no_line += 1
        if self._no_line == len(self.lines):
            self._no_line -= 1
            return ""

        l = self.lines[self._no_line]
        l1 = l.strip()
        if not l1 or l1[0] == '#':
            return self._next_line()
        else:
            return l

    def next_line_iter(self):
        l = self._next_line()
        while l:
            yield l
            l = self._next_line()

    def errors(self):
        nb_lines = len(self.lines)
        for id, warning in self.warnings:
            if id < nb_lines:
                print("== Line %d: %s"%(id, self.lines[id]))
                print(warning)
            else:
                print(id, " ", warning)
    ############################################################################
    ### Parsing of the MTG code
    ### That's the real stuff...
    ############################################################################

    def code(self):
        """
        Parse the code and populate the MTG.
        """
        l = self._next_line()

        if not l.startswith('MTG'):
            self.warnings.append((self._no_line, "MTG section not found."))

        l = self._next_line()
        if not l.startswith('ENTITY-CODE') and not l.startswith('TOPO'):
            self.warnings.append((self._no_line, "ENTITY-CODE or TOPO not found."))

        l = l.split('#')[0]
        features = l.split()[1:]
        self._nb_features = len(features)
        self._feature_head = []
        for feature in features:
            if feature not in self._features:
                self.warnings.append((self._no_line, "Error in ENTITY-CODE: Feature %s is unknown."%feature))
            else:
                self._feature_head.append(feature)

        code_topo = l[:l.find(features[0])] if self._nb_features else l[:]
        nb_cols = len(code_topo.split('\t'))
        self._feature_slice = slice(nb_cols-1, nb_cols-1+self._nb_features)

        self.preprocess_code()
        self.build_mtg()



    def preprocess_line(self, s, diff_space, indent, nb_spaces, edge_type):
        """
        Preprocess a line.
        """
        if (debug):
            print('line :%s, nb_spaces: %d, diff_space: %d, edge_type:%s, %s'%(s,
                                                                               nb_spaces,
                                                                               diff_space,
                                                                               str(edge_type),
                                                                               str(indent)))
        if diff_space == 0:
            if s.startswith('^') or s.startswith('*'):
                s = s[1:]
            elif edge_type:
                elt = ''
                if edge_type[-1] in ['+', '/']:
                    elt = ']'
                if s[0] in ['+','/']:
                    if elt == ']':
                        edge_type.pop()
                    edge_type.append(s[0])
                    s = elt+'[' + s
            elif s[0] in ['+']:
                edge_type.append(s[0])

        elif diff_space > 0:
            # indent
            if s.startswith('^'):
                print('ERROR %s'%s)
            indent.append(nb_spaces)
            if s[0] in ['+','/']:
                edge_type.append(s[0])
                s = "[" + s
            else:
                edge_type.append(s[0])
        else:
            # unindent
            brackets = []
            # Close the previous brackets
            while nb_spaces - indent[-1] < 0:
                indent.pop()
                if edge_type:
                    edge = edge_type.pop()
                    if edge in ['+','/']:
                        brackets.append(']')

            # Same case as diff_space == 0
            assert nb_spaces - indent[-1] == 0
            if s.startswith('^'):
                s = s[1:]
            elif edge_type:
                elt=''
                if edge_type[-1] in ['+','/']:
                    elt = ']'
                if s[0] in ['+','/']:
                    if elt == ']':
                        edge_type.pop()
                    edge_type.append(s[0])
                    s = elt+'[' + s

            s = ''.join(brackets+[s])


        return s, edge_type


    def preprocess_code(self):

        code = [l for l in self.lines[self._no_line+1:] if l.strip() and not l.strip().startswith('#')]

        indent = [0]
        edge_type = []
        new_code = []

        #for l in code:
        for l in self.next_line_iter():
            #l = l.expandtabs(4)
            s = l.strip()
            s= s.split()[0]
            # args
            args = l.split('\t')[self._feature_slice]
            params = [ "%s=%s"%(k,v) for k, v in zip(self._feature_head, args) if v.strip()]

            if self.has_line_as_param:
                params.append("_line=%d"%self._no_line)
            if params:
                s = s + "("+','.join(params)+")"

            # build
            nb_spaces = len(l) - len(l.lstrip('\t'))

            diff_space = nb_spaces - indent[-1]

            #s = self.preprocess_line(s, diff_space, indent, nb_spaces, edge_type)
            s, edge_type = self.preprocess_line(s, diff_space, indent, nb_spaces, edge_type)

            new_code.append(s)

        while edge_type:
            edge = edge_type.pop()
            if edge in ['+','/']:
                new_code.append(']')

        self._new_code = ''.join(new_code)
        if debug:
            print(self._new_code)

    def build_mtg(self):
        """
        """
        self.mtg = multiscale_edit(self._new_code, self._symbols, self._features, self.has_date, mtg=self.mtg)
        #self.mtg = multiscale_edit(self._new_code, {}, self._features)

def read_mtg(s, mtg=None, has_date=False):
    """ Create an MTG from its string representation in the MTG format.

    :Parameter:
        - s (string) - a multi-lines string

    :Return: an MTG

    :Example:

    .. code-block:: python

        f = open('test.mtg')
        txt = f.read()

        g = read_mtg(txt)

    .. seealso:: :func:`read_mtg_file`.

    """
    reader = Reader(s, mtg=mtg, has_date=has_date)
    g = reader.parse()
    return g

def read_mtg_file(fn, mtg=None, has_date=False):
    """ Create an MTG from a filename.

    :Usage:

        >>> g = read_mtg_file('test.mtg')

    .. seealso:: :func:`read_mtg`.
    """
    f = open(fn)
    txt = f.read()
    f.close()
    return read_mtg(txt, mtg=mtg, has_date=has_date)


def mtg_display(g, vtx_id, tab='  ', edge_type=None, label=None):
    """
    Test the traversal of an mtg.
    A first step before writing it.
    """
    from . import traversal
    if not edge_type:
        edge_type = g.properties().get('edge_type', {})
    if not label:
        label= g.properties().get('label', {})

    prev = vtx_id
    prev_order = 0
    prev_scale = g.scale(vtx_id)
    for  vid in traversal.iter_mtg(g, vtx_id):
        if prev == vid:
            continue

        name = label.get(vid, vid)

        if vid in edge_type:
            et = edge_type[vid]
        elif prev == g.parent(vid):
            et = '<'
        else:
            et = '?'

        space = ''

        scale = g.scale(vid)
        order = g.order(vid)

        if prev == g.complex(vid):
            et = '/'
            # add one blank line
            space = '^'
        elif prev_scale == scale and et == '<':
            space = '^'

        if scale < prev_scale:
            yield ''

        order = g.order(vid)

        prev = vid
        prev_scale = scale
        prev_order = order
        if order != prev_order:
            indent = 0

        yield (order*tab) +space+et+ name


###############################################################################
# Class and methods to write in the famous MTG file format.
###############################################################################

class Writer(object):
    """
     Write a MTG string from a mtg object.

    The mtg format is composed of a header and the mtg code.
    The header is used to construct and validate the mtg.
    The code contains topology relations and properties.
    """

    def __init__(self, g, header = '' ):

        self.g = g
        self._header = header

    def header(self):
        """
        Build the MTG header from the datastructure.

        An mtg header contains different parts:
            - code:  definition
            - classes: symbol name and scale
            - description: allowed relationship between symbols
            - features: property name and type
        """

        code = self._code()
        classes = self.classes()
        desc = self.description()
        features = self.features()

    def code(self, property_names, nb_tab=None,
             display_id=False, display_scale=False, filter=None):
        """
        Traverse the MTG and write the code.
        """
        if nb_tab is None:
            from .algo import orders
            nb_tab = max(orders(self.g))+1
       
        head = ['MTG :']

        entity = ['ENTITY-CODE']
        entity.extend((nb_tab-1)*[''])
        entity.extend(property_names)
        head.append('\t'.join(entity))

        # Create for each line a string with code and propertie values.
        # TODO : duplication of code from display_mtg and mtg_display.
        labels = self.g.property('label')
        edge_type = self.g.property('edge_type')

        properties = self.g.properties()
        current_vertex = self.g.root
        tab = 0
        prev_scale = 0

        sym_at_col = []

        for vtx in traversal.iter_mtg2(self.g, current_vertex):

            if filter and not filter(self.g, vtx):
                continue

            log('Process ',vtx, self.g.node(vtx).label)

            cur_scale = self.g.scale(vtx)
            if vtx == current_vertex:
                current_vertex = vtx
                prev_scale = cur_scale
                sym_at_col.append(vtx)
                continue

            # Algorithm description:
            # prev_scale >= cur_scale:
            #   1. search the parent
            #   2. if < same column elif + : tab = col+1
            complex = self.g.complex(vtx)
            if current_vertex == complex:
                et = '/'
                if current_vertex != self.g.root:
                    et = '^'+et

                log('  ','Cas / ',self.g.node(current_vertex).label, vtx, et)

            else:
                et = edge_type.get(vtx,'/')
                parent = self.g.parent(vtx)
                possible_et = possible_tab = None
                log('  ','Cas 2:', et, 'parent:',parent, 'sym_at_col: ',sym_at_col)
                for i in range(tab, -1, -1):
                    vc = v = sym_at_col[i]
                    vscale = self.g.scale(v)
                    log('    col '+str(i),cur_scale, v,'scale',vscale)

                    vtx_proj = vtx
                    parent_proj = parent
                    if vscale > cur_scale:
                        # up
                        for j in range(vscale-cur_scale):
                            vc = self.g.complex(vc)
                        #down
                        # Even if the complex are linked together, several solution can coexist
                        vtx_proj = next(self.g.component_roots_at_scale_iter(vtx,scale=vscale))
                        parent_proj = self.g.parent(vtx_proj)

                    if vc == parent and v == parent_proj:
                        log('   ==> cas 1')
                        if et == '<':
                            et = '^'+et
                            tab = i
                        else:
                            if i+1 < nb_tab:
                                tab = i+1
                            else:
                                et = '^'+et
                                tab = i
                        break
                    elif vc == parent:
                        log('   ==> cas 2')
                        if et == '<':
                            possible_et = '^'+et
                            possible_tab = i
                        else:
                            if i+1 < nb_tab:
                                possible_tab = i+1
                            else:
                                possible_et = '^'+et
                                possible_tab = i
                    elif i == 0 and self.g.complex(vc) == self.g.complex(vtx)==self.g.root:
                        if not possible_et:
                            tab = 0
                            break

                else:
                    #print sy
                    log('    Possible Error. Use hypothetic state if possible.')
                    if possible_et and possible_tab:
                        et = possible_et
                        tab = possible_tab
                    else:
                        print(tab)
                        print(sym_at_col)
                        raise Exception("Error in the MTG for vertex %d"%vtx)

            if tab >= nb_tab:
                msg = """There is not enough tabs to store the MTG code.
                Increase the nb_tab variable to at least %d"""
                raise Exception(msg%(nb_B2tab+2))


            # Create a valid line with properties.
            label = labels.get(vtx, str(vtx))

            if not display_id and not display_scale:
                name = '%s%s'%(et,get_label(label))
            elif display_id and display_scale:
                name = '%s%s\t\t\t(id=%d, scale=%d)'%(et,get_label(label),vtx, self.g.scale(vtx))
            elif display_id:
                name = '%s%s\t\t\t(id=%d)'%(et,get_label(label),vtx)
            else:
                name = '%s%s\t\t\t(scale=%d)'%(et,get_label(label),self.g.scale(vtx))

            line = ['']*nb_tab
            line[tab] = name

            log(' -> Add vertex', line[:tab+1], '(%d)'%tab )

            for pname in property_names:
                if vtx in properties[pname]:
                    p = properties[pname].get(vtx,'')
                    line.append(str(p))
                else:
                    line.append('')

            head.append('\t'.join(line))

            current_vertex = vtx

            if len(sym_at_col)==tab:
                sym_at_col.append(vtx)
            else:
                assert len(sym_at_col) > tab
                sym_at_col = sym_at_col[:tab+1]
                sym_at_col[tab] = vtx

        return head


    @staticmethod
    def _code(code='A'):
        """
        Define the MTG code format.
        """
        if code not in ['A', 'B']:
            code = 'A'

        return "CODE :  \tFORM-%s" % code

    @staticmethod
    def _classes(symbols):
        """
        Define the different symbols with their scale.
        symbols is a list of dictionary with specific keys:
            - symbol is the class name
            - scale is an positive integer
            - decomposition (optional) is in [FREE, LINEAR, CONNECTED, +-LINEAR, <-LINEAR, NOTCONNECTED, NONE]
            - indexation (optional) is FREE or CONSECUTIVE
            - definition (optional) is EXPLICIT or IMPLICIT
        """

        klass = ['CLASSES :']

        head = ['SYMBOL', 'SCALE', 'DECOMPOSITION', 'INDEXATION', 'DEFINITION']
        klass.append('\t'.join(head))

        default = dict(decomposition='FREE', indexation='FREE', definition='EXPLICIT')
        template = Template('\t'.join(['$symbol', '$scale', '$decomposition', '$indexation', '$definition']))

        d = default.copy()
        d.update(dict(symbol='$', scale='0', definition='IMPLICIT'))

        s = template.substitute(d)
        klass.append(s)

        for sdict in symbols:
            d = default.copy()
            d.update(sdict)
            s = template.substitute(d)
            klass.append(s)

        return '\n'.join(klass)

    @staticmethod
    def _description(scale_symbol):
        """
        Generate the description header file for a MTG.
        scale_symbol is a dict that associate a scale integer with the different symbols at this scale.
        """

        desc = ['DESCRIPTION :']
        head = ['LEFT', 'RIGHT', 'RELTYPE', 'MAX']
        desc.append('\t'.join(head))

        template = Template('\t'.join(['$left', '$right', '$reltype', '$max']))

        d = {}
        d['max'] = '?'
        scales = sorted(scale_symbol.keys())
        for scale in scales:
            symbols = scale_symbol[scale]
            d['right'] = ','.join(symbols)
            for s in symbols:
                d['left'] = s
                for edge_type in ['<', '+']:
                    d['reltype'] = edge_type
                    l = template.substitute(d)
                    desc.append(l)

        return '\n'.join(desc)

    @staticmethod
    def _features(name_type):
        """
        Generate the Feature header.
        name_type is a list of tuple containing the property name and the associated property type.
        type is INT, REAL, ALPHA or DATE (DD/MM, DD/MM/YY, DD/MM-TIME, DD/MM/YY-TIME).
        """
        predefined_types = ['INT', 'REAL', 'ALPHA', 'DD/MM', 'DD/MM/YY', 'DD/MM-TIME', 'DD/MM/YY-TIME', 'STRING']
        features = ['FEATURES :']
        features.append('\t'.join(['NAME', 'TYPE']))
        for name, type_ in name_type:
            if type_ not in predefined_types:
                warn('The type %s for the feature %s is not allow'%(type_, name), SyntaxWarning)
                continue
            features.append('\t'.join([name, type_]))

        return '\n'.join(features)


    @staticmethod
    def _scale2symbol(scales):
        """
        scale is a dict mapping scale to a list of symbols.
        Returns a list of dict with two keys symbol and scale.
        """
        symbols = []
        for s, classes in scales.items():
            for class_ in classes:
                symbols.append(dict(scale=str(s), symbol=class_))

        return symbols

def write_mtg(g, properties=[], class_at_scale=None, nb_tab=None, display_id=False):
    """ Transform an MTG into a multi-line string in the MTG format.

    This method build a generic header, then traverses the MTG and transform
    each vertex into a line with its label, topoloical relationship and
    specific `properties`.

    :Parameters:

        - `g` (MTG)
        - `properties` (list): a list of tuples associating a property name with its type.
            Only these properties will be written in the out file.


    :Optional Parameters:

        - `class_at_scale` (dict(name->int)): a map between a class name and its scale.
            If `class _at_scale` is None, its value will be computed from `g`.
        - `nb_tab` (int): the number of tabs used to write the code.
        - `display_id` (bool): display the id for each vertex

    :Returns: a list of strings.

    :Example:

    .. code-block:: python

        # Export all the properties defined in `g`.
        # We consider that all the properties are real numbers.

        properties = [(p, 'REAL') for p in g.property_names() if p not in ['edge_type', 'index', 'label']]
        mtg_lines = write_mtg(g, properties)

        # Write the result into a file example.mtg

        filename = 'example.mtg'
        f = open(filename, 'w')
        f.write(mtg_lines)
        f.close()
    """

    w = Writer(g)

    header = [w._code()]
    header.append('')

    if not class_at_scale:
        label = g.property('label')
        class_at_scale = dict(((get_name(lab),g.scale(id)) for id, lab in label.items()))

    scales = {}
    for class_, scale in class_at_scale.items():
        scales.setdefault(scale, []).append(class_)

    symbols = w._scale2symbol(scales)
    class_str = w._classes(symbols)
    header.append(class_str)
    header.append('')

    header.append(w._description(scales))
    header.append('')

    features = w._features(properties)
    header.append(features)
    header.append('')

    property_name = [p[0] for p in properties]
    code = w.code(property_name, nb_tab=nb_tab, display_id=display_id, filter=lambda g,v: True if g.scale(v) <=4 else False)

    header.extend(code)
    header.append('')

    return '\n'.join(header)

def display(g, max_scale=0, display_id=True, display_scale=False, nb_tab=None,**kwds):
    """
    Display MTG
    """
    w = Writer(g)

    if max_scale:
        f = lambda g,v: True if g.scale(v) <= max_scale else False
    else:
        f = None

    code = w.code([], nb_tab=nb_tab, display_id=display_id, display_scale=display_scale, filter=f)

    return '\n'.join(code[2:])

