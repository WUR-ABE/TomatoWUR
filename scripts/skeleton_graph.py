import networkx as nx
from pathlib import Path
import pandas as pd
import numpy as np
import math
import sys
from copy import deepcopy

from dataclasses import dataclass
sys.path.append("")
from scripts import visualize_examples as ve
from scripts.postprocessing_methods import gaussian_weight, find_closest_points, create_new_points
from scripts.calculate_angles import openalea_method, xy_plane_method
from scripts.graph_to_tree import graph_edges_to_tree, get_single_edge_type, direction
from scripts.utils_data import get_rotation_matrix

class SkeletonGraph():
	'''
	A class to represent a skeleton graph for 3D structures.
	Attributes
	----------
	G : networkx.DiGraph
		The directed graph representing the skeleton.
	G_original : networkx.DiGraph
		A copy of the original graph before any filtering.
	df_pc : pandas.DataFrame
		DataFrame containing point cloud data.
	name : str
		Name of the skeleton graph.
	mapping_reverse : dict
		Reverse mapping of node indices after filtering.
	Methods
	-------
	__init__(name=None)
		Initializes the SkeletonGraph object.
	load(nodes, edges, edge_types, df_pc=None, name=None, attributes={})
		Loads the skeleton graph from given nodes, edges, and edge types.
	get_node_order()
		Calculates and assigns the order of nodes in the graph.
	get_node_attribute(attribute="pos")
		Retrieves a specified attribute for all nodes.
	get_edge_attribute(attribute="edge_type")
		Retrieves a specified attribute for all edges.
	get_edges()
		Returns the edges of the graph.
	get_xyz_pointcloud()
		Returns the XYZ coordinates of the point cloud.
	get_colours_pointcloud()
		Returns the RGB colors of the point cloud.
	get_semantic_pointcloud(semantic_name="semantic")
		Returns the semantic labels of the point cloud.
	visualise_graph()
		Visualizes the skeleton graph.
	get_attributes()
		Retrieves all unique attributes from the graph nodes.
	filter(node_order, keep_parents_only, keep_ends_points=True)
		Filters the graph based on node order and parent status.
	edge_from_filtered()
		Updates the edge types in the original graph based on the filtered graph.
	get_edge_type(angle_between_trunk_and_lateral=60)
		Determines the edge types based on the angle between trunk and lateral branches.
	export_as_nodelist(path)
		Saves the graph to a CSV file.
	load_csv(path)
		Loads the graph from a CSV file.
	get_internode_length()
		Calculates the internode lengths and updates the graph.
	add_gt_attributes(location, dict_attributes)
		Adds ground truth attributes to the closest node to a given location.
	get_gt_attributes(attributes_list=["gt_int_length", "gt_ph_angle", "gt_lf_angle"])
		Retrieves specified ground truth attributes from the graph.
	get_angles(node_roder=1)
		Calculates angles between nodes and updates the graph.
	gaussian_smoothing(var0=.25, var1=.25, indices=[0,1], node_order_filtering=True)
		Applies Gaussian smoothing to the node positions.
	main_post_processing(cfg)
		Applies post-processing methods to the graph based on a configuration.
	line_fitting_3d()
		Fits lines to the 3D points in the graph.
	'''
	def __init__(self, name=None) -> None:
		if name is not None:
			self.load_csv(name)
		pass


	def load(self, nodes, edges, edge_types, df_pc=None, name=None, attributes={}) -> None:
		"""
		Load a graph structure with nodes, edges, and edge types.
		Parameters:
		nodes (np.ndarray): An Nx3 array representing the coordinates of the nodes.
		edges (np.ndarray): An Mx2 array where edges[:,0] are parent nodes and edges[:,1] are child nodes.
		edge_types (np.ndarray or list): An Mx1 array or list with edge types, which can be "+", "<" or None.
		df_pc (pd.DataFrame, optional): A DataFrame containing additional point cloud data. Default is None.
		name (str, optional): The name of the graph. Default is None.
		attributes (dict, optional): A dictionary of additional attributes for the nodes. Default is an empty dictionary.
		Returns:
		None
		"""

		# Step 1: Create a directed graph
		self.G = nx.DiGraph()
		self.G_original = None
		self.df_pc = df_pc
		self.name = name

		## check if length nodes is correct
		if sorted(np.unique(edges))!=sorted(np.arange(nodes.shape[0])):
			## possible mismatch vid number and number of nodes remapping:
			mapping = {node: i for i, node in enumerate(np.unique(edges))}
			new_edges = []
			for parent_id, vid in edges:
				new_edges.append([mapping[parent_id], mapping[vid]])
			new_nodes = np.zeros((len(mapping), 3))
			for key, value in mapping.items():
				new_nodes[value] = nodes[key]
			edges = np.array(new_edges)
			nodes = new_nodes
		
		dict_nodes= {}
		# for i in range(len(nodes)):
		# for i, vid in enumerate(nodes):
		dict_nodes[edges[0][0]] = {"pos": nodes[edges[0][0]], "edge_type": "root"}
		dict_nodes[edges[0][0]].update({key: value[0] for key, value in attributes.items()})

		for i, (parent_id, vid) in enumerate(edges):
			if edge_types is not None:
				dict_nodes[vid] = {"pos": nodes[vid], "edge_type": edge_types[i-1]}
			else:
				dict_nodes[vid] = {"pos": nodes[vid], "edge_type": None}

			if attributes is not None:
				temp_attributes = {key: value[vid] for key, value in attributes.items()}
				dict_nodes[vid].update(temp_attributes)
			# dict_nodes[vid] = {"pos": nodes[edges[i][0]], "edge_type": "root"}
			# if i==0:
			# 	dict_nodes[edges[i][0]] = {"pos": nodes[edges[i][0]], "edge_type": "root"}
			# else:
			# 	vid = edges[i-1][1]
			# 	parent_id = edges[i-1][0]
			# 	# if parent_id not in edges[:,1] and parent_id!=edges.min():
			# 	# 	print(f"parent id {parent_id} not in edges, so skipping!!!")
			# 	# 	print("Not a directed graph exitting")
			# 	# 	exit()
			# 	if edge_types is not None:
			# 		dict_nodes[vid] = {"pos": nodes[i], "edge_type": edge_types[i-1]}
			# 	else:
			# 		dict_nodes[vid] = {"pos": nodes[i], "edge_type": None}

		
		# add remaining nodes:
		for vid in np.unique(edges):
			if vid not in dict_nodes.keys():
				dict_nodes[vid] = {"pos": nodes[vid], "edge_type": None}

		#  Step 2: Add nodes with 3D coordinates and edgetype as attributes
		# for node, (pos, edge_type) in dict_nodes.items():
		for node in dict_nodes.keys():
			# self.G.add_node(node, pos=dict_nodes[node]["pos"], node_type=dict_nodes[node]["edge_type"])  # Store coordinates as 'pos' attribute
			self.G.add_node(node,**dict_nodes[node])  # Store coordinates as 'pos' attribute


		## add additional attributes
		for key, value in attributes.items():
			self.G.nodes[node][key] = value[node]

		# Step 3: Add directed edges with 'type' attribute
		for i, u in enumerate(edges):
			parent = edges[i][0]
			child = edges[i][1]
			# if parent not in self.G.nodes() or child not in self.G.nodes():
			# 	print(f"parent {parent} or child {child} not in nodes, so skipping!!!")
			# 	continue
			if edge_types is not None:
				edge_type = edge_types[i]
				self.G.add_edge(parent, child, edge_type=edge_type)
			else:
				self.G.add_edge(parent, child)

		# apply remapping because ordering of nodes is not guaranteed
		temp_copy = self.G.copy()
		mapping = {node: i for i, node in enumerate(self.G.nodes())}
		self.G = nx.relabel_nodes(temp_copy, mapping)

		# if edge_types is None:
		# 	self.get_edge_type()

		pass

	@classmethod
	def from_skeleton_gt_data(cls, skeleton_path, pc_path=None, pc_semantic_path=None):
		"""
		Load the skeleton data from the ground truth file and create a SkeletonGraph object.

		Parameters:
		skeleton_path (str or Path): Path to the ground truth skeleton file.
		pc_path (str or Path, optional): Path to the point cloud file. Default is None.
		pc_semantic_path (str or Path, optional): Path to the point cloud semantic file. Default is None.

		Returns:
		SkeletonGraph: A SkeletonGraph object containing the loaded skeleton data.
		"""

		df_skeleton = pd.read_csv(str(skeleton_path), low_memory=False)
		
		# Optionally load point cloud and semantic information as well
		df_pointcloud = None
		if pc_path is not None:
			df_pointcloud = pd.read_csv(str(pc_path))
		if pc_semantic_path is not None:
			df_semantics = pd.read_csv(str(pc_semantic_path))
			if df_pointcloud is None:
				df_pointcloud = df_semantics
			else:
				df_pointcloud = pd.concat([df_pointcloud, df_semantics], axis=1)

		skeleton_data = df_skeleton.loc[
			~df_skeleton["x_skeleton"].isna(), ["x_skeleton", "y_skeleton", "z_skeleton", "vid", "parentid", "edgetype"]
		]

		nodes = skeleton_data[["x_skeleton", "y_skeleton", "z_skeleton"]].values
		edges = skeleton_data[["parentid", "vid"]].values[1:].astype(int)
		edge_types = skeleton_data["edgetype"].values[1:].astype(str)

		if "gt_int_length" in df_skeleton.columns:
			skeleton_data = df_skeleton.loc[~df_skeleton["x_skeleton"].isna(), ["gt_int_length", "gt_int_diameter", "gt_ph_angle", "gt_lf_angle"]]
			attributes_gt = {
				"gt_int_length": skeleton_data["gt_int_length"].values,
				"gt_int_diameter": skeleton_data["gt_int_diameter"].values,
				"gt_ph_angle": skeleton_data["gt_ph_angle"].values,
				"gt_lf_angle": skeleton_data["gt_lf_angle"].values
			}
		else:
			attributes_gt = {}
		
		S_gt = cls()
		S_gt.load(nodes, edges, edge_types.tolist(), df_pc=df_pointcloud, attributes=attributes_gt)
		S_gt.name = skeleton_path.stem.replace("_skeleton", "")
		S_gt.get_node_order()

		return S_gt

	def get_node_order(self):
		# Step 4: get node order and determine is node is a parent
		# all_nodes_with_attributes = dict(self.G.nodes(data=True))
		# do_calcuation = False
		# for x in all_nodes_with_attributes.values():
		# 	if "node_order" not in x.keys():
		# 		do_calcuation = True
		# 		break

		# if not do_calcuation:
		# 	return

		node_orders = {}
		root = 0  # Assume the root node is order 0
		node_orders[root] = 0  # Root node is order 0
		is_parent = {0: False}

		# Traverse the MTG to calculate orders
		stack = [root]
		while stack:
			parent = stack.pop()
			parent_order = node_orders[parent]

			for child in self.G.successors(parent):
				edge_type = self.G.edges[parent, child].get('edge_type')
				# Determine the child's order based on the edge type
				if edge_type == "<":
					node_orders[child] = parent_order  # Same order as parent
				elif edge_type == "+":
					is_parent[parent] = True
					node_orders[child] = parent_order + 1  # Increase order by 1 for branches
				# print(counter:=counter+1)
				stack.append(child)

		for child in self.G.nodes():
			self.G.nodes[child]['node_order'] = node_orders[child]
			self.G.nodes[child]['is_parent'] = is_parent.get(child, False)
		# Display the counts for each node
		# print("Number of '+' predecessors for each node:", plus_predecessors_count)

		# print(node_order)

	def get_node_attribute(self, attribute = "pos"):
		# a=dict(self.G.nodes(data=True)).values()
		return np.array([x[attribute] for x in dict(self.G.nodes(data=True)).values()])
	
	def get_edge_attribute(self, attribute = "edge_type"):
		return np.array([self.G.edges[x][attribute] for x in self.G.edges])
	
	def get_edges(self):
		return np.array(self.G.edges()) # returns parent, child
	
	def get_xyz_pointcloud(self):
		if self.df_pc is not None:
			return self.df_pc[["x", "y", "z"]].values
		else:
			return None
		
	def get_colours_pointcloud(self):
		if self.df_pc is not None:
			return self.df_pc[["red", "green", "blue"]].values
		else:
			return None

	def get_semantic_pointcloud(self, semantic_name="semantic"):
		if self.df_pc is not None:
			return self.df_pc[semantic_name].values
		else:
			return None

	def visualise_graph(self, show_segmented=False, semantic_name="semantic", **kwargs):
		all_nodes_with_attributes = dict(self.G.nodes(data=True))
		nodes = np.array([x["pos"] for x in all_nodes_with_attributes.values()])

		for x in all_nodes_with_attributes.values():
			if "node_order" not in x.keys():
				self.get_node_order()
				break

		node_order = np.array([x["node_order"] for x in all_nodes_with_attributes.values()])

		parents = np.array([x["is_parent"] for x in all_nodes_with_attributes.values()])

		edges = np.array(self.G.edges())
		edges_type = np.array([self.G.edges[x]["edge_type"] for x in edges])

		attributes = self.get_attributes()

		if show_segmented:
			xyz = self.get_xyz_pointcloud()
			semantic = self.get_semantic_pointcloud(semantic_name)
			bool_array = np.zeros(semantic.shape[0]).astype(np.bool)
			# bool_array = np.bitwise_or(semantic==1 ,semantic==3) # 1=leaves, 2=main stem, 3=pole, 4=side stem

			ve.vis(pc = xyz[~bool_array, :], colors=semantic[~bool_array], nodes=nodes, edges=edges, node_order=node_order, parents=parents, edges_type=edges_type, attributes=attributes, **kwargs)

		else:
			ve.vis(pc = self.get_xyz_pointcloud(),  colors=self.get_colours_pointcloud(), 
		#   distances=self.get_semantic_pointcloud(),
		  nodes=nodes, edges=edges, node_order=node_order, parents=parents, edges_type=edges_type, attributes=attributes, **kwargs)

	def get_attributes(self):
		# Collect all unique keys from the attribute dictionaries
		unique_keys = set()
		all_nodes_with_attributes = dict(self.G.nodes(data=True))
		for attributes in all_nodes_with_attributes.values():
			unique_keys.update(attributes.keys())
		attributes = {}
		for key in unique_keys:
			if key in ["pos",  "node_type"]:
				continue
			attributes[key] = np.array([x.get(key, None) for x in all_nodes_with_attributes.values()])
		return attributes

	def filter(self, node_order, keep_parents_only, keep_ends_points=True):
		""""Filter the graph based on node order and parent status, and updates self.G accordingly.
		Args:
			node_order (int): Maximum node order to keep in the graph.
			is_parent (bool): Whether to keep parent nodes (True) or child nodes (False).
			keep_ends_points (bool): Whether to keep end points (nodes with no successors) with node_order<node_order.
		returns:
			None
		"""
		if self.G_original is None:
			self.G_original = self.G.copy()
		else:
			self.G = self.G_original.copy()
		
		# if node_order==-1:
		# 	return
		# Create a copy of the graph to avoid modifying the original directly
		self.G_filtered = self.G.copy()
		
		# Step 1: Identify nodes to remove based on the filter criteria
		# nodes_to_remove = [
		# 	node for node in self.G_filtered.nodes # do not remove root node
		# 	if self.G_filtered.nodes[node]["node_order"] > node_order
		# 	or (self.G_filtered.nodes[node]["is_parent"] != is_parent and node!=0)
		# ]
		nodes_to_remove = []
		for node in self.G_filtered.nodes:
			if self.G_filtered.nodes[node]["node_order"] > node_order:
				nodes_to_remove.append(node)
			elif self.G_filtered.nodes[node]["is_parent"]==False and node!=0 and keep_parents_only:
				nodes_to_remove.append(node)
			elif keep_parents_only==False:
				continue

	
		# Step 2: Reconnect edges around nodes being removed
		nodes_to_remove2=[]
		for node in nodes_to_remove:
			# Get predecessors and successors of the node
			predecessors = list(self.G_filtered.predecessors(node))
			successors = list(self.G_filtered.successors(node))

			# do not remove if nodes has no successors and node_order<node_order. See keep_ends_points setting.
			if keep_ends_points and len(successors)==0 and self.G_filtered.nodes[node]["node_order"]<=node_order:
				continue
			else:
				nodes_to_remove2.append(node)
		

			# Connect each predecessor to each successor to preserve connectivity
			for pred in predecessors:
				for succ in successors:
					# Only add the edge if it doesn’t already exist
					if not self.G_filtered.has_edge(pred, succ):
						# Copy the edge attributes from the removed node, if needed
						edge_attrs = self.G_filtered.edges[pred, node] if self.G_filtered.has_edge(pred, node) else {}
						self.G_filtered.add_edge(pred, succ, **edge_attrs)

		# Step 3: Remove nodes after reconnecting edges
		temp_copy = self.G_filtered.copy()
		temp_copy.remove_nodes_from(nodes_to_remove2)
		mapping = {node: i for i, node in enumerate(temp_copy.nodes())}
		self.mapping_reverse = {i: node for i, node in enumerate(temp_copy.nodes())}
		self.G_filtered = nx.relabel_nodes(temp_copy, mapping)
		self.G = self.G_filtered

	
	def remove_non_unique_nodes(self):
		"""Function to remove non unique nodes based on float16 node position"""
		nodes = self.get_node_attribute()
		_, indices = np.unique(nodes.astype(np.float16), axis=0, return_index=True)
		indices.sort()
		points_to_remove = np.setdiff1d(np.arange(nodes.shape[0]), indices).tolist()
		self.remove_node_and_update_edge(points_to_remove, do_relabel=True)

	def remove_node_and_update_edge(self, node_id:int | list, do_relabel: bool = True):
		"""
		Remove node_id and reconnect edges accordingly
		"""

		# for nodes in node_id:
		if isinstance(node_id, int):
			node_ids = [node_id]
		else:
			node_ids = list(node_id)

		temp = self.G.copy()
		for nid in np.unique(node_ids):
			for in_src, _ in self.G.in_edges(nid):
				for _, out_dst in self.G.out_edges(nid):
					temp.add_edge(in_src, out_dst)
					temp.edges[in_src, out_dst].update(self.G.edges[in_src, nid])
			temp.remove_node(nid)
			self.G = temp
		# temp_copy = temp.copy()
		

			# mapping = {node: i for i, node in enumerate(temp_copy.nodes())}
			# self.G = nx.relabel_nodes(temp_copy, mapping)
		remove2 = [x for x in self.G.nodes if self.G.nodes[x]=={}]
		[self.G.remove_node(x) for x in remove2]
		if do_relabel:
			self.G = relabel(self.G)
		
		
	def edge_from_filtered(self):

		edges = np.array(self.G.edges())
		for parent, child in edges:
			target_edge_type = self.G.edges[parent, child]["edge_type"]
			# if target_edge_type == "+":
			# 	print("debug")
			end_node = self.mapping_reverse[child]
			start_node = self.mapping_reverse[parent]
			stack = list(self.G_original.predecessors(end_node))
			next_i = end_node

			while stack:
				temp_i = stack.pop()
				if temp_i==start_node:
					self.G_original.edges[temp_i,next_i]["edge_type"] = target_edge_type
					self.G_original.nodes[temp_i]["edge_type"] = target_edge_type
					self.G_original.nodes[temp_i]["is_parent"] = True
					
					break
				self.G_original.nodes[temp_i]["edge_type"] = "<"
				self.G_original.nodes[temp_i]["is_parent"] = False
				self.G_original.edges[temp_i,next_i]["edge_type"] = "<"
				stack+= list(self.G_original.predecessors(temp_i))
				next_i = temp_i
		
		# for i, internode in enumerate(internodes[1:]):
		# 	if i==0:
		# 		continue
		# 	# edge_type = self.G.nodes[i]["edge_type"]
		# 	stack = list(self.G_original.predecessors(self.mapping_reverse[i]))
		# 	next_i = self.mapping_reverse[i]
		# 	while stack:
		# 		temp_i = stack.pop()
		# 		if temp_i==internodes[i-1]:
		# 			self.G_original.edges[temp_i,next_i]["edge_type"] = "<"

		# 			break
		# 		self.G_original.nodes[temp_i]["edge_type"] = "<"
		# 		self.G_original.edges[temp_i,next_i]["edge_type"] = "<"
		# 		stack+= list(self.G_original.predecessors(temp_i))
		# 		next_i = temp_i
		nodes = np.array([x["pos"] for x in dict(self.G_original.nodes(data=True)).values()])

		edges = np.array(self.G_original.edges())
		# edges_type = np.array([x["edge_type"] for x in dict(self.G_original.nodes(data=True)).values()])
		edges_type = np.array([self.G_original.edges[x]["edge_type"] for x in edges])
		# ve.vis(pc = self.get_xyz_pointcloud(), nodes=nodes, edges=edges, edges_type=edges_type)
		self.G = self.G_original.copy()
		self.get_node_order()
			# predecessors =
			# for predecesor in predecessors:
			# 	self.G_original.nodes[self.mapping_reverse[predecesor]]["edge_type"] = "<"
			# 	print("x")



	def get_edge_type(self, **kwargs):
		self.G = graph_edges_to_tree(graph=self.G, **kwargs)
		self.get_node_order()

		# Traverse the MTG to calculate orders
		# children = list(self.G.successors(root))
		# for child in children:
		# 	edge_types[child]="<"
		
		# stack = list(self.G.successors(root))
		# stack = [root_id]
		# while stack:
		# 	parent_id = stack.pop()
		# 	self.get_single_edge_type(parent_id, angle_between_trunk_and_lateral)
		# 	stack+=list(self.G.successors(parent_id))
		# 	new_id = stack.pop()
		# 	pos = self.G.nodes[new_id]["pos"]

		# 	if new_id==root:
		# 		parent_pos = pos - np.array([0, 0, .1])
		# 	else:
		# 		parent = list(self.G.predecessors(new_id))[0]
		# 		parent_pos = self.G.nodes[parent]["pos"]

		# 	children = list(self.G.successors(new_id))
		# 	if len(children)>0:
		# 		langles = []
		# 		for child in children:
		# 			child_pos = self.G.nodes[child]["pos"]

		# 			first_edge_type = '<'
		# 			langle = math.degrees(math.acos(
		# 				round(np.dot(direction(pos - parent_pos), direction(child_pos - pos)),1))) # round for bug fix

		# 			if langle > angle_between_trunk_and_lateral: 
		# 				first_edge_type = '+'
		# 			else:
		# 				langles.append(langle)

		# 			edge_types[child] = first_edge_type
		# 			stack.append(child)

		# 		# if multiple angles are smaller than 60, then largest angle is a branch
		# 		if len(langles)>1:
		# 			edge_types[children[langles.index(max(langles))]] = "+"

		# for child in self.G.nodes():
		# 	self.G.nodes[child]['edge_type'] = edge_types[child]
		# for parent, child in self.G.edges():
		# 	self.G.edges[parent, child]['edge_type'] = edge_types[child]


	def get_single_edge_type(self, method="angle_based", node_id=0, **kwargs):
		"""
		Script to get type of edge by determining angle between vector parent and vector child.
		If angle is larger than angle_between_trunk_and_lateral then edge_type is a branch.
		Else it is a parent branch. However, multiple children have angle <angle_between_trunk_and_lateral.
		Then 
		"""
		self.G = get_single_edge_type(graph=self.G, method=method, node_id=node_id, **kwargs)

		# node_id_pos = self.G.nodes[node_id]["pos"]
		# children = list(self.G.successors(node_id))

		# parent_node_id = list(self.G.predecessors(node_id))
		# if len(parent_node_id)==0: ## root, because it does not have any parents
		# 	parent_node_id_pos = node_id_pos - np.array([0, 0, .1])
		# else:
		# 	parent_node_id_pos = self.G.nodes[parent_node_id[0]]["pos"]

		# edge_types = {}
		# langles = []
		# langles_child = []
		# for child in children:
		# 	child_pos = self.G.nodes[child]["pos"]

		# 	first_edge_type = '<'
		# 	langle = math.degrees(math.acos(
		# 		round(np.dot(direction(node_id_pos - parent_node_id_pos), direction(child_pos - node_id_pos)),1))) # round for bug fix

		# 	if langle > angle_between_trunk_and_lateral: 
		# 		first_edge_type = '+'
		# 	else:
		# 		langles.append(langle)
		# 		langles_child.append(child)

		# 	edge_types[child] = first_edge_type
		# 	# stack.append(child)

		# # if multiple angles are smaller than 60, then all other larger angles are a branch
		# if len(langles)>1:
		# 	index_min_langles = langles.index(min(langles))
		# 	del langles_child[index_min_langles]
		# 	for child in langles_child:
		# 		edge_types[child] = "+"
		# 	# index_min_langles = langles.index(min(langles))
		# 	# for child in children:
		# 	# edge_types[children[langles.index(max(langles))]] = "+"
		
		# for child, edge_type in edge_types.items():
		# 	self.G.nodes[child]['edge_type'] = edge_type #edge_types[child]
		# 	self.G.edges[node_id, child]['edge_type'] = edge_types[child]

		# # for parent, child in self.G.edges():
		# # 	self.G.edges[parent, child]['edge_type'] = edge_types[child]



	def get_nodes_branch_id(self, root_idx=0, branch_id=0):
		## get all nodes with brnach_id 
		return [n for n in list(nx.dfs_tree(self.G, root_idx)) if self.G.nodes[n]["branch_number"]==branch_id]


	def add_branch_number(self, root_idx = 0):
		"""
		Script to add a branch number to each branch
		"""
		nodes_to_count = set(nx.dfs_tree(self.G, root_idx))
		branch_number = 0

		nodes_main_stem = self.get_nodes_with_order_x(node_id=root_idx)

		def add_number(indices, number):
			for index in indices:
				self.G.nodes[index]["branch_number"] = number
				if index in nodes_to_count:
					nodes_to_count.remove(index)
		
		add_number(nodes_main_stem, number=branch_number)
		branch_number += 1

		while len(nodes_to_count)>0:
			node_id = list(nodes_to_count)[0]
			nodes_branch = self.get_nodes_with_order_x(node_id=node_id)
			add_number(nodes_branch, branch_number)
			branch_number+=1


	def export_as_nodelist(self, save_path=None, rename=False, first_row_correction=False):
		# nx.write_gpickle(self.G, path)
		df = pd.DataFrame(self.get_node_attribute("pos"), columns=["x", "y", "z"])
		if rename:
			df = df.rename(columns={"x": "x_skeleton", "y": "y_skeleton", "z": "z_skeleton"})
		edges_array = self.get_edges()
		edge_types_array = self.get_edge_attribute("edge_type")
		df_2 = pd.DataFrame(edges_array, columns=["parentid", "vid"])
		df_2["edgetype"] = edge_types_array
		df_result = pd.concat([df, df_2], axis=1)

		if first_row_correction:
			df_result["edgetype"] = ""
			df_result["vid"] = 0
			df_result["parentid"] = ""
			df_result.loc[1:, "vid"] = edges_array[:,1]
			df_result.loc[1:, "parentid"] = edges_array[:,0]
			df_result.loc[1:, "edgetype"] = edge_types_array

		if save_path is not None:
			if not save_path.parent.exists():
				save_path.parent.mkdir(parents=True)
			df_result.to_csv(save_path, index=False)
		return df_result

	@classmethod
	def from_mtg(cls, mtg_name):
		from skeletonisation_methods.plantscan3d import mtgmanip
		from skeletonisation_methods.plantscan3d import io

		mtg = io.read_mtg_file(mtg_name)
		nodes, edges, edge_type = mtgmanip.mtg2_nodes_edges_edge_types(mtg)
		obj = SkeletonGraph()
		obj.load(nodes, edges, edge_type)
		return obj
	
	@classmethod
	def from_nodelist(cls, node_list_name):
		"""
		Expect a csv with following headers: 
		x, y, z, parentid, vid, edgetype
		"""
		df = pd.read_csv(node_list_name)
		obj = SkeletonGraph()
		obj.load(df[["x", "y", "z"]].values, df[["parentid", "vid"]].dropna().values.astype(int), df["edgetype"].dropna().values)
		return obj
	

	def export_as_mtg(self, save_name="example.mtg"):
		from skeletonisation_methods.plantscan3d import mtgmanip
		from skeletonisation_methods.plantscan3d import io
		mtg = mtgmanip.nodelist2mtg(nodes=self.get_node_attribute("pos"), edges=self.get_edges(), edge_types=self.get_edge_attribute("edge_type"), radius=None)
		
		# properties = [(p, 'REAL') for p in mtg.property_names() if p not in ['edge_type', 'index', 'label']]
		properties = [(p, 'REAL') for p in mtg.property_names() if p not in ['edge_type', 'position', 'index', 'label']]
		mtg_lines = io.write_mtg(mtg, properties)
		# Write the result into a file example.mtg
		f = open(save_name, 'w')
		f.write(mtg_lines)
		f.close()

	
	def load_csv(self, csv_path: Path):
		if not csv_path.exists():
			raise FileNotFoundError(f"Please check file path {csv_path}")
		df = pd.read_csv(str(csv_path), low_memory=False)
		self.load(nodes=df[["x", "y", "z"]].dropna().values, edges=df[["parentid", "vid"]].dropna().astype(int).values, edge_types=df["edgetype"].dropna().values)
		self.name = csv_path.stem

	def get_internode_length(self):
		### calculate internode length, by returning nodes with node order =0 and only parents nodes
		self.get_node_order()

		self.G_internode = self.G.copy()
		# get list of internodes
		# internodes = 		# Step 1: Identify nodes to remove based on the filter criteria
		internodes = [
			node for node in self.G_internode.nodes
			if self.G_internode.nodes[node]["node_order"] == 0
			and self.G_internode.nodes[node]["is_parent"] 
		]
		# if internodes==[]:
		# 	return [], [], []

		internodes_pos = np.array([self.G_internode.nodes[node]["pos"] for node in internodes])
		internodes_dist = np.linalg.norm(internodes_pos[1:]-internodes_pos[:-1], axis=1)

		## add internode length to graph
		for i, node in enumerate(internodes[1:]):
			self.G.nodes[node]["int_length"] = internodes_dist[i]

		return internodes, internodes_pos, internodes_dist
		# self.visualise_graph()
		print("yeah")


	def add_gt_attributes(self, location, dict_attributes):
		# location = "x_skeleton", "y_skeleton", "z_skeleton"
		# dict_attributes = {gt_int_length, gt_int_diameter, gt_ph_angle, gt_lf_angle}
		poses = self.get_node_attribute("pos")

		array = np.linalg.norm(poses - location, axis=1)
		self.G.nodes[array.argmin()].update(dict_attributes)
		pass


	def get_gt_attributes(self, attributes_list = ["gt_int_length", "gt_ph_angle", "gt_lf_angle"]): # "gt_int_diameter"
		gt_values = []
		for node in self.G.nodes():
			temp_attributes = {}
			for attribute in attributes_list:
				if not np.isnan(self.G.nodes[node].get(attribute, np.nan)):
					temp_attributes[attribute] = self.G.nodes[node].get(attribute, None)
			if len(temp_attributes)>0:
				temp_attributes["node"] = node
				gt_values.append(temp_attributes)
		return gt_values


	def reconnect_nodes(self, method="mst"):
		from scripts import nodes_to_graph
		points = self.get_xyz_pointcloud()
		nodes, edges, edge_type = nodes_to_graph.nodes_to_graph(points=points,
												  nodes=self.get_node_attribute(attribute="pos"),
												  method=method)
		self.load(nodes, edges, edge_types=edge_type, df_pc=self.df_pc, name=self.name)
		

	def get_angles(self, node_roder = 1):
		poses = self.get_node_attribute("pos")

		internodes, _, _ = self.get_internode_length()
		lateral_roots = []
		for i in internodes:
			node_roder = self.G.nodes[i]["node_order"]
			list_succesors = list(self.G.successors(i))
			branches = [n for n in list_succesors if self.G.nodes[n]["node_order"]==node_roder+1]
			if len(branches)>0:
				## if there are multiple branches pick largest
			# if len(branches)>1:
				branch = branches[np.argmax([len(nx.dfs_tree(self.G, branch)) for branch in branches])]
				# for branch in branches:
				successors = list(nx.dfs_tree(self.G, branch))
				lateral_root = [n for n in successors if self.G.nodes[n]["node_order"]==node_roder+1]
				if len(lateral_root)>1: # if size of roots = 1, then we do not consider it as a branch
					lateral_roots.append([i, [i]+ lateral_root])
				elif len(lateral_root)==1:
					lateral_roots.append([i, [i, lateral_root[0]]])
						

		# phyto_angle, relangles, rel_angle_index = openalea_method(poses, lateral_roots)
		# print("Phyto angle:", phyto_angle)
		# print("Relative angles:", relangles)
		# print("Relative angle indices:", rel_angle_index)
		phyto_angle, relangles, rel_angle_index, xy_points, xy_edges = xy_plane_method(poses, lateral_roots)
		## for debugging
		# ve.vis(self.get_node_attribute("pos"), nodes = xy_points, edges=xy_edges)
		# print("Phyto angle:", phyto_angle)
		# print("Relative angles:", relangles)

		for i, lateral_root in enumerate(lateral_roots):
			self.G.nodes[lateral_root[0]]["phyllotactic_angle_id"] = lateral_root[1][1]

		## add internode length to graph
		for i, node in enumerate(rel_angle_index):
			self.G.nodes[node]["ph_angle_xaxis"] = phyto_angle[i]
			self.G.nodes[node]["ph_angle"] = relangles[i]

		####################### for debugging visualize lines
		# temp_nodes = np.array([[x.pos, x.pos+x.dir*x.extend] for x in lateral_lines]).reshape(-1, 3)
		# temp_edges = np.array([[i, i+1] for i in range(0, len(temp_nodes), 2)])
		# folder = Path(r"W:\PROJECTS\VisionRoboticsData\GARdata\datasets\tomato_plant_segmentation") / "20240607_summerschool_csv" / "annotations"
		# file_name = folder / self.name / (self.name + ".csv")
		# ve.vis(pc = pd.read_csv(str(file_name), low_memory=False)[["x", "y", "z"]], nodes=temp_nodes, edges=temp_edges)
		
		## get branching angle 
		for j in lateral_roots[:-1]:
			current_node = j[0]
			angle_branch_id = j[1][1]
			nextnode_id = internodes[internodes.index(current_node)+1]

			pos_branch = poses[angle_branch_id]
			pos_parent_node = poses[current_node]
			pos_next_internode = poses[nextnode_id]

			angle = math.degrees(math.acos(
				round(np.dot(direction(pos_branch - pos_parent_node), direction(pos_next_internode - pos_parent_node)),3)))
		
		## add internode length to graph
		# for node, angle_branch_id, angle in branch_angles:
			self.G.nodes[current_node]["lf_angle"] = angle
			self.G.nodes[current_node]["leaf_angle_branch_id"] = angle_branch_id
			self.G.nodes[current_node]["leaf_angle_nextnode_id"] = nextnode_id



		# lateral_roots = sum([[n for n in self.G.successors(i) if self.G.edge_type(n) == '+'] for i in internodes],[])

		# r = mtg.roots(scale=mtg.max_scale())[0]
		# positions = mtg.property('position')
		# trunk_nodes = mtg.Axis(r)
		# if degree > 1:
		# 	trunk_line = NurbsEstimate(positions,trunk_nodes,degree)
		# else:
		# 	trunk_line = Line.estimate(positions,trunk_nodes)
		# lateral_roots = sum([[n for n in mtg.children(i) if mtg.edge_type(n) == '+'] for i in trunk_nodes],[])
		# lateral_lines = [Line.estimate(positions,mtg.Axis(lr))  for lr in lateral_roots]
		# nodelength = [norm(positions[mtg.parent(lateral_roots[i])]-positions[mtg.parent(lateral_roots[i+1])]) for i in range(len(lateral_roots)-1)]
		
		# return trunk_line, lateral_lines, nodelength

	def gaussian_smoothing(self, **kwargs):
		""""Filter the graph based on node order and parent status, and updates self.G accordingly.
		inputs:
			var0 (float): variance for node
			var1 (float): variance for parent and child nodes
			indices (list): indices to smooth. if [0, 1] only x and y will be smoothed
			node_order_filtering: if True only nodes with same node order will be smoothed
		returns:
			None
		"""
		from scripts.postprocessing_methods import gaussian_smoothing
		self.G = gaussian_smoothing(self.G, **kwargs)

		# It is possible that by apply smoothing non unique nodes are generated. 
		# remove those if they exist
		self.remove_non_unique_nodes()


	def main_post_processing(self, cfg_post_processing):
		if cfg_post_processing is None:
			return
		

		for method in cfg_post_processing.get("methods", []):
			temp = cfg_post_processing[method]
			key = list(temp.keys())[0]
			print("Applying post processing method: ", key)
			try:
				if temp[key] is not None:
					getattr(self, key)(**temp[key])
				else:
					getattr(self, key)()
			except AttributeError:
				print(f"Cannot find function: {key}")


			# if method == "gaussian":
			# 	self.gaussian_smoothing(cfg["post_processing"]["gaussian"]["var0"], cfg["post_processing"]["gaussian"]["var1"])

	def get_nodes_with_order_x(self, node_id, node_order_offset: int = 0, parents_only: bool = False):
		"""
		Get all nodes succesors of node_id with same node_order is offset is 0
		"""

		node_order = self.G.nodes[node_id]["node_order"] + node_order_offset
		if parents_only:
			return [n for n in list(nx.dfs_tree(self.G, node_id)) if (self.G.nodes[n]["node_order"]==node_order and  self.G.nodes[n]["is_parent"])]
		else:
			return [n for n in list(nx.dfs_tree(self.G, node_id)) if self.G.nodes[n]["node_order"]==node_order]


	def simplify(self, distance_threshold: float = 0.001, merge_th: float = np.inf, alpha_th: float = 0.9):
		"""
		Script to simplify Graph, based on AdTree
		https://github.com/tudelft3d/AdTree/blob/main/AdTree/skeleton.cpp#420
		Two methds to simplify graph:
		if node_degrees = 2, determine distance between parent and child of current node.
		if distance is whitin distance threshold remove node if is not a parent node.

		if node_degrees =>2 check whether we can simplify the childs of current node. 
		By calculating distrance from child_i to child_j and vica versa
		merge_th# currently set to infite... but could be anything
		alpha_th = similarity between two chid vectors

		"""

		# distance_threshold = 0.001
		nodes_to_remove = []

		nodes = list(self.G.nodes)
		processed_nodes = []
		while nodes:
			node = nodes[0]
			nodes= nodes[1:]

			processed_nodes.append(node)


		# for node in self.G.nodes:
			pCurrent = self.G.nodes[node]["pos"]
			node_degree = nx.degree(self.G, node)
			if node_degree==1 or self.G.nodes[node].get("edge_type")=="root":
				## node is beginning or end point skip... for now
				continue

			elif node_degree==2: ## node with only parent and end node
				node_parent = list(self.G.predecessors(node))[0]
				pParent = self.G.nodes[node_parent]["pos"]

				node_child = list(self.G.successors(node))[0]
				pChild = self.G.nodes[node_child]["pos"]

				temp = np.cross(pCurrent - pParent, pCurrent - pChild)
				distance = np.linalg.norm(temp) / np.linalg.norm(pParent - pChild)
				if distance<distance_threshold and not self.G.nodes[node]["is_parent"]:
					self.remove_node_and_update_edge(node, do_relabel=False)
					nodes_to_remove.append(node)
			
			elif node_degree>2:
				## function to merge two childs if they are similar
				nodes_child = list(self.G.successors(node))
				min_value = np.inf

				sourceV, targetV = None, None

				for i in range(len(nodes_child)-1):
					for j in range(i+1, len(nodes_child)):
						id_i = nodes_child[i]
						id_j = nodes_child[j]

						pos_i = self.G.nodes[id_i]["pos"]
						pos_j = self.G.nodes[id_j]["pos"]

						merge_i2j = compute_merge_value(pCurrent, pos_i, pos_j, alpha_th)
						merge_j2i = compute_merge_value(pCurrent, pos_j, pos_i, alpha_th)

						if merge_i2j < merge_j2i and merge_i2j<min_value:
							min_value = merge_i2j
							sourceV = id_i
							targetV = id_j
						elif  merge_j2i < merge_i2j and merge_j2i<min_value:
							min_value = merge_j2i
							sourceV = id_i
							targetV = id_j

				if min_value<merge_th and sourceV is not None: # distance too large, you don't want to mrege
					self.merge_vertices_and_update_edges(sourceV, targetV, relabel=False)
					nodes.remove(sourceV)
					nodes.remove(targetV)
				else:
					pass

		self.G = relabel(self.G)


	def line_fitting_3d(self, root_idx: int = 0, line_fit_method="spline"):
		"""
		Script to fit a line through every branch and determine interect with parent branch
		
		"""

		self.add_branch_number()

		## node positions and branches
		positions = self.get_node_attribute()
		branches = self.get_node_attribute("branch_number")

		new_parents = []
		new_parents.append(positions[root_idx])
		new_edges = []

		## start with root branch
		nodes = self.get_nodes_branch_id(branch_id=0)
		points = positions[nodes]

		all_new_points, _ = create_new_points(points, method=line_fit_method)
		all_new_points = np.empty((0, 3))

		mapping = {}	
		edge_mapping = []
		points_to_remove = []
		## get all parents
		parents_list = self.get_nodes_with_order_x(root_idx, parents_only=True)
		while len(parents_list)>0:
			## get first parent_id
			new_parent_id = len(new_parents) -1
			parent_id = parents_list[0]
			parents_list = parents_list[1:]

			# get all nodes of branch parent_id
			nodes_parent_branch = self.get_nodes_branch_id(branch_id=self.G.nodes[parent_id]["branch_number"])
			parent_points, _ = create_new_points(positions[nodes_parent_branch], method=line_fit_method)

			## for current parent id get child with nodeis+1
			nodes_with_id1= [n for n in self.G.successors(parent_id) if self.G.nodes[n]["node_order"]>self.G.nodes[parent_id]["node_order"]] #self.get_nodes_with_order_x(parent_id, node_order_offset=1)

			## fit a line through every unique branch connted to parent_id (mostly=1)
			unique_branches = np.unique(branches[nodes_with_id1])
			for unique_number in unique_branches:
				## get all nodes and parent nodes
				temp_nodes = self.get_nodes_branch_id(branch_id=unique_number)
				temp_nodes_parents = [x for x in temp_nodes if self.G.nodes[x]["is_parent"]]
				
				## add "end_points"
				last_node = self.get_nodes_with_order_x(temp_nodes[0])[-1]
				if not self.G.nodes[last_node]["is_parent"]:
					mapping[last_node] = positions[last_node]
					# if there are parent nodes in temp_nodes connect last node to that
					if len(temp_nodes_parents)>0:
						edge_mapping.append([temp_nodes_parents[-1], last_node])
					else: ## connect to parent_id
						edge_mapping.append([parent_id, last_node])

				## fit a line through all_nodes of current_branch
				fitered, extra_polated_child_points = create_new_points(positions[temp_nodes], method=line_fit_method)
				## if not exrapolated
				if extra_polated_child_points is None:
					parents_list2 = self.get_nodes_with_order_x(temp_nodes[0], parents_only=True)
					parents_list += parents_list2
			
					# update parent position
					if parent_id not in mapping:
						mapping[parent_id] = positions[parent_id]
					else: 
						old_pos = mapping[parent_id]
						mapping[parent_id] = (old_pos + positions[parent_id])/2

					## update edges
					if len(temp_nodes_parents)>0:
						edge_mapping.append([parent_id, temp_nodes_parents[0]])
					else:
						edge_mapping.append([parent_id, last_node])

					continue
					# new_edges.append((new_parent_id, len(new_parents)))
					# new_parents.append(positions[parent_id])

					# continue
				else: 
					# if extra polated we are interested in intersection child nodes and parent nodes.
					# parent node pose is updated on intersection
					extra_polated_child_points= extra_polated_child_points[::-1]
					new_pose, idx, child_idx = find_closest_points(parent_points, extra_polated_child_points)

					all_new_points = np.vstack([all_new_points, extra_polated_child_points[:child_idx]])

					# update parent position
					if parent_id not in mapping:
						mapping[parent_id] = new_pose
					else:
						old_pos = mapping[parent_id]
						mapping[parent_id] = (old_pos + new_pose)/2

					## check whether the intercepted pose is not to close to any other nodes in parent branch
					remove_close_points = True
					remove_close_th = 0.01 # remove points within 1 cm
					if remove_close_points:
						nodes_parent_branch_array = np.array(nodes_parent_branch)
						nodes_parent_branch_array = nodes_parent_branch_array[nodes_parent_branch_array!=parent_id]
						dist = np.linalg.norm(positions[nodes_parent_branch_array.tolist()] - mapping[parent_id], axis=1)
						points_to_remove += nodes_parent_branch_array[dist<remove_close_th].tolist()


					## update edges
					if len(temp_nodes_parents)>0:
						edge_mapping.append([parent_id, temp_nodes_parents[0]])
					else:
						edge_mapping.append([parent_id, last_node])

					new_edges.append((new_parent_id, len(new_parents)))
					new_parents.append(new_pose)

					## additional step to add end points for every branch, add end point if it is not a parent
					# last_node = self.get_nodes_with_order_x(temp_nodes[0])[-1]
					# if not self.G.nodes[last_node]["is_parent"]:
					# 	new_parents.append(positions[last_node])
					# 	new_edges.append((len(new_parents)-1, len(new_parents)))

					# parents_list+= self.get_nodes_with_order_x(temp_nodes[0], parents_only=True)
					parents_list2 = self.get_nodes_with_order_x(temp_nodes[0], parents_only=True)
					parents_list += parents_list2
			
		# update al node positions
		nodes = []
		key_mapping = {}
		for i, (key, pos) in enumerate(mapping.items()):
			key_mapping[key] = i
			nodes.append(pos)
			self.G.nodes[key]["pos"] = pos
	
		## add extrapolated points:
		# for i, pos in enumerate(all_new_points):
		# 	self.G.add_node(len(self.G.nodes), **{"pos": pos, "edge_type": None})

		edge_list = []
		for edge in edge_mapping:
			edge_list.append([key_mapping[edge[0]], key_mapping[edge[1]]])

		points_to_remove = [x for x in points_to_remove if not (self.G.nodes[x]["is_parent"] or self.G.nodes[x]["edge_type"]=="root")]
		self.remove_node_and_update_edge(points_to_remove, do_relabel=True)
		# It is possible that after line fitting non unique nodes are generated. 
		# remove those if they exist
		self.remove_non_unique_nodes()



	def merge_vertices_and_update_edges(self, id_source, id_target, 
			weight_source: float = 0.5,
			weight_target: float = 0.5,
			relabel: bool = True):
		"""
		Merge two vertices in the graph, update their edges, and optionally relabel the graph.
			Parameters:
				id_source (int): ID of the first vertex to merge.
				id_target (int): ID of the second vertex to merge.
				weight_source (float, optional): Weight for the position of the source vertex. Default is 0.5.
				weight_target (float, optional): Weight for the position of the target vertex. Default is 0.5.
				relabel (bool, optional): Whether to relabel the graph after merging. Default is True.
			Returns:
				None		
		"""
		parent_source = list(self.G.predecessors(id_source))[0]
		parent_target = list(self.G.predecessors(id_target))[0]

		if parent_source!=parent_target:
			print("Cannot vertices with different parent")
			return

		new_node_id = max(list(self.G.nodes))+1
		new_edges = []
		new_edges.append([parent_source, new_node_id])

		## remove old edges and add childs to new_edge_list
		childs_source = list(self.G.successors(id_source))
		for child_source in childs_source:
			new_edges.append([new_node_id, child_source])
			self.G.remove_edge(id_source, child_source)

		childs_target = list(self.G.successors(id_target))
		for child_target in childs_target:
			new_edges.append([new_node_id, child_target])
			self.G.remove_edge(id_target, child_target)
		# remove parent edge
		self.G.remove_edge(parent_source, id_source)
		self.G.remove_edge(parent_target, id_target)

		# calculate new position
		new_node_pos = self.G.nodes[id_source]["pos"]*weight_source + self.G.nodes[id_target]["pos"]*weight_target / (weight_source  + weight_target)

		# remove old nodes and add new
		self.G.remove_node(id_source)
		self.G.remove_node(id_target)
		self.G.add_node(new_node_id, **{"pos": new_node_pos, "edge_type": None})

		# update new edges 
		for parent, child in new_edges:
			self.G.add_edge(parent, child)
		for unique_parent in np.unique(np.array(new_edges)[:,0]):
			self.get_single_edge_type(node_id=unique_parent)

		if relabel:
			self.G = relabel(self.G)


def compute_merge_value(current_pos, i_pos, j_pos, alpha_th: float = 0.9):
	"""
	Function taht determines whether the posisition i_pos and j_pos should be merged.
	Based on the angle between i and j if exceeds alpha_threshold, and length
	of both sectors is roughly similar. 
	Returns distance from i_pos to vector_j using sin
	"""

	d_value = np.inf

	vector_i = i_pos - current_pos
	vector_j = j_pos - current_pos

	length_i = np.linalg.norm(vector_i)
	length_j = np.linalg.norm(vector_j)

	dir_i  = vector_i / length_i
	dir_j = vector_j / length_j

	alpha = np.dot(dir_i, dir_j)

	radius_target = 1

	if alpha>alpha_th:
		ratio = length_i / length_j
		if ratio>= 0.5 and ratio<=2:
			return length_i*math.sin(math.acos(alpha)) / radius_target
		## length is not similar so do not merge
		return d_value

	else:
		return d_value


def relabel(graph):
	"""
	Script to relabel graph for visualisation with polyscope.
	"""
	mapping = {node: i for i, node in enumerate(graph.nodes())}
	return nx.relabel_nodes(graph, mapping)


if __name__=="__main__":
	
	from scripts import config
	cfg = config.init_config("PAPER/config.yaml")
	
	node_list_name = "Resources/output_skeleton_paper3/0-paper-2Dto3D/voxel/Harvest_01_PotNr_80.csv"
	node_list_name = "/home/agro/w-drive-vision/GARdata/datasets/tomato_plant_segmentation/TomatoWUR_4dataTU/EXPERIMENTS_PAPER3/0-paper-2Dto3D/xu/Harvest_01_PotNr_80.csv"
	save_folder= "/home/agro/w-drive-vision/GARdata/datasets/tomato_plant_segmentation/TomatoWUR_4dataTU/EXPERIMENTS_PAPER3/figures_M&M/Harvest_01_PotNr_55.csv"

	obj = SkeletonGraph.from_nodelist(node_list_name)
	obj.visualise_graph(save_name="no_post_processing.png")
	# obj.get_edge_type()
	obj.get_node_order()
	# obj.gaussian_smoothing(var0=.25, var1=.25, indices=[0,1], node_order_filtering=True, num_children=2)
	obj.get_edge_type()
	# obj.visualise_graph()
	# exit()
	nodes = obj.get_node_attribute()
	_, indices = np.unique(nodes.astype(np.float16), axis=0, return_index=True)
	indices.sort()
	obj.line_fitting_3d()
	obj.get_edge_type()
	obj.visualise_graph()




	obj.simplify()
	# obj.gaussian_smoothing(var0=.25, var1=1, indices=[0,1], node_order_filtering=False, num_children=3)


	# obj.line_fitting_3d()
	obj.get_edge_type()
	# obj.filter(np.inf, True)
	obj.visualise_graph(save_name="post_processing.png")



