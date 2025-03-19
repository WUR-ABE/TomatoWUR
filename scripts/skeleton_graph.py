import networkx as nx
from pathlib import Path
import pandas as pd
import numpy as np
import math
import sys
sys.path.append("")
from scripts import visualize_examples as ve

def direction(v):
	return v / np.linalg.norm(v)

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
	get_gt_attributes(attributes_list=["gt_int_length", "gt_ph_angle", "gt_br_angle"])
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

		if edge_types is None:
			self.get_edge_type()

		pass


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
		a=dict(self.G.nodes(data=True)).values()
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

	def visualise_graph(self):
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

		ve.vis(pc = self.get_xyz_pointcloud(), nodes=nodes, edges=edges, node_order=node_order, parents=parents, edges_type=edges_type, attributes=attributes)

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
		self.get_edge_type()
		# self.get_node_order()
		
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



	def get_edge_type(self, angle_between_trunk_and_lateral=60):
		edge_types = {0: "root"}
		root = 0
		# Traverse the MTG to calculate orders
		# children = list(self.G.successors(root))
		# for child in children:
		# 	edge_types[child]="<"
		
		# stack = list(self.G.successors(root))
		stack = [root]
		while stack:
			new_id = stack.pop()
			# if new_id==4 or new_id==5:
			# 	print("x")
			pos = self.G.nodes[new_id]["pos"]

			if new_id==root:
				parent_pos = pos - np.array([0, 0, .1])
			else:
				parent = list(self.G.predecessors(new_id))[0]
				parent_pos = self.G.nodes[parent]["pos"]

			children = list(self.G.successors(new_id))
			if len(children)>0:
				langles = []
				for child in children:
					child_pos = self.G.nodes[child]["pos"]

					first_edge_type = '<'
					langle = math.degrees(math.acos(
						round(np.dot(direction(pos - parent_pos), direction(child_pos - pos)),3))) # round for bug fix

					if langle > angle_between_trunk_and_lateral: 
						first_edge_type = '+'
					else:
						langles.append(langle)

					edge_types[child] = first_edge_type
					stack.append(child)

				# if multiple angles are smaller than 60, then largest angle is a branch
				if len(langles)>1:
					edge_types[children[langles.index(max(langles))]] = "+"

		for child in self.G.nodes():
			self.G.nodes[child]['edge_type'] = edge_types[child]
		for parent, child in self.G.edges():
			self.G.edges[parent, child]['edge_type'] = edge_types[child]
		self.get_node_order()


	def export_as_nodelist(self, path):
		# nx.write_gpickle(self.G, path)
		df = pd.DataFrame(self.get_node_attribute("pos"), columns=["x", "y", "z"])
		df_2 = pd.DataFrame(self.get_edges(), columns=["parentid", "vid"])
		df_2["edgetype"] = self.get_edge_attribute("edge_type")
		df_result = pd.concat([df, df_2], axis=1)
		if not path.parent.exists():
			path.parent.mkdir(parents=True)
		df_result.to_csv(path, index=False)

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

	
	def load_csv(self, path):
		df = pd.read_csv(str(path), low_memory=False)
		self.load(nodes=df[["x", "y", "z"]].dropna().values, edges=df[["parentid", "vid"]].dropna().astype(int).values, edge_types=df["edgetype"].dropna().values)
		self.name = path.stem

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
		# dict_attributes = {gt_int_length, gt_int_diameter, gt_ph_angle, gt_br_angle}
		poses = self.get_node_attribute("pos")

		array = np.linalg.norm(poses - location, axis=1)
		self.G.nodes[array.argmin()].update(dict_attributes)
		pass

	def get_gt_attributes(self, attributes_list = ["gt_int_length", "gt_ph_angle", "gt_br_angle"]): # "gt_int_diameter"
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
						

		from scripts.calculate_angles import openalea_method, xy_plane_method
		# phyto_angle, relangles, rel_angle_index = openalea_method(poses, lateral_roots)
		# print("Phyto angle:", phyto_angle)
		# print("Relative angles:", relangles)
		# print("Relative angle indices:", rel_angle_index)
		phyto_angle, relangles, rel_angle_index, xy_points, xy_edges = xy_plane_method(poses, lateral_roots)
		## for debugging
		# ve.vis(self.get_node_attribute("pos"), nodes = xy_points, edges=xy_edges)
		# print("Phyto angle:", phyto_angle)
		# print("Relative angles:", relangles)


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
		branch_angles = []
		for j in lateral_roots[:-1]:
			pos_branch = poses[j[1][1]]
			pos_parent_node = poses[j[0]]
			pos_next_internode = poses[internodes[internodes.index(j[0])+1]]
			angle = math.degrees(math.acos(
				round(np.dot(direction(pos_branch - pos_parent_node), direction(pos_next_internode - pos_parent_node)),3)))
			branch_angles.append([j[0], angle])	
		
		## add internode length to graph
		for node, angle in branch_angles:
			self.G.nodes[node]["br_angle"] = angle

		print("x")


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

	def gaussian_smoothing(self, var0=.25, var1=.25, indices=[0,1], node_order_filtering=True):
		""""Filter the graph based on node order and parent status, and updates self.G accordingly.
		inputs:
			var0 (float): variance for node
			var1 (float): variance for parent and child nodes
			indices (list): indices to smooth. if [0, 1] only x and y will be smoothed
			node_order_filtering: if True only nodes with same node order will be smoothed
		returns:
			None
		"""

		nprop = dict()
		gw0 = gaussian_weight(0, var0)
		gw1 = gaussian_weight(1, var1)
		# self.visualise_graph()
		for node in self.G.nodes():
			value = self.G.nodes[node]["pos"]
			node_order = self.G.nodes[node]["node_order"]
			nvalues = [value * gw0]
			parent = list(self.G.predecessors(node))
			if parent!=[]:
				if node_order_filtering==False:
					nvalues.append(self.G.nodes[parent[0]]["pos"] * gw1)
				elif self.G.nodes[parent[0]]["node_order"] == node_order:
					nvalues.append(self.G.nodes[parent[0]]["pos"] * gw1)
			children = list(self.G.successors(node))
			# children = [child for child in children if self.G.nodes[child]["edge_type"] == '<']
			if node_order_filtering:
				children = [child for child in children if self.G.nodes[child]["node_order"] == node_order]

			for child in children:
				nvalues.append(self.G.nodes[child]["pos"] * gw1)
			nvalue = sum(nvalues[1:], nvalues[0]) / sum([gw0 + (len(nvalues) - 1) * gw1])
			nprop[node] = nvalue
		for node in nprop.keys():
			self.G.nodes[node]["pos"][indices] = nprop[node][indices]	
		# self.visualise_graph()

	def main_post_processing(self, cfg):
		if cfg.get("post_processing", None) is None:
			return

		for method in cfg["post_processing"].get("methods", []):
			temp = cfg["post_processing"][method]
			key = list(temp.keys())[0]
			print("Applying post processing method: ", key)
			if temp[key] is not None:
				getattr(self, key, lambda: "Unknown action")(**temp[key])
			else:
				getattr(self, key, lambda: "Unknown action")()


			# if method == "gaussian":
			# 	self.gaussian_smoothing(cfg["post_processing"]["gaussian"]["var0"], cfg["post_processing"]["gaussian"]["var1"])


	def line_fitting_3d(self):
		idx = 0 # 0 is root
		successors = list(nx.dfs_tree(self.G, 0))
		node_roder = self.G.nodes[idx]["node_order"]


		new_poses = []
		
		lateral_root = [n for n in successors if self.G.nodes[n]["node_order"]==node_roder]
		points = self.get_node_attribute("pos")[lateral_root]
		method = "poly1d"
		# main_stem_xyz = create_new_points(points, method=method)

		internodes, _, _ = self.get_internode_length()
		lateral_roots = []
		for i in internodes:
			node_roder = self.G.nodes[i]["node_order"]
			list_succesors = list(self.G.successors(i))
			branches = [n for n in list_succesors if self.G.nodes[n]["node_order"]==node_roder+1]
			branch = branches[np.argmax([len(nx.dfs_tree(self.G, branch)) for branch in branches])]

			# for branch in branches:
			# branch = branches[np.argmax([len(nx.dfs_tree(self.G, branch)) for branch in branches])]
			successors = list(nx.dfs_tree(self.G, branch))
			idx_side_shoot = [n for n in successors if self.G.nodes[n]["node_order"]==node_roder+1]
			if len(idx_side_shoot)<=3:
				continue
			query_z = self.get_node_attribute("pos")[idx_side_shoot][0, 2]
			points_bool = np.logical_and(points[:, 2] < (query_z+ 0.10) , points[:, 2] > (query_z - 0.1))
			main_stem_xyz = create_new_points(points[points_bool], method=method)

			points2 = self.get_node_attribute("pos")[idx_side_shoot][:10]
			new_points2 = create_new_points(points2, method)
	
			# Extract x, y, z coordinates
			new_pose, idx = find_closest_points(main_stem_xyz, new_points2)
			# ve.vis(self.get_xyz_pointcloud(), nodes=np.vstack([main_stem_xyz, new_points2]), root_idx=idx)

			new_poses.append([i, new_pose])
		for idx, new_pose in new_poses:
			self.G.nodes[idx]["pos"] = new_pose	
		self.G_original = self.G.copy()
		# self.visualise_graph()





def find_closest_points(parent_points, child_points):
	# Find the closest points in the source to the target points
	from scipy.spatial import cKDTree
	tree = cKDTree(parent_points)
	distance = np.inf
	idx = 0
	for child in child_points:
		temp_distance, temp_idx = tree.query(child)
		if tree.query(child)[0]<distance:
			distance = temp_distance
			idx = temp_idx
	new_pose = parent_points[idx]

	return new_pose, idx



def create_new_points(points, method = "spline", **kwargs):
	if len(points)<3:
		return points
	elif method == "spline":
		from scipy.interpolate import splprep, splev
		x, y, z = points[:, 0], points[:, 1], points[:, 2]

		# Use splprep to fit a parametric spline
		tck, u = splprep([x, y, z], s=0)  # `s=0` means no smoothing, use `s>0` for smoothing

		# Generate new parameter values for extrapolation
		u_new = np.linspace(-0.01, 1.01, 100)  # Extrapolate beyond [0, 1] range of u

		# Evaluate the spline at new parameter values
		x_new, y_new, z_new = splev(u_new, tck)
		xyz_new = np.column_stack([x_new, y_new, z_new])
	elif method=="poly1d":
		xyz_new = create_points_poly(points)
	return xyz_new

def create_points_poly(points):
	# Extract x, y, z coordinates
	x, y, z = points[:, 0], points[:, 1], points[:, 2]

	# Define parameter t (e.g., cumulative arc length or just indices)
	t = np.arange(len(points))
	# t  = np.linspace(-0.1, 1.1, 100)  # Extrapolate beyond [0, 1] range of u

	# Fit polynomials for x(t), y(t), z(t)
	degree = 1  # Degree of polynomial
	px = np.polyfit(t, x, degree)
	py = np.polyfit(t, y, degree)
	pz = np.polyfit(t, z, degree)

	# Create polynomial functions
	fx = np.poly1d(px)
	fy = np.poly1d(py)
	fz = np.poly1d(pz)

	t_new = np.linspace(t[0]-int(0.5*len(points)), t[-1] + int(0.1*len(points)), 100)  # Extrapolate beyond original range

	# Predict new points
	x_new = fx(t_new)
	y_new = fy(t_new)
	z_new = fz(t_new)
	xyz_new = np.column_stack([x_new, y_new, z_new])
	return xyz_new
					

def gaussian_weight(x, var):
	from math import exp, sqrt, pi
	return exp(-x ** 2 / (2 * var)) / sqrt(2 * pi * var) # corrected from opeanalea


# def gaussian_filter(mtg, propname, considerapicalonly=True):
# 	prop = mtg.property(propname)
# 	nprop = dict()
# 	gw0 = gaussian_weight(0, 1)
# 	gw1 = gaussian_weight(1, 1)
# 	for vid, value in list(prop.items()):
# 		nvalues = [value * gw0]
# 		parent = mtg.parent(vid)
# 		if parent and parent in prop:
# 			nvalues.append(prop[parent] * gw1)
# 		children = mtg.children(vid)
# 		if considerapicalonly: children = [child for child in children if mtg.edge_type(child) == '<']
# 		for child in children:
# 			if child in prop:
# 				nvalues.append(prop[child] * gw1)

# 		nvalue = sum(nvalues[1:], nvalues[0]) / sum([gw0 + (len(nvalues) - 1) * gw1])
# 		nprop[vid] = nvalue

# 	prop.update(nprop)


if __name__=="__main__":
	
	from scripts import config
	cfg = config.Config("config.yaml")
	
	dt_graph_dir = Path("Resources/output_skeleton") / cfg.skeleton_method

	plant_nr = "Harvest_01_PotNr_95"
	file_name = cfg.data.pointcloud_dir / (plant_nr + ".csv")

	data = pd.read_csv(str(file_name), low_memory=False)
	# 	# if load_skeleton_data:
	# skeleton_data = data.loc[
	# 	~data["x_skeleton"].isna(), ["x_skeleton", "y_skeleton", "z_skeleton", "vid", "parentid", "edgetype"]
	# ]
	# nodes = skeleton_data[["x_skeleton", "y_skeleton", "z_skeleton"]].values
	# edges = skeleton_data[["parentid", "vid"]].values[1:].astype(int)
	# edge_types = skeleton_data["edgetype"].values[1:].astype(str)

	# obj = SkeletonGraph(file_name)
	# obj.load(nodes, edges, edge_types)
	# obj.get_node_order()
	# obj.get_angles()

	S_pred = SkeletonGraph()
	S_pred.load_csv(dt_graph_dir / (file_name.stem + ".csv"))
	S_pred.df = data[['x', 'y', 'z']]
	S_pred.get_node_order()
	S_pred.export_as_mtg()
	# S_pred.visualise_graph()
	S_pred.filter(np.inf, True)
	S_pred.edge_from_filtered()
	# S_pred.get_edge_type(70)
	# S_pred.visualise_graph()
	S_pred.line_fitting_3d()

	# S_pred.main_post_processing(cfg=config)


	# obj.visualise_graph()
	# # obj.filter(2, True)
	# obj.G = obj.G_filtered
	# obj.visualise_graph()
