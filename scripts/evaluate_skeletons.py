from __future__ import annotations

from pathlib import Path
# from wurTomato import WurTomatoData
import pickle
import polyscope as ps
import numpy as np
from natsort import natsorted
from dataclasses import dataclass
import copy

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import pandas as pd
import json
from scripts.utils_skeletonisation import load_json
from scripts.utils_data import create_skeleton_gt_data
from scripts.skeleton_graph import SkeletonGraph
import scripts.visualize_examples as ve
from scripts.calculate_metrics import Metrics

## TODO fix assignmetn problem

class skeleton_matching_bart():
	""" Match predicted nodes to ground truth nodes based on euclidean distance.
	
	"""

	def __init__(self, S_gt, S_pred, method="boogaard", threshold = 0.02, node_order=None) -> None:
		self.S_gt = S_gt
		self.S_pred = S_pred
		self.method = method
		self.matched_indices = None # np.array(n x [gt_index, pred_index])
		self.false_threshold = threshold
		self.node_order = node_order

	def match(self):
		if self.method == "hmm":
			self.match_hmm()
		elif self.method == "roel":
			self.match_roel()
		elif self.method=="boogaard":
			self.match_boogaard()
		elif self.method=="oks":
			self.oks()
		else:
			print("Method not implemented")


	def match_hmm(self):
		from plant_registration_4d import skeleton_matching as skm
		from plant_registration_4d import skeleton as skel

		S1 = skel.Skeleton(XYZ=self.S_gt.get_node_attribute("pos")*100, edges=self.S_gt.get_edges())
		S2 = skel.Skeleton(XYZ=self.S_pred.get_node_attribute("pos")*100, edges=self.S_pred.get_edges())

		params = {'weight_e': 0.01, 'match_ends_to_ends': False,  'use_labels' : False, 'label_penalty' : 1, 'debug': True}
		corres = skm.skeleton_matching(S1, S2, params)

		self.matched_indices = corres

		return corres

	def match_roel(self):
		gt_nodes = self.S_gt.get_node_attribute("pos")
		dt_nodes = self.S_pred.get_node_attribute("pos")

		cost_matrix = distance_matrix(gt_nodes, dt_nodes, p=2)  # p=2 for euclidean dist
		# cost_matrix[cost_matrix > self.false_threshold ] = np.inf
		row_ind, col_ind = linear_sum_assignment(cost_matrix)

		average_tp_error_meters = cost_matrix[row_ind, col_ind].mean()
		self.matched_indices = np.array([row_ind, col_ind]).T
		self.matched_indices = self.matched_indices[cost_matrix[row_ind, col_ind] <= self.false_threshold]



	def match_boogaard(self):
		"""Match predicted nodes to ground truth nodes based on euclidean distance.
		TP if distance is below threshold, FP otherwise. If not match then FN."""

		gt_nodes = self.S_gt.get_node_attribute("pos")
		gt_node_order = self.S_gt.get_node_attribute("node_order")
		dt_nodes = self.S_pred.get_node_attribute("pos")
		dt_node_order = self.S_pred.get_node_attribute("node_order")

		cost_matrix = distance_matrix(gt_nodes, dt_nodes, p=2)  # p=2 for euclidean dist

		matched = []
		matched_indices = []
		for i, gt_node in enumerate(gt_nodes):

			candidates = np.where(cost_matrix[i, :] < self.false_threshold)[0]
			if len(candidates) == 0:
				# FN 
				continue
			arg_sor = cost_matrix[i, candidates].argsort()
			candidates = candidates[arg_sor]

			for candidate in candidates:
				if candidate in matched:
					continue
				# check whether match is optimal
				current_dist = cost_matrix[i, candidate]
				if cost_matrix[:, candidate].min() < current_dist:
					continue
				else:
					# TP
					matched.append(candidate)
					matched_indices.append([i, candidate, gt_node_order[i], dt_node_order[candidate]])
					break
		self.matched_indices = np.array(matched_indices)
		if self.matched_indices.size == 0:
			self.average_tp_error_meters = np.nan
		else:
			self.average_tp_error_meters = cost_matrix[self.matched_indices[:, 0], self.matched_indices[:, 1]].mean()



	def oks(self):
		print("wip")


class GraphPairs:
	def __init__(self, gt_graphs, dt_graphs) -> None:
		# pairs : list[tuple[Graph, Graph]]
		self.pairs = []
		for gt_graph in gt_graphs:
			matching_dt_graph = None
			for dt_graph in dt_graphs:
				if gt_graph.name==dt_graph.name:
					matching_dt_graph = dt_graph
			if matching_dt_graph is not None:
				self.pairs.append((gt_graph, matching_dt_graph))
			else:
				print(f"No predictions found for {gt_graph.name}, skipping...")
		return

	def plot_pair(self, index):
		self.pairs[index][0].plot(other=graph_pairs.pairs[index][1])
		return


@dataclass
class Graph:
	filename: str
	nodes: np.array  # shape (N, 3)
	edges: np.array  # shape (M, 2)
	pcd : np.array | None = None # shape (X, 2), where usually X >> N, like 40.000 or something

	def plot(
		self,
		other: Graph | None,
		own_name="ground_truth",
		other_name="prediction",
		match_threshold_meters=0.02,
	):
		ps.init()
		ps.set_up_dir("z_up")
		ps.remove_all_structures()
		# Set the colors for each point cloud
		color1 = [0, 1, 0]  # Green
		color2 = [1, 0, 0]  # Red
		color3 = [0, 1, 1]  # Yellow

		gt_nodes = self.nodes
		dt_nodes = other.nodes

		if self.pcd is not None:
			ps.register_point_cloud("Source Point Cloud", self.pcd, radius=0.001, color=color3)
		ps.register_point_cloud(own_name, self.nodes, radius=0.01, color=color1)
		ps.register_curve_network(
			"Ground truth edges", self.nodes, self.edges, radius=0.005, color=color1
		)
		if other is not None:
			ps.register_point_cloud(other_name, other.nodes, radius=0.01, color=color2)
			ps.register_curve_network(
				"Predicted edges", other.nodes, other.edges, radius=0.005, color=color2
			)
		ps.show()

	def plot_with_metrics(
			self,
			dt_graph : Graph,
			match_threshold_meters=0.02,
	):
		raise NotImplementedError # TODO plot the TP, FP and FN points/edges

	def plot_matches(
		self,
		other: Graph,
		own_name="ground_truth",
		other_name="prediction",
		match_threshold_meters=0.02,
	):
		gt_nodes = self.nodes
		dt_nodes = other.nodes

		cost_matrix = distance_matrix(gt_nodes, dt_nodes, p=2)  # p=2 for euclidean dist
		row_ind, col_ind = linear_sum_assignment(cost_matrix)

		matched_indices = cost_matrix[row_ind, col_ind] <= match_threshold_meters

		matching_edges = np.array([[i, j] for (i, j) in zip(row_ind, col_ind)])


		ps.init()
		ps.set_up_dir("z_up")
		ps.remove_all_structures()

		# Set the colors for each point cloud
		color1 = [0, 1, 0]  # Green
		color2 = [1, 0, 0]  # Red
		color3 = [0, 0, 1]  # Blue

		ps.register_point_cloud(own_name, self.nodes, radius=0.01, color=color1)
		ps.register_point_cloud(other_name, other.nodes, radius=0.01, color=color2)
		for matching_edge in matching_edges:
			match_nodes = np.array(
				[gt_nodes[matching_edge[0]], dt_nodes[matching_edge[1]]]
			)

			if matching_edge[0] == 6 and matching_edge[1] == 4:
				print("interesting")
			ps.register_curve_network(
				f"match{matching_edge[0]},{matching_edge[1]}",
				match_nodes,
				np.array([[0, 1]]),
				radius=0.005,
				color=color3,
			)
		ps.show()
		return
	
	def save(self, path):
		for name, attribute in self.__dict__.items():
			name = ".".join((name, "pkl"))
			with open("/".join((path, name)), "wb") as f:
				pickle.dump(attribute, f)

	@classmethod
	def load(cls, path):
		my_model = {}
		for name in cls.__annotations__:
			file_name = ".".join((name, "pkl"))
			with open("/".join((path, file_name)), "rb") as f:
				my_model[name] = pickle.load(f)
		return cls(**my_model)


def load_dt_graphs(dt_graph_dir: Path, pickeled_predictions=False, split="test"):
	assert dt_graph_dir.is_dir()
	dt_graphs = []

	# if pickeled_predictions:
	# 	dt_graph_paths = natsorted(dt_graph_dir.glob("*.pkl"))
	# 	with open(f"3DTomatoDataset/20240607_summerschool_csv/{split}.json") as f:
	# 		filenames = json.load(f)
	# 	split_stems = [Path(f["sem_seg_file_name"]).stem for f in filenames]
	# 	dt_graph_paths = [ f for f in dt_graph_paths if f.stem in split_stems]
	# 	for dt_graph_path in dt_graph_paths:
	# 		with open(dt_graph_path, "rb") as f:
	# 			dt_graph = pickle.load(f)
	# 		dt_graphs.append(dt_graph)
	# else:
	dt_graphs = []
	dt_graph_paths = natsorted(dt_graph_dir.glob("*.csv"))

	annot_folder = Path(r"W:\PROJECTS\VisionRoboticsData\GARdata\datasets\tomato_plant_segmentation\20240607_summerschool_csv")
	with open(annot_folder / (f"{split}.json"), "r") as f:
		filenames = json.load(f)
	split_stems = [Path(f["sem_seg_file_name"]).stem for f in filenames]
	dt_graph_paths = [ f for f in dt_graph_paths if f.stem in split_stems]

	for dt_graph_path in dt_graph_paths:
		S_pred = SkeletonGraph()
		S_pred.load_csv(dt_graph_path)
		dt_graphs.append(S_pred)
	# else:
	# 	annot_folder = Path(r"W:\PROJECTS\VisionRoboticsData\GARdata\datasets\tomato_plant_segmentation\20240607_summerschool_csv")
	# 	with open(annot_folder / (f"{split}.json"), "r") as f:
	# 		filenames = json.load(f)
	# 	split_stems = [Path(f["sem_seg_file_name"]).stem for f in filenames]
	# 	dt_graph_paths = [ f for f in dt_graph_paths if f.stem in split_stems]
		
	# 	for dt_graph_path in dt_graph_paths:
	# 		df = pd.read_csv(dt_graph_path)
	# 		df = df.loc[df["class_pred"] == 4, ["x", "y", "z"]]
	# 		S_pred = convert_segmentation2skeleton(df, "dbscan")
	# 		dt_graph = Graph(
	# 			filename=dt_graph_path.stem, nodes=S_pred.XYZ, edges=np.array(S_pred.edges)
	# 		)

			# ve.vis(pc= df[["x", "y", "z"]], nodes=np.array(S_pred.XYZ), edges=np.array(S_pred.edges))

			# S_pred = SkeletonGraph(S_pred.XYZ, np.array(S_pred.edges), edge_types=None)
			# S_pred.name = dt_graph_path.stem
			# S_pred.get_edge_type()
			# S_pred.get_node_order()
			# S_pred.filter(2, True)
			# dt_graphs.append(S_pred)
		# Classnames:
		# 0 is leaf
		# 1 is main stem
		# 2 is pole
		# 3 is side stem
		# 4 is nodes
	return dt_graphs



class Evaluation():
	
	def __init__(self, gt_path_dir=None, 
			  dt_graph_dir=None, 
			  gt_json=None,

			  cfg={"json_name": "test"}):
		
		self.gt_path_dir = gt_path_dir
		self.dt_path_dir = dt_graph_dir
		self.save_folder = self.dt_path_dir
		self.cfg = cfg
		self.gt_json = gt_json


		self.filter = True
		self.filter_node_order = np.inf
		self.node_order_eval_list = [0,1,2,3] #which nodes to evaluate, if -1 then all
		self.parents_only = True
		self.false_threshold = 0.02 # meters

	def calculate_chamfer_distance(self, gt_nodes, dt_nodes):
		"""Calculate the chamfer distance between two sets of nodes.
		Args:
			gt_nodes (np.array): Ground truth nodes
			dt_nodes (np.array): Detected nodes
		Returns:
			float: Chamfer distance
		"""
		############# slow calculation
		# sum_g = 0
		# for gt_node in gt_nodes:
		# 	dist = np.linalg.norm(dt_nodes - gt_node,axis=1)
		# 	sum_g+=np.min(dist)
		# sum_g = sum_g / len(gt_nodes)

		# sum_d = 0
		# for dt_node in dt_nodes:
		# 	dist = np.linalg.norm(gt_nodes - dt_node,axis=1)
		# 	sum_d+=np.min(dist)
		# sum_d = sum_d / len(dt_nodes)

		# Calculate the sum of minimum distances from each ground truth node to the detected nodes
		distances = np.linalg.norm(dt_nodes[:, np.newaxis] - gt_nodes, axis=2)
		sum_g = np.sum(np.min(distances, axis=0)) / len(gt_nodes)

		distances = np.linalg.norm(gt_nodes[:, np.newaxis] - dt_nodes, axis=2)
		sum_d = np.sum(np.min(distances, axis=0)) / len(dt_nodes)

		cd = sum_g + sum_d

		return cd


	def evaluate_pairs(self, graph_pairs=None, vis=False, evaluate_gt=False):
		# if graph_pairs is None:
		# 	gt_graphs = self.load_all_gt_data()
		# 	dt_graphs = load_dt_graphs(self.dt_path_dir, pickeled_predictions=False, split=self.split)
		# 	graph_pairs = GraphPairs(gt_graphs, dt_graphs).pairs


		node_metrics_all = {
			"TP": 0,
			"FP": 0,
			"FN": 0,
			"nanmean_match_distance": [],
		}
		edge_metrics_all = {
			"TP": 0,
			"FP": 0,
			"FN": 0,
		}
		
		per_graph_metrics = []

		# df = pd.DataFrame(columns=["file_name",  
		# 					 "TP", "FP", "FN", "Precision", "Recall", "CD",
		# 					 "TP_edges", "FP_edges", "FN_edges", "Precision_edges", "Recall_edges"])
		df = pd.DataFrame()

		for gt_name in self.load_all_gt_filenames():

			S_gt = self.load_gt_data(gt_name)
			if evaluate_gt:
				S_pred = copy.deepcopy(S_gt)
			else:
				S_pred = self.load_pred_data(S_gt.name)

			node_metrics_pair, edge_metrics_pair, trait_metrics_pair = self.evaluate_single(
				S_gt, S_pred, vis=vis
			)
			# Create a new DataFrame with the new row
			# new_row = pd.DataFrame([{"file_name": gt_name.stem, **node_metrics_pair, **edge_metrics_pair}])
			new_row = pd.DataFrame([{"file_name": gt_name.stem, **node_metrics_pair, **edge_metrics_pair, **trait_metrics_pair}])

			# Concatenate the new row to the existing DataFrame
			df = pd.concat([df, new_row], ignore_index=True)

		################ node metrics
		# Create a dictionary of metrics for all nodes
		metrics = ["TP", "FP", "FN", "Precision", "Recall", "CD"]
		df_nodes = pd.DataFrame(metrics, columns=["metric"])

		# Create a list of node order evaluation strings
		node_order_eval_list_str = [""] + ["_" + str(x) for x in self.node_order_eval_list]

		# Iterate over each node order evaluation string
		for str_node_order in node_order_eval_list_str:
			# Sum the TP, FP, and FN columns for the current node order
			df_temp = df[["TP" + str_node_order, "FP" + str_node_order, "FN" + str_node_order]].sum()
			
			# Calculate Precision and Recall
			tp = df_temp["TP" + str_node_order]
			fp = df_temp["FP" + str_node_order]
			fn = df_temp["FN" + str_node_order]
			
			precision = tp / (tp + fp) if (tp + fp) > 0 else 0
			recall = tp / (tp + fn) if (tp + fn) > 0 else 0
			
			# Calculate Chamfer Distance (CD)
			cd = df["CD"].mean() if str_node_order == "" else ""
			
			# Append the calculated metrics to the new DataFrame
			df_nodes["node_order" + str_node_order] = [tp, fp, fn, precision, recall, cd]

		# Reset the index of the new DataFrame
		df_nodes.reset_index(drop=True, inplace=True)

		print("---" * 20)
		print("NODE METRICS")
		print(df_nodes[["metric", "node_order"]])

		################ edge metrics
		# Sum the TP, FP, and FN columns for edges
		df_temp = df[["TP_edges", "FP_edges", "FN_edges"]].sum()

		# Calculate Precision and Recall for edges
		precision_edges = df_temp["TP_edges"] / (df_temp["TP_edges"] + df_temp["FP_edges"]) if (df_temp["TP_edges"] + df_temp["FP_edges"]) > 0 else 0
		recall_edges = df_temp["TP_edges"] / (df_temp["TP_edges"] + df_temp["FN_edges"]) if (df_temp["TP_edges"] + df_temp["FN_edges"]) > 0 else 0

		# Create a dictionary with the metrics
		metrics_dict = {
			"TP_edges": df_temp["TP_edges"],
			"FP_edges": df_temp["FP_edges"],
			"FN_edges": df_temp["FN_edges"],
			"Precision_edges": precision_edges,
			"Recall_edges": recall_edges
		}

		# Create a DataFrame from the dictionary
		df_edges = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])

		print("---" * 20)
		print("EDGES METRICS")
		print(df_edges)

		################ trait metrics
		df_traits = pd.DataFrame()

		for trait in self.traits:
			df[trait+ "_MAE"] = ""
			df[trait+ "_MSE"] = ""
			df[trait+ "_RMSE"] = ""
			df[trait+ "_MAPE"] = ""

			temp_gt = []
			temp_dt = []
			sum_n = 0
			sum_n_notmatched = 0
			for index, row in df.iterrows():
				dummy_gt = row[trait]["gt"]
				dummy_dt = row[trait]["dt"]
				temp_metrics =  Metrics(y_pred=dummy_dt, gt=dummy_gt).return_dataframe()
				df.iloc[index, df.columns.get_loc(trait+ "_MAE")] = temp_metrics["MAE"][0]
				df.iloc[index, df.columns.get_loc(trait+ "_MSE")] = temp_metrics["MSE"][0]
				df.iloc[index, df.columns.get_loc(trait+ "_RMSE")] = temp_metrics["RMSE"][0]
				df.iloc[index, df.columns.get_loc(trait+ "_MAPE")] = temp_metrics["MAPE"][0]

				temp_gt.extend(row[trait]["gt"])
				temp_dt.extend(row[trait]["dt"])
				sum_n += row[trait]["counter"]
				sum_n_notmatched += row[trait]["counter_notmatched"]
			new_row = Metrics(y_pred=temp_dt, gt=temp_gt).return_dataframe()
			new_row["N"] = sum_n
			new_row["N_notmatched"] = sum_n_notmatched
			df_traits = pd.concat([df_traits, new_row], ignore_index=True)
		
		## delete old trait columns
		df = df.drop(columns=self.traits)

		df_traits["trait"] = self.traits
		## Reordering the columns
		df_traits = df_traits[['trait'] + [col for col in df_traits.columns if col != 'trait']]

		metric_file_node = self.save_folder / "metrics_node.csv"
		metric_file_edge = self.save_folder / "metrics_edge.csv"
		metrics_per_plant = self.save_folder / "metrics_per_plant.csv"
		metric_file_traits = self.save_folder / "metrics_plant_traits.csv"

		if not self.save_folder.exists():
			self.save_folder.mkdir(parents=True)

		df_nodes.to_csv(metric_file_node, index=False)
		df_edges.to_csv(metric_file_edge, index=False)
		df_traits.to_csv(metric_file_traits, index=False)

		print("-"*100, "\n", df_nodes.to_string(index=False))
		print("-"*100, "\n", df_edges.to_string(index=False))
		print("-"*100, "\n", df_traits[["trait", "MAE", "MAPE", "N", "N_notmatched"]].to_string(index=False))

		df_metrics_per_plant = pd.DataFrame(df)
		df_metrics_per_plant.to_csv(metrics_per_plant, index=False)
		print(f"Result are in file://{metric_file_edge.resolve()}")

		return df_nodes, df_edges, df_metrics_per_plant


	def evaluate_single(self, 
		gt_graph: Graph, dt_graph: Graph, vis=False
	):
		"""Match xyz nodes based on euclidean distance."""
		print(gt_graph.name)

		gt_nodes = gt_graph.get_node_attribute("pos")
		dt_nodes = dt_graph.get_node_attribute("pos")
		gt_node_order = gt_graph.get_node_attribute("node_order")
		dt_node_order= dt_graph.get_node_attribute("node_order")

		dummy = skeleton_matching_bart(gt_graph, dt_graph, threshold=self.false_threshold, node_order=gt_graph.get_node_attribute("node_order"))
		dummy.match()

		matched_indices = dummy.matched_indices
		average_tp_error_meters = dummy.average_tp_error_meters
		

		################################### node evaluation
		TP = len(matched_indices)
		FP = len(dt_nodes) - TP
		FN = len(gt_nodes) - TP

		precision = TP / (TP + FP)
		recall = TP / (TP + FN)

		node_metrics = {
			"TP": TP,
			"FP": FP,
			"FN": FN,
			"Precision": precision,
			"Recall": recall,
			# "mean_match_distance": average_tp_error_meters,
			# "CD": chamfer_distance,
		}

		## Calculate TP, FP, FN for each node order
		for x in self.node_order_eval_list:
			if x == self.node_order_eval_list[-1]:
				gt_x = matched_indices[:, 2] >= x
				pred_x = matched_indices[:, 3] >= x
				gt_count = np.sum(gt_node_order >= x)
				pred_count = np.sum(dt_node_order >= x)
			else:
				gt_x = matched_indices[:, 2] == x
				pred_x = matched_indices[:, 3] == x
				gt_count = np.sum(gt_node_order == x)
				pred_count = np.sum(dt_node_order == x)

			TP_x = np.sum(gt_x & pred_x)
			FP_x = pred_count - TP_x
			FN_x = gt_count - TP_x

			node_metrics[f"TP_{x}"] = TP_x
			node_metrics[f"FP_{x}"] = FP_x
			node_metrics[f"FN_{x}"] = FN_x


		## counts zero order
		# bools =  gt_graph.get_node_attribute("node_order")==0
		# matched_indices[:, 0]
		

		# df = pd.DataFrame(gt_nodes, columns=["x", "y", "z"])
		# df["TP"] = 0
		# df.loc[matched_indices[:, 0], "TP"] = 1
		# # df["TP"].iloc[matched_indices[:, 0]] .sum()

		# df["FP"] = 0
		# df["FN"] = 0
		# df.loc[df["TP"]==0, "FN"] = 1
		# df["node_order"] = gt_graph.get_node_attribute("node_order")

		chamfer_distance = 0
		chamfer_distance = self.calculate_chamfer_distance(gt_nodes, dt_nodes)
		node_metrics["CD"] = chamfer_distance

		# node_metrics = {
		# 	"TP": TP,
		# 	"FP": FP,
		# 	"FN": FN,
		# 	"Precision": precision,
		# 	"Recall": recall,
		# 	# "mean_match_distance": average_tp_error_meters,
		# 	"CD": chamfer_distance,
		# }
		# for x in self.node_order_eval_list:
		# 	if x!=self.node_order_eval_list[-1]:
		# 		bools = df["node_order"]==x
		# 	else: # to include larger than
		# 		bools = df["node_order"]>=x
		# 	node_metrics["TP_"+str(x)] = int(df.loc[bools, "TP"].sum())
		# 	node_metrics["FP_"+str(x)] = int(df.loc[bools, "FP"].sum())
		# 	node_metrics["FN_"+str(x)] = int(df.loc[bools, "FN"].sum())

			# df.loc[df["node_order"]==x, "FP"] = df.loc[df["node_order"]==x, "FP"].sum()
			# df.loc[df["node_order"]==x, "FN"] = df.loc[df["node_order"]==x, "FN"].sum()

		################################### edge & trait evaluation
		gt_edges = gt_graph.get_edges()
		dt_edges = dt_graph.get_edges()

		edge_tp = 0

		if matched_indices.size == 0:
			gt_nodes_matched = []
			dt_nodes_matched = []
		else:
			gt_nodes_matched = matched_indices[:, 0]
			dt_nodes_matched = matched_indices[:, 1]
		

		################################### trait metrics 
		self.traits = ["gt_int_length", "gt_ph_angle", "gt_lf_angle"] # "gt_int_diameter"
		dt_graph.get_internode_length()
		dt_graph.get_angles()
		trait_metrics = {x:{"gt":[], "dt":[], "counter":0, "counter_notmatched":0} for x in self.traits}
		temp = gt_graph.get_gt_attributes()
		temp_nodes = [x["node"] for x in temp]

		# for every node, if the node has a positive match calculate the error, else counter number of missed

		for gt_edge in gt_edges:
			if gt_edge[0] not in gt_nodes_matched or gt_edge[1] not in gt_nodes_matched:
				if gt_edge[1] in temp_nodes:
					indexi = temp_nodes.index(gt_edge[1])
					trait = temp[indexi]
					for trait_name in self.traits:
						if trait.get(trait_name) is None:
							continue
						trait_metrics[trait_name]["counter_notmatched"] = trait_metrics[trait_name].get("counter_notmatched", 0) + 1
				continue

			matching_dt_edge = np.array(
				[
					dt_nodes_matched[gt_nodes_matched == gt_edge[0]][0],
					dt_nodes_matched[gt_nodes_matched == gt_edge[1]][0],
				]
			)
			# remember, undirected graph
			if matching_dt_edge in dt_edges or matching_dt_edge[..., ::-1] in dt_edges:
				edge_tp += 1

				if gt_edge[1] in temp_nodes:
					indexi = temp_nodes.index(gt_edge[1])
					trait = temp[indexi]
					dt_node = matched_indices[matched_indices[:,0]==gt_edge[1], 1][0]
					dt_traits = dt_graph.G.nodes[dt_node]
					for trait_name in self.traits:
						if trait.get(trait_name) is None:
							continue
						dt_value = dt_traits.get(trait_name.replace("gt_", "")) ## check because upper dt_trait will not have 
						if dt_value is not None:
							trait_metrics[trait_name]["dt"].append(dt_value)
							trait_metrics[trait_name]["gt"].append(trait[trait_name])
							trait_metrics[trait_name]["counter"] = trait_metrics[trait_name].get("counter", 0) + 1
						else:
							trait_metrics[trait_name]["counter_notmatched"] = trait_metrics[trait_name].get("counter_notmatched", 0) + 1
			else:
				if gt_edge[1] in temp_nodes:
					indexi = temp_nodes.index(gt_edge[1])
					trait = temp[indexi]
					for trait_name in self.traits:
						if trait.get(trait_name) is None:
							continue
						trait_metrics[trait_name]["counter_notmatched"] = trait_metrics[trait_name].get("counter_notmatched", 0) + 1

		edge_fn = len(gt_edges) - edge_tp
		edge_fp = len(dt_edges) - edge_tp

		if edge_tp==0: # to prevent division by zero error
			edge_precision = 0
			edge_recall = 0
		else:
			edge_recall = edge_tp / (edge_tp + edge_fn)
			edge_precision = edge_tp / (edge_tp + edge_fp)
			# "TP_edges", "FP_edges", "FN_edges", "Precision_edge", "Recall_edge"
		edge_metrics = {
			"TP_edges": int(edge_tp),
			"FP_edges": int(edge_fp),
			"FN_edges": int(edge_fn),
			"Precision_edges": int(edge_precision),
			"Recall_edges": int(edge_recall),
		}

		
		# for trait in temp:
		# 	node = trait["node"]
		# 	if node in gt_nodes_matched:
		# 		dt_node = matched_indices[matched_indices[:,0]==node, 1][0]
		# 		dt_traits = dt_graph.G.nodes[dt_node]
		# 		for trait_name in traits:
		# 			if trait.get(trait_name) is None:
		# 				continue
		# 			dt_value = dt_traits.get(trait_name.replace("gt_", "")) ## check because upper dt_trait will not have 
		# 			if dt_value is not None:
		# 				trait_metrics[trait_name]["dt"].append(dt_value)
		# 				trait_metrics[trait_name]["gt"].append(trait[trait_name])
		# 				trait_metrics[trait_name]["counter"] = trait_metrics[trait_name].get("counter", 0) + 1
		# 	else:
		# 		for trait_name in traits:
		# 			if trait.get(trait_name) is None:
		# 				continue
		# 			trait_metrics[trait_name]["counter_notmatched"] = trait_metrics[trait_name].get("counter_notmatched", 0) + 1

		# for trait in traits:
		# 	gt_trait = gt_graph.get_node_attribute(trait)
		# 	dt_trait = dt_graph.get_node_attribute(trait.replace("gt_", ""))
		# 	trait_error = np.abs(gt_trait - dt_trait).mean()
		# 	trait_metrics[trait] = trait_error

		if vis:
			ve.vis_evaluation(gt_graph, dt_graph, matched_indices)
			
		return node_metrics, edge_metrics, trait_metrics


	def evaluate_pred(self, S_pred=None, pred_name=None, vis=False, evaluate_gt=False):
		"""Find related ground truth and return accuracy"""
		lists_gt_names = list(self.gt_path_dir.rglob("*.csv"))
		lists_gt_name_stem = [x.stem for x in lists_gt_names]
		indexi = lists_gt_name_stem.index(pred_name)
		if indexi==[]:
			print(f"No ground truth csv found for {pred_name}")
			return
		
		print("Evaluating", pred_name)

		S_gt = self.load_gt_data(lists_gt_names[indexi])
		if evaluate_gt: ## evaluate the ground truth traits
			S_pred = copy.deepcopy(S_gt)
		elif S_pred is None:
			S_pred = self.load_pred_data(lists_gt_names[indexi])


		node_metric, edge_metric, trait_metrics = self.evaluate_single(S_gt, S_pred, vis=vis)
		self.print_metric(node_metric)
		self.print_metric(edge_metric)
		self.print_trait_metric(trait_metrics)


	def load_all_gt_filenames(self):
		file_names = load_json(self.gt_json)
		return natsorted([self.gt_json.parent / f["skeleton_file_name"] for f in file_names])


	def load_gt_data(self, gt_name):
		print("Loading", gt_name)
		S_gt = create_skeleton_gt_data(gt_name)
		S_gt.get_node_order()
		# S_gt.visualise_graph()
		S_gt.filter(self.filter_node_order, self.parents_only)
		# S_gt.visualise_graph()
		# test = S_gt.get_gt_attributes()

		return S_gt
	

	def load_pred_data(self, plant_id):
		S_pred = SkeletonGraph()
		S_pred.load_csv(self.dt_path_dir / (plant_id + ".csv"))
		S_pred.get_node_order()
		# S_pred.filter(self.filter_node_order, self.parents_only)
		# S_pred.get_edge_type()
		# S_pred.edge_from_filtered()
		# S_pred.line_fitting_3d()
		# S_pred.filter(self.filter_node_order, self.parents_only)
		S_pred.main_post_processing(self.cfg)

		return S_pred
	
	def print_metric(self, metric):
		for k, v in metric.items():
			print(f"{k:<10}{v:>8.3f}")
		print("---" * 20)

	def print_trait_metric(self, trait_metrics):
		for k, v in trait_metrics.items():
			if v["gt"]==[]:
				print(f"{k:<10}, is empty, counter: {v['counter']}, counter_notmatched: {v['counter_notmatched']}")
				continue
			print(f"{k:<10}, MAE: {np.mean(np.abs(np.subtract(v['gt'], v['dt'])))}, counter: {v['counter']}, counter_notmatched: {v['counter_notmatched']}")
		print("---" * 20)

	def show(self, metric="Precision", ascending=True):
		df = pd.read_csv(self.dt_path_dir / "metrics_per_plant.csv")
		df = df.sort_values(by=metric, ascending=ascending)
		for row in df[:5].iterrows():
			print(row[1]["file_name"])
			self.evaluate_pred(pred_name=row[1]["file_name"], vis=True)

if __name__ == "__main__":
	from scripts import config
	cfg = config.Config("config.yaml")
	dt_graph_dir = Path("Resources/output_skeleton") / config["skeleton_method"]

	obj = Evaluation(cfg.pointcloud_dir, dt_graph_dir, cfg=config)
	# obj.filter_node_order = 0
	# obj.parents_only = False
	# obj.evaluate_pairs_per_nodeorder(vis=False)
	# obj.evaluate_pairs(vis=False, evaluate_gt=config["evaluation"]["evaluate_gt"])
	obj.evaluate_pred(pred_name="Harvest_01_PotNr_95", vis=True, evaluate_gt=config["evaluation"]["evaluate_gt"])
	# obj.evaluate_pred(pred_name="Harvest_01_PotNr_95", vis=True, evaluate_gt=True)


	# S_pred = SkeletonGraph("test_graph.csv")


