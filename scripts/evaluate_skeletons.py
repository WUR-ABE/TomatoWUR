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
import sys
sys.path.append("")
from scripts.utils_skeletonisation import load_json
from scripts.utils_data import create_skeleton_gt_data
from scripts.skeleton_graph import SkeletonGraph
import scripts.visualize_examples as ve
from scripts.calculate_metrics import Metrics

## TODO fix assignmetn problem

class skeleton_matching():
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


	# def match_hmm(self):
	# 	from plant_registration_4d import skeleton_matching as skm
	# 	from plant_registration_4d import skeleton as skel

	# 	S1 = skel.Skeleton(XYZ=self.S_gt.get_node_attribute("pos")*100, edges=self.S_gt.get_edges())
	# 	S2 = skel.Skeleton(XYZ=self.S_pred.get_node_attribute("pos")*100, edges=self.S_pred.get_edges())

	# 	params = {'weight_e': 0.01, 'match_ends_to_ends': False,  'use_labels' : False, 'label_penalty' : 1, 'debug': True}
	# 	corres = skm.skeleton_matching(S1, S2, params)

	# 	self.matched_indices = corres

	# 	return corres

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




class Evaluation():
	
	def __init__(self, gt_path_dir=None, 
			  dt_graph_dir=None, 
			  gt_json=None,
			  evaluate_gt: bool=False,
			  filter=None,
			  filter_node_order = np.inf,
			  node_order_eval_list: list=[0,1,2,3],
			  parents_only: bool=True,
			  false_threshold: float=0.02,
			  vis: bool=False,
			  exp_name: str = "",
			  cfg={"json_name": "test"},
			  post_processing: dict | None = None):
		
		self.gt_path_dir = gt_path_dir
		self.dt_path_dir = dt_graph_dir
		self.save_folder = self.dt_path_dir
		self.cfg = cfg
		self.cfg_post_processing = post_processing
		self.gt_json = gt_json

		self.evaluate_gt = evaluate_gt
		# self.filter = filter
		self.filter_node_order = filter_node_order
		self.node_order_eval_list = node_order_eval_list #which nodes to evaluate, if -1 then all
		self.parents_only = parents_only
		self.false_threshold = false_threshold # meters
		self.vis = vis
		self.exp_name = exp_name

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


	def evaluate_pairs(self, graph_pairs=None, vis=False):
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

		for gt_name, pc_name, semantic_name in self.load_all_gt_filenames():

			S_gt = self.load_gt_data(gt_name, pc_name, semantic_name)
			# if pc_name.stem=="Harvest_01_PotNr_95":
			# 	print("DEBUGGING")
			# else:
			# 	continue
			
			if self.evaluate_gt:
				S_pred = copy.deepcopy(S_gt)
				self.pred_name = Path(gt_name).name
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
		metrics = ["TP", "FP", "FN", "Precision", "Recall", "F-score", "CD"]
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

			f_score = 2*precision*recall / (precision + recall)
			
			# Calculate Chamfer Distance (CD)
			cd = df["CD"].mean() if str_node_order == "" else ""
			
			# Append the calculated metrics to the new DataFrame
			df_nodes["node_order" + str_node_order] = [tp, fp, fn, precision, recall, f_score, cd]

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

		f_score_edges = 2*precision_edges*recall_edges / (precision_edges+recall_edges)

		# Create a dictionary with the metrics
		metrics_dict = {
			"TP_edges": df_temp["TP_edges"],
			"FP_edges": df_temp["FP_edges"],
			"FN_edges": df_temp["FN_edges"],
			"Precision_edges": precision_edges,
			"Recall_edges": recall_edges,
			"F-score_edges": f_score_edges
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

		metric_file_node = self.save_folder / f"metrics_node{self.exp_name}.csv"
		metric_file_edge = self.save_folder / f"metrics_edge{self.exp_name}.csv"
		metrics_per_plant = self.save_folder / f"metrics_per_plant{self.exp_name}.csv"
		metric_file_traits = self.save_folder / f"metrics_plant_traits{self.exp_name}.csv"

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

	def determine_tp_fp_fn_edges(self, gt_nodes_matched, dt_nodes_matched, dt_edges, gt_query_edge):
		if gt_query_edge[0] not in gt_nodes_matched or gt_query_edge[1] not in gt_nodes_matched:
			return np.empty(0)

		## get dt edge
		matching_dt_edge = np.array(
			[
				dt_nodes_matched[gt_nodes_matched == gt_query_edge[0]][0],
				dt_nodes_matched[gt_nodes_matched == gt_query_edge[1]][0],
			]
		)
		# remember, undirected graph. if match exist in dt_edges it is a TP
		if (dt_edges==matching_dt_edge).all(1).any() or (dt_edges==matching_dt_edge[..., ::-1]).all(1).any():
			return matching_dt_edge
		return np.empty(0)

	def evaluate_single(self, 
		gt_graph, dt_graph, vis=False
	):
		"""Match xyz nodes based on euclidean distance."""
		print(gt_graph.name)

		gt_nodes = gt_graph.get_node_attribute("pos")
		dt_nodes = dt_graph.get_node_attribute("pos")
		gt_node_order = gt_graph.get_node_attribute("node_order")
		dt_node_order= dt_graph.get_node_attribute("node_order")

		dummy = skeleton_matching(S_gt=gt_graph,
							S_pred=dt_graph,
							method="boogaard",
							threshold=self.false_threshold,
							node_order=gt_graph.get_node_attribute("node_order"))
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
			"F-score": 2*precision*recall / (precision+recall),
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


		################################### Chamfer Distance
		chamfer_distance = 0
		chamfer_distance = self.calculate_chamfer_distance(gt_nodes, dt_nodes)
		node_metrics["CD"] = chamfer_distance

		################################### edge & trait evaluation
		gt_graph.get_angles()
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
		gt_measurements: list[dict] = gt_graph.get_gt_attributes()
		nodes_with_gt = [x["node"] for x in gt_measurements]

		# for every node, if the node has a positive match calculate the error, else counter number of missed
		for gt_edge in gt_edges:
			if gt_edge[0] not in gt_nodes_matched or gt_edge[1] not in gt_nodes_matched:
				# check whether edge has gt measurement, if true update counter_notmatched
				if gt_edge[1] in nodes_with_gt:
					indexi = nodes_with_gt.index(gt_edge[1])
					trait = gt_measurements[indexi]
					for trait_name in self.traits:
						if trait.get(trait_name) is None:
							continue
						if gt_graph.G.nodes[gt_edge[1]].get("leaf_angle_branch_id", None) is None:
							## bug fix. The highest leaves do not have a new internode so therefore those should be skipped
							continue
						trait_metrics[trait_name]["counter_notmatched"] = trait_metrics[trait_name].get("counter_notmatched", 0) + 1
				continue

			## get dt edge
			matching_dt_edge = np.array(
				[
					dt_nodes_matched[gt_nodes_matched == gt_edge[0]][0],
					dt_nodes_matched[gt_nodes_matched == gt_edge[1]][0],
				]
			)
			# remember, undirected graph. if match exist in dt_edges it is a TP
			# if matching_dt_edge in dt_edges or matching_dt_edge[..., ::-1] in dt_edges:
			if (dt_edges==matching_dt_edge).all(1).any() or (dt_edges==matching_dt_edge[..., ::-1]).all(1).any():
				edge_tp += 1

				if gt_edge[1] in nodes_with_gt:
					indexi = nodes_with_gt.index(gt_edge[1])
					trait = gt_measurements[indexi]
					prev_dt_node = matched_indices[matched_indices[:,0]==gt_edge[0], 1][0]
					dt_node = matched_indices[matched_indices[:,0]==gt_edge[1], 1][0]
					dt_traits = dt_graph.G.nodes[dt_node]
					for trait_name in self.traits:
						if trait.get(trait_name) is None:
							continue
						dt_value = dt_traits.get(trait_name.replace("gt_", "")) ## check because upper dt_trait will not have 
						## 
						if trait_name=="gt_ph_angle":
							edge_of_trait0= [int(gt_edge[0]), gt_graph.G.nodes[gt_edge[0]]["phyllotactic_angle_id"]]
							temp_matched_edge0 = self.determine_tp_fp_fn_edges(gt_nodes_matched,
								dt_nodes_matched, dt_edges, gt_query_edge=edge_of_trait0)

							edge_of_trait1= [int(gt_edge[1]), gt_graph.G.nodes[gt_edge[1]]["phyllotactic_angle_id"]]
							temp_matched_edge1 = self.determine_tp_fp_fn_edges(gt_nodes_matched,
								dt_nodes_matched, dt_edges, gt_query_edge=edge_of_trait1)
							
							if temp_matched_edge0.size==0 or temp_matched_edge1.size==0 or dt_value is None:
								trait_metrics[trait_name]["counter_notmatched"] = trait_metrics[trait_name].get("counter_notmatched", 0) + 1
							elif temp_matched_edge0[1]==dt_graph.G.nodes[prev_dt_node].get("phyllotactic_angle_id", None) and temp_matched_edge1[1]==dt_graph.G.nodes[dt_node].get("phyllotactic_angle_id", None):
								trait_metrics[trait_name]["dt"].append(dt_value)
								trait_metrics[trait_name]["gt"].append(trait[trait_name])
								trait_metrics[trait_name]["counter"] = trait_metrics[trait_name].get("counter", 0) + 1
							else:
								trait_metrics[trait_name]["counter_notmatched"] = trait_metrics[trait_name].get("counter_notmatched", 0) + 1
						elif trait_name=="gt_lf_angle":
							if gt_graph.G.nodes[gt_edge[1]].get("leaf_angle_branch_id", None) is None:
								## bug fix. The highest leaves do not have a new internode so therefore those should be skipped
								continue
							# edge_of_trait0= [int(gt_edge[1]), gt_graph.G.nodes[gt_edge[1]]"leaf_angle_branch_id", np.inf)]
							edge_of_trait0= [int(gt_edge[1]), gt_graph.G.nodes[gt_edge[1]]["leaf_angle_branch_id"]]

							temp_matched_edge0 = self.determine_tp_fp_fn_edges(gt_nodes_matched,
								dt_nodes_matched, dt_edges, gt_query_edge=edge_of_trait0)

							edge_of_trait1= [int(gt_edge[1]), gt_graph.G.nodes[gt_edge[1]]["leaf_angle_nextnode_id"]]
							temp_matched_edge1 = self.determine_tp_fp_fn_edges(gt_nodes_matched,
								dt_nodes_matched, dt_edges, gt_query_edge=edge_of_trait1)
							
							if temp_matched_edge0.size==0 or temp_matched_edge1.size==0 or dt_value is None:
								trait_metrics[trait_name]["counter_notmatched"] = trait_metrics[trait_name].get("counter_notmatched", 0) + 1
							# elif temp_matched_edge0[1]==dt_graph.G.nodes[prev_dt_node].get("phyllotactic_angle_id", None) and temp_matched_edge1[1]==dt_graph.G.nodes[dt_node].get("phyllotactic_angle_id", None):
							else:
								trait_metrics[trait_name]["dt"].append(dt_value)
								trait_metrics[trait_name]["gt"].append(trait[trait_name])
								trait_metrics[trait_name]["counter"] = trait_metrics[trait_name].get("counter", 0) + 1
							# else:
							# 	trait_metrics[trait_name]["counter_notmatched"] = trait_metrics[trait_name].get("counter_notmatched", 0) + 1

						elif trait_name=="gt_int_length":
							if dt_value is not None:
								trait_metrics[trait_name]["dt"].append(dt_value)
								trait_metrics[trait_name]["gt"].append(trait[trait_name])
								trait_metrics[trait_name]["counter"] = trait_metrics[trait_name].get("counter", 0) + 1
							else:
								## despite TP it might be possible that node order is incorrect so no match possible.
								trait_metrics[trait_name]["counter_notmatched"] = trait_metrics[trait_name].get("counter_notmatched", 0) + 1
						else:
							NotImplementedError(f"Evaluation of trait name {trait_name} not yet implemented")
			else:
				# check whether edge has gt measurement, if true update counter_notmatched
				if gt_edge[1] in nodes_with_gt:
					indexi = nodes_with_gt.index(gt_edge[1])
					trait = gt_measurements[indexi]
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
			"Precision_edges": float(edge_precision),
			"Recall_edges": float(edge_recall),
			"F-score_edges": float(2*edge_precision*edge_recall / (edge_precision + edge_recall)),
		}

		if vis:
			self.pred_name = Path(self.pred_name)
			save_name = self.pred_name.parent / (self.pred_name.stem + f"{self.exp_name}.png")
			ve.vis_evaluation(gt_graph, dt_graph, matched_indices, save_name=save_name)
			
		return node_metrics, edge_metrics, trait_metrics


	def evaluate_pred(self, pred_name=None, vis=False, evaluate_gt=False):
		"""Find related ground truth and return accuracy"""
		self.load_all_gt_filenames()
		gt_name, pc_name = None, None
		for gt_name_1, pc_name_1, semantic_name_1 in self.load_all_gt_filenames():
			if pc_name_1.stem!=pred_name.stem:
				continue
			gt_name = gt_name_1
			pc_name = pc_name_1
			semantic_name = semantic_name_1

		if gt_name is None or pc_name is None:
			raise FileNotFoundError("In evaluatin line 770 gt or pc name not found")

		S_gt = self.load_gt_data(gt_name, pc_name, semantic_name)

		print("Evaluating", pred_name)

		if evaluate_gt: ## evaluate the ground truth traits
			S_pred = copy.deepcopy(S_gt)
			self.pred_name = Path(gt_name.name + ".csv") 
		else:
			S_pred = self.load_pred_data(pred_name.stem)
			

		node_metric, edge_metric, trait_metrics = self.evaluate_single(S_gt, S_pred, vis=vis)
		self.print_metric(node_metric)
		self.print_metric(edge_metric)
		self.print_trait_metric(trait_metrics)


	def load_all_gt_filenames(self):
		file_names = load_json(self.gt_json)
		return natsorted([(self.gt_json.parent / f["skeleton_file_name"], self.gt_json.parent / f["file_name"], self.gt_json.parent / f["sem_seg_file_name"]) for f in file_names])


	def load_gt_data(self, gt_name, pc_name, semantic_name=None):
		print("Loading", gt_name)
		S_gt = create_skeleton_gt_data(gt_name, pc_name, pc_semantic_path=semantic_name)
		# S_gt.df_pc = pd.read_csv(pc_name)
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
		S_pred.main_post_processing(self.cfg_post_processing)

		S_pred.filter(self.filter_node_order, self.parents_only)
		# S_pred.get_edge_type()
		# S_pred.edge_from_filtered()
		# S_pred.line_fitting_3d()
		# S_pred.filter(self.filter_node_order, self.parents_only)
		self.pred_name = self.dt_path_dir / (plant_id + ".csv")


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
	cfg = config.init_config(cfg_filename="config.yaml")

	## original tomatowur
	# cfg = config.Config("config.yaml")
	# dt_graph_dir = Path("Resources/output_skeleton") / config["skeleton_method"]
	# obj = Evaluation(cfg.pointcloud_dir, dt_graph_dir, cfg=config)
	# obj.evaluate_pred(pred_name="Harvest_01_PotNr_95", vis=True, evaluate_gt=config["evaluation"]["evaluate_gt"])

	obj = Evaluation(cfg.data.pointcloud_dir, cfg.save_folder, **cfg.evaluate)

	# obj.filter_node_order = 0
	# obj.parents_only = False
	# obj.evaluate_pairs_per_nodeorder(vis=False)
	# obj.evaluate_pairs(vis=False, evaluate_gt=config["evaluation"]["evaluate_gt"])
	obj.evaluate_pred(pred_name=Path("Harvest_01_PotNr_95"), vis=True, evaluate_gt=False)
	# obj.evaluate_pred(pred_name="Harvest_01_PotNr_95", vis=True, evaluate_gt=True)


	# S_pred = SkeletonGraph("test_graph.csv")


