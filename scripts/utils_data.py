# utils_data.py
import pandas as pd
from scripts import skeleton_graph

def create_skeleton_gt_data(skeleton_path, pc_path=None, pc_semantic_path=None):
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
		attributes_gt = None
	
	S_gt = skeleton_graph.SkeletonGraph()
	S_gt.load(nodes, edges, edge_types, df_pc=df_pointcloud, attributes=attributes_gt)
	S_gt.name = skeleton_path.stem.replace("_skeleton", "")
	S_gt.get_node_order()

	return S_gt
