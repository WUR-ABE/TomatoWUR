# utils_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import json
import open3d as o3d
import cv2
import shutil

from . import skeleton_graph

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


def nerfstudio_writer(nerfstudio_dict: dict, rgb_images_path: list, seg_images_path: list, xyz: np.ndarray, add_masks: bool=False, output_nerf: Path=Path("nerf")):
	"""
	Writes data for Nerfstudio format, including transforms, sparse point cloud, and images/masks.
	Args:
		nerfstudio_dict (dict): Dictionary containing Nerfstudio related items such camera intrinsic and extrinsics. 
		rgb_images_path (list): List of file paths to RGB images.
		seg_images_path (list): List of file paths to segmentation images.
		xyz (np.ndarray): Numpy array of 3D points for the sparse point cloud.
		add_masks (bool, optional): Whether to include segmentation masks. Defaults to False.
		output_nerf (Path, optional): Output directory for Nerfstudio data. Defaults to "nerf".
	Returns:
		None
	"""
	
	
	if not output_nerf.exists():
		output_nerf.mkdir()
	sparse_output_ply = output_nerf / "sparse_pointcloud.ply"
	
	## add file_path_to dict
	for i, x in enumerate(nerfstudio_dict["frames"]):
		x["file_path"] =  f"images/frame_{str(i+1).zfill(5)}.png"
		if add_masks:
			x["mask_path"] = f"masks/{str(i+1).zfill(5)}.png"

	nerfstudio_dict["ply_file_path"] = str(sparse_output_ply.name)

	with open(output_nerf / "transforms.json", "w") as f:
		json.dump(nerfstudio_dict, f, indent=4)

	# Create an Open3D point cloud object
	point_cloud = o3d.geometry.PointCloud()
	# Assign points to the point cloud and save as ply
	point_cloud.points = o3d.utility.Vector3dVector(xyz)
	o3d.io.write_point_cloud(str(sparse_output_ply), point_cloud)
	print(f"Sparse point cloud saved to {sparse_output_ply}")
	

	img_dirs = ["images", "images_2", "images_4", "images_8"]
	mask_dirs = [x.replace("images", "masks") for x in img_dirs]
	for (img_dir, mask_dir) in zip(img_dirs, mask_dirs):
		nerf_path = output_nerf / img_dir
		if not nerf_path.exists():
			nerf_path.mkdir()

		if add_masks:
			nerf_path = output_nerf / mask_dir
			if not nerf_path.exists():
				nerf_path.mkdir()


	for i, (img_path, img_seg_path) in enumerate(zip(rgb_images_path, seg_images_path)):
		output_rgb = output_nerf / img_dirs[0] / f"frame_{i+1:05d}.png"
		output_mask = output_nerf / img_dirs[0].replace("images", "masks") / f"{i+1:05d}.png"

		shutil.copy(src = str(img_path) , dst=str(output_rgb))
		if add_masks:
			img_seg = cv2.imread(str(img_seg_path), -1)
			img_seg[img_seg>0]=255
			cv2.imwrite(str(output_mask), img_seg)
			# shutil.copy(src = str(img_seg_path) , dst=str(output_mask))
		continue

		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_seg = cv2.imread(img_seg_path, -1)
		img_seg[img_seg>0]=255

		# Ensure the segmentation layer is a single channel
		if len(img_seg.shape) > 2:
			img_seg = img_seg[:, :, 0]

		# # Stack the image and segmentation layer
		# combined_array = np.dstack((img, img_seg))

		# # Save the combined image as a PNG
		current_image = Image.fromarray(img, mode="RGB")
		# combined_image.save(output_nerf / img_dirs[0] / f"frame_{i+1:05d}.png")
		# # Save resized versions
		# current_image = combined_image
		for scale in [2, 4, 8]:
			new_size = (current_image.width // 2, current_image.height // 2)
			current_image = current_image.resize(new_size, Image.LANCZOS)
			current_image.save(output_nerf / f"images_{scale}" / f"frame_{i+1:05d}.png")
