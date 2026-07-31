from pathlib import Path
import numpy as np
import pandas as pd
import open3d as o3d
from scipy import ndimage
from skimage.morphology import skeletonize #, thin, medial_axis
import networkx as nx

from scripts import utils_skeletonisation
from scripts import visualize_examples as ve


def fill_pc_using_dilation(points: np.ndarray, voxel_size_stage_1: int, num_dilations_erosions: int = 4, **kwargs):
	"""
	Fills a voxel grid based on input point cloud data using Open3D, applies morphological dilation and erosion, and returns the filled voxel coordinates.

	Args:
		points (np.ndarray): Nx3 array of 3D point coordinates representing the point cloud.
		voxel_size_stage_1 (float): The size of each voxel in the grid.
		num_dilations_erosions (int, optional): Number of dilation and erosion operations to apply. Default is 4.
		**kwargs: Additional keyword arguments (currently unused).

	Returns:
		filled_pc (np.ndarray): filled point cloud.
		voxel_grid.origin (np.ndarray): The origin of the voxel grid.
		voxel_grid.voxel_size (float): The size of each voxel in the grid.

	Notes:
		- Uses Open3D for voxelization and SciPy ndimage for morphological operations.
		- The function first dilates and then erodes the binary voxel grid to fill gaps in the point cloud.
	"""

	num_dilations = int(num_dilations_erosions)
	num_erosions = int(num_dilations_erosions)

	## filling up point cloud
	custom_fill = True
	if custom_fill:
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(points)
		voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=voxel_size_stage_1)
		voxel_points = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])

		voxel_min = np.floor(np.min(voxel_points, axis=0)).astype(int)
		voxel_max = np.ceil(np.max(voxel_points, axis=0)).astype(int)
		grid_shape = (voxel_max - voxel_min + 1)
		# grid_shape = (voxel_max - voxel_min)

		# Initialize a 3D binary grid
		binary_grid = np.zeros(grid_shape, dtype=bool)
		indices = (voxel_points - voxel_min).astype(int)  # Shift points to the grid's minimum corner
		binary_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True


		skel_temp = np.zeros(binary_grid.shape, dtype=bool)
		# size = skel_temp.size
		kern_size = 3
		element = np.ones((kern_size,kern_size,kern_size)).astype(binary_grid.dtype)
		# done = False
		for j in range(num_dilations):
			binary_grid = ndimage.binary_dilation(binary_grid, element)
		for j in range(num_erosions):
			binary_grid = ndimage.binary_erosion(binary_grid, element)
		# print(binary_grid.sum())
			# eroded = ndimage.binary_erosion(binary_grid, element)
			# temp = binary_grid - temp
		new_pc = np.argwhere(binary_grid)
		# new_pc = np.argwhere(skel_temp)
		filled_pc = np.floor((new_pc + voxel_min)*voxel_grid.voxel_size + voxel_grid.origin + [1,1,1])

		# save_name="/home/agro/w-drive-vision/GARdata/datasets/tomato_plant_segmentation/TomatoWUR_4dataTU/EXPERIMENTS_PAPER3/0-paper-2Dto3D/voxel/step2.png"
		# ve.vis(filled_pc, save_name)      

		return filled_pc, voxel_grid.origin, voxel_grid.voxel_size
	

def create_binary_grid(new_pc, voxel_size = 4):
	"""
	Converts a point cloud into a binary 3D grid using voxelization.

	Args:
		new_pc (np.ndarray): Input point cloud as an (N, 3) array of 3D coordinates.
		voxel_size (float, optional): Size of each voxel. Defaults to 4.

	Returns:
		tuple: 
			- binary_grid (np.ndarray): 3D boolean array representing occupied voxels.
			- binary_grid_copy (np.ndarray): Copy of the binary grid.
			- voxel_points (np.ndarray): Array of voxel grid indices for occupied voxels.
			- voxel_grid.origin (np.ndarray): Origin of the voxel grid.
			- voxel_min (np.ndarray): Minimum voxel grid index (bounding box corner).
	"""
	pc = o3d.geometry.PointCloud()
	pc.points = o3d.utility.Vector3dVector(new_pc)

	# Create voxel grid from the mesh
	# voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
	voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=voxel_size)

	# Assuming voxel_grid is your generated voxel grid
	voxel_centers = [voxel.grid_index for voxel in voxel_grid.get_voxels()]
	voxel_points = np.array(voxel_centers)# * voxel_grid.voxel_size  # Scale to real-world coordinates

	# Step 1: Convert voxel points to a binary 3D grid array
	# Determine the bounding box of the voxel grid
	voxel_min = np.floor(np.min(voxel_points, axis=0)).astype(int)
	voxel_max = np.ceil(np.max(voxel_points, axis=0)).astype(int)
	grid_shape = (voxel_max - voxel_min + 1)
	# grid_shape = (voxel_max - voxel_min)

	# Initialize a 3D binary grid
	binary_grid = np.zeros(grid_shape, dtype=bool)

	# Fill in the binary grid with voxel centers
	indices = (voxel_points - voxel_min).astype(int)  # Shift points to the grid's minimum corner
	binary_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
	binary_grid_copy = binary_grid.copy()

	## TODO visualize new_pc and voxel_points
	# np.savetxt("!binary_grid.csv", (np.argwhere(binary_grid) + voxel_min) * voxel_size + [voxel_size,voxel_size,voxel_size], delimiter=",")
	# np.savetxt("!new_pc.csv", new_pc, delimiter=",")
	# np.savetxt("!voxel_points.csv", voxel_points*voxel_grid.voxel_size + [4,4,4], delimiter=",")

	return binary_grid, binary_grid_copy, voxel_points, voxel_grid.origin, voxel_min


def find_connected_points(skeleton: np.ndarray):
	"""
	Find points connected to a given point in a 3x3x3 neighborhood in a binary 3D grid.

	Args:
		skeleton (np.ndarray): A 3D binary grid (1 for point, 0 for empty space).

	Returns:
		np.ndarray: Array of edges, where each edge is a pair of indices into the skeleton points array.
	"""

	points = np.argwhere(skeleton)
	edges = []

	temp = np.array([
		[0, 1, 0],
		[1, 0, 0],
		[0, -1, 0],
		[-1, 0, 0],
		[0,0,1],
		[0,0,-1],
		[1,1,0],
		[1,0,1],
		[0,1,1],
		[1,-1,0],
		[1,0,-1],
		[0,1,-1],
		[-1,1,0],
		[-1,0,1],
		[0,-1,1],
		[1,1,1],
		[1,1,-1],
		[1,-1,1],
		[1,-1,-1],
		[-1,1,1],
		[-1,1,-1],
		[-1,-1,1],
		[-1,-1,-1]], dtype=np.int8)
	
	max_points = points.max(0)
	for i, point in enumerate(points): 
		# Define the range for the neighborhood (3x3x3)
		for item in temp:
			new_pos = point + item
			if np.any(new_pos > max_points): ## to prevent error
				continue
			# Check if neighbor is within bounds and is a '1'
			if skeleton[new_pos[0], new_pos[1], new_pos[2]]:
				indexi = np.all((np.argwhere(skeleton)==new_pos),1).argmax()
				edges.append([i, indexi])
	edges = np.array(edges)
	edges = np.unique([sorted(e) for e in edges], axis=0)
	## copyp from connected compontens geodesic skeleton
	# ve.vis(nodes=points, edges=edges)
	return edges


def main(points_filtered, root_idx=None, voxel_size_stage_1=2, 
		 voxel_size_stage_2=4, return_pc="default", nodes2edges=None, **kwargs):
	
	"""
	Fills and skeletonizes a 3D point cloud using voxel-based dilation and thinning.
	Args:
		points_filtered (np.ndarray): Filtered point cloud coordinates.
		root_idx (int, optional): Index of the root node in the filtered points. Defaults to 0.
		voxel_size_stage_1 (int, optional): Voxel size for the initial dilation stage. Defaults to 2.
		voxel_size_stage_2 (int, optional): Voxel size for the skeletonization stage. Defaults to 4.
		return_pc (str, optional): Output format option. Defaults to "default".
		**kwargs: Additional arguments for internal processing functions.
	Returns:
		tuple: (new_nodes, new_edges, new_root_idx)
			new_nodes (np.ndarray): Skeleton node coordinates.
			new_edges (np.ndarray): Directed edges between skeleton nodes.
			new_root_idx (int): Index of the root node in the skeleton.
	"""
	
	points = (points_filtered*1000).astype(int) ## convert to mm space
	filled_pc, voxel_origin, _ = fill_pc_using_dilation(points, voxel_size_stage_1=voxel_size_stage_1,
												**kwargs)
	
	binary_grid, binary_grid_copy, voxel_points, voxel_origin, voxel_min = create_binary_grid(filled_pc, voxel_size_stage_2)

	# np.sort(np.argwhere(binary_grid))
	# ve.vis(points_filtered, nodes=points/1000)
	# ve.vis(points, nodes=(np.argwhere(binary_grid) + voxel_min) * voxel_size_stage_1) #+ [voxel_size,voxel_size,voxel_size])

	# skeleton = skeletonize_3d(binary_grid)
	skeleton = skeletonize(binary_grid)
	edges = find_connected_points(skeleton)

	nodes = np.floor((np.argwhere(skeleton) + voxel_min)*voxel_size_stage_2 + voxel_origin + [1,1,1])
	nodes = nodes / 1000
	if nodes2edges is not None:
		return nodes, _, root_idx

	raise NotImplementedError
	## TODO fix circular loops and root_idx

	# edge_nodes_idx = np.linalg.norm(nodes - points_filtered[root_idx], axis=1).argmin()

	## if voxel size is small it is possible that we will have unconnected components
	G = nx.Graph()
	for e in edges:
		G.add_edge(e[0], e[1])
	components = list(nx.connected_components(G))
	largest_component = max(components, key=len)
	G=G.subgraph(largest_component).copy()
	G.remove_edges_from(list(nx.selfloop_edges(G)))
	edges = np.array(G.edges())
	edge_nodes_idx = np.linalg.norm(nodes[edges[:,0]] - points_filtered[root_idx], axis=1).argmin()
	
	## convert to directed graph
	temp_edges = utils_skeletonisation.undirected2directed(edges, root=edge_nodes_idx)
	# temp_edges = np.array(list(nx.bfs_edges(G, source=edge_nodes_idx)))


	## not all nodes are connected. Those unconnected nodes need to be remove. And edges need to be updated
	new_idx = np.sort(np.unique(temp_edges.flatten()))
	new_edges = temp_edges.copy()
	for i, idx in enumerate(new_idx):
		new_edges[new_edges==idx] = i
	new_nodes = nodes[new_idx]
	new_root_idx = np.where(new_idx==edge_nodes_idx)[0][0]
	
	## visualisation material and methods
	# save_name="/home/agro/w-drive-vision/GARdata/datasets/tomato_plant_segmentation/TomatoWUR_4dataTU/EXPERIMENTS_PAPER3/0-paper-2Dto3D/voxel/step3.png"
	# ve.vis(points_filtered, nodes=new_nodes, edges=new_edges, root_idx=new_root_idx, save_name=save_name)

	return new_nodes, new_edges, new_root_idx


if __name__ == "__main__":
	
	############ tomato dataset
	folder = Path(r"W:\PROJECTS\VisionRoboticsData\GARdata\datasets\tomato_plant_segmentation\20240607_summerschool_csv\annotations")
	plant_name = "Harvest_03_PotNr_407"
	file_name = folder / plant_name / (plant_name + ".csv")

	# # name = "Dense_point_cloud_8.csv"
	# # file_name = Path(r"W:\PROJECTS\VisionRoboticsData\ExxactRobotics\tomato_plant_johan_series8\Reconstruction_aligned_pc") / name
	df = pd.read_csv(str(file_name), delimiter = ",", low_memory=False)
	bool_array = np.bitwise_or(df["semantic"]==1 ,df["semantic"]==3)
	# df = df.loc[~bool_array]

	points_df = df.loc[~bool_array, ["x", "y", "z"]].values
	main(df[["x", "y", "z"]].values, points_df)
	# file_name = r"W:\PROJECTS\VisionRoboticsData\GARdata\datasets\tomato_plant_segmentation\20240402_colourmesh_gt_v2\Harvest_01_PotNr_55_mesh.ply"
	# pc = o3d.io.read_triangle_mesh(file_name)

	################## babette dataset
	# name = "Dense_point_cloud_1_cleaned.csv"
	# file_name = Path(r"W:\PROJECTS\VisionRoboticsData\ExxactRobotics\tomato_plant_johan_series8\Reconstruction_aligned_pc") / name
	# df = pd.read_csv(str(file_name), delimiter = ",")
	# bool_array = np.bitwise_or(df["pred"]==1 ,df["pred"]==3) # leaf, main stem, pole, side stem
	# df = df.loc[bool_array]

	# points = df[["//X", "Y", "Z"]].values * 10

############################################### OLD code

# def fill_voxel_using_mesh(file_name):
# 	import pymeshlab
# 	ms = pymeshlab.MeshSet()
# 	ms.load_new_mesh(file_name)  # Replace with your file path

# 	# Estimate normals
# 	# You can specify options like number of neighbors for estimation
# 	ms.apply_filter('compute_normal_for_point_clouds', k=50)  # k is the number of nearest neighbors

# 	# Optional: Reorient normals to make them consistent (useful for smooth surfaces)
# 	# ms.apply_filter('invert_faces_orientation')

# 	# Save the point cloud with normals to a new file
# 	# ms.save_current_mesh("point_cloud_with_normals.ply")

# 	# Print the first few normals (for verification)
# 	normals = ms.current_mesh().vertex_normal_matrix()
# 	points = ms.current_mesh().vertex_matrix()

# 	a = pd.DataFrame(points_df, columns=["x", "y", "z"])
# 	b = pd.DataFrame(np.hstack([points, normals]), columns=["x", "y", "z", "nx", "ny", "nz"])
# 	c = pd.merge(a, b, on=["x", "y", "z"], how="inner")

# 	points = c[["x", "y", "z"]].values
# 	normals = c[["nx", "ny", "nz"]].values


######### fill voxels using normals
# ve.vis_multiple_pc([points, np.argwhere(binary_grid)+voxel_min])
# exit()
# ve.vis_multiple_pc([points, new_points, new_points2,  new_points3])
# ve.vis_multiple_pc([points, new_points, new_points2,  new_points3, new_points4])
# factor = 100
# factor = 0.05
# new_points = points - normals * 0.05/factor
# new_points2 = points - normals * 0.1/factor
# new_points3 = points - normals * 0.15/factor
# new_pc = np.vstack([points, new_points, new_points2, new_points3])



# def custom_made_skeletonisation(binary_grid):
# 	custom_method = False
# 	if custom_method:
# 		from scipy import ndimage
# 		skel_temp = np.zeros(binary_grid.shape, dtype=bool)
# 		size = skel_temp.size
# 		kern_size = 3
# 		element = np.ones((kern_size,kern_size,kern_size)).astype(binary_grid.dtype)
# 		done = False
# 		while( not done):
# 			eroded = ndimage.binary_erosion(binary_grid, element)
# 			temp = ndimage.binary_dilation(eroded, element)
# 			# temp = binary_grid - temp
# 			temp = np.logical_xor(binary_grid, temp)
# 			skel_temp = np.bitwise_or(skel_temp, temp)
# 			binary_grid = eroded.copy()
		
# 			zeros = size - binary_grid.sum()
# 			if zeros==size:
# 				done = True

# 		skeleton = skel_temp



# def skeleton_to_nodes(skeleton, voxel_points, voxel_min, ):
# 	kern_size = 3

# 	# print(voxel_points.shape, skeleton.sum())
# 	# skeleton = thin(binary_grid)
# 	# print(voxel_points.shape, skeleton.shape)
# 	# skeleton = medial_axis(binary_grid)


# 	# Step 3: Convert the skeletonized binary grid back to voxel coordinates
# 	# skeleton_points = np.argwhere(skeleton) + voxel_min  # Shift back to original coordinates

# 	# Step 4: Convert skeleton points to point cloud and save
# 	# skeleton_pcd = o3d.geometry.PointCloud()
# 	# skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_points * voxel_grid.voxel_size)
# 	# o3d.io.write_point_cloud("skeletonized_voxel_grid.ply", skeleton_pcd)

# 	# Perform convolution to count the number of 1's in each 3x3x3 neighborhood
# 	dummy  = np.zeros(skeleton.shape)
# 	indices = np.argwhere(skeleton)
# 	# ve.vis(indices, nodes=voxel_points)


# 	dummy[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
# 	kernel =  np.ones((kern_size,kern_size,kern_size)).astype(int)
# 	kernel[1,1,1] = 10
# 	conv_result = ndimage.convolve(dummy, kernel, mode='constant', cval=0)

# 	# Apply the threshold condition
# 	# Set points to zero where the neighborhood sum is less than 4
# 	filtered_grid = np.where(conv_result>=13, skeleton, 0)

# 	filtered_grid_correct = np.logical_or(filtered_grid, np.where(conv_result==2, skeleton, 0))
# 	# filtered_grid = np.logical_and(filtered_grid, skeleton)

# 	## due to the convolutions, multiple points at almost similar location are created.
# 	## we need to remove thos points. by finding unique clusters based on label
# 	large_clustters = np.argwhere(filtered_grid)
# 	from scipy.ndimage import label
# 	dummy2 = np.zeros(skeleton.shape)
# 	dummy2[large_clustters[:, 0], large_clustters[:, 1], large_clustters[:, 2]] = 1
# 	temp, num_features = label(dummy2, structure=np.ones((kern_size,kern_size,kern_size)).astype(int))
# 	new_clusters = [np.argwhere(temp==i)[0] for i in range(1, num_features+1)]

# 	new_clusters = np.array(new_clusters)
# 	## add ending points
# 	# new_clusters = np.vstack([new_clusters, np.argwhere(np.where(conv_result==2, skeleton, 0))])
# 	new_clusters = np.vstack([new_clusters, np.argwhere(np.where(conv_result==11, skeleton, 0))])

# 	return new_clusters


# Function to detect branch points and endpoints
# def detect_points(skeleton):
#     """Detect branch points and endpoints."""
#     from scipy.ndimage import convolve

#     # Define a convolution kernel to count neighbors
#     kernel = np.array([[1, 1, 1],
#                        [1, 10, 1],
#                        [1, 1, 1]])
	
#     neighbors = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
	
#     # Branch points: More than 3 neighbors
#     branch_points = (neighbors > 12) & skeleton
	
#     # Endpoints: Exactly 1 neighbor
#     endpoints = (neighbors == 11) & skeleton
	
#     return branch_points, endpoints


