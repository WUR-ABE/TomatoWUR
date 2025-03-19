import numpy as np
# import pandas as pd
# from pathlib import Path
# import open3d as o3d
# import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

# from plant_registration_4d import skeleton
# from plant_registration_4d import visualize as vis
# from plant_registration_4d import skeleton_matching as skm


# def get_only_parent_nodes(skeleton_data, max_node_order=2):
# 	"""
# 	This function is a bit complitated, but what it does it that is only gets nodes that have a child,
# 	also known as parent nodes.
# 	Furthermore an additional filtering is done to only select nodes of order x.
# 	It returns a skeleton with updates edges indexes to be able to use the 4d_plant_registration library


# 	"""
# 	parentids = skeleton_data.loc[skeleton_data["edgetype"] == "+", "parentid"].values
# 	vidids = skeleton_data.loc[skeleton_data["edgetype"] == "+", ["vid", "parentid"]].values
# 	vidids= vidids.ravel()

# 	vid_dict = {}
# 	edges_dict = {}
# 	for x in skeleton_data.index[::-1]:
# 		if x == 0 or x == 1:
# 			print("debug started")

# 		root = False
# 		parentid = skeleton_data.iloc[x].parentid
# 		counter = 0
# 		if skeleton_data.edgetype.iloc[x] == "+":  # to correct for itself as well
# 			counter += 1
# 		while not root and x != 0:
# 			# while not root:
# 			temp = skeleton_data[skeleton_data["vid"] == parentid]
# 			if temp.edgetype.iloc[0] == "+":
# 				counter += 1
# 			parentid = temp.parentid.iloc[0]

# 			if (
# 				skeleton_data.iloc[x].parentid in parentids
# 				and skeleton_data.iloc[x].parentid not in edges_dict.keys()
# 				and parentid in parentids
# 			):
# 				edges_dict[skeleton_data.iloc[x].parentid] = parentid

# 			if np.isnan(parentid) or parentid == "":
# 				root = True
# 				vid_dict[skeleton_data.iloc[x].vid] = counter
# 				# vid_dict[x]=counter

# 	if parentids[0] == 0:  # to correct for if first node is a branch:
# 		vid_dict[0] = 0
# 	edges_dict = dict(sorted(edges_dict.items()))

# 	parent_nodes_only = skeleton_data[skeleton_data["vid"].isin(parentids)][["x_skeleton", "y_skeleton", "z_skeleton"]].values
# 	parent_nodes_order = np.array([vid_dict[x] for x in skeleton_data[skeleton_data["vid"].isin(parentids)]["vid"].values])
# 	# parent_nodes_only = skeleton_data[skeleton_data["vid"].isin(vidids)][["x_skeleton", "y_skeleton", "z_skeleton"]].values
# 	# parent_nodes_order = np.array([vid_dict[x] for x in skeleton_data[skeleton_data["vid"].isin(vidids)]["vid"].values])


# 	parent_nodes_only = parent_nodes_only[parent_nodes_order <= max_node_order]
# 	parent_nodes_order = parent_nodes_order[parent_nodes_order <= max_node_order]

# 	# edges =  skeleton_data[skeleton_data['vid'].isin(parentids)][["vid", "parentid"]].values
# 	## renumber edges
# 	remap_dict = {}
# 	remap_dict[list(edges_dict.items())[0][1]] = 0
# 	edges_new = []
# 	counter = 0
# 	for i, (vid, parentid) in enumerate(edges_dict.copy().items()):
# 		if vid_dict[vid] <= max_node_order:
# 			remap_dict[vid] = counter + 1
# 			counter += 1
# 		else:
# 			edges_dict.pop(vid)

# 			print("intersting")
# 	# for i, (vid, parentid) in enumerate(edges_dict.items()):
# 	#     remap_dict[vid]=i+1
# 	for vid_new, parentid_new in edges_dict.items():
# 		edges_new.append(np.array([remap_dict[vid_new], remap_dict[parentid_new]]))

# 	return parent_nodes_only, parent_nodes_order, edges_new


# def convert_segmentation2skeleton(df, clustering="dbscan", visualize=False):
# 	if clustering == "dbscan":
# 		from sklearn.cluster import DBSCAN

# 		# Apply DBSCAN
# 		dbscan = DBSCAN(eps=0.02, min_samples=5, algorithm="auto")
# 		labels = dbscan.fit_predict(df[["x", "y", "z"]].values)
# 	elif clustering == "hdbscan":
# 		import hdbscan

# 		hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
# 		labels = hdbscan_clusterer.fit_predict(df[["x", "y", "z"]].values)

# 	else:
# 		print("clustering method unknown check %s", clustering)

# 	# Color mapping for clusters
# 	max_label = labels.max()
# 	colors = (np.random.rand(max_label, 3) * 255).astype(int)
# 	colors = np.zeros((len(df), 3))
# 	for x in range(max_label + 1):
# 		colors[labels == x] = np.random.rand(3)  # .astype(int)

# 	df["labels"] = labels
# 	# df.groupby("labels").mean(["x", "y", "z"]).to_csv("test_pred.csv", index=False)
# 	S_pred = reconstruct_tree(df.groupby("labels").mean(["x", "y", "z"]).values, df)
# 	if visualize:
# 		# for debugging
# 		# Convert to Open3D Point Cloud
# 		point_cloud = o3d.geometry.PointCloud()
# 		point_cloud.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].values)
# 		point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 		# Visualize the result
# 		o3d.visualization.draw_geometries([point_cloud], window_name="DBSCAN Clustering")
# 	return S_pred


# def evaluate_skeleton(S_gt, S_pred, method="1", visualize=False):
# 	th = 0.02  # cm

# 	## evaluation using linear sum assignment
# 	if method == "1":
# 		cost_matrix = distance_matrix(S_gt.XYZ, S_pred.XYZ)

# 		row_ind, col_ind = linear_sum_assignment(cost_matrix)  # , maximize=th) # improved version of hungarion algorithm

# 		TP = (cost_matrix[row_ind, col_ind] <= th).sum()
# 		FP = (cost_matrix[row_ind, col_ind] >= th).sum()
# 		FN = len(S_gt.XYZ) - TP

# 		## create dataframe for visualisation.
# 		df_result = pd.DataFrame(S_gt.XYZ, columns=["x", "y", "z"])
# 		df_result[["blue", "green", "red"]] = [0, 0, 0]
# 		dummy = np.full(cost_matrix.shape[0], False)
# 		dummy[row_ind[cost_matrix[row_ind, col_ind] <= th]] = True
# 		df_result.loc[dummy, ["blue", "green", "red"]] = [0, 255, 0]  # TP
# 		df_result.loc[~dummy, ["blue", "green", "red"]] = [0, 0, 255]  # FN
# 		##
# 		df_result_pred = pd.DataFrame(S_pred.XYZ, columns=["x", "y", "z"])
# 		df_result_pred[["blue", "green", "red"]] = [0, 0, 0]
# 		dummy = np.full(cost_matrix.shape[1], False)
# 		dummy[col_ind[cost_matrix[row_ind, col_ind] < th]] = True
# 		df_result_pred.loc[dummy, ["blue", "green", "red"]] = [0, 200, 0]  # TP
# 		df_result_pred.loc[~dummy, ["blue", "green", "red"]] = [255, 0, 0]  # FP
# 		df_result = pd.concat([df_result, df_result_pred])
# 		print("x")
# 	else:
# 		## evaluation, might be suboptimal because of for loop
# 		df_result = []
# 		df_pred_copy = S_pred.XYZ.copy()

# 		TP = 0
# 		FP = 0
# 		FN = 0

# 		# TODO improve the fact that it is possible that a point is assigned to a suboptimal points as there can be two nodes within 2cm
# 		# TODO add evaluation of edges
# 		for query_point in S_gt.XYZ:
# 			dist = np.linalg.norm(df_pred_copy - query_point, axis=1)
# 			dist_order = np.argsort(dist)
# 			# [k, idx, _] = pcd_tree.search_radius_vector_3d(query_point, radius)
# 			if dist[dist_order[0]] <= th:
# 				TP += 1

# 				df_result.append(query_point.tolist() + [0, 255, 0])  # BGR
# 				df_result.append(df_pred_copy[dist_order[0]].tolist() + [0, 200, 0])  # BGR
# 				df_pred_copy = df_pred_copy[dist_order[1:]]
# 			else:
# 				FN += 1
# 				df_result.append(query_point.tolist() + [0, 0, 255])  # BGR
# 		for x in df_pred_copy:
# 			df_result.append(x.tolist() + [255, 0, 0])  # BGR

# 		FP = len(df_pred_copy)
# 		# visualisation of errors
# 	print("TP", TP)
# 	print("FP", FP)
# 	print("FN", FN)

# 	if visualize:
# 		# Visualize the result
# 		point_cloud = o3d.geometry.PointCloud()
# 		point_cloud.points = o3d.utility.Vector3dVector(df_result[["x", "y", "z"]].values)
# 		point_cloud.colors = o3d.utility.Vector3dVector(df_result[["blue", "green", "red"]].values)

# 		o3d.visualization.draw_geometries([point_cloud], window_name="TP green, FN Red, FP in blue")
# 		# pd.DataFrame(df_result, columns=["x", "y", "z", "blue", "green", "red"]).to_csv("df_result.csv", index=False)
# 	print("Finished")


# def reconstruct_tree(points, df):
# 	# Step 1: Calculate the pairwise distances between all points
# 	distance_matrix = squareform(pdist(points))

# 	# Step 2: Compute the Minimum Spanning Tree (MST)
# 	mst_matrix = minimum_spanning_tree(distance_matrix)

# 	# Step 3: Convert the MST to a list of edges
# 	mst_edges = np.transpose(mst_matrix.nonzero())
# 	mst_edges =[np.array([i, j]) for i, j in mst_edges]

# 	a = np.array(mst_edges)
		
# 	# created directed graph
# 	start_point = points[:,2].argmin()
# 	processed = []
# 	new_edges = []
# 	indexed = [start_point]
# 	while len(indexed)!=0:
# 		start_point = indexed.pop(0)
# 		if start_point in processed:
# 			continue

# 		connections =list(np.argwhere(a[:,0]==start_point).reshape(-1))
# 		connections += list(np.argwhere(a[:,1]==start_point).reshape(-1))
# 		for x2 in a[connections]:
# 			if x2[0]==start_point:
# 				if x2[1] in processed:
# 					continue
# 				new_edges.append(x2)
# 				indexed.append(x2[1])
# 			else:
# 				if x2[0] in processed:
# 					continue
# 				new_edges.append(x2[::-1])
# 				indexed.append(x2[0])

# 		processed.append(start_point)
# 	mst_edges = np.array(new_edges)

# 	# Print the edges of the MST
# 	print("Edges in the Minimum Spanning Tree:")
# 	for edge in mst_edges:
# 		print(f"Edge between node {edge[0]} and node {edge[1]} with distance {distance_matrix[edge[0], edge[1]]:.2f}")

# 	S = skeleton.Skeleton(points, mst_edges)
# 	return S
# 	fh = plt.figure()
# 	ax = fh.add_subplot(111, projection="3d")

# 	vis.plot_skeleton(ax, S)
# 	vis.plot_pointcloud(ax, df[["x", "y", "z"]].values)
# 	plt.show()
# 	return S

from scripts.skeleton_graph import SkeletonGraph
import json

def load_json(file_name):
	with open(file_name, "r") as f:
		return json.load(f)


# def create_skeleton_gt_data(df_gt):
#     ## TODO fix node order
#     skeleton_data = df_gt.loc[
#         ~df_gt["x_skeleton"].isna(), ["x_skeleton", "y_skeleton", "z_skeleton", "vid", "parentid", "edgetype"]
#     ]
#     parent_nodes_only, parent_node_order, edges = get_only_parent_nodes(skeleton_data)
#     S_gt = skeleton.Skeleton(parent_nodes_only, edges)
#     return S_gt

def fit_line(xyz):
    """Function that receives an Nx3 np array and fits a line using the best vector
    input
    -----
    xyz: numpy (float) Nx3 array with N= xyz coordinates 
    -------
    returns
    vector_up_down: a vector that that desribes the line
    xyz_start:      start punt of vector
    xyz_end:        end punt of vector
    xyz_mean:       mean xyz point of xyz
    """
    # t1=time.time()
    xyz_mean = xyz.mean(axis=0)
    # t2=time.time()
    A= xyz-xyz_mean
    # [U,S,vh] = np.linalg.svd(A.T,full_matrices=False)
    ## Use svd to reduce summarize data into three vectors (vh) select first most important vector
    _,_, vh= np.linalg.svd(A,False)
    try:
        d = vh[0]
        if d[2]>0:
            d = d*-1 ## to make sure start is highest point and end is lowest point
    except IndexError:
        return 'noise' ,[],[],[],[]
    # t3=time.time()
    # d = U[:,0]
    tr = np.dot(d,A.T)
    # t4=time.time()


    tr1 = tr.min()
    tr2 = tr.max()
    # print(np.dot(t1,d))
    xyz_start = xyz_mean+np.dot(tr1,d)
    xyz_end = xyz_mean+np.dot(tr2,d)

    vector_up_down = xyz_end-xyz_start
    # t5=time.time()

    # time_list = [t1,t2,t3,t4,t5]
    # time_string= ['mean','svd','dot1','dot2']
    # for x in range(1,5):
    #     line = time_string[x-1]+' %.2f [ms]'%((time_list[x]-time_list[x-1])*1000)
    #     print(line)


    return 'ok', vector_up_down, xyz_start, xyz_end, xyz_mean

def findBottomCenterRoot(points, semantic, method="center"):
    if method == "center":
        xyz = (points.min(0) + points.max(0))/2
        xyz[2] = points.min(0)[2]
        idx = np.abs(points - xyz).sum(1).argmin(0)
    elif method == "bottom":
        idx = points[:, 2].argmin(0)
    elif method == "line":
        points_main_stem = points[semantic == 2]
        # only get first 10% to deal with skewed plants
        # points_main_stem.sort(key=lambda x: x[2])
        points_main_stem = points_main_stem[np.argsort(points_main_stem[:,2])][:int(0.1*len(points_main_stem))]
        _, _ , xyz_start, xyz_end, _  = fit_line(points_main_stem)
        # get closest point to xyz_start
        idx = np.linalg.norm(points - xyz_end, axis=1).argmin()

        ## for debugging
        # import scripts.visualize_examples as ve
        # ve.vis(pc = points, nodes=points, root_idx=idx)
        # ve.vis(pc = points, nodes=np.array([xyz_start, xyz_end]), edges=np.array([[0, 1]]))
    elif method == "pyransac":
        print("DOES NOT WORK properly, maybe bug fix?")
        # from xu_method import cylinder_fitting
        points_main_stem = points[semantic == 2]
        # only get first 10% to deal with skewed plants
        # points_main_stem.sort(key=lambda x: x[2])
        points_main_stem = points_main_stem[np.argsort(points_main_stem[:,2])][:int(0.1*len(points_main_stem))]
        # w_fit, C_fit, r_fit, fit_err = cylinder_fitting.fit(points_main_stem)
        import pyransac3d as pyrsc
        center, axis, radius, inliers = pyrsc.Cylinder().fit(pts=points_main_stem, thresh=0.05)
        # get min point in the cylinder
        if axis[2] < 0:
            axis = axis*-1
        xyz_end = center - axis* (points_main_stem.max(0) - points_main_stem.min(0))[2]/2
        xyz_start = center + axis* (points_main_stem.max(0) - points_main_stem.min(0))[2]/2
        idx = np.linalg.norm(points - xyz_end, axis=1).argmin()
        import scripts.visualize_examples as ve
        ve.vis(pc = points, nodes=points, root_idx=idx)
        ve.vis(pc = points, nodes=np.array([xyz_start, xyz_end]), edges=np.array([[0, 1]]))

    elif method == "cylinderfit":
        from xu_method import cylinder_fitting
        points_main_stem = points[semantic == 2]
        # only get first 10% to deal with skewed plants
        # points_main_stem.sort(key=lambda x: x[2])
        points_main_stem = points_main_stem[np.argsort(points_main_stem[:,2])][:int(0.1*len(points_main_stem))]
        axis, center, r_fit, fit_err = cylinder_fitting.fit(points_main_stem)
        if axis[2] < 0:
            axis = axis*-1
        xyz_end = center - axis* (points_main_stem.max(0) - points_main_stem.min(0))[2]/2
        xyz_start = center + axis* (points_main_stem.max(0) - points_main_stem.min(0))[2]/2
        idx = np.linalg.norm(points - xyz_end, axis=1).argmin()
        ## for debugging
        # import scripts.visualize_examples as ve
        # ve.vis(pc = points, nodes=points, root_idx=idx)
        # ve.vis(pc = points_main_stem, nodes=np.array([xyz_start, xyz_end]), edges=np.array([[0, 1]]))

    elif method == "circle_fit":
        #! python
        #  == METHOD 2 ==
        from scipy import optimize
        from math import sqrt
        points_main_stem = points[semantic == 2]
        # only get first 10% to deal with skewed plants
        # points_main_stem.sort(key=lambda x: x[2])
        points_main_stem = points_main_stem[np.argsort(points_main_stem[:,2])][:int(0.1*len(points_main_stem))]

        x = points_main_stem[:, 0]
        y = points_main_stem[:, 1]

        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return ((x-xc)**2 + (y-yc)**2)**.5

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = tuple(points_main_stem.mean(0)[:2])
        center_2, ier = optimize.leastsq(f_2, center_estimate)
        xyz_start = np.array([center_2[0], center_2[1], points_main_stem.max(0)[2]])
        xyz_end = np.array([center_2[0], center_2[1], points_main_stem.min(0)[2]])
        import scripts.visualize_examples as ve

        idx = np.linalg.norm(points - xyz_end, axis=1).argmin()
        # ve.vis(pc = points, nodes=points, root_idx=idx)
        # ve.vis(pc = points_main_stem, nodes=np.array([xyz_start, xyz_end]), edges=np.array([[0, 1]]))


    return idx



# if __name__ == "__main__":
# 	import yaml
# 	folder = Path(r"W:\PROJECTS\VisionRoboticsData\GARdata\datasets\tomato_plant_segmentation\20241223_csv_skeletons")
# 	with open("config.yaml", "r") as f:
# 		config = yaml.load(f, Loader=yaml.FullLoader)

	
# 	plant_id = "Harvest_01_PotNr_179"
# 	# plant_id = "Harvest_02_PotNr_237"
# 	# plant_id = "Harvest_03_PotNr_74"
# 	# plant_id = "Harvest_03_PotNr_407"

# 	input_file = folder / "annotations" / plant_id / (plant_id + ".csv")
# 	S_gt = create_skeleton_gt_data(input_file)
# 	S_gt.get_node_order()
# 	S_gt.visualise_graph()

# 	S_gt.get_internode_length()
# 	S_gt.get_angles()
# 	S_gt.apply_post_processing()
	# S_gt.visualise_graph()


# 	S_gt = create_skeleton_gt_data(df_gt)


# 	folder = Path("./Resources/")
# 	input_file = folder / (input_file.stem + ".txt")
# 	df = pd.read_csv(input_file)
# 	df = df.loc[df["class_pred"] == 4, ["x", "y", "z"]]

# 	S_pred = convert_segmentation2skeleton(df, "dbscan")
# #
# 	# evaluate_skeleton(S_gt, S_pred, method="1", visualize=True)
# 	# exit()

# 	# Perform matching
# 	params = {'weight_e':0.01, 'match_ends_to_ends': False,  'use_labels' : False, 'label_penalty' : 1, 'debug': False}
# 	corres = skm.skeleton_matching(S_pred, S_gt, params)
# 	# print("Estimated correspondences: \n", corres)

# 	# visualize results
# 	fh = plt.figure()
# 	ax = fh.add_subplot(111, projection="3d")
# 	vis.plot_skeleton(ax, S_gt, "b", label="GT")
# 	vis.plot_skeleton(ax, S_pred, "r", label="Pred")
# 	vis.plot_skeleton_correspondences(ax, S_gt, S_pred, corres)
# 	# vis.plot_skeleton_correspondences(ax, S_pred, S_gt, corres)

# 	# plt.title("Estimated correspondences between skeletons")
# 	plt.legend()
# 	plt.show()
