
import polyscope as ps
import numpy as np

pc_radius = 0.00025
node_radius = 0.006
skeleton_radius = 0.001
vector_length = 0.02
vector_radius = 0.0005





ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_navigation_style("turntable")

def pred2colors(pred):
    color_mapping = { 
        "Leaf": {
            "rgb_encoding": [
                255,
                50,
                50
            ],
            "class_id": 0
        },
        "Main stem": {
            "rgb_encoding": [
                255,
                225,
                50
            ],
            "class_id": 1
        },
        "Pole": {
            "rgb_encoding": [
                109,
                255,
                50
            ],
            "class_id": 2
        },
        "Side Stem": {
            "rgb_encoding": [
                50,
                167,
                255
            ],
            "class_id": 3
        }
    }
    colors = np.zeros((pred.shape[0], 3))
    for key in color_mapping.keys():
        colors[pred==color_mapping[key]['class_id'], :] = np.array(color_mapping[key]['rgb_encoding'])

    return colors


def add_node_order(ps_cloud, node_order):
    if node_order is not None:
        ps_cloud.add_scalar_quantity("node_order", node_order, enabled=True)

def add_attributes(ps_cloud, attributes):
    if attributes is not None:
        for key, values in attributes.items():
            if isinstance(values[0], str):
                continue
                # ps_cloud3.add_categorical_quantity(key, values, enabled=False)
            else:
                ps_cloud.add_scalar_quantity(key, values, enabled=False)

def add_nodes(name="nodes", nodes=[], root_idx=None, parents=None):
    if nodes is not None:
        ps_cloud3 = ps.register_point_cloud(name, np.asarray(nodes), radius=node_radius) # (N, 3)
        node_colors = np.zeros(nodes.shape)
        node_colors[:, :] = [0.5, 0.5, 0.5] 
        if root_idx is not None:
            node_colors[root_idx, :] = [1, 0, 0]
        if parents is not None:
            node_colors[parents, :] = [0, 1, 0]
        ps_cloud3.add_color_quantity(name, node_colors, enabled=True)
        return ps_cloud3
    else:
        return None

def add_edges(name="Skeleton", nodes=None, edges=None, edges_type=None):
    if edges is not None:
        ps_cloud4 = ps.register_curve_network(name, nodes, edges, radius=skeleton_radius)
        if edges_type is not None:
            colors = np.zeros((edges.shape[0], 3))
            colors[edges_type=="+", :] = np.array([1, 0, 0])
            ps_cloud4.add_color_quantity(name, colors, defined_on='edges', enabled=True)

def add_pc(name="point cloud", pc=None, colors=None, distances=None, normals=None):
    if pc is not None:
        ps_cloud = ps.register_point_cloud(name,np.asarray(pc), radius=pc_radius) # (N, 3)
        if colors is not None:
            ps_cloud.add_color_quantity("rand colors", np.asarray(colors)/255, enabled=True)
        if distances is not None:
            ps_cloud.add_scalar_quantity("distances", np.asarray(distances), cmap="jet", enabled=True)
        if normals is not None:
            ps_cloud.add_vector_quantity("normal", np.asarray(normals), enabled=True, length=vector_length, radius=vector_radius)
        return ps_cloud
    else:
        return None

def vis(pc=None, colors=None, nodes=None, node_order=None, edges=None, edges_type=None, root_idx=None, parents=None, distances=None, normals=None, attributes=None):

    ps.init()
    ## add raw point cloud
    add_pc("point cloud", pc, colors, distances, normals)

    # add nodes
    ps_cloud_nodes = add_nodes("nodes", nodes, root_idx, parents)
    add_node_order(ps_cloud_nodes, node_order)
    add_attributes(ps_cloud_nodes, attributes)

    # add edges
    add_edges("Skeleton", nodes, edges, edges_type)

    ps.show()
    # ps.screenshot("frame_filename.png")


def vis_two_nodes(pc=None, nodes=None, nodes_2=None):
    ps.init()
    if pc is not None:
        ps_cloud = ps.register_point_cloud("point cloud",np.asarray(pc), radius=pc_radius) # (N, 3)
        # ps_cloud.add_color_quantity("rand colors", np.asarray(pc.colors)/255, enabled=True)
    if nodes is not None:
        ps_cloud3 = ps.register_point_cloud("nodes",np.asarray(nodes), radius=node_radius) # (N, 3)
        node_colors = np.zeros(nodes.shape)
        node_colors[:, :] = [0.5, 0.5, 0.5] 
        ps_cloud3.add_color_quantity("nodes", node_colors, enabled=True)
    if nodes_2 is not None:
        ps_cloud_nodes_2= ps.register_point_cloud("nodes_2",np.asarray(nodes_2), radius=node_radius) # (N, 3)
        node_colors = np.zeros(nodes.shape)
        node_colors[:, :] = [1, 0.5, 0.5] 
        ps_cloud_nodes_2.add_color_quantity("nodes_2", node_colors, enabled=True)
    ps.show()

def vis_multiple_pc(list_pc=[]):
    ps.init()
    for i, pc in enumerate(list_pc):
        ps_cloud = ps.register_point_cloud("point cloud_"+str(i),np.asarray(pc), radius=pc_radius) # (N, 3)

    ps.show()

def generate_unique_colors(x):
    import matplotlib.pyplot as plt
    import random
    # Generate x evenly spaced values between 0 and 1
    colors = plt.cm.get_cmap('hsv', x)  # Use the 'hsv' colormap (or another colormap of your choice)
    colors = [colors(i) for i in range(x)]
    random.shuffle(colors)
    return colors

def vis_components(pc=None, components=None):
    ps.init()
    if pc is not None and components is not None:
        ps_cloud = ps.register_point_cloud("point cloud",np.asarray(pc), radius=pc_radius) # (N, 3)
        # ps_cloud.add_color_quantity("rand colors", np.asarray(pc.colors)/255, enabled=True)
        unique_colours = generate_unique_colors(len(components))
        component_colors = np.zeros(pc.shape)
        scalar_values = np.zeros(pc.shape[0])
        for i, x in enumerate(components):
            component_colors[x] = unique_colours[i][:3]
            scalar_values[x] = i
        ps_cloud.add_color_quantity("colors", component_colors, enabled=True)
        ps_cloud.add_scalar_quantity("group_idx", scalar_values, enabled=False)

    ps.show()


def vis_distance(pc=None, distances=None):
    ps.init()
    if pc is not None and distances is not None:
        ps_cloud = ps.register_point_cloud("point cloud",np.asarray(pc), radius=pc_radius) # (N, 3)
        ps_cloud.add_scalar_quantity("geodesic_distances", distances, cmap="jet", enabled=True)

    ps.show()


def vis_mtg(mtg):
    from xu_method.mtgmanip import mtg2pgltree
    ps.init()
    nodes, parents, vertex2node = mtg2pgltree(mtg)
    ps.register_curve_network("Skeleton", np.asarray(nodes), parents, radius=skeleton_radius)
    ps.show()

def vis_evaluation(S_gt, S_pred, matching_edges):
    ps.init()
    ps.set_up_dir("z_up")
    ps.remove_all_structures()

    # Set the colors for each point cloud
    gt_tp_color = [0, 1, 0]  # Green
    pred_tp_color = [0, 0.5, 0]
    fp_color = [1, 0, 0]  # Red
    fn_color = [220/255, 45/255, 0]  # orange

    S_gt_nodes = S_gt.get_node_attribute()
    S_pred_nodes = S_pred.get_node_attribute()

    ############ add GT
    ps.register_point_cloud("point_cloud", S_gt.get_xyz_pointcloud(), radius=pc_radius)

    # ps_gt = ps.register_point_cloud("GT", S_gt_nodes, radius=node_radius)
    ps_gt = add_nodes("nodes_gt", S_gt_nodes)
    color_gt = np.zeros(S_gt_nodes.shape)
    color_gt[:, :] = fn_color
    if matching_edges.size > 0:
        color_gt[matching_edges[:, 0], :] = gt_tp_color
    ps_gt.add_color_quantity("colors", color_gt, enabled=True)

    add_edges("Skeleton", S_gt_nodes, S_gt.get_edges(), S_gt.get_edge_attribute("edge_type"))
    add_attributes(ps_gt, S_gt.get_attributes())

    ############ add PRED
    ps_pred = add_nodes("nodes_pred", S_pred_nodes)
    color_pred = np.zeros(S_pred_nodes.shape)
    color_pred[:, :] = fp_color
    if matching_edges.size > 0:
        color_pred[matching_edges[:, 1], :] = pred_tp_color
    ps_pred.add_color_quantity("colors", color_pred, enabled=True)


    add_edges("Skeleton_pred", S_pred_nodes,S_pred.get_edges(), S_pred.get_edge_attribute("edge_type"))
    add_attributes(ps_pred, S_pred.get_attributes())

    show_matched_edges = False
    if show_matched_edges:
        for matching_edge in matching_edges:
            match_nodes = np.array(
                [S_gt_nodes[matching_edge[0]], S_pred_nodes[matching_edge[1]]]
            )

            if matching_edge[0] == 6 and matching_edge[1] == 4:
                print("interesting")
            ps.register_curve_network(
                f"match{matching_edge[0]},{matching_edge[1]}",
                match_nodes,
                np.array([[0, 1]]),
                radius=0.005,
                color=pred_tp_color,
            )
    ps.show()


def vis_correspondence():
    from pathlib import Path
    import pandas as pd
    import json

    folder = Path(r"W:\PROJECTS\VisionRoboticsData\ExxactRobotics\tomato_plant_johan_series8\Reconstruction_aligned_pc") 
    points_list = []
    not_cleaned_points = []
    nodes_list = []
    edges_list = []
    for file_name in folder.glob("*cleaned_skel.csv"):
        pd_skel = pd.read_csv(file_name)
        nodes = pd_skel[["x", "y", "z"]].values
        edges = pd_skel[["vid", "parent_id"]].dropna().astype(int).values
        # from plant_registration_4d import skeleton as pr4d_skel
        # S_pred = skel.Skeleton(nodes, edges)
        # S_pred_list.append(S_pred)
        nodes_list.append(nodes)
        edges_list.append(edges)

        points_list.append(pd.read_csv(file_name.parent / (file_name.name.replace("_skel",""))))
        not_cleaned_points.append(pd.read_csv(file_name.parent / (file_name.name.replace("_cleaned_skel",""))))

    not_cleaned_points = not_cleaned_points[1:]
    points_list = points_list[1:]
    nodes_list = nodes_list[1:]
    edges_list = edges_list[1:]


    i=0
    corres = np.load("corres.npy")
    print("x")
    iterations = 20
    for ii in range(iterations):
        # create basis ps 
        ps.init()
        if ii<-100:
            ps_start = ps.register_point_cloud("START", not_cleaned_points[0][["//X", "Y", "Z"]], radius=pc_radius) # (N, 3)
            ps_start.add_color_quantity("colors", not_cleaned_points[0][["R", "G", "B"]]/255, enabled=True)
            
            ################3 segmented visualisation
            # not_cleaned_points[0]["pred"] = 0
            # points_list[0]["pred"] = 3
            # merged = pd.merge(not_cleaned_points[0], points_list[0], on=["//X", "Y", "Z"], how="outer")
            # merged["pred"] = 0
            # merged.loc[merged["pred_y"]==3, "pred"] = 3
            # colors = pred2colors(merged["pred"].values)
            # ps_cloud = ps.register_point_cloud("segmented", merged[["//X", "Y", "Z"]], radius=pc_radius) # (N, 3)
            # ps_cloud.add_color_quantity("colors", colors/255, enabled=True)

        elif ii<-100:
        # elif ii==iterations-1:
            ps_end = ps.register_point_cloud("END", not_cleaned_points[1][["//X", "Y", "Z"]], radius=pc_radius)
            ps_end.add_color_quantity("colors", not_cleaned_points[1][["R", "G", "B"]]/255, enabled=True)
            
            ################3 segmented visualisation
            # not_cleaned_points[1]["pred"] = 0
            # points_list[1]["pred"] = 3
            # merged = pd.merge(not_cleaned_points[1], points_list[1], on=["//X", "Y", "Z"], how="outer")
            # merged["pred"] = 0
            # merged.loc[merged["pred_y"]==3, "pred"] = 3
            # colors = pred2colors(merged["pred"].values)
            # ps_cloud = ps.register_point_cloud("segmented", merged[["//X", "Y", "Z"]], radius=pc_radius) # (N, 3)
            # ps_cloud.add_color_quantity("colors", colors/255, enabled=True)

        else:
            # if ii<iterations/2:
            # ps_start = ps.register_point_cloud("START", not_cleaned_points[0][["//X", "Y", "Z"]], radius=0.0005) # (N, 3)
            # ps_start.add_color_quantity("colors", not_cleaned_points[0][["R", "G", "B"]]/255, enabled=True)
            # else:
                # ps_end = ps.register_point_cloud("END", not_cleaned_points[1][["//X", "Y", "Z"]], radius=0.0005)
                # ps_end.add_color_quantity("colors", not_cleaned_points[1][["R", "G", "B"]]/255, enabled=True)

            for i, x in enumerate(zip(nodes_list, edges_list)):
                nodes = x[0]
                edges = x[1]
                # if i==1:
                #     continue
                ps_cloud = ps.register_point_cloud("point cloud"+str(i), points_list[i][["//X", "Y", "Z"]], 
                                                   radius=pc_radius,
                                                   enabled=True) # (N, 3)
                ps_cloud.set_color(np.array([50, 167, 255])/255)  # Set a single color for all points (e.g., red)

                ps_nodes = ps.register_point_cloud("nodes"+str(i), nodes, 
                                                   radius=0.007,
                                                   enabled=True) # (N, 3)
                ps_nodes.set_color(np.array([0, 0, 255])/255)  # Set a single color for all points (e.g., red)

                ps_skeleton = ps.register_curve_network("Skeleton"+str(i), nodes, edges, radius=0.003)
                ps_skeleton.set_color(np.array([15, 255, 0])/255)  # Set a single color for all points (e.g., red)

            ############33 tracking visualisation
            corres_nodes = np.vstack([nodes_list[0][corres[:, 0]], nodes_list[1][corres[:, 1]]])
            size_0 = int(corres_nodes.shape[0]/2)
            corres_edges = np.array([list(range(size_0)), list(range(size_0,corres_nodes.shape[0]))]).T

            ps_corres = ps.register_curve_network("Corres", corres_nodes, corres_edges, radius=skeleton_radius)
            ps_corres.set_color(np.array([227, 25, 50])/255)  # Set a single color for all points (e.g., red)

            
            start = nodes_list[0][corres[:, 0]]
            end = nodes_list[1][corres[:, 1]]

            intermediate = start + (end-start)*ii/iterations
            ps_inter = ps.register_point_cloud("intermediate_pc", intermediate, radius=0.006) # (N, 3)
            ps_inter.set_color(np.array([255, 100, 0])/255)  # Set a single color for all points (e.g., red)

            # ps.show()


        view_name = f"screenshots/view_{ii}.json"
        with open(view_name, "r") as f:
            b = json.load(f)
            newviewMat = np.array(b["viewMat"]).reshape((4,4))

        intrinsics = ps.CameraIntrinsics(fov_vertical_deg=45., aspect=2.)
        extrinsics = ps.CameraExtrinsics(mat=newviewMat)
        params = ps.CameraParameters(intrinsics, extrinsics)
        ps.set_view_camera_parameters(params)

        ## to save camera views
        # my_dict = json.loads(ps.get_view_as_json())
        # with open(f"screenshots/view_{ii}.json", "w") as f:
        #     json.dump(my_dict, f)

        # ps.show()
        save_dir = Path("screenshots") / "pointclouds_6" / "tracking"
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        ps.screenshot(str(save_dir / f"frame_filename_{ii}.png"), transparent_bg=False)
        ps.remove_all_structures()
        print(ii)


if __name__=="__main__":
    vis_correspondence()