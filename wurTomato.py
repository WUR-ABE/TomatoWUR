################################################################
# Author     : Bart van Marrewijk                                  #
# Contact    : bart.vanmarrewijk@wur.nl                         #
#            : andy@wired-wrong.co.uk                          #
# Date       : 30-05-2024                                      #
# Description: Written for use as part of the 2024 CDT summer  #
#            : school. Theme 6 - Phenotyping and beyond colour #
################################################################

# Usage: Run python from command line
#      : from LastSTRAW import LastStrawData
#      : help(LastStrawData)

import os
import argparse
import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
import json

from torch.utils.data import Dataset
from tqdm import tqdm
import requests
from zipfile import ZipFile
from pathlib import Path
import pandas as pd
# import natsort
import polyscope as ps
# import yaml

from scripts.utils_data import create_skeleton_gt_data
from scripts.utils_skeletonisation import findBottomCenterRoot#convert_segmentation2skeleton, evaluate_skeleton
from scripts import skeleton_graph
from scripts import visualize_examples as ve
from scripts import evaluate_semantic_segmentation
from scripts.evaluate_skeletons import Evaluation
from scripts import config

from skeletonisation_methods.plantscan3d import xu

semantic_id2rgb_colour = {
    1: [255, 50, 50],
    2: [255, 225, 50],
    3: [109, 255, 50],
    4: [50, 167, 255],
    5: [167, 50, 255],
}
# Find the maximum semantic ID to determine the size of the array
max_id = max(semantic_id2rgb_colour.keys())
# Create an array where index corresponds to the semantic ID
rgb_array = np.zeros((max_id + 1, 3), dtype=np.uint8)
# Populate the array with the RGB values
for key, value in semantic_id2rgb_colour.items():
    rgb_array[key] = value

from omegaconf import dictconfig


class WurTomatoData(Dataset):
    """
    LastStrawData inherits from Pytorch dataset class.
    LastStrawData(path: str | None [, argument=value (optional)])

        Arguments    : Values / types
        -----------------------------
        down_sample  : float | 0 (default: 0)
        url          : str
        folder       : str
        download_file: str
        check_folder : str

    Description:
    The LastStrawData class imports the LastSTRAW data either from
    a URL or from a given path (folder). It can also visualise the
    point cloud data using open3D. If a path is given it should
    point directly to numpy xyz files. A list of these files is
    generated in the order governed by the operating system call
    (import os) os.listdir(path).

    The file format is:
        X Y Z R G B class instance (each field separated by space)
        X,Y and Z: signed float
        R,G and B: unsigned int8
        class    : int64
        instance : int64
        comments : lines starting with // (such as a header line)

    This class is based upon, but extensively modified, a code
    example given by LastSTRAW at https://lcas.github.io/LAST-Straw/

    Author     : Andy Perrett
    Contact    : aperrett@lincoln.ac.uk
               : andy@wired-wrong.co.uk
    Date       : 30-05-2024
    Description: Written for use as part of the 2024 CDT summer
               : school. Theme 6 - Phenotyping and beyond colour

    Example usage:

    from LastSTRAW import LastStrawData

    URL = \"https://lcas.lincoln.ac.uk/nextcloud/index.php/s/omQY9ciP3Wr43GH/download\"
    FOLDER = \"/tmp/\"
    CHECK_FOLDER = \"LAST-Straw/\"
    DOWNLOAD_FILE = \"download.zip\"
    VOXEL_SIZE = 0
    DATA_DIR = None
    #DATA_DIR = \'/home/andy/Documents/CDT summer school/LAST-Straw/LAST-Straw/\'

    def main():

        lastStraw = LastStrawData(data_dir=DATA_DIR,
                                    down_sample=VOXEL_SIZE,
                                    url = URL,
                                    folder=FOLDER,
                                    check_folder = CHECK_FOLDER,
                                    download_file=DOWNLOAD_FILE)

        pc, rgb, labels = lastStraw[0]

        lastStraw.visualise(0)

        # Load each scan
        # for pc, rgb, _ in lastStraw:
        #     pointC = o3d.geometry.PointCloud()
        #     pointC.points = o3d.utility.Vector3dVector(pc)
        #     pointC.colors = o3d.utility.Vector3dVector(rgb)
        #     lastStraw.visualise(pointC)

    if __name__ == \'__main__\':
        main()
    """

    def __init__(self, **kwargs):
        config_data = config.Config("config.yaml")

        # self._set_attributes(config_data)
        self.__dict__.update(config_data.__dict__)

        # If the data folder can not be found then ask to download the data
        if not (self.project_dir / self.project_code).exists():
            user_input = input(f"Data not found {self.project_dir}. Do you want to download the data? (y/n): ").strip().lower()
            if user_input == 'y':
                self.__download()
                self.__unzip()
            else:
                raise FileNotFoundError("Data not found and download not initiated.")

        ## open annotation file
        with open(self.data.json_path, "r") as f:
            self.dataset = json.load(f)
        for x in self.dataset:
            for key, value in x.items():
                x[key] = self.data.json_path.parent / value
                if not x[key].is_file():
                    print("warning file is missing")

        self.S_gt = None
        print("Successfully loaded the WURTomato dataset")

    # # Download LastSTRAW data file in zip format
    def __download(self):
        """
        If the unzipped files exist do not download. If they do not
        exist then download the zip file
        """
        print(self.project_dir)
        if not (self.project / self.checkFolder).is_dir():
            # if not (self.folder / self.downloadFile).is_file():
            # print("Downloading: " + self.downloadFile + " to folder: " + self.folder + " from: " + self.url)
            print("This may take a while (TomatoWUR is 4.8GB)...")
            response = requests.get(self.url, stream=True)
            if response.status_code == 200:
                # Open a local file with write-binary ('wb') mode
                with open(self.folder / self.downloadFile, "wb") as file:
                    for chunk in tqdm(response.iter_content(chunk_size=8192)):  # chunk size speeds up progress
                        if chunk:  # Filter out keep-alive new chunks
                            file.write(chunk)
                    print("File downloaded successfully.")
            else:
                print(f"Failed to download file. Status code: {response.status_code}")
        else:
            print("File already download and extracted.")

    # Taken from https://www.geeksforgeeks.org/unzipping-files-in-python/
    def __unzip(self):
        """
        If data zip file has been download, extract all files
        and delete downloaded zip file
        """
        if (self.project_dir / self.downloadFile).is_file():
            if not (self.project_dir).is_dir():
                print(f"Extracting: {self.project_dir / self.downloadFile}")
                with ZipFile(str(self.project_dir / self.downloadFile), "r") as zObject:
                    zObject.extractall(path=str(self.folder))
                print(f"Deleting {self.project_dir / self.downloadFile}")
                os.remove(str(self.project_dir / self.downloadFile))

    def __load_graph(self, index):
        if self.S_gt is None or self.S_gt.name != self.dataset[index]["file_name"].stem:
            self.S_gt = create_skeleton_gt_data(self.dataset[index]["skeleton_file_name"], pc_path=self.dataset[index]["file_name"], pc_semantic_path=self.dataset[index]["sem_seg_file_name"])
        return self.S_gt

    # Loads xyz of point cloud
    def __load_xyz_array(self, index):
        # Loads the data from an .xyz file into a numpy array.
        self.__load_graph(index)
        return self.S_gt.get_xyz_pointcloud()

    def __load_xyz_semantic_array(self, index):
        self.__load_graph(index)
        return self.S_gt.get_semantic_pointcloud()

    def get_filtered_data(self, index):
        self.__load_graph(index)
        pcd = self.S_gt.get_xyz_pointcloud()
        semantic = self.S_gt.get_semantic_pointcloud()
        bool_array = np.bitwise_or(semantic==1 ,semantic==3) # 1=leaves, 2=main stem, 3=pole, 4=side stem

        return pcd[~bool_array, :], semantic[~bool_array]

    # Loads point cloud from file in Numpy array. Returns point cloud
    # def __load_as_o3d_cloud(self, index):
    #     # Loads the data from an .xyz file into an open3d point cloud object.
    #     data, labels_available = self.__load_as_array(index)
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = o3d.utility.Vector3dVector(data[["x", "y", "z"]].values)
    #     pc.colors = o3d.utility.Vector3dVector(data[["red", "green", "blue"]].values)
    #     labels = None
    #     skeleton_data = None
    #     if labels_available:
    #         labels = data[["semantic", "leaf_stem_instances", "leaf_instances", "stem_instances"]].values
    #         labels = data[["semantic_with_nodes", "leaf_stem_instances", "leaf_instances", "stem_instances"]].values

    #         # if load_skeleton_data:
    #         skeleton_data = data.loc[
    #             ~data["x_skeleton"].isna(), ["x_skeleton", "y_skeleton", "z_skeleton", "vid", "parentid", "edgetype"]
    #         ]
    #     return pc, labels_available, labels, skeleton_data

    # Saves the point cloud data - TODO NOTE untested
    # def save_data_as_xyz(self, data, fileName):
    #     # To save your own data in the same format as we used, you can use this function.
    #     # Edit as needed with more or fewer columns.
    #     with open(self.path + fileName, 'w') as f:
    #         f.write("//X Y Z R G B class instance\n")
    #         np.savetxt(f, data, fmt=['%.6f', '%.6f', '%.6f','%d', '%d', '%d'])
    #     return

    # Return number of data files
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        self.scan_index = 0
        return self
    
    def __next__(self):
        if self.scan_index < len(self):
            # pointCloud, labels_available, labels, skeleton_data = self.__load_as_o3d_cloud(self.scan_index)
            data = self.__load_graph(self.scan_index)
            self.scan_index += 1
            return data
    
        else:
            raise StopIteration

    def __getitem__(self, index):
        return self.__load_graph(index)

    def visualise(self, index=0):
        self.__load_graph(index)
        print(f'Visualising {self.dataset[index]["file_name"].stem}')
        ve.vis(pc = self.S_gt.get_xyz_pointcloud(), colors=self.S_gt.get_colours_pointcloud())

    def visualise_semantic(self, index, semantic_name= "semantic"):
        self.__load_graph(index)
        print(f'Visualising semantic {self.dataset[index]["file_name"].stem}')
        labels = self.S_gt.get_semantic_pointcloud(semantic_name=semantic_name)
        colours = rgb_array[labels.astype(int)].copy()
        ve.vis(pc = self.S_gt.get_xyz_pointcloud(), colors=colours)

        ## visualising semantics with nodes
        # labels = self.S_gt.get_semantic_pointcloud(semantic_name="semantic_with_nodes")
        # colours = rgb_array[labels.astype(int)].copy()
        # ve.vis(pc = self.S_gt.get_xyz_pointcloud(), colors=colours)

    def visualise_skeleton(self, index, parent_nodes_only=True):
        print(f'Visualising skeleton {self.dataset[index]["file_name"].stem}')
        self.__load_graph(index)
        self.S_gt.visualise_graph()


    # def visualise_inference(self, txt_name="./Resources/predictions/Harvest_01_PotNr_179.txt"):
    #     if isinstance(txt_name, str):
    #         txt_name = Path(txt_name)

    #     # Create skeleton object
    #     df_pred = pd.read_csv(txt_name)
    #     print("changed classes!!!")
    #     df_pred2 = df_pred.loc[df_pred["class_pred"] == 4, ["x", "y", "z"]]
    #     S_pred = convert_segmentation2skeleton(df_pred2, "dbscan", visualize=True)

    #     ## Load GT skeleton object
    #     try:
    #         indexi = list(self.scans_dict.keys()).index(txt_name.stem)
    #     except ValueError:
    #         print(f"{txt_name.stem} not in dataset, exitting")
    #         exit()

    #     pc, labels_available, labels, skeleton_data = self.__load_as_o3d_cloud(indexi)
    #     S_gt = create_skeleton_gt_data(skeleton_data)

    #     colors = rgb_array[labels[:, 0].astype(int)].copy()/255
    #     import polyscope as ps
    #     ps.init()
    #     ps_cloud = ps.register_point_cloud("coloured_pc",np.asarray(pc.points), radius=0.002) # (N, 3)
    #     ps_cloud.add_color_quantity("rand colors", np.asarray(pc.colors)/255, enabled=True)

    #     ps_cloud2= ps.register_point_cloud("semantic_pc",np.asarray(pc.points), radius=0.002) # (N, 3)
    #     ps_cloud2.add_color_quantity("rand colors", colors, enabled=True)


    #     colors_pred = rgb_array[(df_pred["class_pred"].values+1).astype(int)].copy()/255
    #     ps_cloud3 = ps.register_point_cloud("pred_semantic",np.asarray(pc.points), radius=0.002) # (N, 3)
    #     ps_cloud3.add_color_quantity("rand colors", colors_pred, enabled=True)

    #     # ps_cloud4 = ps.register_point_cloud("GT skeleton", S_gt.XYZ, radius=0.005) # (N, 3)
    #     ps.register_curve_network("GT Skeleton", S_gt.XYZ, np.array(S_gt.edges), radius=0.01)

    #     ps.show()

    #     evaluate_skeleton(S_gt, S_pred, method="1", visualize=True)
    #     # exit()

    #     # Perform matching
    #     # params = {'weight_e':0.01, 'match_ends_to_ends': False,  'use_labels' : False, 'label_penalty' : 1, 'debug': False}
    #     # corres = skm.skeleton_matching(S_pred, S_gt, params)
    #     # print("Estimated correspondences: \n", corres)

    #     # visualize results
    #     fh = plt.figure()
    #     ax = fh.add_subplot(111, projection="3d")
    #     vis.plot_skeleton(ax, S_gt, "b", label="GT")
    #     vis.plot_skeleton(ax, S_pred, "r", label="Pred")
    #     # vis.plot_skeleton_correspondences(ax, S_gt, S_pred, corres)
    #     # vis.plot_skeleton_correspondences(ax, S_pred, S_gt, corres)

    #     # plt.title("Estimated correspondences between skeletons")
    #     plt.legend()
    #     plt.show()

    def run_semantic_evaluation(self):
        dt_graph_dir = Path("./Resources/output_semantic_segmentation")
        obj = evaluate_semantic_segmentation.EvaluationSemantic(dt_graph_dir=dt_graph_dir, gt_json=self.data.json_path)
        obj.evaluate_pairs()


    def run_skeleton_evaluation(self):
        # folder = Path(self.cfg["folder"])
        dt_graph_dir = Path("Resources/output_skeleton") / self.cfg["skeleton_method"]
        obj = Evaluation(self.data.pointcloud_dir, dt_graph_dir, gt_json=self.data.json_path)
        obj.evaluate_pairs(vis=False, evaluate_gt=self.cfg["evaluation"]["evaluate_gt"])


    def nodes2edges(self, points, nodes, method="geodesic", **kwargs):
        if method == "geodesic":
            raise NotImplementedError
            visualize_examples.vis(points, nodes=nodes)
            from geoskel.geodesic_skeleton import GeodesicSkeleton
            gs = GeodesicSkeleton(points, nodes)
            gs.compute_geodesics()
            gs.construct_graph()
            # visualize_examples.vis(points, nodes=gs.nodes, edges=np.array(gs.edge_list))

            return gs.nodes, np.array(gs.edge_list), None
        elif method == "xu":

            nodes, edges, edge_type = xu.xu_method_connect_points(nodes, kwargs["parents"], kwargs["mtg"])
            return nodes, edges, edge_type
        elif method == "mst":
            import mistree as mist
            import networkx as nx
            mst = mist.GetMST(x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2])
            degree, edge_length, branch_length, branch_shape, edge_index, branch_index = mst.get_stats(
                include_index=True, k_neighbours=15)
            # Convert to Graph
            
            return nodes, edge_index.T, None
            
            print("come on Bart implement this")
        else:
            raise NotImplementedError
    
        
    def run_semantic_segmentation(self):
        semseg_url = "https://github.com/WUR-ABE/2D-to-3D_segmentation"
        print(f"Not implemented, please have look at following git: f{semseg_url}")


    def run_skeletonisation(self, method = "xu", visualise=False):

        save_folder = Path("Resources/output_skeleton")

        for i in tqdm(range(len(self))):
            print(f'Running skeletonisation on {self.dataset[i]["file_name"]}')
            # if self.dataset[i]["file_name"].stem!="Harvest_01_PotNr_293":
            #     continue
            pcd = self.__load_xyz_array(i)
            semantic = self.__load_xyz_semantic_array(i)
            pcd_filtered, semantic_filtered = self.get_filtered_data(i)

            root_idx = findBottomCenterRoot(pcd_filtered, semantic_filtered, method=self.cfg["root_method"])

            if self.cfg["skeleton_method"]=="xu":
                binaratio = self.cfg["xu"]["binratio"]
                n_neighbors = self.cfg["xu"]["n_neighbors"]

                positions, parents, mtg = xu.xu_method(pcd_filtered, root_idx=root_idx, binratio=binaratio, nearest_neighbour=n_neighbors, vis=False)
                nodes, edges, edge_type = self.nodes2edges(pcd_filtered, positions, method=self.cfg["xu"]["nodes2edges"], parents=parents, mtg=mtg)
                save_name = save_folder / "xu" / (self.dataset[i]["file_name"].stem+".csv")


                # exit()

            elif self.cfg["skeleton_method"]=="som":
                from skeletonisation_methods.som import som

                nodes = som.som_method(pcd_filtered, root_idx=root_idx, cfg=self.cfg["som"])
                nodes, edges, edge_type = self.nodes2edges(pcd_filtered, nodes, method=self.cfg["som"]["nodes2edges"])
                save_name = save_folder / "som" / (self.dataset[i]["file_name"].stem+".csv")


            elif self.cfg["skeleton_method"]=="voxel":
                import skeletonisation_methods.voxel.fill_voxel as fill_voxel
                nodes, edges, root_idx = fill_voxel.main(pcd, pcd_filtered,  root_idx=root_idx, voxel_size = self.cfg["voxel"]["voxel_size"],
                                                  return_pc=self.cfg["voxel"]["nodes2edges_input"])
                edge_type = None
                S_pred = skeleton_graph.SkeletonGraph()
                S_pred.load(nodes, edges, edge_types=edge_type, df=pd.DataFrame(pcd, columns=["x", "y", "z"]), name=self.dataset[i]["file_name"].stem)
                S_pred.get_node_order()
                S_pred.gaussian_smoothing(var0=0.3, var1=0.3, indices=[0,1,2], node_order_filtering=False)
                S_pred.gaussian_smoothing(var0=0.25, var1=0.25, indices=[0,1], node_order_filtering=False)

                S_pred.get_edge_type(angle_between_trunk_and_lateral=70)
                edges = S_pred.get_edges()
                nodes = S_pred.get_node_attribute()
                # S_pred.visualise_graph()
                # nodes, edges, edge_type = self.nodes2edges(input_pc, nodes, method=self.cfg["voxel"]["nodes2edges"])
                save_name = save_folder / "voxel" / (self.dataset[i]["file_name"].stem+".csv")

            elif self.cfg["skeleton_method"]=="laplacian":
                from skeletonisation_methods.pc_skeletor.pc_skeletor import LBC
                lbc = LBC(pcd_filtered, **self.cfg["laplacian"]["settings_lbc"])
                lbc.extract_skeleton()
                nodes = np.asarray(lbc.contracted_point_cloud.voxel_down_sample(0.002).points)
                edge_nodes_idx = np.linalg.norm(nodes - pcd_filtered[root_idx], axis=1).argmin()
                nodes = np.vstack([nodes[edge_nodes_idx], np.delete(nodes, edge_nodes_idx, axis=0)])
                nodes, edges, edge_type = self.nodes2edges(pcd_filtered, nodes, method=self.cfg["laplacian"]["nodes2edges"])
                import scripts.visualize_examples as ve
                from geoskel.geodesic_skeleton import undirected2directed
                edges = undirected2directed(edges)
                save_name = save_folder / "laplacian" / (self.dataset[i]["file_name"].stem+".csv")

                # ve.vis(pcd_filtered, nodes=nodes, edges=edges ,root_idx=0)

            S_pred = skeleton_graph.SkeletonGraph()
            S_pred.load(nodes, edges, edge_types=edge_type, df_pc=pd.DataFrame(pcd, columns=["x", "y", "z"]), name=self.dataset[i]["file_name"].stem)
            S_pred.get_node_order()
            if visualise:
                S_pred.visualise_graph()
            S_pred.export_as_nodelist(save_name)

            # exit()


        # visualize_examples.vis(points[~bool_array, :])
        # points = xyz[["x", "y", "z"]].values
        # bool_array = np.zeros(len(points), dtype=bool)
        # points = xyz[["x", "y", "z"]].values

        # name = "Dense_point_cloud_7_cleaned.csv"
        # file_name = Path(r"W:\PROJECTS\VisionRoboticsData\ExxactRobotics\tomato_plant_johan_series8\Reconstruction_aligned_pc") / name
        # xyz = pd.read_csv(str(file_name), delimiter = ",")

        # points = xyz[["//X", "Y", "Z"]].values

        # colors = visualize_examples.pred2colors(xyz["pred"].values) 
        # bool_array = xyz["pred"]==0

        # bool_array = points[:,0]==np.inf


        

            # S_gt = create_skeleton_gt_data(skeleton_data

    def run_debugging(self, i):
        # pc, labels_available, labels, skeleton_data = self.__load_as_o3d_cloud(i)
        # points = np.asarray(pc.points)

        plant_nr = "Harvest_02_PotNr_27"
        file_name = Path(r"W:\PROJECTS\VisionRoboticsData\GARdata\datasets\tomato_plant_segmentation\20240607_summerschool_csv\annotations") / plant_nr / (plant_nr+".csv")
        df = pd.read_csv(file_name, delimiter = ",", low_memory=False)
        points = df[["x", "y", "z"]].values
        labels = df[["semantic"]].values
        bool_array = np.bitwise_or(labels[:,0]==1 ,labels[:,0]==3)
        # visualize_examples.vis(points[~bool_array, :])
        # points = xyz[["x", "y", "z"]].values
        # bool_array = np.zeros(len(points), dtype=bool)
        # points = xyz[["x", "y", "z"]].values

        # name = "Dense_point_cloud_7_cleaned.csv"
        # file_name = Path(r"W:\PROJECTS\VisionRoboticsData\ExxactRobotics\tomato_plant_johan_series8\Reconstruction_aligned_pc") / name
        # xyz = pd.read_csv(str(file_name), delimiter = ",")

        # points = xyz[["//X", "Y", "Z"]].values

        # colors = visualize_examples.pred2colors(xyz["pred"].values) 
        # bool_array = xyz["pred"]==0

        # bool_array = points[:,0]==np.inf


        nodes, root_idx, edges, edge_types, mtg = xu.xu_method(points[~bool_array, :], binratio=15, vis=True)

        S_pred = skeleton_graph.SkeletonGraph(nodes, edges, edge_types, pcd=points[~bool_array], name=file_name.stem)
        # S_pred.get_edge_type()
        S_pred.get_node_order()
        S_pred.filter(2, True)
        S_pred.save_graph("test_graph.csv")

        from scripts.evaluate_skeletons import Evaluation
        obj = Evaluation()
        obj.evaluate_pred(S_pred, file_name.stem, vis=True)
        exit()

        from xu_method.mtgmanip import nodelist2mtg
        mtg = nodelist2mtg(nodes=nodes, edges=edges, edge_types=edge_types)
        from xu_method import io
        print(mtg.is_valid())
        print(mtg)

        for id in mtg.vertices():
            print(mtg[id])

        mtg_lines = io.write_mtg(mtg, properties = [(p, 'REAL') for p in mtg.property_names() if p not in ['edge_type', 'index', 'label', 'position']])
        f = open('test.mtg', 'w')
        f.write(mtg_lines)
        f.close()

        from xu_method.mtgmanip import mtg2_nodes_edges_edge_types
        nodes, edges, edge_types = mtg2_nodes_edges_edge_types(mtg)
        visualize_examples.vis(points, nodes=nodes, edges=edges, edges_type=edge_types, root_idx=mtg.root)


    
        # io.write_mtg(mtg, str(file_name.parent / (file_name.stem+"_mtg.txt")))

        # visualize_examples.vis(points[~bool_array, :], nodes)
        df = pd.DataFrame(nodes, columns=["x", "y", "z"])
        df["vid"] = np.nan
        df["parent_id"] = np.nan
        df["vid"][:len(edges[:,0])] = edges[:, 1]
        df["parent_id"][:len(edges[:,0])] = edges[:, 0]
        df.to_csv(str(file_name.parent / (file_name.stem+"_skel.csv")), index=False)

        # S_pred = pr4d_skel.Skeleton(nodes, edges)
        # S_gt = skeleton.Skeleton(parent_nodes_only, edges)


        
        ################## optimise using sthochastic 
        stochastic_optim = False
        if stochastic_optim:
            from skeleton_refinement.stochastic_registration import perform_registration
            # refined_skel = perform_registration(points[~bool_array, :], nodes, max_iterations=100, tolerance=0.001/1000, alpha=2/1000, beta=2/1000)
            # visualize_examples.vis(points[~bool_array, :], refined_skel)

            temp_points = points[~bool_array, :]*1000
            tep_skel = nodes*1000
            refined_skel = perform_registration(temp_points, tep_skel, max_iterations=50, tolerance=0.001)
            temp_points = temp_points/1000
            refined_skel = refined_skel/1000
            # visualize_examples.vis(None, refined_skel)
            # visualize_examples.vis_two_nodes(temp_points, nodes, refined_skel)



        print("x")

    def run_geodesic(self, i):
        from geoskel.geodesic_skeleton import GeodesicSkeleton

        ## running example of geodesic solution, credits to Kyle Fogarty <ktf25@cam.ac.uk>
        pc, labels_available, labels, skeleton_data = self.__load_as_o3d_cloud(i)

        gs = GeodesicSkeleton(np.asarray(pc.points), skeleton_data[["x_skeleton", "y_skeleton", "z_skeleton"]].values)
        gs.compute_geodesics()
        gs.construct_graph()

        ps.init()
        ps.register_point_cloud('Graph Nodes', gs.nodes)
        ps.register_point_cloud("Point Cloud", gs.plc)
        ps.register_curve_network("Skeleton", gs.nodes, np.array(gs.edge_list))
        ps.show()

        # predicted_graph = Graph(
        #     nodes=gs.nodes, edges=np.array(gs.edge_list), filename=dt_graph.filename
        # )
        # with open(
        #     (output_dir / predicted_graph.filename).with_suffix(".pkl"), "wb"
        # ) as f:
        #     pickle.dump(predicted_graph, f)
        # print("x")

    # def run_registration(self):
    #     from plant_registration_4d import visualize as vis

    #     folder = Path(r"W:\PROJECTS\VisionRoboticsData\ExxactRobotics\tomato_plant_johan_series8\Reconstruction_aligned_pc") 
    #     S_pred_list = []
    #     for file_name in folder.glob("*skel.csv"):
    #         skel = pd.read_csv(file_name)
    #         nodes = skel[["x", "y", "z"]].values
    #         edges = skel[["vid", "parent_id"]].dropna().astype(int).values
    #         from plant_registration_4d import skeleton as pr4d_skel
    #         S_pred = pr4d_skel.Skeleton(nodes, edges)
    #         S_pred_list.append(S_pred)
    #     S1 = S_pred_list[0]
    #     S2 = S_pred_list[1]


    #     # set matching params
    #     match_params = {'weight_e': 0.01,
    #                     'match_ends_to_ends': True,
    #                     'use_labels' : False,
    #                     'label_penalty' : 1,
    #                     'debug': False}

    #     # set registration params
    #     reg_params = {'num_iter': 20,
    #                 'w_rot' : 100,
    #                 'w_reg' : 100,
    #                 'w_corresp' : 1,
    #                 'w_fix' : 1,
    #                 'fix_idx' : [],
    #                 'R_fix' : [np.eye(3)],
    #                 't_fix' : [np.zeros((3,1))],
    #                 'use_robust_kernel' : True,
    #                 'robust_kernel_type' : 'cauchy',
    #                 'robust_kernel_param' : 2,
    #                 'debug' : False}

    #     # iterative procedure params
    #     params = {'num_iter' : 5,
    #             'visualize': True,
    #             'match_params': match_params,
    #             'reg_params': reg_params}

    #     # call register function
    #     from plant_registration_4d.iterative_registration import iterative_registration
    #     from plant_registration_4d import non_rigid_registration as nrr


    #     T12, corres = iterative_registration(S1, S2, params)

    #     # % Apply registration params to skeleton
    #     S2_hat = nrr.apply_registration_params_to_skeleton(S1, T12)
    #     from plant_registration_4d import visualize as vis_skel
    #     fh = plt.figure()
    #     vis_skel.plot_skeleton(fh, S1,'b')
    #     vis_skel.plot_skeleton(fh, S2_hat,'k')
    #     vis_skel.plot_skeleton(fh, S2,'r')
    #     vis.plot_skeleton_correspondences(fh, S2_hat, S2, corres)
    #     plt.title("Skeleton registration results.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise Wur Tomato Data.")

    # Add arguments for each visualization option
    parser.add_argument("--visualise", type=int, help="Visualise data at given index")
    parser.add_argument("--visualise_semantic", type=int, help="Visualise semantic data at given index")
    parser.add_argument("--visualise_skeleton", type=int, help="Visualise skeleton data at given index")
    # parser.add_argument("--visualise_inference", type=str, help="Visualise inference from given file path")
    # parser.add_argument("--run_geodesic", type=int, help="Run geodesic example")
    parser.add_argument("--run_semseg_evaluation", action='store_true', help="Run evaluation example")
    parser.add_argument("--run_skeleton_evaluation", action='store_true', help="Run evaluation example")
    # skeletonisation
    parser.add_argument("--run_skeleton", action='store_true', help="debugging")
    ## debugging
    parser.add_argument("--run_debugging", type=int, help="debugging")
    parser.add_argument("--run_registration", action='store_true', help="debugging")


    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of WurTomatoData
    obj = WurTomatoData()

    # visualissation
    if args.visualise is not None:
        obj.visualise(args.visualise)
    elif args.visualise_semantic is not None:
        obj.visualise_semantic(args.visualise_semantic)
    elif args.visualise_skeleton is not None:
        obj.visualise_skeleton(args.visualise_skeleton)
    ## run evaluation
    elif args.run_semseg_evaluation:
        obj.run_semantic_evaluation()
    elif args.run_skeleton_evaluation:
        obj.run_skeleton_evaluation()
    ## run skeletonisation example
    elif args.run_skeleton:
        obj.run_skeletonisation()

    # elif args.run_debugging is not None:
    #     obj.run_debugging(args.run_debugging)
    # elif args.run_evaluation:
    #     obj.run_evaluation()
    # elif args.run_registration:
    #     obj.run_registration()
    

# if __name__=="__main__":
#     obj = WurTomatoData()
#     # obj.visualise(0)
#     # obj.visualise_semantic(0)
#     # obj.visualize_skeleton(0)
#     # obj.visualize_inference("./work_dir/debug/result/Harvest_01_PotNr_179.txt")
