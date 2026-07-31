################################################################
# Author     : Bart van Marrewijk
# Contact    : bart.vanmarrewijk@wur.nl
# Date       : 19-05-2026
# Description: Code related to the TomatoWUR dataset, including 
# skeletonisation methods of point cloud to plant traits paper 
################################################################

# Usage: see if__name__


import os
import sys
import argparse
import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
import json
import natsort

import networkx as nx

from torch.utils.data import Dataset
from tqdm import tqdm
import requests
from zipfile import ZipFile
from pathlib import Path
import pandas as pd
# import natsort
import polyscope as ps
# from omegaconf import dictconfig
# import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.utils_skeletonisation import findBottomCenterRoot, undirected2directed#convert_segmentation2skeleton, evaluate_skeleton
from scripts import skeleton_graph
from scripts import visualize_examples as ve
from scripts import evaluate_semantic_segmentation
from scripts.evaluate_skeletons import Evaluation
from scripts import config
from scripts.nodes_to_graph import nodes_to_graph

from skeletonisation_methods.plantscan3d import xu
import time



def create_folder(folder_name):
    folder_name = Path(folder_name)
    if not folder_name.exists():
        folder_name.mkdir()

class WurTomatoData(Dataset):
    """

    Description:
    loading and visualidation TomatoWUR dataset: DOI 

    https://github.com/WUR-ABE/TomatoWUR

    Author     : Bart M. van Marrewijk
    Contact    : bart.vanmarrewijk@wur.nl
    Date       : 19-05-2026

    Example usage:

    obj = WurTomatoData()
    ## visualize point cloud
    obj.visualise(index=0)

    ## visualise_semantic
    obj.visualise_semantic(index=0)

    obj.visualise_skeleton(index=0)
    obj.run_semantic_evaluation()
    obj.run_skeleton_evaluation()
    obj.run_skeletonisation(visualise=False)
    """

    def __init__(self, **kwargs):
        config_data = config.init_config(**kwargs)
        ## set to self.
        for key, value in config_data.items():
            setattr(self, key, value)

        self.cfg = config_data
        # self._set_attributes(config_data)
        # self.__dict__.update(config_data.__dict__)
        # self.__dict__.update(config_data.__dict__["_content"])

        # If the data folder can not be found then ask to download the data
        if not (self.project_dir / self.project_code).exists():
            user_input = input(f"Data not found {self.project_dir / self.project_code}. Do you want to download the data? (y/n): ").strip().lower()
            if user_input == 'y':
                self.__download()
                self.__unzip()
            else:
                raise FileNotFoundError("Data not found and download not initiated.")

        ## open annotation file
        with open(self.data.json_path, "r") as f:
            self.dataset = json.load(f)

        # Apply natsort to self.dataset based on "file_name"
        self.dataset = natsort.natsorted(self.dataset, key=lambda x: str(x["file_name"]))
        for x in self.dataset:
            for key, value in x.items():
                if key=="images" or key=="images_seg" or key=="genotype":
                        continue
                x[key] = self.data.json_path.parent / value
                if not x[key].is_file():
                    print(f"warning {x[key]} is missing")

        self.S_gt = None
        self.camera_specs = None

        if not self.cfg.save_folder.exists():
            self.cfg.save_folder.mkdir(True, True)
        print(f"Successfully loaded the WURTomato dataset split: {self.cfg.data.json_split} N={len(self.dataset)}")

    # # Download LastSTRAW data file in zip format
    def __download(self):
        """
        If the unzipped files exist do not download. If they do not
        exist then download the zip file
        """
        print(self.project_dir)
        if not (self.project_dir / self.project_code).is_dir():
            if not self.project_dir.exists():
                self.project_dir.mkdir()

            self.downloadFile = "temp.zip"
            if (self.project_dir / self.downloadFile).is_file():
                print("Already downloaded but not unzipped")
                return

            response = requests.get("https://" + str(self.url), stream=True)
            if response.status_code == 200:
                print("Downloading, this may take a while (TomatoWUR is 3.4GB)...")
                total_size = int(response.headers.get('content-length', 0))  # Get total size in bytes
                block_size = 8192  # Or whatever chunk size you want
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

                with open(self.project_dir / self.downloadFile, "wb") as file:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            file.write(chunk)
                            progress_bar.update(len(chunk))
                progress_bar.close()
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
            if not (self.project_dir / self.project_code).is_dir():
                print(f"Extracting: {self.project_dir / self.downloadFile}")
                with ZipFile(str(self.project_dir / self.downloadFile), "r") as zObject:
                    file_list = zObject.namelist()
                    total_files = len(file_list)
                    progress_bar = tqdm(total=total_files, unit='file', desc="Extracting files")
                    for file in file_list:
                        zObject.extract(file, path=str(self.project_dir))
                        progress_bar.update(1)
                    progress_bar.close()

                new_zip_file = self.project_dir / (self.project_code + ".zip")
                print(f"Extracting: {new_zip_file}")
                with ZipFile(new_zip_file, "r") as zObject:
                    file_list = zObject.namelist()
                    total_files = len(file_list)
                    progress_bar = tqdm(total=total_files, unit='file', desc="Extracting files")
                    for file in file_list:
                        zObject.extract(file, path=str(self.project_dir))
                        progress_bar.update(1)
                    progress_bar.close()
                print(f"Deleting {new_zip_file}")
                os.remove(str(new_zip_file))

    def load_graph(self, index):
        if self.S_gt is None or self.S_gt.name != self.dataset[index]["file_name"].stem:
            self.S_gt = skeleton_graph.SkeletonGraph.from_skeleton_gt_data(self.dataset[index]["skeleton_file_name"], pc_path=self.dataset[index]["file_name"], pc_semantic_path=self.dataset[index]["sem_seg_file_name"])
        return self.S_gt
    
    def get_index_by_name(self, name="Harvest_02_PlantNr_27"):
        id_dict = {}
        for i, item in enumerate(self.dataset):
            id_dict[item["file_name"].stem] = i
        return id_dict[name]

    # Loads xyz of point cloud
    def load_xyz_array(self, index):
        # Loads the data from an .xyz file into a numpy array.
        self.load_graph(index)
        return self.S_gt.get_xyz_pointcloud()

    def load_xyz_semantic_array(self, index):
        self.load_graph(index)
        return self.S_gt.get_semantic_pointcloud()

    def get_filtered_data(self, index):
        self.load_graph(index)
        pcd = self.S_gt.get_xyz_pointcloud()
        semantic = self.S_gt.get_semantic_pointcloud()
        bool_array = np.bitwise_or(semantic==1 ,semantic==3) # 1=leaves, 2=main stem, 3=pole, 4=side stem

        return pcd[~bool_array, :], semantic[~bool_array]

    # Return number of data files
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        self.scan_index = 0
        return self
    
    def __next__(self):
        if self.scan_index < len(self):
            # pointCloud, labels_available, labels, skeleton_data = self.__load_as_o3d_cloud(self.scan_index)
            data = self.load_graph(self.scan_index)
            self.scan_index += 1
            return data
    
        else:
            raise StopIteration

    def __getitem__(self, index):
        return self.load_graph(index)

    def visualise(self, index=0):
        self.load_graph(index)
        print(f'Visualising {self.dataset[index]["file_name"].stem}')
        ve.vis(pc = self.S_gt.get_xyz_pointcloud(), colors=self.S_gt.get_colours_pointcloud(), save_name=self.cfg.save_folder/"image.png")

    def visualise_semantic(self, index=0, semantic_name= "leaf_stem_instances"):
        self.load_graph(index)
        print(f'Visualising semantic {self.dataset[index]["file_name"].stem}')
        labels = self.S_gt.get_semantic_pointcloud(semantic_name=semantic_name)
        if "semantic" in semantic_name:
            ve.vis(pc = self.S_gt.get_xyz_pointcloud(), colors=labels)
        else:
            ve.vis(pc = self.S_gt.get_xyz_pointcloud(), scalars=labels)

    def visualise_input(self, index=0, semantic_name= "semantic"):
        self.load_graph(index)

        pcd_filtered, semantic_filtered = self.get_filtered_data(index)
        print(f'Visualising semantic {self.dataset[index]["file_name"].stem}')
        ve.vis(pc = pcd_filtered, colors=semantic_filtered, save_name=self.cfg.save_folder/"image.png")

        ## visualising semantics with nodes
        # labels = self.S_gt.get_semantic_pointcloud(semantic_name="semantic_with_nodes")
        # colours = rgb_array[labels.astype(int)].copy()
        # ve.vis(pc = self.S_gt.get_xyz_pointcloud(), colors=colours)

    def create_images_giphy(self):
        # Loop to rotate and capture frames
        n_frames = 36
        for i in range(n_frames):
            angle_deg = i * (360 / n_frames)
            # Set view by rotating around z axis
            # Example: rotate camera around the z-axis at a fixed radius
            radius = 1  # Adjust as needed
            center = np.mean(self.S_gt.get_xyz_pointcloud(), axis=0)
            angle_rad = np.deg2rad(angle_deg)
            camera_position = center + radius * np.array([np.cos(angle_rad), np.sin(angle_rad), 0.5])
            up_dir = np.array([0, 0, 1])
            ps.look_at_dir(camera_location=camera_position, target=center, up_dir=up_dir)

            # Draw the scene and save a screenshot
            ps.screenshot(f"frames/frame_{i:03d}.png", transparent_bg=False)

        # Optional: close viewer
        ps.clear_user_callback()

    def visualise_skeleton(self, index=0, parent_nodes_only=False, apply_filtering=False):
        print(f'Visualising skeleton {self.dataset[index]["file_name"].stem}')
        self.load_graph(index)
        ## for graph in section 2
        if parent_nodes_only:
            self.S_gt.filter(keep_parents_only=True, node_order=100)
        apply_filtering=False
        if apply_filtering: 
            self.S_gt.gaussian_smoothing()
            self.S_gt.get_edge_type()
            self.S_gt.line_fitting_3d()
            self.S_gt.get_edge_type()
            self.S_gt.simplify()
            self.S_gt.get_edge_type()

        self.S_gt.visualise_graph(save_name=self.cfg.save_folder/"image.png", show_segmented=True)


    def run_semantic_evaluation(self, dt_graph_dir = Path("./Resources/output_semantic_segmentation")):
        obj = evaluate_semantic_segmentation.EvaluationSemantic(dt_graph_dir=dt_graph_dir, gt_json=self.data.json_path)
        obj.evaluate_pairs()


    def run_skeleton_evaluation(self):
        # folder = Path(self.cfg["folder"])
        print(f"Evaluating skeletons in: {self.cfg.save_folder}")
        obj = Evaluation(self.data.pointcloud_dir, self.cfg.save_folder, **self.cfg.evaluate)
        obj.evaluate_pairs(vis=self.cfg.evaluate.vis)
    
    def run_single_skeleton_evaluation(self, plant_id="Harvest_01_PotNr_80"):
        # folder = Path(self.cfg["folder"])
        print(f"Evaluating skeletons in: {self.cfg.save_folder}")
        obj = Evaluation(self.data.pointcloud_dir, self.cfg.save_folder, **self.cfg.evaluate)
        save_name = self.cfg.save_folder / (plant_id+".csv")
        if self.cfg.debug_plant:
            save_name = self.cfg.save_folder / ( self.cfg.debug_plant+".csv")
                       
        obj.evaluate_pred(pred_name=save_name, vis=True)

            
    def run_semantic_segmentation(self):
        semseg_url = "https://github.com/WUR-ABE/2D-to-3D_segmentation"
        print(f"Not implemented, please have look at following git: f{semseg_url}")

    def run_skeletonisation_optimisation(self):
        config_copy = self.cfg.copy()
        combos = config.create_config_list(self.cfg[self.cfg["skeleton_method"]])
        self.save_folder = self.save_folder / "optimisation"
        optimisation_folder = self.save_folder
        create_folder(self.save_folder)
        if not self.save_folder.exists():
            self.save_folder.mkdir()
        with open(self.save_folder / "optimisation_settings.json", "w") as f:
            json.dump(combos, f, indent=4)

        for i, combi in enumerate(combos):
            self.cfg.save_folder = optimisation_folder / str(i)
            create_folder(self.cfg.save_folder)
            self.cfg[self.cfg["skeleton_method"]].update(**combi)
            self.run_skeletonisation()
            self.run_skeleton_evaluation()
  

    def run_skeletonisation(self, visualise=False):
        # save_folder = Path("Resources/output_skeleton_paper3")
        config.save_config(self.cfg, self.save_folder / "config.yaml")

        speed_test = []
        for i in tqdm(range(len(self))):
            print(f'Running skeletonisation on {self.dataset[i]["file_name"]}')
            if self.cfg.debug_plant:
                if self.dataset[i]["file_name"].stem!=self.cfg.debug_plant:
                    continue
            pcd = self.load_xyz_array(i)
            semantic = self.load_xyz_semantic_array(i)
            pcd_filtered, semantic_filtered = self.get_filtered_data(i)

            t0=time.time()
            if self.root_method=="gt":
                root_pos = self.S_gt.G.nodes[0]["pos"]
                pcd_filtered = np.vstack([root_pos, pcd_filtered])
                semantic_filtered = np.insert(semantic_filtered, 0, 2)
                root_idx = 0
                # raise NotImplementedError
            else:
                root_idx = findBottomCenterRoot(pcd_filtered, semantic_filtered, method=self.root_method)
            print("root_position", pcd_filtered[root_idx])

            if self.cfg["skeleton_method"]=="xu":
                binaratio = self.cfg["xu"]["binratio"]
                n_neighbors = self.cfg["xu"]["n_neighbors"]

                positions, parents, mtg = xu.xu_method(pcd_filtered, root_idx=root_idx, binratio=binaratio, nearest_neighbour=n_neighbors, vis=False)
                nodes, edges, _ = nodes_to_graph(pcd_filtered, positions, root_idx=root_idx, method=self.cfg["xu"]["nodes2edges"], parents=parents, mtg=mtg)
                S_pred = skeleton_graph.SkeletonGraph()
                S_pred.load(nodes, edges, edge_types=None, df_pc=pd.DataFrame(pcd, columns=["x", "y", "z"]), name=self.dataset[i]["file_name"].stem)
                S_pred.get_edge_type(**self.cfg["xu"]["graph2tree"])

                save_name = self.cfg.save_folder / (self.dataset[i]["file_name"].stem+".csv")

            elif self.cfg["skeleton_method"]=="som":
                from skeletonisation_methods.som import som
                self.cfg["som"]["init"]["x"] = round(len(pcd_filtered)/self.cfg["som"]["points_per_node"]) ## empirecally
                self.cfg["som"]["init"]["y"] = 1

                nodes = som.som_method(pcd_filtered, cfg=self.cfg["som"])
                nodes, edges, _ = nodes_to_graph(pcd_filtered, nodes, root_idx=root_idx, method=self.cfg["som"]["nodes2edges"])

                S_pred = skeleton_graph.SkeletonGraph()
                S_pred.load(nodes, edges, edge_types=None, df_pc=pd.DataFrame(pcd, columns=["x", "y", "z"]), name=self.dataset[i]["file_name"].stem)
                S_pred.get_edge_type(**self.cfg["som"]["graph2tree"])

                save_name = self.cfg.save_folder / (self.dataset[i]["file_name"].stem+".csv")
                # ve.vis(nodes=nodes, edges=edges)

            elif self.cfg["skeleton_method"]=="voxel":
                import skeletonisation_methods.voxel.fill_voxel as fill_voxel
                nodes, edges, root_idx = fill_voxel.main(pcd_filtered,
                                                        root_idx=root_idx,
                                                         **self.cfg.voxel)
                                                #     voxel_size = self.cfg["voxel"]["voxel_size"],
                                                #   return_pc=self.cfg["voxel"]["nodes2edges_input"])
                if self.cfg["voxel"]["nodes2edges"] is not None:
                    nodes, edges, edge_type = nodes_to_graph(pcd_filtered, nodes, root_idx=root_idx, method=self.cfg["voxel"]["nodes2edges"])

                S_pred = skeleton_graph.SkeletonGraph()
                S_pred.load(nodes, edges, edge_types=None, df_pc=pd.DataFrame(pcd, columns=["x", "y", "z"]), name=self.dataset[i]["file_name"].stem)
                S_pred.get_edge_type(**self.cfg["voxel"]["graph2tree"])
                # S_pred.gaussian_smoothing(var0=0.3, var1=0.3, indices=[0,1,2], node_order_filtering=False)
                # S_pred.gaussian_smoothing(var0=0.25, var1=0.25, indices=[0,1], node_order_filtering=False)

                # S_pred.get_edge_type(angle_between_trunk_and_lateral=60)
                # edges = S_pred.get_edges()
                # nodes = S_pred.get_node_attribute()
                # S_pred.visualise_graph()
                # nodes, edges, edge_type = nodes_to_graph(input_pc, nodes, method=                # nodes, edges, edge_type = nodes_to_graph(input_pc, nodes, method=self.cfg["voxel"]["nodes2edges"])
                save_name = self.cfg.save_folder / (self.dataset[i]["file_name"].stem+".csv")

            elif self.cfg["skeleton_method"]=="laplacian":
                from skeletonisation_methods.pc_skeletor.pc_skeletor import LBC
                lbc = LBC(pcd_filtered, **self.cfg["laplacian"]["settings_lbc"])
                lbc.extract_skeleton()
                lbc.extract_topology()
                nodes = np.asarray(lbc.skeleton.points)

                print("BART 20250805 changed root_dix from none to root_idx in laplacian!!")
                nodes, edges, _ = nodes_to_graph(pcd_filtered, nodes, root_idx=root_idx, method=self.cfg["laplacian"]["nodes2edges"])

                S_pred = skeleton_graph.SkeletonGraph()
                S_pred.load(nodes, edges, edge_types=None, df_pc=pd.DataFrame(pcd, columns=["x", "y", "z"]), name=self.dataset[i]["file_name"].stem)
                S_pred.get_edge_type(**self.cfg["laplacian"]["graph2tree"])
                
                ## including simplified skeleton by LBC method:                
                # nodes, edges, edge_type = nodes_to_graph(pcd_filtered, np.asarray(lbc.topology.points), method=self.cfg["laplacian"]["nodes2edges"])
                
                ## DEBUGGING
                debug = False
                # debug=True
                if debug:
                    # ve.vis(pc=pcd_filtered)
                    # ve.vis(pc=np.asarray(lbc.contracted_point_cloud.points))
                    # ve.vis(pc=np.asarray(lbc.skeleton.points))
                    # temp = skeleton_graph.relabel(lbc.skeleton_graph)
                    # ve.vis(pc=pcd_filtered, nodes=np.array([temp.nodes[node]['pos'] for node, degree in temp.degree()]),edges=np.asarray(temp.edges))
                    # ve.vis(pc=pcd_filtered, nodes=np.asarray(lbc.topology.points),
                    #        edges=np.asarray(lbc.topology_graph.edges))
                    ## graph after simplication 
                    # nodes = np.asarray(lbc.topology.points)
                    # edges = undirected2directed(np.asarray(lbc.topology_graph.edges))
                    ve.vis(pc=pcd_filtered, nodes=nodes, edges=edges, edges_type=S_pred.get_attributes()["edge_type"])
                
                save_name = self.cfg.save_folder / (self.dataset[i]["file_name"].stem+".csv")

            elif self.cfg["skeleton_method"]=="laplacian_semantic":
                from skeletonisation_methods.pc_skeletor.pc_skeletor import SLBC
                temp_dict = {'trunk': pcd_filtered[semantic_filtered==2,:], 'branches': pcd_filtered[semantic_filtered==4,:]}
                lbc = SLBC(temp_dict, **self.cfg["laplacian_semantic"]["settings_lbc"])
                lbc.extract_skeleton()
                lbc.extract_topology()
                nodes = np.asarray(lbc.skeleton.points)
                nodes, edges, _ = nodes_to_graph(pcd_filtered, nodes, root_idx=root_idx, method=self.cfg["laplacian_semantic"]["nodes2edges"])

                S_pred = skeleton_graph.SkeletonGraph()
                S_pred.load(nodes, edges, edge_types=None, df_pc=pd.DataFrame(pcd, columns=["x", "y", "z"]), name=self.dataset[i]["file_name"].stem)
                S_pred.get_edge_type(**self.cfg["laplacian_semantic"]["graph2tree"])

                ## including simplified skeleton by LBC method:                
                # nodes, edges, edge_type = nodes_to_graph(pcd_filtered, np.asarray(lbc.topology.points), method=self.cfg["laplacian"]["nodes2edges"])
                
                save_name = self.cfg.save_folder / (self.dataset[i]["file_name"].stem+".csv")

            elif self.cfg["skeleton_method"]=="adtree":
                from skeletonisation_methods.adtree import Adtree_debugging
                save_name = self.cfg.save_folder / (self.dataset[i]["file_name"].stem+".csv")
                Adtree_debugging.run_adtree(adtree_path=self.cfg.adtree.adtree_path, xyz_array=pcd_filtered,
                                            output_name=save_name)
            else:
                raise NotImplementedError(f'{self.cfg["skeleton_method"]} method not implemented.')
                exit()
            ## determine speed of algorithm
            speed_test.append(time.time()-t0)
            
            S_pred.get_node_order()
            # ve.vis(pc=pcd, nodes=nodes, edges=edges, edges_type=S_pred.get_attributes()["edge_type"][1:])

            if visualise:
                S_pred.df_pc["semantic"] = self.load_xyz_semantic_array(i)
                S_pred.visualise_graph(save_name=self.cfg.save_folder / (self.dataset[i]["file_name"].stem+"_input.png"), show_segmented=False)
            print(f"saving skeleton to {save_name}")
            S_pred.export_as_nodelist(save_path=save_name)
        
        print(f'avg speed {self.cfg["skeleton_method"]} is: {np.mean(speed_test):0.2f} [s]')
        


    def get_2d_images(self, index=0):
        ##TODO fix folder,
        ## get for loop with images
        images_path = [self.data.json_path.parent / x for x in self.dataset[index]["images"]]
        images_seg_path = [self.data.json_path.parent / x for x in self.dataset[index]["images_seg"]]
        return images_path, images_seg_path
        

    def load_camera_specs(self):
        """
        Loads the camera specifications from the calibration folder
        Attributes:
            camera_specs (CameraClass): An instance of CameraClass containing the camera specifications.
        """
        from scripts import camera_calib
        if self.camera_specs is None:
            self.camera_specs = camera_calib.CameraClass(calib_folder=self.data.camera_poses_dir) 

    def voxel_carving(self, index=0):
        """
        Create 3D point clouds using voxel carving (high similarity with original data but not exactly the same)
        """
        if self.camera_specs is None:
            self.load_camera_specs()
        print("Staring voxel carving methodology")
        from scripts import voxel_carving
        _, img_seg_list = self.get_2d_images(index)
        voxel_carving.custom_voxel_carving(self.camera_specs, img_folder_or_list=img_seg_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loading wurTomato dataset")
    parser.add_argument("--config", type=str, default= "config.yaml", help="debugging")

    # Parse the arguments
    args, unknown = parser.parse_known_args()

    # Create an instance of WurTomatoData
    obj = WurTomatoData(cfg_filename=args.config, overrides=unknown)
    print(obj.cfg.run_mode)

    for run_mode in obj.cfg.run_mode:
        if run_mode=="voxel_carving":
            obj.voxel_carving()
        elif run_mode=="skeletonisation":
            obj.run_skeletonisation(visualise=obj.vis_skeleton)
        elif run_mode=="evaluate":
            obj.run_skeleton_evaluation()
        elif run_mode=="evaluate_semantic":
            obj.run_semantic_evaluation()
        elif run_mode=="optimisation":
            obj.run_skeletonisation_optimisation()
        elif run_mode=="visualise":
            for i, obj_i in enumerate(obj):
                print(i, obj_i.name)
                obj_i.visualise_graph()
        elif run_mode=="visualise_semantic":
            obj.visualise_semantic()
        elif run_mode=="visualise_skeleton":
            obj.visualise_skeleton()
        elif run_mode=="evaluate_single":
            obj.run_single_skeleton_evaluation(obj.cfg.debug_plant)
            print(f"WARNING Please check run_mode: {run_mode}")
        else:
            print("runmodes is not valid, please check")
    
