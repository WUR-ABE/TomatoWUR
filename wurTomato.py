################################################################
# Author     : Bart van Marrewijk                              #
# Contact    : bart.vanmarrewijk@wur.nl                        #
# Date       : 30-05-2024                                      #
# Description: Code related to the TomatoWUR dataset           #
################################################################

# Usage: see if__name__


import os
import sys
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
# from omegaconf import dictconfig
# import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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



class WurTomatoData(Dataset):
    """

    Description:
    loading and visualidation TomatoWUR dataset: DOI 

    https://github.com/WUR-ABE/TomatoWUR

    Author     : Bart M. van Marrewijk
    Contact    : bart.vanmarrewijk@wur.nl
    Date       : 20-03-2025

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
        config_data = config.Config("config.yaml")

        # self._set_attributes(config_data)
        self.__dict__.update(config_data.__dict__)

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
        for x in self.dataset:
            for key, value in x.items():
                if key=="images" or key=="images_seg" or key=="genotype":
                        continue
                x[key] = self.data.json_path.parent / value
                if not x[key].is_file():
                    print(f"warning {x[key]} is missing")

        self.S_gt = None
        self.camera_specs = None
        print("Successfully loaded the WURTomato dataset")

    # # Download LastSTRAW data file in zip format
    def __download(self):
        """
        If the unzipped files exist do not download. If they do not
        exist then download the zip file
        """
        print(self.project_dir)
        if not (self.project_dir / self.project_code).is_dir():
            self.downloadFile = "temp.zip"
            if (self.project_dir / self.downloadFile).is_file():
                print("Already downloaded but not unzipped")
                return
            # if not (self.folder / self.downloadFile).is_file():
            # print("Downloading: " + self.downloadFile + " to folder: " + self.folder + " from: " + self.url)
            print("This may take a while (TomatoWUR is 3.36GB)...")

            response = requests.get("https://" + str(self.url), stream=True)
            if response.status_code == 200:
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

    def __load_graph(self, index):
        if self.S_gt is None or self.S_gt.name != self.dataset[index]["file_name"].stem:
            self.S_gt = create_skeleton_gt_data(self.dataset[index]["skeleton_file_name"], pc_path=self.dataset[index]["file_name"], pc_semantic_path=self.dataset[index]["sem_seg_file_name"])
        return self.S_gt

    # Loads xyz of point cloud
    def load_xyz_array(self, index):
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


    def run_semantic_evaluation(self):
        dt_graph_dir = Path("./Resources/output_semantic_segmentation")
        obj = evaluate_semantic_segmentation.EvaluationSemantic(dt_graph_dir=dt_graph_dir, gt_json=self.data.json_path)
        obj.evaluate_pairs()


    def run_skeleton_evaluation(self):
        # folder = Path(self.cfg["folder"])
        dt_graph_dir = Path("Resources/output_skeleton") / self.cfg["skeleton_method"]
        obj = Evaluation(self.data.pointcloud_dir, dt_graph_dir, gt_json=self.data.json_path)
        obj.evaluate_pairs(vis=False, evaluate_gt=self.cfg["evaluation"]["evaluate_gt"])


    def nodes2edges(self, points, nodes, method="xu", **kwargs):
        if method == "xu":
            nodes, edges, edge_type = xu.xu_method_connect_points(nodes, kwargs["parents"], kwargs["mtg"])
            return nodes, edges, edge_type
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
            pcd = self.load_xyz_array(i)
            semantic = self.__load_xyz_semantic_array(i)
            pcd_filtered, semantic_filtered = self.get_filtered_data(i)

            root_idx = findBottomCenterRoot(pcd_filtered, semantic_filtered, method=self.cfg["root_method"])

            if self.cfg["skeleton_method"]=="xu":
                binaratio = self.cfg["xu"]["binratio"]
                n_neighbors = self.cfg["xu"]["n_neighbors"]

                positions, parents, mtg = xu.xu_method(pcd_filtered, root_idx=root_idx, binratio=binaratio, nearest_neighbour=n_neighbors, vis=False)
                nodes, edges, edge_type = self.nodes2edges(pcd_filtered, positions, method=self.cfg["xu"]["nodes2edges"], parents=parents, mtg=mtg)
                save_name = save_folder / "xu" / (self.dataset[i]["file_name"].stem+".csv")

            else:
                raise NotImplementedError(f'{self.cfg["skeleton_method"]} method not implemented.')
                exit()

            S_pred = skeleton_graph.SkeletonGraph()
            S_pred.load(nodes, edges, edge_types=edge_type, df_pc=pd.DataFrame(pcd, columns=["x", "y", "z"]), name=self.dataset[i]["file_name"].stem)
            S_pred.get_node_order()
            if visualise:
                S_pred.visualise_graph()
            print(f"saving skeleton to {save_name}")
            S_pred.export_as_nodelist(save_name)



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
    # obj.voxel_carving()
    # exit()

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
