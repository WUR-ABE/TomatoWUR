# voxel_carving.py
import cv2
import open3d as o3d

from pathlib import Path
import numpy as np
import pandas as pd
import natsort

# from convert_marvin_pointclouds import load_marvin_calibration
# from convert_2Dto3D_tools import pointcloud_utils
"""The maxi marvin code is not publically available, but the following code snippet is a open3d based voxel carving implementation.
Note that this implementation is not optimized for speed. 
"""

def custom_voxel_carving(obj, img_folder_or_list, cubic_size = [400, 400, 800], voxel_size=2, save_name=None):
    # set voxel grid size and resolution. Increase voxel_size to speed up.
    
    # voxel_size = 5

    print("Creating VoxelGrid with size %.2f, wait a few minutes..." % voxel_size)
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size[0],
        height=cubic_size[1],
        depth=cubic_size[2],
        voxel_size=voxel_size,
        # origin=np.array([-cubic_size[0] / 2.0, -cubic_size[1] / 2.0, -cubic_size[2] / 2.0]),
        origin=np.array([0, 0, 0], dtype=float),
        color=np.array([0, 0, 0], dtype=float))
    
    if isinstance(img_folder_or_list, Path):
        img_list = natsort.natsorted(img_folder_or_list.glob("*preseg*.png"))
    else:
        assert isinstance(img_folder_or_list, list), "img_folder_or_list should be a list"
        img_list = img_folder_or_list

    for cam_name in img_list:
        # if "coloured" in cam_name.stem:
        #     continue
        bgr_img = cv2.imread(str(cam_name))
        mask = np.ones((1920, 1080), dtype=np.float32)*1 # must be float otherwise does not work...
        mask[np.all(bgr_img==[0,0,0], axis=2)] = 0

        cam_id = str(int(cam_name.stem.split("_")[-1].replace("cam_","")))
        params = o3d.camera.PinholeCameraParameters()
        fx, cx, fy, cy = obj.get_fx_cx_fy_cy(cam_id)
        height, width = obj.get_height_width(cam_id)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # params.intrinsic.set_intrinsics(1080, 1920, fx, fy, cx, cy)
        params.intrinsic = intrinsics
        tf = obj.get_tf(cam_id)
        tf[:3,3]*=1000 ## to convert to mm
        params.extrinsic = tf
        # True is actually better, but False matches MaxiMarvin
        voxel_carving.carve_silhouette(o3d.geometry.Image(mask), params, keep_voxels_outside_image=False) 

        print(voxel_carving)

    # create voxel points 
    xyz_array = np.asarray([voxel_carving.origin + pt.grid_index*voxel_carving.voxel_size for pt in voxel_carving.get_voxels()])
    df = pd.DataFrame(xyz_array, columns=["x", "y", "z"])/1000

    # to be compatible with marvin output we need to reproject as well  
    # for cam_num in range(0, 15):
    #     cam_num = str(cam_num)
    #     points = reproject_points(df[["x", "y", "z"]].values.astype(float), cam_num)
    #     df["x" + cam_num] = points[:, 0]
    #     df["y" + cam_num] = points[:, 1]
    # pointcloud_utils.save_df_pointcloud("keep_voxels_outside_image.ply", df)
    from scripts import visualize_examples as ve
    ve.vis(pc=xyz_array)

    if save_name is not None:
        save_name = Path(save_name)
        df.to_csv(save_name, index=False)
        print("saved point cloud as", save_name)

if __name__=="__main__":
    from scripts import camera_calib

    obj = camera_calib.CameraClass("config.yaml")