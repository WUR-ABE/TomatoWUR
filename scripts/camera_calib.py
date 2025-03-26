# camera_calib.py
import json
from pathlib import Path
import numpy as np
import copy

class CameraParams():
    height: None
    width: None
    fx = None
    fy = None
    cx = None
    cy = None
    tf = None

    def set_fx_cx_fy_cy(self, fx, cx, fy, cy):
        self.fx = fx
        self.cx = cx
        self.fy = fy
        self.cy = cy

    def set_height_width(self, height, width):
        self.height = height
        self.width = width

    def set_tf(self, tf):
        self.tf = np.array(tf)
        assert self.tf.shape == (4, 4), "Transformation matrix tf must be 4x4"


class CameraClass():

    def __init__(self, calib_folder=None):
        self.cam_dict = {}
        self.camera_dict={}
        self.cam_ids = list(range(0,15))
        if calib_folder is None:
            self.calib_folder = Path("TomatoWUR") / "camera_poses"
        else:
            self.calib_folder  = Path(calib_folder)
        self.load_cams()

    def load_cams(self):
        [self.load_cam(x) for x in self.cam_ids]

    def load_cam(self, cam_id):
        with open(self.calib_folder  / (str(cam_id)+ ".json"), "r") as f:
            data = json.load(f)

        cam_id_obj = CameraParams()
        cam_id_obj.set_fx_cx_fy_cy(data["intrinsics"]["fx"], data["intrinsics"]["cx"], 
                                   data["intrinsics"]["fy"], data["intrinsics"]["cy"],
                                   )
        cam_id_obj.set_height_width(data["intrinsics"]["height"], data["intrinsics"]["width"])
        cam_id_obj.set_tf(data["extrinsics"])
        self.camera_dict[str(cam_id)] = cam_id_obj
        print(f"sucessfully loaded {cam_id}")

    def get_intrinsics(self, cam_id):
        if cam_id in self.camera_dict:
            cam_params = self.camera_dict[cam_id]
            intrinsics_matrix = np.array([
                [cam_params.fx, 0, cam_params.cx],
                [0, cam_params.fy, cam_params.cy],
                [0, 0, 1]
            ])
            return intrinsics_matrix
        else:
            raise ValueError(f"Camera ID {cam_id} not found")

    def get_fx_cx_fy_cy(self, cam_id):
        if cam_id in self.camera_dict:
            cam_params = self.camera_dict[cam_id]
            return cam_params.fx, cam_params.cx, cam_params.fy, cam_params.cy
        else:
            raise ValueError(f"Camera ID {cam_id} not found")
        
    def get_height_width(self, cam_id):
        if cam_id in self.camera_dict:
            cam_params = self.camera_dict[cam_id]
            return cam_params.height, cam_params.width
        else:
            raise ValueError(f"Camera ID {cam_id} not found")

    def get_tf(self, cam_id):
        if cam_id in self.camera_dict:
            cam_params = self.camera_dict[cam_id]
            return cam_params.tf
        
    def get_o3d_tf(self, cam_id):
        """returns tf in mm"""
        if cam_id in self.camera_dict:
            cam_params = self.camera_dict[cam_id]
            tf = copy.copy(cam_params.tf)
            tf[:3,3]*=1000 ## to convert to mm
            return tf
        


if __name__=="__main__":
    obj = CameraClass(calib_folder="/home/agro/w-drive-vision/GARdata/datasets/tomato_plant_segmentation/20241223_csv_skeletons/camera_poses")
    obj.load_cams()