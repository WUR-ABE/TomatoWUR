# camera_calib.py
import json
from pathlib import Path
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

def T_opencv_to_opengl(T: np.ndarray) -> np.ndarray:
    """
    Convert T from OpenCV convention to OpenGL convention.

    - OpenCV
        - +X: Right
        - +Y: Down
        - +Z: The view direction, pointing forward and away from the camera
        - Used in: OpenCV, COLMAP, camtools default
    - OpenGL
        - +X: Right
        - +Y: Up
        - +Z: The negative view direction, pointing back and away from the camera
        - -Z: The view direction
        - Used in: OpenGL, Blender, Nerfstudio
          https://docs.nerf.studio/quickstart/data_conventions.html#coordinate-conventions

    Args:
        T: Extrinsic matrix (world-to-camera) of shape (4, 4) in OpenCV convention.

    Returns:
        Extrinsic matrix (world-to-camera) of shape (4, 4) in OpenGL convention.
    """
    # sanity.assert_T(T)
    # pose = T_to_pose(T)
    # pose = pose_opencv_to_opengl(pose)
    # T = pose_to_T(pose)
    T = np.copy(T)
    T[:, 2] *= -1
    T = T[:, [1, 0, 2, 3]]
    T[1:3, 0:4] *= -1
    return T

def unmirrored_marvin_output2open3d(tf=None, quat=None, trans=None):
    """Also works for open3d2unmirrored_marvin_output"""
    # copied from camtools R_t_to_C
    if tf is not None:
        r, t = tf[:3, :3], tf[:3, 3]
        new_trans = -r.T @ t
        new_tf = np.eye(4)
        new_tf[:3, :3] = r.T
        new_tf[:3, 3] = new_trans
        return new_tf
    elif quat is not None and trans is not None:
        r = R.from_quat(quat).as_matrix()
        new_trans = -r.T @ trans
        new_quat = R.from_quat(quat).inv().as_quat()
        return new_quat, new_trans

    else:
        print("tf or quat/trans not specified in unmirored_marvin_output2open3d...")

def temp_debugging(rotation, translation, keep_original_world_coordinate=False):
    # rotation = qvec2rotmat(im_data.qvec)

    # translation = im_data.tvec.reshape(3, 1)
    w2c = np.concatenate([rotation, translation.reshape(3, 1)], 1)
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
    c2w = w2c #np.linalg.inv(w2c)
    # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
    c2w[0:3, 1:3] *= -1
    if not keep_original_world_coordinate:
        c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[2, :] *= -1
    return c2w

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
        self.camera_dict_nerfstudio={}

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
        data_nerf = copy.deepcopy(data[ "nerfstudio"])
        data = data["open3d"]

        cam_id_obj = CameraParams()
        cam_id_obj.set_fx_cx_fy_cy(data["intrinsics"]["fx"], data["intrinsics"]["cx"], 
                                   data["intrinsics"]["fy"], data["intrinsics"]["cy"],
                                   )
        cam_id_obj.set_height_width(data["intrinsics"]["height"], data["intrinsics"]["width"])
        cam_id_obj.set_tf(data["extrinsics"])
        self.camera_dict_nerfstudio[str(cam_id)] = {**data_nerf["intrinsics"], "transform_matrix": data_nerf["extrinsics"]}

        print(f"Sucessfully loaded cams in open3d world2cam format {cam_id}")

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
        
    
        
    def get_nerfstudio_format(self):
        temp_dict  = {}
        temp_dict["camera_model"] = "OPENCV"
        temp_dict["orientation_override"] = "none"
        temp_dict["frames"] = []

        frames = []  # Initialize frames as an empty list
        
        for x in self.cam_ids:
            # cam_params = self.camera_dict.get(str(x))
            # if cam_params:
            #     # Get OpenGL Projection Matrix
            #     # projection_matrix = open3d_intrinsics_to_opengl_projection(cam_params.fx,
            #     #                                                             cam_params.fy, 
            #     #                                                             cam_params.cx, 
            #     #                                                             cam_params.cy, cam_params.width, cam_params.height)

            #     tf = self.get_tf(str(x)) ## I think this is cam to world see output coordinate system

            #     # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c But according to this w2c

            #     # tf2 = unmirrored_marvin_output2open3d(tf.copy()) # == np.lingal.inv(tf)
            #     # visualize_coordinate_system(rotation_matrix=tf, save_name="nerf/vis_coordinate_system/"+str(x)+".txt")
            #     w2c = np.linalg.inv(tf)
            #     visualize_coordinate_system(rotation_matrix=w2c, save_name="nerf/vis_coordinate_system/"+str(x)+".txt")
            #     # https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/process_data/colmap_utils.py

            #     # 180-degree rotation matrix about the X-axis
            #     rot_x_180 = np.array([[1, 0, 0, 0],
            #                             [0, -1, 0, 0],
            #                             [0, 0, -1, 0],
            #                             [0, 0, 0, 1]])


            #     tf =  tf @ rot_x_180

            #     # tf[:3,:3] = flip_rotation_axes(tf[:3,:3], False, True, True)
            #     # tf = temp_debugging(rotation=tf[:3,:3], translation=tf[:3, 3], keep_original_world_coordinate=True)
            #     # tf[3, 0]  = tf[3, 0] * -1
            #     # if not 
            #     # tf[:3, 1] = tf[:3, 1]*-1 # y axis
            #     # tf[:3, 2] = tf[:3, 2]*-1 # z axis

            #     # tf[:3, 3] *= 1000  # Convert to mm
            #     # rotmatrix = flip_rotation_axes(tf[:3,:3].copy(), flip_x=False, flip_y=True, flip_z=True)
            #     # tf[:3,:3] = rotmatrix
            #     # view_matrix = tf

            #     # print("Projection Matrix:\n", projection_matrix)
            #     print("View Matrix:\n", view_matrix)


            #     frame = {
            #         "h": cam_params.height,
            #         "w": cam_params.width,
            #         "file_path": f"images/frame_{str(x+1).zfill(5)}.png",
            #         "fl_x": cam_params.fx,
            #         "fl_y": cam_params.fy,
            #         "cx": cam_params.cx,
            #         "cy": cam_params.cy,
            #         "transform_matrix": view_matrix.tolist()
            #         # "transform_matrix": cam_params.tf.tolist()

            #     }
            frame = self.camera_dict_nerfstudio[str(x)]
            frames.append(frame)
    
        temp_dict["frames"] = frames
        return temp_dict
    
    def to_colmap(self, xyz, save_folder=Path("")):
        params_list = []
        for x in self.cam_ids:
            x = str(x)
            import open3d as o3d
            params = o3d.camera.PinholeCameraParameters()
            fx, cx, fy, cy = self.get_fx_cx_fy_cy(x)
            height, width = self.get_height_width(x)
            intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

            # params.intrinsic.set_intrinsics(1080, 1920, fx, fy, cx, cy)
            params.intrinsic = intrinsics
            tf = self.get_tf(x)
            # tf[:3,3]*=1000 ## to convert to mm
            params.extrinsic = tf
            params_list.append(params)
        from scripts import calib2colmap
        calib2colmap.camera_params2colmap(params_list, xyz, save_folder)

def flip_rotation_axes(rotation_matrix, flip_x=False, flip_y=False, flip_z=False):
    """
    Flip the specified axes of a 3x3 rotation matrix.

    Args:
    rotation_matrix (np.array): The original 3x3 rotation matrix.
    flip_x (bool): Whether to flip the X-axis.
    flip_y (bool): Whether to flip the Y-axis.
    flip_z (bool): Whether to flip the Z-axis.

    Returns:
    np.array: The rotation matrix after flipping the specified axes.
    """
    flipped_matrix = rotation_matrix.copy()

    if flip_x:
        flipped_matrix[1:3, :] = -flipped_matrix[1:3, :]

    if flip_y:
        flipped_matrix[[0, 2], :] = -flipped_matrix[[0, 2], :]

    if flip_z:
        flipped_matrix[:, [0, 1]] = -flipped_matrix[:, [0, 1]]

    return flipped_matrix
        
def open3d_extrinsics_to_opengl_view(extrinsics):
    # Open3D uses right-handed: +Z forward, OpenGL: -Z forward
    flip_z = np.diag([1, -1, -1, 1])
    view_matrix = np.linalg.inv(extrinsics @ flip_z)
    return view_matrix

def open3d_intrinsics_to_opengl_projection(fx, fy, cx, cy, width, height, near=0.1, far=100.0):
    # fx, fy = intrinsics.get_focal_length()
    # cx, cy = intrinsics.get_principal_point()

    left = -cx * near / fx
    right = (width - cx) * near / fx
    bottom = -(height - cy) * near / fy
    top = cy * near / fy

    projection_matrix = np.array([
        [2 * near / (right - left), 0, (right + left) / (right - left), 0],
        [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]
    ])
    return projection_matrix


def create_points(axis, size, xyz):
    axis = axis / np.linalg.norm(axis)
    return np.linspace([0, 0, 0], [axis], 100).squeeze(1) * size + xyz

def visualize_coordinate_system(rotation_matrix, size=0.1, save_name=None, gray=False):
    import pandas as pd
    """visualize coordinates system, size is in meters, so with
    suize=0.1 x_axis is 10 """

    x_bgr = [0, 0, 255]
    y_bgr = [0, 255, 0]
    z_bgr = [255, 0, 0]

    if gray:
        x_bgr = [127, 127, 127]
        y_bgr = [127, 127, 127]
        z_bgr = [127, 127, 127]

    x_axis = rotation_matrix[:3, 0]
    y_axis = rotation_matrix[:3, 1]
    z_axis = rotation_matrix[:3, 2]

    if len(rotation_matrix[:, 0]) == 4:
        xyz = rotation_matrix[:3, 3]
    else:
        xyz = np.array([0, 0, 0])

    # print(matrix)
    x_points = create_points(x_axis, size, xyz)
    y_points = create_points(y_axis, size, xyz)
    z_points = create_points(z_axis, size, xyz)

    csv_matrix = np.zeros((300, 6), dtype=np.float32)
    csv_matrix[:100, :3] = x_points
    csv_matrix[:100, 3:] = x_bgr

    csv_matrix[100:200, :3] = y_points
    csv_matrix[100:200, 3:] = y_bgr

    csv_matrix[200:300, :3] = z_points
    csv_matrix[200:300, 3:] = z_bgr
    
    # for visualisatoin of extra cams in gray(paper)
    # csv_matrix[:, :3] = csv_matrix[:, :3]/1000
    # csv_matrix[:, 3:] = [127, 127, 127]


    df = pd.DataFrame(csv_matrix, columns=["X", "Y", "Z", "Blue", "Green", "Red"])
    if save_name is not None:
        df.to_csv(str(save_name), index=False)
    return df


if __name__=="__main__":
    obj = CameraClass(calib_folder="/home/agro/w-drive-vision/GARdata/datasets/tomato_plant_segmentation/20241223_csv_skeletons/camera_poses")
    obj.load_cams()