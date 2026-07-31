import struct
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from pathlib import Path

def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)

def camera_params2colmap(params_list, xyz, output_name=Path("")):
    # Assume we have:
    # params_list = [...]  # List of o3d.camera.PinholeCameraParameters
    # xyz = np.loadtxt("xyz.txt")  # 3D point cloud (Nx3)

    ### Writing `cameras.bin`
    # with open(output_name / "cameras.bin", "wb") as f:
    #     f.write(struct.pack("<I", len(params_list)))  # Number of cameras
    #     for i, params in enumerate(params_list):
    #         intrinsic = params.intrinsic
    #         width, height = intrinsic.width, intrinsic.height
    #         fx, fy = intrinsic.get_focal_length()
    #         cx, cy = intrinsic.get_principal_point()

    #         f.write(struct.pack("<I", i + 1))  # Camera ID
    #         f.write(struct.pack("<I", 1))  # Model ID (1 = PINHOLE)
    #         f.write(struct.pack("<II", width, height))  # Width, Height
    #         f.write(struct.pack("<dddd", fx, fy, cx, cy))  # Intrinsics

    with open(output_name / "cameras.bin", "wb") as fid:
        write_next_bytes(fid, len(params_list), "Q")
        # for _, cam in cameras.items():
        for i, params in enumerate(params_list):
            intrinsic = params.intrinsic
            width, height = intrinsic.width, intrinsic.height
            fx, fy = intrinsic.get_focal_length()
            cx, cy = intrinsic.get_principal_point()
            model_id = 1 #CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [i+1, model_id, width, height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in [fx, fy, cx, cy]:
                write_next_bytes(fid, float(p), "d")

    ### Writing `images.bin`
    with open(output_name /"images.bin", "wb") as f:
        f.write(struct.pack("<I", len(params_list)))  # Number of images
        for i, params in enumerate(params_list):
            extrinsic = params.extrinsic
            Rmat, t = extrinsic[:3, :3], extrinsic[:3, 3]

            # Convert R matrix to quaternion
            quat = R.from_matrix(np.array(Rmat)).as_quat()  # (qx, qy, qz, qw)
            qw, qx, qy, qz = quat[-1], quat[0], quat[1], quat[2]  # COLMAP order

            f.write(struct.pack("<I", i + 1))  # Image ID
            f.write(struct.pack("<ddddddd", qw, qx, qy, qz, t[0], t[1], t[2]))  # Pose
            f.write(struct.pack("<I", i + 1))  # Camera ID
            f.write(struct.pack("<I", 0))  # Number of 2D points (set to 0 for now)
            f.write(f"frame_{i+1:05d}.png".encode() + b"\x00")  # Image filename

    ### Writing `points3D.bin`
    with open(output_name /"points3D.bin", "wb") as f:
        f.write(struct.pack("<I", len(xyz)))  # Number of points
        for i, point in enumerate(xyz):
            x, y, z = point
            rgb = (255, 255, 255)  # Default white color
            error = 1.0  # Default reprojection error
            f.write(struct.pack("<Q", i))  # Point ID
            f.write(struct.pack("<ddd", x, y, z))  # XYZ
            f.write(struct.pack("<BBB", *rgb))  # Color
            f.write(struct.pack("<d", error))  # Error
            f.write(struct.pack("<I", 0))  # Number of observations (0 for now)