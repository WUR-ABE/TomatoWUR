from pathlib import Path
import pandas as pd
import subprocess
import tempfile
import numpy as np
import scripts.visualize_examples as ve

# TODO further testing of Adtree needed

def debug_adtree():
    adtree_build_folder = Path("./AdTree/build/")

    points = pd.read_csv(adtree_build_folder / "delaunay_points.txt", delimiter=" ", names=["index", "x", "y", "z", "lengthsubtree"])
    xyz = points[["x", "y", "z"]].values
    edges = pd.read_csv(adtree_build_folder / "delaunay_edges.txt", delimiter=" ", names=["edge1", "edge2", "nweight"])

    ve.vis(nodes=xyz, edges=edges[["edge1", "edge2"]], attributes={"lengthsubtree": points["lengthsubtree"].values}, edge_attributes={"nweight": edges["nweight"].values})


    points = pd.read_csv(adtree_build_folder / "mst_points.txt", delimiter=" ", names=["index", "x", "y", "z", "lengthsubtree"])
    xyz = points[["x", "y", "z"]].values
    distances = points["lengthsubtree"].values
    edges = pd.read_csv(adtree_build_folder / "mst_edges.txt", delimiter=" ", names=["edge1", "edge2", "nweight"])

    ve.vis(nodes=xyz, edges=edges[["edge1", "edge2"]], attributes={"lengthsubtree": distances}) #, edge_attributes={"nweight": edges["nweight"].values})

def run_adtree(adtree_path, xyz_array=np.random.rand(100,3), output_name: Path=Path("")):
    # point_cloud = np.random.rand(100, 3)  # Example point cloud with 100 points
    temp_xyz_path = output_name.parent / (output_name.stem + ".xyz")
    if xyz_array.shape[0] > 10000:
        sampled_indices = np.random.choice(xyz_array.shape[0], 10000, replace=False)
        xyz_array = xyz_array[sampled_indices]
    np.savetxt(temp_xyz_path, xyz_array, fmt="%.6f")

    # Run the AdTree command using subprocess
    adtree_command = [
        str(adtree_path),
        str(temp_xyz_path),
        str(output_name.parent / output_name.stem)
    ]
    subprocess.run(adtree_command)


    ##
    print("bart")

if __name__=="__main__":
    # run_adtree()
    debug_adtree()