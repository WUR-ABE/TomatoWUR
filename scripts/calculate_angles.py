## angles
## partly copied from plantscan3d angleanalyses.py

import numpy as np
from math import degrees


def openalea_method(poses: np.ndarray, lateral_roots: list):
    lateral_lines = [Line.estimate(poses, lr[1]) for lr in lateral_roots]
    phyto_angle = [mangle(l.dir) for l in lateral_lines]
    # phyto_angle = phylo_angles(branches=lateral_lines)
    # calculate relative angles as in ground truth measurements:
    relangles = relative_angles(phyto_angle, ccw=True)
    rel_angle_index = [x[0] for x in lateral_roots][1:]

    return phyto_angle, relangles, rel_angle_index
		
def xy_plane_method(poses: np.ndarray, lateral_roots: list):
    """
    Calculate angles and relative angles for given poses and lateral roots.
    This method computes the angles of lateral roots in the xy-plane and their relative angles.
    Parameters:
    poses (numpy.ndarray): An array of positions.
    lateral_roots (list): A list of list, where each tuple contains indices of lateral roots.
    Returns:
    tuple: A tuple containing:
        - phyto_angle (list): A list of angles of the lateral roots in the xy-plane.
        - relangles (list): A list of relative angles between consecutive lateral roots.
        - rel_angle_index (list): A list of indices corresponding to the relative angles.
    """
    ref_vector = np.array([1,0]) # represents x-axis
    branch_points =  np.array([poses[lr[1][1]] for lr in lateral_roots])
    parent_points = np.array([poses[lr[1][0]] for lr in lateral_roots])


    lines = branch_points - parent_points
    lateral_lines = np.divide(lines.T, np.linalg.norm(lines,axis=1)).T

    dot_product = np.clip(np.dot(lateral_lines[:,:2], ref_vector), -1.0, 1.0) #[:,:2] # only because we only focus on xy component

    ## TODO ASK GERT
    lateral_lines_2d = np.divide(lines[:,:2].T, np.linalg.norm(lines[:,:2],axis=1)).T
    dot_product = np.clip(np.dot(lateral_lines_2d, ref_vector), -1.0, 1.0)

    ## for debugging visualise lines:
    temp = np.zeros(lateral_lines.shape)
    temp[:,:2] = lateral_lines[:,:2]
    xy_line_points = parent_points + 0.1*temp
    xy_points = np.vstack([parent_points, xy_line_points])
    xy_edges = np.array([(i, len(parent_points)+i) for i in range(len(parent_points))])



    angles = np.arccos(dot_product)
    def cross_product_2d(a, b):
        return a[0] * b[1] - a[1] * b[0]
    ## rectify orientation
    for i, angle in enumerate(angles):
        temp = cross_product_2d(lateral_lines[i,:2], ref_vector)
        if temp > 0: #counter clockwise
            pass
        else:
            angles[i] = -1*angle
    phyto_angle = np.rad2deg(angles)

    # calculate relative angles similar as ground truth measurements
    relangles = relative_angles(phyto_angle, ccw=True)
    rel_angle_index = [x[0] for x in lateral_roots][1:]

    return phyto_angle, relangles, rel_angle_index, xy_points, xy_edges


def direction(v):
	return v / np.linalg.norm(v)


def angle(v1, v2, axis=None):
    """
    Calculates the angle between two vectors in 2D or 3D space.
    
    Parameters:
        v1 (array-like): First vector.
        v2 (array-like): Second vector.
        axis (array-like, optional): Axis for the angle computation in 3D space (only relevant for 3D vectors).
    
    Returns:
        float: The angle in radians between v1 and v2.
    """
    # # Example usage:
    # v1_2d = [1, 0]
    # v2_2d = [0, 1]
    # print("Angle between v1 and v2 in 2D:", angle(v1_2d, v2_2d))  # Should be π/2 radians or 90 degrees

    # v1_3d = [1, 0, 0]
    # v2_3d = [0, 1, 0]
    # axis = [0, 0, 1]
    # print("Angle between v1 and v2 in 3D around axis:", angle(v1_3d, v2_3d, axis))  # Should be π/2 radians or 90 degrees

    # plantscan3d frequenlty (vector_branch, [1,0, 0], [0, 0, 1])

    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Normalize the vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        raise ValueError("Input vectors must not be zero.")
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    # Compute the dot product
    dot_product = np.dot(v1, v2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure within [-1, 1] to handle numerical issues

    # Compute the angle
    angle_radians = np.arccos(dot_product)

    # Handle axis if provided (for 3D vectors)
    if axis is not None:
        if len(v1) != 3 or len(v2) != 3 or len(axis) != 3:
            raise ValueError("For axis-based calculations, all vectors must be 3D.")
        
        axis = np.array(axis)
        cross_product = np.cross(v1, v2)
        if np.dot(cross_product, axis) < 0:
            angle_radians = -angle_radians
    
    return angle_radians


def relative_angles(angles, ccw=True):
    ## counter clock wise [ccw] = positive between angles, otherwise negative

    lastangle = angles[0]
    relangles = []
    for a in angles[1:]:
        if ccw:
            while a < lastangle:
                a += 360
            ra = a-lastangle
        else:
            while a > lastangle:
                a -= 360
            ra = lastangle-a            
        relangles.append(ra)
        lastangle = a
    return relangles


class Line:
    def __init__(self, pos, dir, extend):
        self.pos = pos
        self.dir = dir
        self.extend = extend
    def __repr__(self):
        return 'Line('+str(self.pos)+','+str(self.dir)+','+str(self.extend)+')'
    
    @staticmethod
    def estimate(lpoints, idx):
        # lpoints = [positions[n] for n in nodes]
        # idx = list(range(len(nodes)))
        # pos = centroid_of_group(lpoints,idx)
        pos = np.mean(lpoints[idx], axis=0)

        dir = direction(pointset_orientation_vpython(lpoints,idx))
        # if np.dot(pos-positions[nodes[0]],dir) < 0: dir *= -1
        if np.dot(pos-lpoints[idx[0]],dir) < 0: dir *= -1

        extend = max([abs(np.dot(p-pos,dir)) for p in lpoints])
        return Line(pos,dir,extend)


def pointset_orientation_vpython(points, indices=None):
    """
    Computes the principal axis of variation in a set of points.

    Parameters:
        points (list or np.ndarray): List of points in 2D or 3D space.
        indices (list, optional): Indices of points to include.

    Returns:
        np.ndarray: The principal axis (eigenvector) of the points.
    """
    if indices is not None:
        points = np.array(points)[indices]
    else:
        points = np.array(points)

    # Center the points
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point

    # Compute the covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)

    # Find the principal axis
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    return principal_axis

def mangle(d, rd=np.array([1, 0, 0]), td=np.array([0,0,1])):
    # red=refdir
    # td=trunkdir
    a = angle(d,rd,td)
    da = degrees(a)
    return da

def phylo_angles(trunk=None, branches=None):
    # refdir = np.array([1,0,0])
    # trunkdir = np.array([0,0,1])
    return [mangle(l.dir) for l in branches]

    # if isinstance(trunk,Line):
    #     trunkdir = trunk.dir
    #     refdir = get_ref_dir(trunkdir) 
    #     print('Angle taken from initial direction', refdir,'rotating around', trunkdir)
    #     return [mangle(l.dir,refdir,trunkdir) for l in branches]
    # else:
    #     result = []
    #     for l in branches:
    #         initpos = l.pos-l.dir*l.extend
    #         cp, u = trunk.findClosest(initpos)
    #         trunkdir = trunk.getTangentAt(u)
    #         refdir = get_ref_dir(trunkdir)
    #         result.append(mangle(l.dir,refdir,trunkdir))
        # return result