# utils_data.py
import numpy as np
import pandas as pd


def get_rotation_matrix(vector: np.ndarray, angle_deg: float):
	"""
	Generate a rotation matrix using Rodrigues' rotation formula.
	Args:
		vector (np.ndarray): Unit rotation axis as a 3D vector [ux, uy, uz].
		angle_deg (float): Rotation angle in degrees.
	Returns:
		np.ndarray: 3x3 rotation matrix.
	Example:
		>>> axis = np.array([0, 0, 1])
		>>> rotation_matrix = get_rotation_matrix(axis, 45)
	"""
	ux, uy, uz = vector.tolist()
	c = np.cos(angle_deg / 57.3)
	s = np.sin(angle_deg / 57.3)
	
	R = np.array([
			[c + ux**2*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
			[uy*ux*(1-c) + uz*s,  c + uy**2*(1-c),    uy*uz*(1-c) - ux*s],
			[uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s, c + uz**2*(1-c)]
		])
	return R