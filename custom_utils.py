from scipy.spatial.transform import Rotation as R
import numpy as np
import math 

def quaternion_to_rpy(quaternion):
    # Reorder quaternion from [w, x, y, z] to [x, y, z, w] for scipy.spatial.transform
    quaternion_reordered = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    # Convert to Euler angles (roll, pitch, yaw)
    r = R.from_quat(quaternion_reordered)
    rpy = r.as_euler('xyz', degrees=True)  # Euler angles in degrees
    return rpy

def average(lst):
    return sum(lst) / len(lst) if lst else 0  # Check to avoid division by zero

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)

