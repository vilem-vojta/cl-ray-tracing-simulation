import numpy as np

def emission(n):
    """
    Generate random unit vectors distributed uniformly on a hemisphere.

    Args:
        n (int): number of unit vectors

    Returns:
        np.ndarray: Nx3 array of unit vectors.
    """
    phi = np.random.uniform(0, 2 * np.pi, size = n)
    cos_theta = np.random.uniform(0, 1, size = n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    
    vectors = np.column_stack((x, y, z))

    return vectors

def rotate_vectors(vectors, m_tilt_x_deg, m_tilt_y_deg, m_tilt_z_deg):
    """
    Applies 3D rotation matrices to a set of vectors.

    Args:
        vectors (np.ndarray): Nx3 array of photon trajectories or hit points.
        m_tilt_x_deg (float): Roll angle in degrees.
        m_tilt_y_deg (float): Pitch angle in degrees.
        m_tilt_z_deg (float): Yaw angle in degrees.
    
    Returns:
        np.ndarray: Rotated vectors.
    """
    x_rad = np.radians(m_tilt_x_deg)
    y_rad = np.radians(m_tilt_y_deg)
    z_rad = np.radians(m_tilt_z_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x_rad), -np.sin(x_rad)],
        [0, np.sin(x_rad), np.cos(x_rad)]
    ])

    Ry = np.array([
        [np.cos(y_rad), 0, np.sin(y_rad)],
        [0, 1, 0],
        [-np.sin(y_rad), 0, np.cos(y_rad)]
    ])

    Rz = np.array([
        [np.cos(z_rad), -np.sin(z_rad), 0],
        [np.sin(z_rad),  np.cos(z_rad), 0],
        [0, 0, 1]
    ])

    #the order of the rotations matters, but for small angles, the difference is negligible
    R = Rz @ Ry @ Rx
    
    return vectors @ R.T