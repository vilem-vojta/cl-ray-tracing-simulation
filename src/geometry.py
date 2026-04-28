import numpy as np
from vector_math import rotate_vectors

def find_intersection(origins, directions, f=2.5e-3):
    """
    Calculates the intersection points of photon rays with the paraboloid mirror.
    The analytical solution solves the quadratic equation for the intersection 
    of a 3D line (ray) and a paraboloid surface: x = (y^2 + z^2) / (4f) - f.
    
    Args:
        origins (np.ndarray): Nx3 array of ray starting positions.
        directions (np.ndarray): Nx3 array of ray direction vectors.
        f (float): Focal length of the parabolic mirror in meters.
        
    Returns:
        tuple: (hit_points, valid_directions) containing only the rays that 
               successfully hit the physical mirror surface.
    """
    a = 1 / (4 * f)
    
    x0 = origins[:, 0]
    y0 = origins[:, 1]
    z0 = origins[:, 2]

    vx = directions[:, 0]
    vy = directions[:, 1]
    vz = directions[:, 2]

    # Coefficients for the quadratic equation: A*t^2 + B*t + C = 0
    A = a * (vy**2 + vz**2)
    B = 2 * a * (y0 * vy + z0 * vz) - vx
    C = a * (y0**2 + z0**2) - f - x0

    D = B**2 - 4 * A * C
    D = np.maximum(D, 0) # Prevent negative values due to floating-point inaccuracies

    sqrt_D = np.sqrt(D)
    
    # Calculate both possible roots of the quadratic equation
    t1 = (-B + sqrt_D) / (2 * A)
    t2 = (-B - sqrt_D) / (2 * A)

    # Keep only the valid forward intersections (t > 0)
    t1_valid = np.where(t1 > 0, t1, np.inf)
    t2_valid = np.where(t2 > 0, t2, np.inf)
    t_hit = np.minimum(t1_valid, t2_valid)

    hit_points = origins + t_hit[:, np.newaxis] * directions

    focus_dist = 0.5e-3
    hole_diam = 0.6e-3
    x_max = 13.25e-3

    hit_x = hit_points[:, 0]
    hit_y = hit_points[:, 1]
    hit_z = hit_points[:, 2]

    # Spatial mask: Constrains the infinite mathematical paraboloid to real physical dimensions
    valid_mask = (hit_z >= focus_dist) & \
                 ((hit_x**2 + hit_y**2) >= (hole_diam / 2)**2) & \
                 (hit_x <= (-f + x_max))
                 # 1. Truncates the bottom of the mirror
                 # 2. Accounts for the central aperture (e-beam hole)
                 # 3. Limits the maximum forward length of the mirror

    return hit_points[valid_mask], directions[valid_mask]


def reflect_rays(directions, hit_points, f=2.5e-3):
    """
    Computes the reflected directional vectors using the vectorial law of reflection.
    
    Args:
        directions (np.ndarray): Nx3 array of incident direction vectors.
        hit_points (np.ndarray): Nx3 array of intersection points on the mirror.
        f (float): Focal length of the parabolic mirror in meters.
        
    Returns:
        np.ndarray: Nx3 array of reflected direction vectors.
    """
    a = 1 / (4 * f)
    
    # Calculate the surface normal vectors using the gradient of the paraboloid equation
    ny = hit_points[:, 1] * 2 * a
    nz = hit_points[:, 2] * 2 * a
    nx = np.full(ny.shape, -1)

    normals = np.column_stack((nx, ny, nz))

    # Normalize the normal vectors to unit length
    lengths = np.sqrt(nx**2 + ny**2 + nz**2)
    normals_normalized = normals / lengths[:, np.newaxis]

    dot_product = np.sum(directions * normals_normalized, axis=1)

    # Apply the vectorial law of reflection: r = v - 2(v . n)n
    directions_reflected = directions - 2 * dot_product[:, np.newaxis] * normals_normalized

    return directions_reflected


def intersect_detector(hit_points, directions_reflected, x_det=0.5, det_radius=0.01, m_tilt_x_deg=0, m_tilt_y_deg=0, m_tilt_z_deg=0):
    """
    Calculates the intersection of the reflected rays with a planar circular detector.
    
    Args:
        hit_points (np.ndarray): Nx3 array of reflection points on the mirror.
        directions_reflected (np.ndarray): Nx3 array of reflected vectors.
        x_det (float): Distance of the detector plane along the x-axis.
        det_radius (float): Effective acceptance radius of the detector.
        m_tilt_x_deg, m_tilt_y_deg, m_tilt_z_deg (float): Mechanical misalignment angles.
        
    Returns:
        tuple: (y_det, z_det) coordinates of photons successfully hitting the active detector area.
    """
    # Rotate the reflection points and vectors back to the global laboratory coordinate system
    if m_tilt_x_deg or m_tilt_y_deg or m_tilt_z_deg:
        hit_points = rotate_vectors(hit_points, m_tilt_x_deg, m_tilt_y_deg, m_tilt_z_deg)
        directions_reflected = rotate_vectors(directions_reflected, m_tilt_x_deg, m_tilt_y_deg, m_tilt_z_deg)
    
    xh = hit_points[:, 0]
    yh = hit_points[:, 1]
    zh = hit_points[:, 2]
    
    vx = directions_reflected[:, 0]
    vy = directions_reflected[:, 1]
    vz = directions_reflected[:, 2]

    # Filter out rays traveling backwards or parallel to the detector plane
    forward_mask = vx > 1e-12
    xh, yh, zh = xh[forward_mask], yh[forward_mask], zh[forward_mask]
    vx, vy, vz = vx[forward_mask], vy[forward_mask], vz[forward_mask]

    # Parameter "t" for the intersection with the plane at x = x_det
    t_det = (x_det - xh) / vx
    y_det = yh + t_det * vy
    z_det = zh + t_det * vz

    # Apply the effective aperture mask (circular detector filter)
    dist_from_center = np.sqrt(y_det**2 + z_det**2)
    on_detector_mask = dist_from_center <= det_radius

    return y_det[on_detector_mask], z_det[on_detector_mask]