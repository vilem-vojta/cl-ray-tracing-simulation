import numpy as np
import matplotlib.pyplot as plt
from geometry import find_intersection, reflect_rays, intersect_detector
from vector_math import emission, rotate_vectors

def run_simulation(n_photons, x_det,
                   det_radius,
                   starting_position=(0, 0, 0),
                   m_tilt_x_deg=0,
                   m_tilt_y_deg=0,
                   m_tilt_z_deg=0, 
                   create_plot=True,
                   verbose=True):
    """
    Executes the core Monte Carlo ray-tracing simulation for a single emission point.
    
    This function generates isotropic photon trajectories, simulates their interaction 
    with a misaligned parabolic mirror, and calculates the final collection efficiency 
    at the downstream detector plane.

    Args:
        n_photons (int): Total number of photons to emit.
        x_det (float): Distance to the detector plane along the optical x-axis (meters).
        det_radius (float): Effective acceptance radius of the detector (meters).
        starting_position (list): [x, y, z] coordinates of the emission origin.
        m_tilt_x_deg (float): Mirror roll misalignment in degrees.
        m_tilt_y_deg (float): Mirror pitch misalignment in degrees.
        m_tilt_z_deg (float): Mirror yaw misalignment in degrees.
        create_plot (bool): If True, renders a 2D scatter plot of the detector hits.
        verbose (bool): If True, prints step-by-step efficiency metrics to the console.
        
    Returns:
        tuple: (hit_points, y_final, efficiency)
            - hit_points (np.ndarray): Coordinates of valid mirror reflections.
            - y_final (np.ndarray): y-coordinates of photons hitting the detector.
            - efficiency (float): Overall collection efficiency in percent.
    """
    if verbose:
        print(f"Running simulation with {n_photons} photons.")

    # Generate the initial emission origin for all photons
    origins = np.array([starting_position] * n_photons)
    
    # Generate isotropic directional vectors (restricted to the upper hemisphere)
    directions = emission(n_photons)

    # Inverse rotation: We rotate the emission source instead of the complex mirror surface.
    # This mathematical trick allows us to use the standard, unrotated paraboloid equation 
    # for the analytical intersection checks, vastly improving computational efficiency.
    if m_tilt_x_deg or m_tilt_y_deg or m_tilt_z_deg:
        origins = rotate_vectors(origins, -m_tilt_x_deg, -m_tilt_y_deg, -m_tilt_z_deg)
        directions = rotate_vectors(directions, -m_tilt_x_deg, -m_tilt_y_deg, -m_tilt_z_deg)

    # Calculate analytical intersections with the parabolic mirror
    hit_points, directions_in = find_intersection(origins, directions)
    
    efficiency_hit = len(hit_points) / n_photons * 100
    if verbose:
        print(f"Mirror hit by {len(hit_points)} photons - {efficiency_hit:.2f} %.")

    # Early exit if the beam entirely misses the mirror
    if len(hit_points) == 0:
        if verbose: print("Zero photons hit the mirror")
        return np.array([]), np.array([]), 0.0

    # Compute the reflected vectors using the vectorial law of reflection
    directions_out = reflect_rays(directions_in, hit_points)

    # Forward rotation back to the global laboratory frame is handled inside intersect_detector,
    # followed by the evaluation of effective aperture limits.
    y_final, z_final = intersect_detector(hit_points, directions_out, x_det, det_radius, m_tilt_x_deg, m_tilt_y_deg, m_tilt_z_deg)
    
    # Final detection efficiency calculation
    efficiency = (len(y_final) / n_photons) * 100
    if verbose:
        print(f"Detector hit by {len(y_final)} photons. Total efficiency: {efficiency:.2f} %")

    # Render the 2D scatter plot representing the physical detector surface
    if create_plot:
        plt.figure(figsize=(8, 8))
        # Multiply by 1000 to convert spatial coordinates from meters to millimeters
        plt.scatter(y_final * 1000, z_final * 1000, s=0.5, alpha=0.5, color="blue")
        
        # Plot the effective detector boundary (acceptance aperture ring)
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(det_radius * 1000 * np.cos(theta), det_radius * 1000 * np.sin(theta), color="red", linestyle="--")

        plt.title(f"Detector scatter plot: roll = {m_tilt_x_deg}°, pitch = {m_tilt_y_deg}°, yaw = {m_tilt_z_deg}°\nCollection efficiency: {efficiency:2.0f} %", fontsize=15, fontweight="bold")
        plt.xlabel("y [mm]", fontsize=15, fontweight="bold")
        plt.ylabel("z [mm]", fontsize=15, fontweight="bold")
        plt.tick_params(axis="both", labelsize=14)
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    return hit_points, y_final, efficiency

if __name__ == "__main__":
    run_simulation(
        n_photons=100000, 
        x_det=0.3, 
        det_radius=0.01,
        m_tilt_x_deg=0,
        m_tilt_y_deg=0,
        m_tilt_z_deg=0
    )
