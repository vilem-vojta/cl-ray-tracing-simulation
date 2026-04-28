import numpy as np
import matplotlib.pyplot as plt
from simulation_engine import run_simulation

"""
Mirror degrees of freedom:

The xy-plane is parallel to the sample surface and perpendicular to the electron beam. 
The z-axis is parallel to the e-beam.
The mirror reflects the photons in the positive "x" direction towards the detector.

Roll  - movement in the yz-plane (twist around the main optical axis).
Pitch - movement in the xz-plane (tilt up/down towards the detector).
Yaw   - movement in the xy-plane (pan left/right, parallel to the sample surface).
"""

n_photons = 5000
x_det = 0.9 # m
det_radius = 0.0125 # m
m_tilt_x_deg = 0 # roll in degrees
m_tilt_y_deg = 0 # pitch in degrees
m_tilt_z_deg = 0 # yaw in degrees 
steps = 100
dist_from_origin = 166e-6/2 # m 

def multiple_simulations(
        n_photons,
        x_det,
        det_radius,
        m_tilt_x_deg,
        m_tilt_y_deg,
        m_tilt_z_deg,
        steps,
        dist_from_origin,
        show_plot=True
    ):
    """
    Executes a 2D spatial raster scan to generate a collection efficiency map.
    
    This function mimics the operation of an SEM by rastering the CL emission origin 
    across a defined X-Y grid (the Field of View) and evaluating the relative 
    photon collection efficiency at each discrete spatial coordinate.

    Args:
        n_photons (int): Number of rays generated per spatial grid point.
        x_det (float): Axial distance of the detector from the origin in meters.
        det_radius (float): Effective acceptance radius of the detector in meters.
        m_tilt_x_deg (float): Roll misalignment angle of the mirror in degrees.
        m_tilt_y_deg (float): Pitch misalignment angle of the mirror in degrees.
        m_tilt_z_deg (float): Yaw misalignment angle of the mirror in degrees.
        steps (int): Grid resolution (number of evaluation points along one axis).
        dist_from_origin (float): Half-width of the simulated field of view in meters.
        show_plot (bool): If True, renders the resulting efficiency map using Matplotlib.

    Returns:
        np.ndarray: A 2D array (steps x steps) containing the calculated efficiencies.
    """
    
    # Generate the spatial grid for the Field of View (FOV)
    x_range = np.linspace(-dist_from_origin, dist_from_origin, steps)
    y_range = np.linspace(-dist_from_origin, dist_from_origin, steps)
    efficiency_map = np.zeros((steps, steps))

    # Raster scan: Iterating through every spatial coordinate in the FOV
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            # The z-coordinate remains 0, ensuring constant sample focus/height
            _, _, eff = run_simulation(
                n_photons,
                x_det,
                det_radius,
                starting_position=[x, y, 0],
                m_tilt_x_deg=m_tilt_x_deg,
                m_tilt_y_deg=m_tilt_y_deg,
                m_tilt_z_deg=m_tilt_z_deg,
                create_plot=False,
                verbose=False
            )
            efficiency_map[i, j] = eff

    if show_plot:
        # Convert the extent limits from meters to micrometers for plotting
        extent_um = [-dist_from_origin*1e6, dist_from_origin*1e6, -dist_from_origin*1e6, dist_from_origin*1e6]
        
        # Normalize the map so the maximum efficiency found equals 100%
        max_eff = np.max(efficiency_map)
        if max_eff > 0:
            relative_eff = (efficiency_map / max_eff) * 100
        else:
            relative_eff = efficiency_map
            
        # Render the 2D spatial map
        plt.figure(figsize=(10, 8))
        plt.imshow(relative_eff, extent=extent_um, origin="lower", cmap="viridis")
        
        cbar = plt.colorbar()
        cbar.set_label("Relative efficiency [%]", fontsize=20, fontweight="bold")
        cbar.ax.tick_params(labelsize=20)
        cbar.set_ticks(np.linspace(20, 100, 5)) # Fixed ticks (20, 40, 60, 80, 100)
        
        plt.title(f"Collection efficiency - simulation\nroll: {m_tilt_x_deg}°, pitch: {m_tilt_y_deg}°, yaw: {m_tilt_z_deg}°", fontsize=22, fontweight="bold")
        plt.xlabel("x [µm]", fontsize=20, fontweight="bold")
        plt.ylabel("y [µm]", fontsize=20, fontweight="bold")
        plt.tick_params(axis="both", labelsize=20)
        
        plt.show()

    return efficiency_map

if __name__ == "__main__":
    multiple_simulations(
        n_photons,
        x_det,
        det_radius,
        m_tilt_x_deg,
        m_tilt_y_deg,
        m_tilt_z_deg,
        steps,
        dist_from_origin,
        show_plot=True
    )