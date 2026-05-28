import warp as wp
from dataclasses import dataclass

@dataclass
class SimParams:
    """A container for simulation parameters."""
    nx: int
    ny: int
    viscosity: float
    initial_density: float = 1.0
    # Add other parameters like boundary velocities as needed

class LBMFlowField2D:
    """
    Manages all GPU data arrays for a 2D LBM simulation.
    This class handles the state of the fluid field.
    """
    def __init__(self, params: SimParams, device):
        self.params = params
        self.device = device
        self.num_nodes = params.nx * params.ny

        # Grid dimensions
        self.nx = params.nx
        self.ny = params.ny
        self.viscosity = params.viscosity
        
        # Relaxation time (tau) and frequency (omega)
        self.tau = 3.0 * self.viscosity + 0.5
        self.omega = 1.0 / self.tau

        # --- GPU Data Arrays ---
        # Node flags (0: Fluid, 1: Wall, etc.)
        self.flags = wp.zeros(shape=(self.nx, self.ny), dtype=wp.int32, device=device)

        # We use a double-buffering scheme for moments
        # Moments are: [rho, ux, uy, pixx, piyy, pixy]
        num_moments = 6
        self.moments_pre = wp.zeros(shape=(self.nx, self.ny, num_moments), dtype=wp.float32, device=device)
        self.moments_post = wp.zeros(shape=(self.nx, self.ny, num_moments), dtype=wp.float32, device=device)
                
        # Fluid force acting on nodes (e.g., from immersed boundaries)
        self.fluid_force = wp.zeros(shape=(self.nx, self.ny, 2), dtype=wp.float32, device=device)

    def swap_moments(self):
        """Swaps the pre- and post-moment buffers for the next time step."""
        self.moments_pre, self.moments_post = self.moments_post, self.moments_pre

    def reset(self):
        """Resets all data arrays on the GPU to zero."""
        self.flags.zero_()
        self.moments_pre.zero_()
        self.moments_post.zero_()
        self.f_post_collision.zero_()
        self.fluid_force.zero_()