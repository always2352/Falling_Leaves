import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import yaml
import argparse

from lbm_constants import LBMD2Q9
from lbm_field import SimParams, LBMFlowField2D
from solid_solver import LeafRigidBody2D
import lbm_kernels as kernels

def init_fluid_state(field: LBMFlowField2D):
    """initialize fluid state (rho=1.0, u=0.0)"""
    wp.launch(
        kernel=kernels.initialize_kernel,
        dim=(field.nx, field.ny),
        inputs=[field.moments_pre, field.flags, field.params.initial_density],
        device=field.device
    )
    
    wp.launch(
        kernel=kernels.initialize_kernel,
        dim=(field.nx, field.ny),
        inputs=[field.moments_post, field.flags, field.params.initial_density],
        device=field.device
    )

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def main():
    parser = argparse.ArgumentParser(description="LBM Falling Leaf FSI Simulation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim_cfg = cfg['simulation']
    fluid_cfg = cfg['fluid']
    leaf_cfg = cfg['leaf']
    out_cfg = cfg['output']

    wp.init()
    device = wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded config from: {args.config}")

    # ==============================================================================
    # 1. Simulation Settings
    # ==============================================================================
    nx, ny = sim_cfg['nx'], sim_cfg['ny']
    dt = sim_cfg['dt']
    dx_lbm = sim_cfg['dx_lbm']
 
    params = SimParams(nx=nx, ny=ny, viscosity=fluid_cfg['viscosity'], initial_density=fluid_cfg['density'])
    
    # physical units to LBM units conversion factors
    cl = dx_lbm    # length scale
    ct = dt  # time scale
    cf = params.initial_density * cl**3 / ct**2  # force scale

    fluid_field = LBMFlowField2D(params, device)
    
    flags_np = np.zeros((nx, ny), dtype=np.int32)
    
    # ---------------------------------------------------------
    # set boundary conditions (Outlet)
    # ---------------------------------------------------------
    # Bottom Boundary (y = 0) -> Outlet Down
    flags_np[:, 0] = 4
    
    # Top Boundary (y = ny - 1) -> Outlet Up
    flags_np[:, ny-1] = 3
    
    # Left Boundary (x = 0) -> Outlet Left
    flags_np[0, 1:ny-1] = 5
    
    # Right Boundary (x = nx - 1) -> Outlet Right
    flags_np[nx-1, 1:ny-1] = 6
    
    fluid_field.flags.assign(flags_np)

    init_fluid_state(fluid_field)

    # ==============================================================================
    # 2. initialize Solid Solver
    # ==============================================================================
    leaf_b = leaf_cfg['a'] / leaf_cfg['aspect_ratio']
    solid_solver = LeafRigidBody2D(
        device=device,
        a=leaf_cfg['a'],             
        b=leaf_b,         
        num_pts=leaf_cfg['num_pts'],       
        rho=leaf_cfg['rho'],    
        com_offset=leaf_cfg['com_offset'], 
        initial_pos=leaf_cfg['initial_pos'],
        initial_rot_deg=leaf_cfg['initial_rot_deg']
    )

    solid_solver.gravity = np.array(sim_cfg['gravity'], dtype=np.float32)

    # ==============================================================================
    # 3. visualization (Matplotlib)
    # ==============================================================================
    output_dir = out_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 10))
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect('equal')
    
    # fluid velocity magnitude figure
    img = ax.imshow(np.zeros((ny, nx)), origin='lower', cmap='jet', vmin=0, vmax=0.3)
    
    # render solid boundary edges as lines
    lines = [ax.plot([], [], 'k-', lw=1.5)[0] for _ in range(solid_solver.num_boundary_edges)]

    com_scatter = ax.scatter([], [], color='red', s=12, edgecolor='black', linewidth=0.5, zorder=5, label='CoM')
    geom_scatter = ax.scatter([], [], color='cyan', s=12, edgecolor='black', linewidth=0.5, zorder=5, label='Geom Center')
    ax.legend(loc='upper right')

    sim_time = 0.0
    
    def update_plot(frame):
        nonlocal solid_solver, fluid_field, device, sim_time

        for _ in range(out_cfg['substeps']):

            # ------------------------------------------------------------------
            # Step A: solid solver update and give fluid solver the updated state
            # ------------------------------------------------------------------
            solid_solver.update_gpu_buffers()

            # ------------------------------------------------------------------
            # Step B: fluid solver update
            # ------------------------------------------------------------------
            wp.launch(
                kernel=kernels.streaming_and_collision_kernel,
                dim=(nx, ny),
                inputs=[
                    fluid_field.moments_pre,
                    fluid_field.moments_post,
                    fluid_field.flags,
                    fluid_field.fluid_force,
                    
                    solid_solver.v_pos_gpu,
                    solid_solver.v_vel_gpu,
                    solid_solver.v_force_gpu, 
                    solid_solver.edges_gpu,
                    solid_solver.num_boundary_edges,
                    
                    nx, ny,
                    fluid_field.omega,      
                    cl, ct, cf,
                ],
                device=device
            )

            # Swap LBM buffers for the next step
            fluid_field.swap_moments()
            
            wp.launch(
                kernel=kernels.apply_outlet_kernel,
                dim=(nx, ny),
                inputs=[
                    fluid_field.moments_pre,
                    fluid_field.flags,
                    nx, 
                    ny
                ],
                device=device
            )

            # ------------------------------------------------------------------
            # Step C: apply fluid forces to solid and forward rigid body dynamics
            # ------------------------------------------------------------------
            solid_solver.apply_forces_and_forward_rigid(dt)
            sim_time += dt

        # --- rendering ---
        # 1. fluid velocity magnitude
        moments_cpu = fluid_field.moments_pre.numpy() # Shape: (nx, ny, 6)
        ux = moments_cpu[:, :, 1]
        uy = moments_cpu[:, :, 2]
        speed = np.sqrt(ux**2 + uy**2)
        if np.any(np.isnan(speed)):
            print("\033[31mWarning: NaN values detected in speed field!\033[0m")
            exit(-1)
        img.set_data(speed.T) 

        # 2. rigid body edges
        v_pos_global = solid_solver.get_global_vertices()
        for i, edge in enumerate(solid_solver.boundary_edges):
            p1, p2 = v_pos_global[edge[0]], v_pos_global[edge[1]]
            lines[i].set_data([p1[0] / dx_lbm, p2[0] / dx_lbm], [p1[1] / dx_lbm, p2[1] / dx_lbm])

        # 3. Calculate Global Centers and Update Scatters
        R = solid_solver.get_rotation_matrix()
        com_global = solid_solver.c_pos
        geom_global = com_global - R @ solid_solver.com_offset

        com_grid = com_global / dx_lbm
        geom_grid = geom_global / dx_lbm

        com_scatter.set_offsets([com_grid])
        geom_scatter.set_offsets([geom_grid])

        ax.set_title(f"FSI Frame {frame} | Solid Pos: ({solid_solver.c_pos[0]:.1f}, {solid_solver.c_pos[1]:.1f})")
        frame_filename = os.path.join(output_dir, f"frame_{frame:04d}.png")
        fig.savefig(frame_filename, bbox_inches='tight')
        
        return [img, com_scatter, geom_scatter] + lines

    # ==============================================================================
    # 4. animation loop
    # ==============================================================================
    print("Starting simulation...")
    ani = FuncAnimation(fig, update_plot, frames=out_cfg['total_frames'], interval=50, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()