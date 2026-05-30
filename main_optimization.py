import os
import yaml
import argparse
import numpy as np
import torch
import warp as wp

from lbm_constants import LBMD2Q9
from lbm_field import SimParams, LBMFlowField2D
from solid_solver import LeafRigidBody2D, LeafRigidBody2D_Torch
import lbm_kernels as kernels

class WarpToTorchBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, warp_array, wp_tape):
        """
        Warp to PyTorch GPU Tensor
        """
        ctx.warp_array = warp_array
        ctx.wp_tape = wp_tape
        
        torch_tensor = wp.to_torch(warp_array)
        return torch_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Warp to Pytorch
        """
        warp_array = ctx.warp_array
        
        if grad_output is not None:
            wp_grad_view = wp.from_torch(grad_output.contiguous(), dtype=wp.vec2)
            
            warp_array.grad = wp.zeros_like(warp_array)
            wp.array_copy(dest=warp_array.grad, src=wp_grad_view)
            
        return None, None

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
    
def main_optimization():
    parser = argparse.ArgumentParser(description="LBM Falling Leaf FSI Simulation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim_cfg = cfg['simulation']
    fluid_cfg = cfg['fluid']
    leaf_cfg = cfg['leaf']
    out_cfg = cfg['output']

    wp.init()
    device_wp = wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")
    device_torch = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    com_offset_torch = torch.tensor([0.05, 0.0], dtype=torch.float32, device=device_torch, requires_grad=True)
    optimizer = torch.optim.Adam([com_offset_torch], lr=0.01)

    for epoch in range(2):
        
        # ==============================================================================
        # 2. instantiate Fluid Field
        # ==============================================================================
        fluid_field = LBMFlowField2D(params, device_wp)
        flags_np = np.zeros((nx, ny), dtype=np.int32)
        flags_np[:, 0] = 4
        flags_np[:, ny-1] = 3      
        flags_np[0, 1:ny-1] = 5    
        flags_np[nx-1, 1:ny-1] = 6      
        fluid_field.flags.assign(flags_np)
        init_fluid_state(fluid_field)

        # ==============================================================================
        # 3. initialize Solid Solver
        # ==============================================================================
        leaf_b = leaf_cfg['a'] / leaf_cfg['aspect_ratio']
        solid_solver = LeafRigidBody2D_Torch(
            device_torch=device_torch,
            device_wp=device_wp,
            a=leaf_cfg['a'],             
            b=leaf_b,         
            num_pts=leaf_cfg['num_pts'],       
            rho=leaf_cfg['rho'],    
            com_offset=com_offset_torch,
            initial_pos=leaf_cfg['initial_pos'],
            initial_rot_deg=leaf_cfg['initial_rot_deg']
        )
        solid_solver.gravity = torch.tensor(sim_cfg['gravity'], dtype=torch.float32, device=device_torch)
        u_wind = fluid_cfg['crosswind_velocity']

        # ==============================================================================
        # 4. Flow Field Pre-blowing
        # ==============================================================================
        PRE_BLOW_STEPS = 400 * 5
        print(f"\n--- Epoch {epoch:02d} ---")
        print(f"Pre-blowing fluid field with crosswind for {PRE_BLOW_STEPS} steps...")
        
        for _ in range(PRE_BLOW_STEPS):
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
                device=device_wp
            )
            fluid_field.swap_moments()
            wp.launch(
                kernel=kernels.apply_outlet_inlet_kernel,
                dim=(nx, ny),
                inputs=[
                    fluid_field.moments_pre, 
                    fluid_field.flags, 
                    nx, ny,
                    u_wind],
                device=device_wp
            )

        # ==============================================================================
        # 5. Differentiable Optimization -- Forward Calculation
        # ==============================================================================
        history = []
        solid_solver._generate_ellipse()
        solid_solver._compute_mass_properties()

        print(f"Simulating {out_cfg['total_frames']} frames of leaf free fall...")
            
        for frame in range(out_cfg['total_frames']):

            for _ in range(out_cfg['substeps']):
                solid_solver.update_gpu_buffers()

                substep_tape = wp.Tape()
                with substep_tape:
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
                        device=device_wp
                    )

                    fluid_field.swap_moments()

                    wp.launch(
                        kernel=kernels.apply_outlet_inlet_kernel,
                        dim=(nx, ny),
                        inputs=[
                            fluid_field.moments_pre,
                            fluid_field.flags,
                            nx, 
                            ny,
                            u_wind
                        ],
                        device=device_wp
                    )

                v_force_torch = WarpToTorchBridge.apply(solid_solver.v_force_gpu, substep_tape)

                pos_prev_node = solid_solver.c_pos.clone()
                vel_prev_node = solid_solver.c_vel.clone()
                ang_prev_node = solid_solver.angle.clone()
                omg_prev_node = solid_solver.omega.clone()

                solid_solver.apply_forces_and_forward_rigid(v_force_torch, dt)

                pos_next_node = solid_solver.c_pos.clone()
                vel_next_node = solid_solver.c_vel.clone()
                ang_next_node = solid_solver.angle.clone()
                omg_next_node = solid_solver.omega.clone()

                history.append({
                    'wp_tape': substep_tape, 
                    'v_force_wp_ref': solid_solver.v_force_gpu,
                    'torch_graph': {
                        'inputs': (pos_prev_node, vel_prev_node, ang_prev_node, omg_prev_node, v_force_torch),
                        'outputs': (pos_next_node, vel_next_node, ang_next_node, omg_next_node)
                    }
                })

        #######  Need to check, currently grad seems like not connected completely #######
        
        # ==============================================================================
        # 6. Differentiable Optimization -- Backward Propagation
        # ==============================================================================

        # optimizer.zero_grad()  

        # grad_c_pos_upstream = torch.tensor([0.0, -1.0], device=device_torch)
        # grad_c_vel_upstream = torch.zeros(2, device=device_torch)
        # grad_angle_upstream = torch.tensor(0.0, device=device_torch)
        # grad_omega_upstream = torch.tensor(0.0, device=device_torch)

        # for step_idx in reversed(range(len(history_history))):
        #     step_data = history_history[step_idx]

        #     torch_sub_graph = step_data['torch_graph']

        #     sub_grads = torch.autograd.grad(
        #         outputs=torch_sub_graph['outputs'],
        #         inputs=torch_sub_graph['inputs'],
        #         grad_outputs=(grad_c_pos_upstream, grad_c_vel_upstream, grad_angle_upstream, grad_omega_upstream),
        #         allow_unused=True
        #     )

        #     grad_c_pos_upstream = sub_grads[0] if sub_grads[0] is not None else torch.zeros_like(grad_c_pos_upstream)
        #     grad_c_vel_upstream = sub_grads[1] if sub_grads[1] is not None else torch.zeros_like(grad_c_vel_upstream)
        #     grad_angle_upstream = sub_grads[2] if sub_grads[2] is not None else torch.zeros_like(grad_angle_upstream)
        #     grad_omega_upstream = sub_grads[3] if sub_grads[3] is not None else torch.zeros_like(grad_omega_upstream)

        #     current_f_grad_torch = sub_grads[4] if sub_grads[4] is not None else torch.zeros_like(v_force_torch)
            
        #     wp_grad_view = wp.from_torch(current_f_grad_torch.contiguous(), dtype=wp.vec2)
        #     wp.array_copy(dest=step_data['grad_holder'], src=wp_grad_view)

        #     step_data['v_force_gpu_ref'].grad = step_data['grad_holder']
        #     step_data['wp_tape'].backward()

        # solid_solver.com_offset.grad = torch.autograd.grad(
        #     outputs=solid_solver.angle, inputs=solid_solver.com_offset, 
        #     grad_outputs=grad_angle_upstream, allow_unused=True
        # )[0]

        # print(f" Final Optimization Gradient Successfully Formed: {solid_solver.com_offset.grad.cpu().numpy()}")

if __name__ == "__main__":
    main_optimization()