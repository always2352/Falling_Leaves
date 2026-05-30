import yaml
import argparse
import numpy as np
import torch
import warp as wp

from lbm_field import SimParams, LBMFlowField2D
from solid_solver import LeafRigidBody2D_Torch
import lbm_kernels as kernels

def clone_warp_array(src, requires_grad=None):
    return wp.clone(src, requires_grad=requires_grad)


def torch_grad_or_zero(grad, reference):
    if grad is None:
        return torch.zeros_like(reference)
    return grad


def wp_grad_or_zero(grad, reference):
    if grad is None:
        return wp.zeros_like(reference, requires_grad=False)
    return wp.clone(grad, requires_grad=False)


def wp_array_grad_to_torch(wp_array, reference_torch):
    if wp_array.grad is None:
        return torch.zeros_like(reference_torch)
    return wp.to_torch(wp_array.grad).clone()


def run_fsi_substep_with_history(
    fluid_field,
    solid_solver,
    nx,
    ny,
    dt,
    cl,
    ct,
    cf,
    u_wind,
    device_wp,
):
    pos_prev = solid_solver.c_pos.detach().clone().requires_grad_(True)
    vel_prev = solid_solver.c_vel.detach().clone().requires_grad_(True)
    angle_prev = solid_solver.angle.detach().clone().requires_grad_(True)
    omega_prev = solid_solver.omega.detach().clone().requires_grad_(True)

    solid_solver.c_pos = pos_prev
    solid_solver.c_vel = vel_prev
    solid_solver.angle = angle_prev
    solid_solver.omega = omega_prev

    v_pos_torch, v_vel_torch = solid_solver.get_boundary_kinematics()

    moments_pre_in = clone_warp_array(fluid_field.moments_pre, requires_grad=True)
    moments_post_buffer = clone_warp_array(fluid_field.moments_post, requires_grad=True)

    v_pos_wp = wp.from_torch(v_pos_torch.detach().contiguous(), dtype=wp.vec2, requires_grad=True)
    v_vel_wp = wp.from_torch(v_vel_torch.detach().contiguous(), dtype=wp.vec2, requires_grad=True)
    v_force_wp = wp.zeros(
        shape=(v_pos_torch.shape[0],),
        dtype=wp.vec2,
        device=device_wp,
        requires_grad=True,
    )

    current_pre = moments_pre_in
    current_post = moments_post_buffer

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=kernels.streaming_and_collision_kernel,
            dim=(nx, ny),
            inputs=[
                current_pre,
                current_post,
                fluid_field.flags,
                fluid_field.fluid_force,
                v_pos_wp,
                v_vel_wp,
                v_force_wp,
                solid_solver.edges_gpu,
                solid_solver.num_boundary_edges,
                nx,
                ny,
                fluid_field.omega,
                cl,
                ct,
                cf,
            ],
            device=device_wp,
        )

        current_pre, current_post = current_post, current_pre

        wp.launch(
            kernel=kernels.apply_outlet_inlet_kernel,
            dim=(nx, ny),
            inputs=[
                current_pre,
                fluid_field.flags,
                nx,
                ny,
                u_wind,
            ],
            device=device_wp,
        )

    force_torch_input = wp.to_torch(v_force_wp).detach().clone().requires_grad_(True)

    solid_solver.apply_forces_and_forward_rigid(force_torch_input, dt)

    fluid_field.moments_pre = current_pre
    fluid_field.moments_post = current_post

    return {
        "rigid_graph": {
            "inputs": (
                pos_prev,
                vel_prev,
                angle_prev,
                omega_prev,
                force_torch_input,
                solid_solver.com_offset,
            ),
            "outputs": (
                solid_solver.c_pos,
                solid_solver.c_vel,
                solid_solver.angle,
                solid_solver.omega,
            ),
        },
        "kinematics_graph": {
            "inputs": (
                pos_prev,
                vel_prev,
                angle_prev,
                omega_prev,
                solid_solver.com_offset,
            ),
            "outputs": (
                v_pos_torch,
                v_vel_torch,
            ),
        },
        "warp_graph": {
            "tape": tape,
            "v_pos_wp": v_pos_wp,
            "v_vel_wp": v_vel_wp,
            "v_force_wp": v_force_wp,
            "moments_pre_in": moments_pre_in,
            "moments_pre_out": current_pre,
        },
    }


def backward_through_history(history, loss, solid_solver, com_offset_torch):
    grad_c_pos_upstream, grad_c_vel_upstream, grad_angle_upstream, grad_omega_upstream = torch.autograd.grad(
        outputs=loss,
        inputs=(solid_solver.c_pos, solid_solver.c_vel, solid_solver.angle, solid_solver.omega),
        retain_graph=True,
        allow_unused=True,
    )

    grad_c_pos_upstream = torch_grad_or_zero(grad_c_pos_upstream, solid_solver.c_pos)
    grad_c_vel_upstream = torch_grad_or_zero(grad_c_vel_upstream, solid_solver.c_vel)
    grad_angle_upstream = torch_grad_or_zero(grad_angle_upstream, solid_solver.angle)
    grad_omega_upstream = torch_grad_or_zero(grad_omega_upstream, solid_solver.omega)

    grad_com_total = torch.zeros_like(com_offset_torch)
    fluid_grad_upstream = wp.zeros_like(history[-1]["warp_graph"]["moments_pre_out"], requires_grad=False)

    for step_idx in reversed(range(len(history))):
        step_data = history[step_idx]
        rigid_graph = step_data["rigid_graph"]
        kinematics_graph = step_data["kinematics_graph"]
        warp_graph = step_data["warp_graph"]

        rigid_grads = torch.autograd.grad(
            outputs=rigid_graph["outputs"],
            inputs=rigid_graph["inputs"],
            grad_outputs=(
                grad_c_pos_upstream,
                grad_c_vel_upstream,
                grad_angle_upstream,
                grad_omega_upstream,
            ),
            retain_graph=True,
            allow_unused=True,
        )

        grad_force_torch = torch_grad_or_zero(rigid_grads[4], rigid_graph["inputs"][4])
        grad_com_total = grad_com_total + torch_grad_or_zero(rigid_grads[5], com_offset_torch)

        grads = {
            warp_graph["v_force_wp"]: wp.from_torch(
                grad_force_torch.detach().contiguous(),
                dtype=wp.vec2,
                requires_grad=False,
            ),
            warp_graph["moments_pre_out"]: fluid_grad_upstream,
        }
        warp_graph["tape"].backward(grads=grads)

        grad_v_pos_torch = wp_array_grad_to_torch(warp_graph["v_pos_wp"], kinematics_graph["outputs"][0])
        grad_v_vel_torch = wp_array_grad_to_torch(warp_graph["v_vel_wp"], kinematics_graph["outputs"][1])

        kinematics_grads = torch.autograd.grad(
            outputs=kinematics_graph["outputs"],
            inputs=kinematics_graph["inputs"],
            grad_outputs=(grad_v_pos_torch, grad_v_vel_torch),
            retain_graph=(step_idx > 0),
            allow_unused=True,
        )

        grad_c_pos_upstream = torch_grad_or_zero(rigid_grads[0], rigid_graph["inputs"][0]) + torch_grad_or_zero(
            kinematics_grads[0], kinematics_graph["inputs"][0]
        )
        grad_c_vel_upstream = torch_grad_or_zero(rigid_grads[1], rigid_graph["inputs"][1]) + torch_grad_or_zero(
            kinematics_grads[1], kinematics_graph["inputs"][1]
        )
        grad_angle_upstream = torch_grad_or_zero(rigid_grads[2], rigid_graph["inputs"][2]) + torch_grad_or_zero(
            kinematics_grads[2], kinematics_graph["inputs"][2]
        )
        grad_omega_upstream = torch_grad_or_zero(rigid_grads[3], rigid_graph["inputs"][3]) + torch_grad_or_zero(
            kinematics_grads[3], kinematics_graph["inputs"][3]
        )
        grad_com_total = grad_com_total + torch_grad_or_zero(kinematics_grads[4], com_offset_torch)

        fluid_grad_upstream = wp_grad_or_zero(warp_graph["moments_pre_in"].grad, warp_graph["moments_pre_in"])

    com_offset_torch.grad = grad_com_total.detach().clone()
    return grad_com_total

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


def compute_optimization_loss(history, solid_solver, initial_pos, loss_cfg):
    final_pos = solid_solver.c_pos
    initial_pos_torch = torch.tensor(initial_pos, dtype=torch.float32, device=final_pos.device)

    x_transport = final_pos[0] - initial_pos_torch[0]
    y_drop = initial_pos_torch[1] - final_pos[1]

    omega_sq_terms = []
    angle_sq_terms = []
    for step_data in history:
        omega_k = step_data["rigid_graph"]["outputs"][3]
        angle_k = step_data["rigid_graph"]["outputs"][2]
        omega_sq_terms.append(omega_k * omega_k)
        angle_sq_terms.append(angle_k * angle_k)

    mean_omega_sq = torch.stack(omega_sq_terms).mean() if omega_sq_terms else torch.zeros((), device=final_pos.device)
    mean_angle_sq = torch.stack(angle_sq_terms).mean() if angle_sq_terms else torch.zeros((), device=final_pos.device)

    w_x_transport = float(loss_cfg.get("x_transport", 1.0))
    w_y_drop = float(loss_cfg.get("y_drop", 0.25))
    w_omega_reg = float(loss_cfg.get("omega_reg", 0.01))
    w_angle_reg = float(loss_cfg.get("angle_reg", 0.0))

    loss = (
        -w_x_transport * x_transport
        + w_y_drop * y_drop
        + w_omega_reg * mean_omega_sq
        + w_angle_reg * mean_angle_sq
    )

    components = {
        "x_transport": x_transport.detach(),
        "y_drop": y_drop.detach(),
        "mean_omega_sq": mean_omega_sq.detach(),
        "mean_angle_sq": mean_angle_sq.detach(),
        "w_x_transport": w_x_transport,
        "w_y_drop": w_y_drop,
        "w_omega_reg": w_omega_reg,
        "w_angle_reg": w_angle_reg,
    }
    return loss, components


def main_optimization():
    parser = argparse.ArgumentParser(description="LBM Falling Leaf FSI Simulation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim_cfg = cfg['simulation']
    fluid_cfg = cfg['fluid']
    leaf_cfg = cfg['leaf']
    out_cfg = cfg['output']
    opt_cfg = cfg.get('optimization', {})
    loss_cfg = opt_cfg.get('loss_weights', {})

    wp.init()
    use_cuda = wp.is_cuda_available() and torch.cuda.is_available()
    shared_device = "cuda:0" if use_cuda else "cpu"
    device_wp = wp.get_device(shared_device)
    device_torch = torch.device(shared_device)

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

    num_epochs = opt_cfg.get('epochs', 2)

    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        
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
        PRE_BLOW_STEPS = sim_cfg.get('pre_blow_steps', 400 * 5)
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
        solid_solver._generate_ellipse()
        solid_solver._compute_mass_properties()
        history = []

        print(f"Simulating {out_cfg['total_frames']} frames of leaf free fall...")
            
        for frame in range(out_cfg['total_frames']):

            for _ in range(out_cfg['substeps']):
                history.append(
                    run_fsi_substep_with_history(
                        fluid_field=fluid_field,
                        solid_solver=solid_solver,
                        nx=nx,
                        ny=ny,
                        dt=dt,
                        cl=cl,
                        ct=ct,
                        cf=cf,
                        u_wind=u_wind,
                        device_wp=device_wp,
                    )
                )

        # ==============================================================================
        # 6. Differentiable Optimization -- Backward Propagation
        # ==============================================================================
        loss, loss_components = compute_optimization_loss(
            history=history,
            solid_solver=solid_solver,
            initial_pos=leaf_cfg['initial_pos'],
            loss_cfg=loss_cfg,
        )
        grad_com = backward_through_history(history, loss, solid_solver, com_offset_torch)
        optimizer.step()

        grad_value = None if grad_com is None else grad_com.detach().cpu().numpy()
        print(
            f"loss={loss.item():.6f}, final_pos={solid_solver.c_pos.detach().cpu().numpy()}, "
            f"x_transport={loss_components['x_transport'].item():.6f}, "
            f"y_drop={loss_components['y_drop'].item():.6f}, "
            f"omega_reg={loss_components['mean_omega_sq'].item():.6f}, "
            f"com_offset={com_offset_torch.detach().cpu().numpy()}, grad={grad_value}"
        )

if __name__ == "__main__":
    main_optimization()