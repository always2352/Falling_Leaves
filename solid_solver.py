import os
import numpy as np
import torch
import warp as wp

class LeafRigidBody2D:
    def __init__(self, device, a, b, num_pts, rho, com_offset, initial_pos, initial_rot_deg):
        self.device = device
        self.rho = rho
        self.a = a
        self.b = b
        self.com_offset = np.array(com_offset, dtype=np.float32)
        self.num_vertices = num_pts
        self.num_boundary_edges = num_pts

        self.c_pos = np.array(initial_pos, dtype=np.float32)
        self.c_vel = np.zeros(2, dtype=np.float32)
        self.angle = np.deg2rad(initial_rot_deg)
        self.omega = 0.0

        self._generate_ellipse()
        self._compute_mass_properties()

        self.v_force_global = np.zeros_like(self.v_pos_local)
        self.sum_force_global = np.zeros(2, dtype=np.float32)
        self.sum_torque_local = 0.0
        self.gravity = np.zeros(2, dtype=np.float32)

        self.v_pos_gpu = wp.zeros(shape=(self.num_vertices,), dtype=wp.vec2, device=device)
        self.v_vel_gpu = wp.zeros(shape=(self.num_vertices,), dtype=wp.vec2, device=device)
        self.v_force_gpu = wp.zeros(shape=(self.num_vertices,), dtype=wp.vec2, device=device)
        self.edges_gpu = wp.array(self.boundary_edges.flatten(), dtype=wp.int32, device=device)

        self.update_gpu_buffers()

    def _generate_ellipse(self):
        t = np.linspace(0, 2 * np.pi, self.num_vertices, endpoint=False)
        x_local = self.a * np.cos(t) - self.com_offset[0]
        y_local = self.b * np.sin(t) - self.com_offset[1]

        self.v_rest = np.column_stack((x_local, y_local)).astype(np.float32)
        self.v_pos_local = self.v_rest.copy()

        edges = [[i, (i + 1) % self.num_vertices] for i in range(self.num_vertices)]
        self.boundary_edges = np.array(edges, dtype=np.int32)

    def _compute_mass_properties(self):
        area = np.pi * self.a * self.b
        self.mass = self.rho * area

        # mass is evenly distributed on the boundary node
        self.v_mass = np.full(self.num_vertices, self.mass / self.num_vertices, dtype=np.float32)
        self.inertia = self.mass * (self.a**2 + self.b**2) / 4.0

        # Parallel Axis Theorem
        i_geom = self.mass * (self.a**2 + self.b**2) / 4.0
        offset_sq_distance = self.com_offset[0]**2 + self.com_offset[1]**2
        self.inertia = i_geom + self.mass * offset_sq_distance

    def get_rotation_matrix(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([[c, -s], [s, c]])
   
    def get_global_vertices(self):
        R = self.get_rotation_matrix()
        return (R @ self.v_pos_local.T).T + self.c_pos

    def update_gpu_buffers(self):
        R = self.get_rotation_matrix()
        v_pos_global = (R @ self.v_pos_local.T).T + self.c_pos

        v_rot_local = np.zeros_like(self.v_pos_local)
        v_rot_local[:, 0] = -self.omega * self.v_pos_local[:, 1]
        v_rot_local[:, 1] = self.omega * self.v_pos_local[:, 0]
        v_vel_global = self.c_vel + (R @ v_rot_local.T).T

        self.v_pos_gpu.assign(v_pos_global.astype(np.float32))
        self.v_vel_gpu.assign(v_vel_global.astype(np.float32))
        self.v_force_gpu.zero_()
   
    def apply_forces_and_forward_rigid(self, dt):
        self.v_force_global = self.v_force_gpu.numpy()
        self.sum_force_global = np.sum(self.v_force_global, axis=0) + self.mass * self.gravity

        v_pos_global = self.get_global_vertices()
        r = v_pos_global - self.c_pos
        self.sum_torque = np.sum(r[:, 0] * self.v_force_global[:, 1] - r[:, 1] * self.v_force_global[:, 0])

        self.c_vel += (self.sum_force_global / self.mass) * dt
        self.c_pos += self.c_vel * dt

        self.omega += (self.sum_torque / self.inertia) * dt
        self.angle += self.omega * dt
        
class LeafRigidBody2D_Torch:
    def __init__(self, device, a, b, num_pts, rho, com_offset, initial_pos, initial_rot_deg):
        self.device = torch.device(device)
        self.rho = rho
        self.a = a
        self.b = b

        self.com_offset = torch.tensor(com_offset, dtype=torch.float32, device=self.device)
        self.num_vertices = num_pts
        self.num_boundary_edges = num_pts

        self.c_pos = torch.tensor(initial_pos, dtype=torch.float32, device=self.device)
        self.c_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.angle = torch.tensor(np.deg2rad(initial_rot_deg), dtype=torch.float32, device=self.device)
        self.omega = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        self._generate_ellipse()
        self._compute_mass_properties()

        self.v_force_global = torch.zeros((self.num_vertices, 2), dtype=torch.float32, device=self.device)
        self.sum_force_global = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.sum_torque = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.gravity = torch.zeros(2, dtype=torch.float32, device=self.device)

        self.v_pos_gpu = torch.zeros((self.num_vertices, 2), dtype=torch.float32, device=self.device)
        self.v_vel_gpu = torch.zeros((self.num_vertices, 2), dtype=torch.float32, device=self.device)
        self.v_force_gpu = torch.zeros((self.num_vertices, 2), dtype=torch.float32, device=self.device)

        self.edges_gpu = torch.tensor(self.boundary_edges.flatten(), dtype=torch.int32, device=self.device)

        self.update_gpu_buffers()

    def _generate_ellipse(self):
        t = np.linspace(0, 2 * np.pi, self.num_vertices, endpoint=False)
        x_local = self.a * torch.cos(t) - self.com_offset[0]
        y_local = self.b * torch.sin(t) - self.com_offset[1]

        self.v_rest = torch.stack((x_local, y_local), dim=1).to(torch.float32)
        self.v_pos_local = self.v_rest.clone()

        edges = [[i, (i + 1) % self.num_vertices] for i in range(self.num_vertices)]
        self.boundary_edges = np.array(edges, dtype=np.int32)

    def _compute_mass_properties(self):
        area = np.pi * self.a * self.b
        self.mass = torch.tensor(self.rho * area, dtype=torch.float32, device=self.device)
        self.v_mass = torch.full((self.num_vertices,), self.mass / self.num_vertices, dtype=torch.float32, device=self.device)
        
        i_geom = self.mass * (self.a**2 + self.b**2) / 4.0
        offset_sq_distance = self.com_offset[0]**2 + self.com_offset[1]**2
        self.inertia = i_geom + self.mass * offset_sq_distance

    def get_rotation_matrix(self):
        c, s = torch.cos(self.angle), torch.sin(self.angle)
        return torch.stack([
            torch.stack([c, -s]),
            torch.stack([s, c])
        ])
   
    def get_global_vertices(self):
        R = self.get_rotation_matrix()
        return (R @ self.v_pos_local.T).T + self.c_pos

    def update_gpu_buffers(self):
        R = self.get_rotation_matrix()
        v_pos_global = (R @ self.v_pos_local.T).T + self.c_pos

        v_rot_local = torch.zeros_like(self.v_pos_local)
        v_rot_local[:, 0] = -self.omega * self.v_pos_local[:, 1]
        v_rot_local[:, 1] = self.omega * self.v_pos_local[:, 0]
        v_vel_global = self.c_vel + (R @ v_rot_local.T).T

        self.v_pos_gpu.copy_(v_pos_global)
        self.v_vel_gpu.copy_(v_vel_global)
        self.v_force_gpu.zero_()
   
    def apply_forces_and_forward_rigid(self, dt):
        v_force_torch = wp.to_torch(self.v_force_gpu) 
        self.v_force_global.copy_(v_force_torch)
        
        self.sum_force_global = torch.sum(self.v_force_global, dim=0) + self.mass * self.gravity

        v_pos_global = self.get_global_vertices()
        r = v_pos_global - self.c_pos

        self.sum_torque = torch.sum(r[:, 0] * self.v_force_global[:, 1] - r[:, 1] * self.v_force_global[:, 0])

        self.c_vel += (self.sum_force_global / self.mass) * dt
        self.c_pos += self.c_vel * dt

        self.omega += (self.sum_torque / self.inertia) * dt
        self.angle += self.omega * dt