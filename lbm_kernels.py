import warp as wp
from lbm_constants import LBMD2Q9

# ==============================================================================
# GPU Device Functions
# ==============================================================================


@wp.func
def line_segment_intersect(
    p1: wp.vec2, p2: wp.vec2,
    p3: wp.vec2, p4: wp.vec2
):
    """
    Calculates the intersection of two 2D line segments, p1-p2 and p3-p4.
    """
    ### TODO 3a: Line Segment Intersection (2 pts) ###
    # You need to solve linear equations to find the intersection point.
    # Return the parameters t and u where:
    # Intersection point = p1 + t * (p2 - p1) = p3 + u * (p4 - p3)
    # If the segments do not intersect, return t = -1.0 and u = -1.0 as a sentinel value.
    r = p2 - p1
    s = p4 - p3
    
    denom = r[0] * s[1] - r[1] * s[0]
    
    if abs(denom) < 1e-6:
        return wp.vec2(-1.0, -1.0)
    
    p3_p1 = p3 - p1
    
    t = (p3_p1[0] * s[1] - p3_p1[1] * s[0]) / denom
    u = (p3_p1[0] * r[1] - p3_p1[1] * r[0]) / denom
    
    if (t >= 0.0 and t <= 1.0) and (u >= 0.0 and u <= 1.0):
        return wp.vec2(t, u)
    else:
        return wp.vec2(-1.0, -1.0)


@wp.func
def find_fluid_link_solid_intersection(
    fluid_p1: wp.vec2,
    fluid_p2: wp.vec2,
    solid_v_pos: wp.array(dtype=wp.vec2),
    solid_edges: wp.array(dtype=wp.int32),
    num_edges: wp.int32
):
    """
    Finds the closest intersection between a fluid link and the solid mesh.
    Returns: hit_edge_idx, barycentric_u, distance_t
    """
    min_t = float(2.0)
    hit_idx = int(-1)
    hit_u = float(0.0)

    for i in range(num_edges):
        v0_idx = solid_edges[i * 2]
        v1_idx = solid_edges[i * 2 + 1]
        solid_p1 = solid_v_pos[v0_idx]
        solid_p2 = solid_v_pos[v1_idx]

        res_intersect = line_segment_intersect(fluid_p1, fluid_p2, solid_p1, solid_p2)
        t = res_intersect[0]
        u = res_intersect[1]

        if t >= 0.0 and t < min_t:
            min_t = t
            hit_idx = i
            hit_u = u

    return hit_idx, hit_u, min_t


@wp.func
def calculate_f_dist_from_moments(
    rho: wp.float32, ux: wp.float32, uy: wp.float32,
    pi_xx: wp.float32, pi_yy: wp.float32, pi_xy: wp.float32,
    # w: wp.array(dtype=wp.float32),
    idx: wp.int32
):
    """Matches mlCalDistributionD2Q9ALL / AtIndex"""
    A0 = rho
    Ax = ux * A0
    Ay = uy * A0
    Axx = rho * pi_xx
    Ayy = rho * pi_yy
    Axy = rho * pi_xy

    Axxy = -2.0 * rho * uy * ux * ux + 2.0 * Axy * ux + Axx * uy
    Axyy = -2.0 * rho * ux * uy * uy + 2.0 * Axy * uy + Ayy * ux
    Axxyy = 0.0

    w = LBMD2Q9.device_w
    
    if idx == 0:
        return wp.max(0.0, w[0] * (A0 - (3.0*Axx)/2.0 +
                      (9.0*Axxyy)/4.0 - (3.0*Ayy)/2.0))
    elif idx == 1:
        return wp.max(0.0, w[1] * (A0 + 3.0*Ax + 3.0*Axx -
                      (9.0*Axyy)/2.0 - (9.0*Axxyy)/2.0 - (3.0*Ayy)/2.0))
    elif idx == 2:
        return wp.max(0.0, w[2] * (A0 - (3.0*Axx)/2.0 -
                      (9.0*Axxy)/2.0 + 3.0*Ay - (9.0*Axxyy)/2.0 + 3.0*Ayy))
    elif idx == 3:
        return wp.max(0.0, w[3] * (A0 - 3.0*Ax + 3.0*Axx +
                      (9.0*Axyy)/2.0 - (9.0*Axxyy)/2.0 - (3.0*Ayy)/2.0))
    elif idx == 4:
        return wp.max(0.0, w[4] * (A0 - (3.0*Axx)/2.0 +
                      (9.0*Axxy)/2.0 - 3.0*Ay - (9.0*Axxyy)/2.0 + 3.0*Ayy))
    elif idx == 5:
         return wp.max(0.0, w[5] * (A0 + 3.0*Ax + 3.0*Axx + 9.0 *
                      Axy + 9.0*Axxy + 9.0*Axyy + 3.0*Ay + 9.0*Axxyy + 3.0*Ayy))
    elif idx == 6:
        return wp.max(0.0, w[6] * (A0 - 3.0*Ax + 3.0*Axx - 9.0 *
                      Axy + 9.0*Axxy - 9.0*Axyy + 3.0*Ay + 9.0*Axxyy + 3.0*Ayy))
    elif idx == 7:
        return wp.max(0.0, w[7] * (A0 - 3.0*Ax + 3.0*Axx + 9.0 *
                      Axy - 9.0*Axxy - 9.0*Axyy - 3.0*Ay + 9.0*Axxyy + 3.0*Ayy))
    elif idx == 8:
        return wp.max(0.0, w[8] * (A0 + 3.0*Ax + 3.0*Axx - 9.0 *
                      Axy - 9.0*Axxy + 9.0*Axyy - 3.0*Ay + 9.0*Axxyy + 3.0*Ayy))


@wp.func
def calculate_pi_after_collision(
    rho: wp.float32, ux: wp.float32, uy: wp.float32,
    fx: wp.float32, fy: wp.float32, omega: wp.float32,
    pi_xx_pre: wp.float32, pi_yy_pre: wp.float32, pi_xy_pre: wp.float32
):
    """Matches mlGetPIAfterCollision"""
    pixx_part = (pi_xx_pre - pi_yy_pre) / 2.0
    piyy_part = (pi_yy_pre - pi_xx_pre) / 2.0
    RU2 = rho * ux * ux
    RV2 = rho * uy * uy

    pi_xx_post = rho / 3.0 + pixx_part * (1.0 - omega) + RU2 / 2.0 + RV2 / 2.0 + RU2 * \
        omega / 2.0 - RV2 * omega / 2.0 + fx * ux + \
        0.5 * (1.0 - omega) * (fx * ux - fy * uy)
    pi_yy_post = rho / 3.0 + piyy_part * (1.0 - omega) + RU2 / 2.0 + RV2 / 2.0 - RU2 * \
        omega / 2.0 + RV2 * omega / 2.0 + fy * uy + \
        0.5 * (1.0 - omega) * (fy * uy - fx * ux)
    pi_xy_post = pi_xy_pre - pi_xy_pre * omega + ux * uy * \
        rho * omega + (1.0 - 0.5 * omega) * (fy * ux + fx * uy)

    return pi_xx_post, pi_yy_post, pi_xy_post

# ==============================================================================
# GPU Kernels
# ==============================================================================

@wp.kernel
def initialize_kernel(
    moments_pre: wp.array(dtype=wp.float32, ndim=3),
    flags: wp.array(dtype=wp.int32, ndim=2),
    initial_density: wp.float32
):
    """Initializes moments to equilibrium with zero velocity."""
    i, j = wp.tid()
    
    rho = initial_density
    ux, uy = 0.0, 0.0
    
    # Equilibrium Pi tensor for zero velocity
    pixx = 0.0 
    piyy = 0.0
    pixy = 0.0

    moments_pre[i, j, 0] = rho
    moments_pre[i, j, 1] = ux
    moments_pre[i, j, 2] = uy
    moments_pre[i, j, 3] = pixx
    moments_pre[i, j, 4] = piyy
    moments_pre[i, j, 5] = pixy

vec9 = wp.vec(length=9, dtype=wp.float32)

@wp.kernel
def streaming_and_collision_kernel(
    moments_pre: wp.array(dtype=wp.float32, ndim=3),
    moments_post: wp.array(dtype=wp.float32, ndim=3),
    flags: wp.array(dtype=wp.int32, ndim=2),
    fluid_force: wp.array(dtype=wp.float32, ndim=3),
    solid_v_pos: wp.array(dtype=wp.vec2),
    solid_v_vel: wp.array(dtype=wp.vec2),
    solid_v_force: wp.array(dtype=wp.vec2),
    solid_edges: wp.array(dtype=wp.int32),
    num_edges: wp.int32,
    nx: wp.int32, ny: wp.int32,
    Omega: wp.float32,
    cl: wp.float32, ct: wp.float32, cf: wp.float32,    
):
    i, j = wp.tid()
    if flags[i, j] != 0: return

    rho_cur, ux_cur, uy_cur = moments_pre[i, j, 0], moments_pre[i, j, 1], moments_pre[i, j, 2]
    pixx_cur, piyy_cur, pixy_cur = moments_pre[i, j, 3], moments_pre[i, j, 4], moments_pre[i, j, 5]

    pop = vec9(0.0)
    for k in range(9):
        dx, dy, inv_k = LBMD2Q9.device_ex[k], LBMD2Q9.device_ey[k], LBMD2Q9.device_inv[k]
        isrc, jsrc = i - dx, j - dy
        not_cross = (isrc >= 0 and isrc < nx and jsrc >= 0 and jsrc < ny)
        isrc_per, jsrc_per = (isrc + nx) % nx, (jsrc + ny) % ny
        flag_back = flags[isrc_per, jsrc_per]
        is_blocked = False
        f_leaving_current = calculate_f_dist_from_moments(rho_cur, ux_cur, uy_cur, pixx_cur, piyy_cur, pixy_cur, inv_k)

        if flag_back == 1: # static walls
            pop[k] = wp.max(0.0, f_leaving_current)
            is_blocked = True
        elif not_cross:
            fluid_p1 = wp.vec2(float(i) * cl, float(j) * cl)
            fluid_p2 = wp.vec2(float(isrc_per) * cl, float(jsrc_per) * cl)
            hit_idx, hit_u, hit_t = find_fluid_link_solid_intersection(fluid_p1, fluid_p2, solid_v_pos, solid_edges, num_edges)

            if hit_idx != -1:
                is_blocked = True
                ### TODO 3b: FSI Coupling (6 pts) ###
                # 1. Interpolate solid velocity `v_solid` at hit point `hit_u`.
                # 2. Scale `v_solid` to LBM units (`ux`, `uy`).
                # 3. Modify Pi tensor for moving boundary: `pixx_s = pixx_cur + ux*ux - ux_cur*ux_cur ...`
                # 4. Calculate `fout` using calculate_f_dist_from_moments. Assign it to `pop[k]`.
                # 5. Compute momentum exchange force: `fx = (fin*(-dx-ux) - fout*(dx-ux)) * cf`.
                # 6. Apply force to solid vertices using `wp.atomic_add`.
                v0_idx = solid_edges[hit_idx * 2]
                v1_idx = solid_edges[hit_idx * 2 + 1]
                v_solid = (1.0 - hit_u) * solid_v_vel[v0_idx] + hit_u * solid_v_vel[v1_idx]
                
                u_wall = v_solid * (ct / cl)
                ux_wall = u_wall[0]
                uy_wall = u_wall[1]

                pixx_s = pixx_cur + rho_cur*(ux_wall * ux_wall - ux_cur * ux_cur)
                piyy_s = piyy_cur + rho_cur*(uy_wall * uy_wall - uy_cur * uy_cur)
                pixy_s = pixy_cur + rho_cur*(ux_wall*uy_wall - ux_cur*uy_cur)

                f_out = calculate_f_dist_from_moments(rho_cur, ux_wall, uy_wall, pixx_s, piyy_s, pixy_s, k)
                f_in = calculate_f_dist_from_moments(rho_cur, ux_cur, uy_cur, pixx_cur, piyy_cur, pixy_cur, inv_k)
                pop[k] = wp.max(0.0, f_out)

                dx_f = float(dx)
                dy_f = float(dy)
                fx = (f_in * (-dx_f - ux_cur) - f_out * (dx_f - ux_cur)) * cf
                fy = (f_in * (-dy_f - uy_cur) - f_out * (dy_f - uy_cur)) * cf

                force = wp.vec2(fx, fy)
                wp.atomic_add(solid_v_force, v0_idx, (1.0 - hit_u) * force)
                wp.atomic_add(solid_v_force, v1_idx, hit_u * force)

        if not is_blocked:
            rho_src = moments_pre[isrc_per, jsrc_per, 0]
            ux_src = moments_pre[isrc_per, jsrc_per, 1]
            uy_src = moments_pre[isrc_per, jsrc_per, 2]
            pixx_src = moments_pre[isrc_per, jsrc_per, 3]
            piyy_src = moments_pre[isrc_per, jsrc_per, 4]
            pixy_src = moments_pre[isrc_per, jsrc_per, 5]
            f_incoming = calculate_f_dist_from_moments(rho_src, ux_src, uy_src, pixx_src, piyy_src, pixy_src, k)
            pop[k] = f_incoming

    # 2. Macroscopic Update
    rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + \
        pop[4] + pop[5] + pop[6] + pop[7] + pop[8]
    FX = fluid_force[i, j, 0]
    FY = fluid_force[i, j, 1]
    invRho = 1.0 / rhoVar

    ux_new = ((pop[1] - pop[3] + pop[5] - pop[6] -
              pop[7] + pop[8]) + 0.5 * FX) * invRho
    uy_new = ((pop[2] - pop[4] + pop[5] + pop[6] -
              pop[7] - pop[8]) + 0.5 * FY) * invRho

    pixx_new = pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[8]
    piyy_new = pop[2] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8]
    pixy_new = pop[5] - pop[6] + pop[7] - pop[8]

    # 3. Collision Relaxation

    pixx_rel, piyy_rel, pixy_rel = calculate_pi_after_collision(
        rhoVar, ux_new, uy_new, FX, FY, Omega, pixx_new, piyy_new, pixy_new
    )

    # Convert back to normalized moments
    pixx_final = (pixx_rel * invRho - LBMD2Q9.cs2)
    piyy_final = (piyy_rel * invRho - LBMD2Q9.cs2)
    pixy_final = (pixy_rel * invRho)

    # Write to Post Moments
    moments_post[i, j, 0] = rhoVar
    moments_post[i, j, 1] = ux_new + FX * invRho / 2.0
    moments_post[i, j, 2] = uy_new + FY * invRho / 2.0
    moments_post[i, j, 3] = pixx_final
    moments_post[i, j, 4] = piyy_final
    moments_post[i, j, 5] = pixy_final


# ==============================================================================
# Outlet Boundary Condition Kernel
# ==============================================================================

@wp.kernel
def apply_outlet_kernel(
    moments: wp.array(dtype=wp.float32, ndim=3),
    flags: wp.array(dtype=wp.int32, ndim=2),
    nx: wp.int32, 
    ny: wp.int32,
    u_inlet: wp.float32
):
    """
    Applies outlet boundary conditions by extrapolating velocities from the interior 
    fluid nodes to the outlet boundary nodes.
    
    Fluid = 0
    OutletUp = 3
    OutletDown = 4
    InletLeft = 5
    OutletRight = 6
    """
    x, y = wp.tid()

    if x < 0 or x >= nx or y < 0 or y >= ny:
        return

    flag = flags[x, y]

    # --- Outlet Up (Top Boundary) ---
    if flag == 3: # OutletUp
        if y - 1 >= 0:
            if flags[x, y - 1] == 0: # Check if interior node is Fluid
                up_vel = moments[x, y - 1, 2] # Get uy from interior
                if up_vel > 0.0:
                    moments[x, y, 2] = up_vel  # Set uy at outlet
                moments[x, y, 4] = up_vel * up_vel # Set piyy approximation

    # --- Outlet Down (Bottom Boundary) ---
    elif flag == 4: # OutletDown
        if y + 1 < ny:
            if flags[x, y + 1] == 0:
                down_vel = moments[x, y + 1, 2] # Get uy from interior
                if down_vel < 0.0:
                    moments[x, y, 2] = down_vel
                moments[x, y, 4] = down_vel * down_vel

     # --- Outlet Left (Left Boundary) ---
    elif flag == 5: # OutletLeft
        if x + 1 < nx:
            if flags[x + 1, y] == 0:
                left_vel = moments[x + 1, y, 1] # Get ux from interior
                if left_vel < 0.0:
                    moments[x, y, 1] = left_vel
                moments[x, y, 3] = left_vel * left_vel

    # --- Outlet Right (Right Boundary) ---
    elif flag == 6: # OutletRight
        if x - 1 >= 0:
            if flags[x - 1, y] == 0:
                right_vel = moments[x - 1, y, 1] # Get ux from interior
                if right_vel > 0.0:
                    moments[x, y, 1] = right_vel
                moments[x, y, 3] = right_vel * right_vel

@wp.kernel
def apply_outlet_inlet_kernel(
    moments: wp.array(dtype=wp.float32, ndim=3),
    flags: wp.array(dtype=wp.int32, ndim=2),
    nx: wp.int32, 
    ny: wp.int32,
    u_inlet: wp.float32
):
    """
    Applies outlet boundary conditions by extrapolating velocities from the interior 
    fluid nodes to the outlet boundary nodes.
    
    Fluid = 0
    OutletUp = 3
    OutletDown = 4
    InletLeft = 5
    OutletRight = 6
    """
    x, y = wp.tid()

    if x < 0 or x >= nx or y < 0 or y >= ny:
        return

    flag = flags[x, y]

    # --- Outlet Up (Top Boundary) ---
    if flag == 3: # OutletUp
        if y - 1 >= 0:
            if flags[x, y - 1] == 0: # Check if interior node is Fluid
                up_vel = moments[x, y - 1, 2] # Get uy from interior
                if up_vel > 0.0:
                    moments[x, y, 2] = up_vel  # Set uy at outlet
                moments[x, y, 4] = up_vel * up_vel # Set piyy approximation

    # --- Outlet Down (Bottom Boundary) ---
    elif flag == 4: # OutletDown
        if y + 1 < ny:
            if flags[x, y + 1] == 0:
                down_vel = moments[x, y + 1, 2] # Get uy from interior
                if down_vel < 0.0:
                    moments[x, y, 2] = down_vel
                moments[x, y, 4] = down_vel * down_vel

    # --- REVISED: Inlet Left (Left Boundary) ---
    elif flag == 5: # InletLeft
        moments[x, y, 1] = u_inlet
        moments[x, y, 2] = 0.0               
        moments[x, y, 3] = u_inlet * u_inlet 
        moments[x, y, 4] = 0.0

    # --- Outlet Right (Right Boundary) ---
    elif flag == 6: # OutletRight
        if x - 1 >= 0:
            if flags[x - 1, y] == 0:
                right_vel = moments[x - 1, y, 1] # Get ux from interior
                if right_vel > 0.0:
                    moments[x, y, 1] = right_vel
                moments[x, y, 3] = right_vel * right_vel