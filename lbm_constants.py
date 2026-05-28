import warp as wp

vec9i = wp.types.vector(length=9, dtype=wp.int32)
vec9f = wp.types.vector(length=9, dtype=wp.float32)

class LBMD2Q9:
    """
    Constants for the D2Q9 Lattice Boltzmann Model.
    Q: Number of discrete velocities.
    ex, ey: Discrete velocity vectors.
    inv: Indices of opposite velocity vectors.
    w: Weights for each velocity direction.
    cs2: Speed of sound squared.
    """
    Q = 9
    cs2 = 1.0 / 3.0

    # Device-side constant arrays
    device_ex = wp.constant(vec9i(0, 1, 0, -1, 0, 1, -1, -1, 1))
    device_ey = wp.constant(vec9i(0, 0, 1, 0, -1, 1, 1, -1, -1))
    device_inv = wp.constant(vec9i(0, 3, 4, 1, 2, 7, 8, 5, 6))
    device_w = wp.constant(vec9f(4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0))
