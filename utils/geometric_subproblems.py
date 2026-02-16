"""
Geometric Subproblems for Robotic Inverse Kinematics

This module contains the core geometric subproblem functions used in solving
inverse kinematics for robotic arms. These functions implement fundamental
geometric operations for robot kinematics based on the theory of Paden-Kahan
subproblems.

The module includes numerical (NumPy) implementations
of each subproblem, along with helper functions

Subproblems included:
- SP0: Rotation about axis to align two vectors (perpendicular case)
- SP1: Rotation about axis to align two general vectors (plane and sphere)
- SP2: Two rotations to align two pairs of vectors (two circles)
- SP3: Rotation to achieve specific distance (circle and sphere)
- SP4: Rotation to satisfy linear constraint (circle and plane)

This method was proposed in Paper: https://www.sciencedirect.com/science/article/pii/S0094114X25000606
Adapted to Python by: Chuizheng Kong

"""
import numpy as np


def sp_0(p, q, k):
    """
    Numerical version of subproblem 0: finds theta such that q = rot(k, theta)*p

    ** assumes k'*p = 0 and k'*q = 0

    Requires that p and q are perpendicular to k. Use subproblem 1 if this is not
    guaranteed.

    Args:
        p: 3x1 numpy array vector before rotation
        q: 3x1 numpy array vector after rotation
        k: 3x1 numpy array rotation axis unit vector

    Returns:
        theta: scalar angle in radians
    """

    # Check that p and q are perpendicular to k (optional assertion for debugging)
    # assert (np.abs(np.dot(k,p)) < eps) and (np.abs(np.dot(k,q)) < eps), \
    #        "k must be perpendicular to p and q"

    norm = np.linalg.norm

    ep = p / norm(p)
    eq = q / norm(q)

    theta = 2 * np.arctan2(norm(ep - eq), norm(ep + eq))

    if np.dot(k, np.cross(p, q)) < 0:
        return -theta

    return theta


def sp_1(p1, p2, k):
    """
    Numerical version of subproblem 1: finds theta such that rot(k, theta)*p1 = p2

    If the problem is well-posed (same norm and k-component), finds exact solution.
    Otherwise, finds least-squares solution that minimizes || rot(k, theta)*p1 - p2 ||

    Args:
        p1: 3x1 numpy array vector before rotation
        p2: 3x1 numpy array vector after rotation
        k: 3x1 numpy array rotation axis unit vector

    Returns:
        theta: scalar angle in radians
        is_LS: boolean flag indicating if solution is least-squares
    """

    # Compute norms
    norm_p1 = np.linalg.norm(p1)
    norm_p2 = np.linalg.norm(p2)

    # Check for least-squares condition
    norm_diff = abs(norm_p1 - norm_p2)

    # Components along k
    k_dot_p1 = np.dot(k, p1)
    k_dot_p2 = np.dot(k, p2)
    k_comp_diff = abs(k_dot_p1 - k_dot_p2)

    # Project vectors onto plane perpendicular to k
    p1_proj = p1 - k_dot_p1 * k
    p2_proj = p2 - k_dot_p2 * k

    # Compute projected norms
    norm_p1_proj = np.linalg.norm(p1_proj)
    norm_p2_proj = np.linalg.norm(p2_proj)
    proj_diff = abs(norm_p1_proj - norm_p2_proj)

    # Check if this is a least-squares problem
    tolerance = 1e-8
    is_LS = (
        (norm_diff > tolerance) or (k_comp_diff > tolerance) or (proj_diff > tolerance)
    )

    # Handle degenerate cases
    if norm_p1_proj < tolerance or norm_p2_proj < tolerance:
        return 0.0, is_LS

    # Normalize projected vectors
    p1_proj_norm = p1_proj / norm_p1_proj
    p2_proj_norm = p2_proj / norm_p2_proj

    # Calculate angle using atan2 approach (similar to sp_0)
    p_diff = p1_proj_norm - p2_proj_norm
    p_sum = p1_proj_norm + p2_proj_norm

    norm_diff_proj = np.linalg.norm(p_diff)
    norm_sum_proj = np.linalg.norm(p_sum)

    # Handle edge case where vectors are identical
    if norm_sum_proj < tolerance:
        return np.pi, is_LS

    theta = 2 * np.arctan2(norm_diff_proj, norm_sum_proj)

    # Check sign using cross product
    cross_p1p2 = np.cross(p1_proj, p2_proj)
    sign_check = np.dot(k, cross_p1p2)

    if sign_check < 0:
        theta = -theta

    return theta, is_LS


def sp_2(p1, p2, k1, k2):
    """
    Numerical version of sp_2 that follows the exact MATLAB reference implementation.

    [theta1, theta2] = sp_2(p1, p2, k1, k2) finds theta1, theta2 such that
        rot(k1, theta1)*p1 = rot(k2, theta2)*p2

    This implementation follows the MATLAB reference exactly:
    % Rescale for least-squares case
    p1_nrm = p1/norm(p1);
    p2_nrm = p2/norm(p2);

    [theta1, t1_is_LS] = subproblem.sp_4(k2, p1_nrm, k1, dot(k2,p2_nrm));
    [theta2, t2_is_LS] = subproblem.sp_4(k1, p2_nrm, k2, dot(k1,p1_nrm));

    % Make sure solutions correspond by flipping theta2
    % Also make sure in the edge case that one angle has one solution and the
    % other angle has two solutions that we duplicate the single solution
    if numel(theta1)>1 || numel(theta2)>1
        theta1 = [theta1(1) theta1(end)];
        theta2 = [theta2(end) theta2(1)];
    end

    Args:
        p1: 3x1 numpy array
        p2: 3x1 numpy array
        k1: 3x1 numpy array with norm(k1) = 1
        k2: 3x1 numpy array with norm(k2) = 1

    Returns:
        theta1: numpy array of theta1 solutions
        theta2: numpy array of theta2 solutions
        is_LS: boolean flag indicating if solution is least-squares
    """

    # Rescale for least-squares case
    p1_nrm = p1 / np.linalg.norm(p1)
    p2_nrm = p2 / np.linalg.norm(p2)

    # Call sp_4 twice as in MATLAB reference
    theta1, t1_is_LS = sp_4(k2, p1_nrm, k1, np.dot(k2, p2_nrm))
    theta2, t2_is_LS = sp_4(k1, p2_nrm, k2, np.dot(k1, p1_nrm))

    # Pair solutions as in MATLAB
    if len(theta1) > 1 or len(theta2) > 1:
        # Duplicate if needed
        if len(theta1) == 1:
            theta1 = np.array([theta1[0], theta1[0]])
        if len(theta2) == 1:
            theta2 = np.array([theta2[0], theta2[0]])
        # MATLAB pairing: theta1 = [theta1(1) theta1(end)]; theta2 = [theta2(end) theta2(1)];
        theta1 = np.array([theta1[0], theta1[-1]])
        theta2 = np.array([theta2[-1], theta2[0]])

    # LS flag
    is_LS = abs(np.linalg.norm(p1) - np.linalg.norm(p2)) > 1e-8 or t1_is_LS or t2_is_LS

    return theta1, theta2, is_LS


def sp_3(p1, p2, k, d):
    """
    Numerical version of sp_3 based on the MATLAB reference implementation.

    Subproblem 3: Circle and sphere

    theta = sp_3(p1, p2, k, d) finds theta such that
        || rot(k, theta)*p1 - p2 || = d
    If there's no solution, minimize the least-squares residual
        | || rot(k, theta)*p1 - p2 || - d |

    If the problem is well-posed, there may be 1 or 2 exact solutions, or 1
    least-squares solution

    The problem is ill-posed if (p1, k) or (p2, k) are parallel

    Parameters:
    -----------
    p1 : array_like, shape (3,)
        3D vector
    p2 : array_like, shape (3,)
        3D vector
    k : array_like, shape (3,)
        3D vector with norm(k) = 1
    d : float
        Scalar value (desired distance)

    Returns:
    --------
    theta : ndarray
        Array of angles (in radians). Shape is (N,) where N is the number of solutions
    is_LS : bool
        True if theta is a least-squares solution, False if exact solutions
    """

    # Convert inputs to numpy arrays
    p1 = np.array(p1).reshape(-1)
    p2 = np.array(p2).reshape(-1)
    k = np.array(k).reshape(-1)

    # Validate input dimensions
    if p1.shape[0] != 3 or p2.shape[0] != 3 or k.shape[0] != 3:
        raise ValueError("p1, p2, and k must be 3D vectors")

    # Following MATLAB reference: [theta, is_LS] = subproblem.sp_4(p2, p1, k, 1/2 * (dot(p1,p1)+dot(p2,p2)-d^2));
    # Calculate the parameter for sp_4
    p1_dot_p1 = np.dot(p1, p1)
    p2_dot_p2 = np.dot(p2, p2)
    d_squared = d * d
    sp4_d_param = 0.5 * (p1_dot_p1 + p2_dot_p2 - d_squared)

    # Call sp_4 with the calculated parameters
    theta, is_LS = sp_4(p2, p1, k, sp4_d_param)

    return theta, is_LS


def sp_4(h, p, k, d):
    """
    Numerical version of sp_4 based on the MATLAB reference implementation.

    Subproblem 4: Circle and plane

    Finds theta such that h' * rot(k, theta) * p = d
    If there's no solution, minimize the least-squares residual
    | h' * rot(k, theta) * p - d |

    If the problem is well-posed, there may be 1 or 2 exact solutions, or 1
    least-squares solution

    The problem is ill-posed if (p, k) or (h, k) are parallel

    Parameters:
    -----------
    h : array_like, shape (3,)
        3D vector with norm(h) = 1
    p : array_like, shape (3,)
        3D vector
    k : array_like, shape (3,)
        3D vector with norm(k) = 1
    d : float
        Scalar value

    Returns:
    --------
    theta : ndarray
        Array of angles (in radians). Shape is (N,) where N is the number of solutions
    is_LS : bool
        True if theta is a least-squares solution, False if exact solutions
    """

    # Convert inputs to numpy arrays and ensure they're column vectors
    h = np.array(h).reshape(-1)
    p = np.array(p).reshape(-1)
    k = np.array(k).reshape(-1)

    # Validate input dimensions
    if h.shape[0] != 3 or p.shape[0] != 3 or k.shape[0] != 3:
        raise ValueError("h, p, and k must be 3D vectors")

    # A_11 = cross(k, p)
    A_11 = np.cross(k, p)

    # A_1 = [A_11 -cross(k, A_11)]
    # This creates a 3x2 matrix where first column is A_11 and second column is -cross(k, A_11)
    A_1 = np.column_stack([A_11, -np.cross(k, A_11)])

    # A = h' * A_1 (this is a 1x2 matrix, but we'll treat as 1D array)
    A = h.T @ A_1  # This gives us a (2,) array

    # b = d - h' * k * (k' * p)
    b = d - np.dot(h, k) * np.dot(k, p)

    # norm_A_2 = dot(A, A) = ||A||^2
    norm_A_2 = np.dot(A, A)

    # x_ls_tilde = A_1' * (h * b)
    x_ls_tilde = A_1.T @ (h * b)  # This gives us a (2,) array

    # Check if we have exact solutions or need least-squares
    if norm_A_2 > b**2:
        # Two exact solutions case
        xi = np.sqrt(norm_A_2 - b**2)

        # x_N_prime_tilde = [A(2); -A(1)] (swap and negate first component)
        x_N_prime_tilde = np.array([A[1], -A[0]])

        # Two solution candidates
        sc_1 = x_ls_tilde + xi * x_N_prime_tilde
        sc_2 = x_ls_tilde - xi * x_N_prime_tilde

        # Compute angles using atan2
        theta = np.array([np.arctan2(sc_1[0], sc_1[1]), np.arctan2(sc_2[0], sc_2[1])])
        is_LS = False

    else:
        # Least-squares solution case
        theta = np.array([np.arctan2(x_ls_tilde[0], x_ls_tilde[1])])
        is_LS = True

    return theta, is_LS


def rot(axis, angle):
    """
    Numerical version of rotation matrix using Rodrigues' formula.

    Args:
        axis: 3x1 numpy array (unit vector)
        angle: scalar angle in radians

    Returns:
        3x3 numpy rotation matrix
    """

    c = np.cos(angle)
    s = np.sin(angle)
    v = 1 - c

    # Skew-symmetric matrix
    k_skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    # Rodrigues' formula
    R = np.eye(3) + s * k_skew + v * k_skew @ k_skew

    return R
