"""
SEW Stereo Kinematics Class

Python equivalent of the MATLAB sew_stereo class for spherical-elbow-wrist 
stereo kinematics calculations.
"""

from typing import Tuple, Union

import numpy as np


def vec_normalize(vec):
    """
    Normalize a vector to unit length.

    Args:
        vec: numpy array or CasADi SX/MX vector

    Returns:
        Normalized vector
    """
    return vec / np.linalg.norm(vec)


def rot(axis, angle):
    """
    Rodrigues rotation formula implementation.

    Args:
        axis: 3x1 rotation axis (should be unit vector)
        angle: rotation angle in radians

    Returns:
        3x3 rotation matrix or CasADi expression
    """
    # NumPy implementation
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


class SEWStereo:
    """
    Spherical-Elbow-Wrist (SEW) Stereo kinematics class.

    This class handles forward and inverse kinematics for a spherical-elbow-wrist
    mechanism with stereo constraints.
    """

    def __init__(self, R: np.ndarray, V: np.ndarray):
        """
        Initialize SEW Stereo object.

        Args:
            R: 3x1 reference direction vector
            V: 3x1 reference velocity/direction vector
        """
        self.R = R
        self.V = V

    def fwd_kin(self, S: np.ndarray, E: np.ndarray, W: np.ndarray) -> float:
        """
        Forward kinematics: compute psi angle from joint positions.

        Args:
            S: 3x1 shoulder position
            E: 3x1 elbow position
            W: 3x1 wrist position

        Returns:
            psi: computed angle in radians
        """
        E_rel = E - S
        W_rel = W - S
        w_hat = vec_normalize(W_rel)

        n_hat_sew = vec_normalize(np.cross(W_rel, E_rel))
        n_hat_ref = vec_normalize(np.cross(w_hat - self.R, self.V))

        # Compute psi using atan2
        cross_product = np.cross(w_hat, n_hat_ref)
        dot_product = np.dot(n_hat_sew, n_hat_ref)
        cross_dot = np.dot(n_hat_sew, cross_product)

        psi = np.arctan2(cross_dot, dot_product)
        return psi

    def alt_fwd_kin(self, S: np.ndarray, E: np.ndarray, W: np.ndarray) -> float:
        """
        Alternative forward kinematics implementation.

        Args:
            S: 3x1 shoulder position
            E: 3x1 elbow position
            W: 3x1 wrist position

        Returns:
            psi: computed angle in radians
        """
        E_rel = E - S
        W_rel = W - S
        w_hat = vec_normalize(W_rel)

        n_sew = np.cross(W_rel, E_rel)
        n_phi = -np.cross(w_hat - self.R, n_sew)

        cross_product = np.cross(-self.R, self.V)
        dot_product = np.dot(n_phi, self.V)
        cross_dot = np.dot(n_phi, cross_product)

        psi = np.arctan2(cross_dot, dot_product)
        return psi

    def inv_kin(
        self, S: np.ndarray, W: np.ndarray, psi: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse kinematics: compute elbow direction and SEW normal from psi.

        Args:
            S: 3x1 shoulder position
            W: 3x1 wrist position
            psi: desired angle in radians

        Returns:
            e_CE: 3x1 elbow direction vector
            n_SEW: 3x1 SEW normal vector
        """
        p_SW = W - S
        e_SW = vec_normalize(p_SW)
        k_r = np.cross(e_SW - self.R, self.V)
        k_x = np.cross(k_r, p_SW)
        e_x = vec_normalize(k_x)

        # Apply rotation
        R_matrix = rot(e_SW, psi)
        e_CE = R_matrix @ e_x

        n_SEW = np.cross(e_SW, e_CE)

        return e_CE, n_SEW

    def jacobian(
        self, S: np.ndarray, E: np.ndarray, W: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian matrices for the SEW mechanism.

        Args:
            S: 3x1 shoulder position
            E: 3x1 elbow position
            W: 3x1 wrist position

        Returns:
            J_e: Jacobian with respect to elbow
            J_w: Jacobian with respect to wrist
        """
        E_rel = E - S
        W_rel = W - S
        w_hat = vec_normalize(W_rel)

        n_ref = np.cross(w_hat - self.R, self.V)

        x_c = np.cross(n_ref, W_rel)
        x_hat_c = vec_normalize(x_c)
        y_hat_c = np.cross(w_hat, x_hat_c)

        # Projection of E onto plane perpendicular to w_hat
        p = (np.eye(3) - np.outer(w_hat, w_hat)) @ E_rel
        p_hat = vec_normalize(p)

        # Jacobian with respect to elbow
        cross_product = np.cross(w_hat, p_hat)
        J_e = cross_product / np.linalg.norm(p)

        # Jacobian with respect to wrist (three components)
        norm_x_c = np.linalg.norm(x_c)
        norm_W = np.linalg.norm(W_rel)
        norm_p = np.linalg.norm(p)

        J_w_1 = (np.dot(w_hat, self.V) / norm_x_c) * y_hat_c
        J_w_2 = (np.dot(w_hat, np.cross(self.R, self.V)) / norm_x_c) * x_hat_c
        J_w_3 = (np.dot(w_hat, E_rel) / norm_W / norm_p) * cross_product

        J_w = J_w_1 + J_w_2 - J_w_3

        return J_e, J_w


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Testing SEW Stereo Kinematics")
    print("=" * 60)

    # Test parameters
    R = np.array([1.0, 0.0, 0.0])
    V = np.array([0.0, 1.0, 0.0])

    # Create SEW stereo object
    sew = SEWStereo(R, V)

    # Test positions
    S = np.array([0.0, 0.0, 0.0])
    E = np.array([1.0, 1.0, 0.5])
    W = np.array([2.0, 0.5, 1.0])

    print(f"Shoulder position: {S}")
    print(f"Elbow position: {E}")
    print(f"Wrist position: {W}")
    print(f"Reference R: {R}")
    print(f"Reference V: {V}")

    # Test forward kinematics
    psi_fwd = sew.fwd_kin(S, E, W)
    psi_alt = sew.alt_fwd_kin(S, E, W)

    print(f"\nForward kinematics:")
    print(f"psi (method 1): {psi_fwd:.6f} rad ({np.degrees(psi_fwd):.2f}°)")
    print(f"psi (method 2): {psi_alt:.6f} rad ({np.degrees(psi_alt):.2f}°)")

    # Test inverse kinematics
    e_CE, n_SEW = sew.inv_kin(S, W, psi_fwd)

    print(f"\nInverse kinematics:")
    print(f"Elbow direction e_CE: {e_CE}")
    print(f"SEW normal n_SEW: {n_SEW}")

    # Test Jacobians
    J_e, J_w = sew.jacobian(S, E, W)

    print(f"\nJacobians:")
    print(f"J_e (elbow): {J_e}")
    print(f"J_w (wrist): {J_w}")
