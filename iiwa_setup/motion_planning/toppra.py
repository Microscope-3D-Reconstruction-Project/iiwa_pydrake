import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import (
    BsplineTrajectory,
    MultibodyPlant,
    PathParameterizedTrajectory,
    Toppra,
    Trajectory,
)


def reparameterize_with_toppra(
    trajectory: Trajectory,
    plant: MultibodyPlant,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    num_grid_points: int = 1000,
) -> PathParameterizedTrajectory:
    """Reparameterize a trajectory/ path with Toppra.

    Args:
        trajectory (Trajectory): The trajectory on which the TOPPRA problem will be
        solved.
        plant (MultibodyPlant): The robot that will follow the solved trajectory. Used
        for enforcing torque and frame specific constraints.
        velocity_limits (np.ndarray): The velocity limits of shape (N,) where N is the
        number of robot joint joints.
        acceleration_limits (np.ndarray): The acceleration limits of shape (N,) where N
        is the number of robot joint joints.
        num_grid_points (int, optional): The number of uniform points along the path to
        discretize the problem and enforce constraints at.

    Returns:
        PathParameterizedTrajectory: The reparameterized trajectory.
    """

    # Print debug info
    print("start time:", trajectory.start_time())
    print("end time:", trajectory.end_time())

    # Create gridpoints
    gridpoints = np.linspace(
        trajectory.start_time(), trajectory.end_time(), num_grid_points
    )

    # # Check path properties at all gridpoints
    # print(f"\nChecking path at {num_grid_points} gridpoints...")
    # q_lower = plant.GetPositionLowerLimits()
    # q_upper = plant.GetPositionUpperLimits()

    # problematic_points = []
    # q_samples = []
    # qdot_samples = []
    # qddot_samples = []
    # t_samples = []

    # for i, t in enumerate(gridpoints):
    #     q = trajectory.value(t).flatten()
    #     qdot = trajectory.EvalDerivative(t, 1).flatten()
    #     qddot = trajectory.EvalDerivative(t, 2).flatten()

    #     # Store for plotting
    #     q_samples.append(q)
    #     qdot_samples.append(qdot)
    #     qddot_samples.append(qddot)
    #     t_samples.append(t)

    #     # Check for issues
    #     issues = []
    #     if np.any(q < q_lower - 1e-6):
    #         issues.append(f"q below lower limit: joints {np.where(q < q_lower)[0]}")
    #     if np.any(q > q_upper + 1e-6):
    #         issues.append(f"q above upper limit: joints {np.where(q > q_upper)[0]}")
    #     if np.any(np.isnan(q)) or np.any(np.isinf(q)):
    #         issues.append("q has NaN/Inf")
    #     if np.any(np.isnan(qdot)) or np.any(np.isinf(qdot)):
    #         issues.append("qdot has NaN/Inf")
    #     if np.any(np.isnan(qddot)) or np.any(np.isinf(qddot)):
    #         issues.append("qddot has NaN/Inf")
    #     if np.linalg.norm(qdot) > 1e6:
    #         issues.append(f"qdot very large: {np.linalg.norm(qdot):.2e}")
    #     if np.linalg.norm(qddot) > 1e6:
    #         issues.append(f"qddot very large: {np.linalg.norm(qddot):.2e}")

    #     if issues:
    #         problematic_points.append((i, t, issues))
    #         if len(problematic_points) <= 5:  # Print first 5
    #             print(f"  Issue at gridpoint {i}/{num_grid_points-1} (t={t:.4f}): {', '.join(issues)}")

    # if problematic_points:
    #     print(f"⚠ Found {len(problematic_points)} problematic gridpoints!")
    #     if len(problematic_points) > 5:
    #         print(f"  (showing first 5, {len(problematic_points)-5} more hidden)")
    # else:
    #     print("✓ All gridpoints look reasonable")

    # # Convert to numpy arrays for plotting
    # q_samples = np.array(q_samples)  # Shape: (num_gridpoints, 7)
    # qdot_samples = np.array(qdot_samples)
    # qddot_samples = np.array(qddot_samples)
    # t_samples = np.array(t_samples)

    # # Create plots
    # fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # # Plot positions
    # for j in range(7):
    #     axes[0].plot(t_samples, q_samples[:, j], label=f'q{j+1}', alpha=0.7)
    #     # Plot joint limits
    #     axes[0].axhline(q_lower[j], color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    #     axes[0].axhline(q_upper[j], color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    # axes[0].set_ylabel('Position (rad)')
    # axes[0].set_title('Joint Positions')
    # axes[0].legend(loc='right', fontsize=8)
    # axes[0].grid(True, alpha=0.3)

    # # Mark problematic points
    # if problematic_points:
    #     prob_times = [p[1] for p in problematic_points]
    #     axes[0].scatter(prob_times, [0]*len(prob_times), color='red', marker='x', s=100, zorder=10, label='Issues')

    # # Plot velocities
    # for j in range(7):
    #     axes[1].plot(t_samples, qdot_samples[:, j], label=f'qdot{j+1}', alpha=0.7)
    #     # Plot velocity limits
    #     axes[1].axhline(-velocity_limits[j], color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    #     axes[1].axhline(velocity_limits[j], color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    # axes[1].set_ylabel('Velocity (rad/s)')
    # axes[1].set_title('Joint Velocities')
    # axes[1].legend(loc='right', fontsize=8)
    # axes[1].grid(True, alpha=0.3)

    # # Plot accelerations
    # for j in range(7):
    #     axes[2].plot(t_samples, qddot_samples[:, j], label=f'qddot{j+1}', alpha=0.7)
    #     # Plot acceleration limits
    #     axes[2].axhline(-acceleration_limits[j], color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    #     axes[2].axhline(acceleration_limits[j], color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    # axes[2].set_xlabel('Time (s)')
    # axes[2].set_ylabel('Acceleration (rad/s²)')
    # axes[2].set_title('Joint Accelerations')
    # axes[2].legend(loc='right', fontsize=8)
    # axes[2].grid(True, alpha=0.3)

    # plt.tight_layout()

    # # Save the plot
    # from pathlib import Path
    # output_dir = Path(__file__).parent.parent.parent / "outputs"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # save_path = output_dir / "toppra_trajectory_analysis.png"
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # print(f"Plot saved to: {save_path}")

    # plt.show(block=False)
    # plt.pause(0.1)

    print(f"\nAttempting TOPPRA with {num_grid_points} gridpoints...")

    toppra = Toppra(
        path=trajectory,
        plant=plant,
        gridpoints=gridpoints,
    )

    toppra.AddJointVelocityLimit(-velocity_limits, velocity_limits)
    toppra.AddJointAccelerationLimit(-acceleration_limits, acceleration_limits)
    time_trajectory = toppra.SolvePathParameterization()
    # if time_trajectory is None:
    #     print("\n❌ TOPPRA failed! Trying with fewer gridpoints...")
    #     # Try with fewer gridpoints to isolate the issue
    #     for num_points in [500, 250, 100, 50]:
    #         print(f"  Retrying with {num_points} gridpoints...")
    #         gridpoints_retry = np.linspace(
    #             trajectory.start_time(), trajectory.end_time(), num_points
    #         )
    #         toppra_retry = Toppra(
    #             path=trajectory,
    #             plant=plant,
    #             gridpoints=gridpoints_retry,
    #         )
    #         toppra_retry.AddJointVelocityLimit(-velocity_limits, velocity_limits)
    #         toppra_retry.AddJointAccelerationLimit(-acceleration_limits, acceleration_limits)
    #         time_trajectory = toppra_retry.SolvePathParameterization()
    #         if time_trajectory is not None:
    #             print(f"✓ Success with {num_points} gridpoints!")
    #             return PathParameterizedTrajectory(trajectory, time_trajectory)

    #     raise RuntimeError("TOPPRA failed to solve the problem even with reduced gridpoints.")

    print("✓ TOPPRA succeeded!")
    return PathParameterizedTrajectory(trajectory, time_trajectory)
