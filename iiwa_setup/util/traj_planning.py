import numpy as np

from pydrake.all import (
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    PiecewisePolynomial,
    Rgba,
)

from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra


def compute_simple_traj_from_q1_to_q2(
    plant,
    q1: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
):
    print("Generating simple trajectory from q1 to q2")
    path = PiecewisePolynomial.FirstOrderHold([0, 1], np.column_stack((q1, q2)))

    print("Updating with TOPPRA to enforce velocity and acceleration limits")
    traj = reparameterize_with_toppra(
        path,
        plant,
        velocity_limits=vel_limits,
        acceleration_limits=acc_limits,
    )

    print("Trajectory generation complete!")
    return traj


def setup_trajectory_optimization_from_q1_to_q2(
    station,
    q1: np.ndarray,
    q2: np.ndarray,
    duration_constraints: tuple[float, float],
    num_control_points: int = 10,
    duration_cost: float = 1.0,
    path_length_cost: float = 1.0,
    visualize_solving: bool = False,
):
    optimization_plant = station.get_optimization_plant()
    internal_plant = station.get_internal_plant()
    internal_context = station.get_internal_plant_context()
    num_q = optimization_plant.num_positions()

    print("Planning initial trajectory from q1 to q2")

    trajopt = KinematicTrajectoryOptimization(num_q, num_control_points, spline_order=4)
    prog = trajopt.get_mutable_prog()

    # ============= Costs =============
    trajopt.AddDurationCost(duration_cost)
    trajopt.AddPathLengthCost(path_length_cost)

    # ============= Bounds =============
    trajopt.AddPositionBounds(
        optimization_plant.GetPositionLowerLimits(),
        optimization_plant.GetPositionUpperLimits(),
    )
    trajopt.AddVelocityBounds(
        optimization_plant.GetVelocityLowerLimits(),
        optimization_plant.GetVelocityUpperLimits(),
    )

    # ============= Constraints =============
    trajopt.AddDurationConstraint(duration_constraints[0], duration_constraints[1])

    # Position
    trajopt.AddPathPositionConstraint(q1, q1, 0.0)
    trajopt.AddPathPositionConstraint(q2, q2, 1.0)
    # Use quadratic consts to encourage q current and q goal
    prog.AddQuadraticErrorCost(np.eye(num_q), q1, trajopt.control_points()[:, 0])
    prog.AddQuadraticErrorCost(np.eye(num_q), q2, trajopt.control_points()[:, -1])

    # Velocity (TOPPRA assumes zero start and end velocities)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1)

    if visualize_solving:

        def PlotPath(control_points):
            """
            Visualize the end-effector path in Meshcat
            """
            rgba = Rgba(0, 1, 0, 1)
            cps = control_points.reshape((num_q, num_control_points))
            # Reconstruct the spline trajectory
            traj = BsplineTrajectory(trajopt.basis(), cps)
            s_samples = np.linspace(0, 1, 100)
            ee_positions = []
            for s in s_samples:
                q = traj.value(s).flatten()
                internal_plant.SetPositions(internal_context, q)
                X_WB = internal_plant.EvalBodyPoseInWorld(
                    internal_context,
                    internal_plant.GetBodyByName("iiwa_link_7"),
                )
                ee_positions.append(X_WB.translation())
            ee_positions = np.array(ee_positions).T  # shape (3, N)
            station.internal_meshcat.SetLine(
                "positions_path",
                ee_positions,
                line_width=0.05,
                rgba=rgba,
            )

        prog.AddVisualizationCallback(PlotPath, trajopt.control_points().reshape((-1,)))

    return trajopt, prog


def add_collision_constraints_to_trajectory(
    station,
    trajopt: KinematicTrajectoryOptimization,
    num_samples: int = 25,
    minimum_distance: float = 0.001,
):
    """
    Add collision avoidance constraints to the trajectory optimization.
    """

    optimization_plant = station.get_optimization_plant()
    optimization_plant_context = (
        station.internal_station.get_optimization_plant_context()
    )

    collision_constraint = MinimumDistanceLowerBoundConstraint(
        optimization_plant,
        minimum_distance,
        optimization_plant_context,
        None,
    )

    evaluate_at_s = np.linspace(0, 1, num_samples)  # TODO: Use a diff value?
    for s in evaluate_at_s:
        trajopt.AddPathPositionConstraint(collision_constraint, s)

    return trajopt


def resolve_with_toppra(
    station,
    trajopt: KinematicTrajectoryOptimization,
    result,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
):
    # Use controller plant because we don't need to check for collisions here
    controller_plant = station.get_iiwa_controller_plant()

    # Reparameterize with TOPPRA
    geometric_path = trajopt.ReconstructTrajectory(result)

    # Plot joint trajectories
    ts = np.linspace(geometric_path.start_time(), geometric_path.end_time(), 100)
    # qs = np.array([geometric_path.value(t) for t in ts])
    # plt.figure()
    # for i in range(qs.shape[1]):
    #     plt.plot(ts, qs[:, i], label=f"Joint {i+1}")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Joint Position [rad]")
    # plt.title("Geometric Path Joint Positions")
    # plt.legend()
    # plt.savefig("output/geometric_path.png")
    # plt.close()

    trajectory = reparameterize_with_toppra(
        geometric_path,
        controller_plant,
        velocity_limits=vel_limits,
        acceleration_limits=acc_limits,
    )

    return trajectory
