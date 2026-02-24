# General imports
import argparse

from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Drake imports
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    JointSliders,
    KinematicTrajectoryOptimization,
    MeshcatPoseSliders,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Rgba,
    RigidTransform,
    Simulator,
    Solve,
    TrajectorySource,
)
from pydrake.systems.drawing import plot_system_graphviz
from termcolor import colored

# Personal files
from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.util.traj_planning import (
    resolve_with_toppra,
    setup_trajectory_optimization_from_q1_to_q2,
)
from iiwa_setup.util.visualizations import draw_sphere
from utils.hemisphere_solver import load_joint_poses_from_csv
from utils.kuka_geo_kin import KinematicsSolver


class State(Enum):
    IDLE = auto()
    PLANNING = auto()
    MOVING = auto()


def plot_path_with_frames(
    path_points,
    path_rots,
    output_path,
    frame_scale=0.01,
    num_frames=10,
):
    """
    Plot a 3D path with coordinate frames along it.

    Args:
        path_points: (3, N) array of positions along path
        path_rots: List of (3, 3) rotation matrices at each point
        hemisphere_pos: (3,) array of hemisphere center position
        hemisphere_radius: Radius of hemisphere
        output_path: Path object where to save the figure
        frame_scale: Scale factor for coordinate frame arrows (meters)
        num_frames: Approximate number of frames to display
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw path
    ax.plot(
        path_points[0, :],
        path_points[1, :],
        path_points[2, :],
        label="Hemisphere Path",
        linewidth=2,
    )

    # Draw coordinate frames along the path (subsample for clarity)
    frame_step = max(1, len(path_rots) // num_frames)
    quiver_length = 0.2
    linewidth = 0.5

    for i in range(0, len(path_rots), frame_step):
        pos = path_points[:, i]
        R = path_rots[i]

        # Extract each axis and scale uniformly
        x_axis = R[:, 0] * frame_scale  # First column
        y_axis = R[:, 1] * frame_scale  # Second column
        z_axis = R[:, 2] * frame_scale  # Third column

        # X axis (red)
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="red",
            arrow_length_ratio=0.2,
            linewidth=linewidth,
            length=quiver_length,
            normalize=False,
        )
        # Y axis (green)
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="green",
            arrow_length_ratio=0.2,
            linewidth=linewidth,
            length=quiver_length,
            normalize=False,
        )
        # Z axis (blue)
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="blue",
            arrow_length_ratio=0.2,
            linewidth=linewidth,
            length=quiver_length,
            normalize=False,
        )

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Generated Path Along Hemisphere with Coordinate Frames")
    ax.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_waypoints_down_optical_axis(
    pose_curr: RigidTransform, num_points: int = 100
):
    """
    Generate waypoints along the optical axis of the current end-effector pose.

    Args:
        pose_curr: RigidTransform of current end-effector pose
        num_points: Number of points to generate along the optical axis

    Returns:
        List of RigidTransform representing the waypoints
    """
    waypoints = []
    for i in range(num_points):
        # Linear interpolation along the optical axis (z-axis of end-effector)
        t = i / (num_points - 1)
        translation = pose_curr.translation() + t * pose_curr.rotation().matrix()[:, 2]
        waypoints.append(RigidTransform(pose_curr.rotation(), translation))
    return waypoints


def generate_IK_solutions_for_path(path_points, path_rots, kinematics_solver, q_init):
    trajectory_joint_poses = []
    q_prev = (
        q_init  # Try to match first point to current joint configuration for smoothness
    )

    for i in range(len(path_points.T)):
        eef_pos = path_points[:, i]  # Shift spiral to be around the hemisphere center
        eef_rot = path_rots[i]  # Use the rotation matrix from the path
        if i == 0:
            Q, elbow_angles = kinematics_solver.IK_for_microscope_multiple_elbows(
                eef_rot, eef_pos, num_elbow_angles=100, track_elbow_angle=True
            )
            q_curr, idx = kinematics_solver.find_closest_solution(
                Q, q_prev, return_index=True
            )
            elbow_angle = elbow_angles[idx]

        else:
            Q = kinematics_solver.IK_for_microscope(  # NOTE: Just using 0 elbow angle for now
                eef_rot, eef_pos, psi=elbow_angle
            )
            # Choose the solution closest to the previous one for smoothness
            q_curr = kinematics_solver.find_closest_solution(Q, q_prev)
        # else:
        #     q_curr = Q[0]  # Just pick the first solution if no previous solution exists

        trajectory_joint_poses.append(q_curr)
        q_prev = q_curr

    trajectory_joint_poses = np.array(trajectory_joint_poses).T  # Shape (7, num_points)

    return trajectory_joint_poses


def main(use_hardware: bool) -> None:
    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    # - add_model:
    #     name: sphere_obstacle
    #     file: package://iiwa_setup/sphere_obstacle.sdf
    # - add_weld:
    #     parent: world
    #     child: sphere_obstacle::sphere_body
    #     X_PC:
    #         translation: [0.5, 0.0, 0.6]
    plant_config:
        # For some reason, this requires a small timestep
        time_step: 0.005
        contact_model: "hydroelastic_with_fallback"
        discrete_contact_approximation: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            lcm_bus: "default"
            control_mode: position_only
    lcm_buses:
        default:
            lcm_url: ""
    """

    # ===================================================================
    # Diagram Setup
    # ===================================================================
    builder = DiagramBuilder()

    # Load scenario
    scenario = LoadScenario(data=scenario_data)
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            use_hardware=use_hardware,
        ),
    )
    # Load all values I use later
    controller_plant = station.get_iiwa_controller_plant()

    # eef pose sliders
    # Get initial end-effector pose from robot's default joint configuration
    plant_context = controller_plant.CreateDefaultContext()
    controller_plant.SetPositions(
        plant_context,
        controller_plant.GetPositions(plant_context),  # Uses YAML-specified positions
    )

    internal_plant = station.get_internal_plant()
    internal_plant_context = internal_plant.CreateDefaultContext()
    initial_eef_pose = internal_plant.GetFrameByName(
        "microscope_tip_link"
    ).CalcPoseInWorld(internal_plant_context)

    # eef pose sliders
    # Set up teleop widgets
    eef_teleop = builder.AddSystem(
        MeshcatPoseSliders(
            station.internal_meshcat,
            lower_limit=[0, -0.5, -np.pi, -0.6, -0.8, 0.0],
            upper_limit=[2 * np.pi, np.pi, np.pi, 0.8, 0.3, 1.1],
            initial_pose=initial_eef_pose,
        )
    )

    # Create dummy constant position source (using station's default position)
    default_position = station.get_iiwa_controller_plant().GetPositions(
        station.get_iiwa_controller_plant().CreateDefaultContext()
    )
    dummy = builder.AddSystem(ConstantVectorSource(default_position))

    # Add connections (using dummy instead of teleop)
    builder.Connect(
        dummy.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    # Visualize internal station with Meshcat
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    # Build diagram
    diagram = builder.Build()

    # ==================================================================
    # Simulator Setup
    # ====================================================================
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Execute Trajectory")
    station.internal_meshcat.AddButton("Stop Simulation")
    execute_trajectory_clicks = 0

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    vel_limits = np.full(7, 0.3)  # rad/s
    acc_limits = np.full(7, 0.3)  # rad/s^2
    prev_state = None
    state = State.IDLE

    # Create trajectory
    eef_pose_teleop_context = eef_teleop.GetMyContextFromRoot(simulator.get_context())
    eef_pose_prev = RigidTransform(
        eef_teleop.get_output_port().Eval(eef_pose_teleop_context)
    )
    eef_pose_latest = RigidTransform(eef_pose_prev)

    kinematics_solver = KinematicsSolver(station)

    # IK computation thread state
    ik_thread = None
    ik_result = {"ready": False, "trajectory": None, "trajectory_start_time": None}

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        # Button Management
        new_execute_trajectory_clicks = station.internal_meshcat.GetButtonClicks(
            "Execute Trajectory"
        )

        if prev_state != state:
            print(colored(f"State changed: {prev_state} -> {state}", "cyan"))
            prev_state = state

        if state == State.IDLE:
            # Check if current update is different from most recent update.
            # If it is, plan traj from current pose to new pose (not latest pose to new pose)
            eef_pose_teleop_context = eef_teleop.GetMyContextFromRoot(
                simulator.get_context()
            )
            eef_pose_current = eef_teleop.get_output_port().Eval(
                eef_pose_teleop_context
            )
            if not eef_pose_current.IsNearlyEqualTo(eef_pose_latest, 1e-6):
                print(
                    colored(
                        "Teleop sliders changed, re-planning trajectory...", "yellow"
                    )
                )

                # Step 1) Solve IK for desired pose
                Q = kinematics_solver.IK_for_microscope_multiple_elbows(
                    eef_pose_current.rotation().matrix(),
                    eef_pose_current.translation(),
                )

                # Step 2) Find IK closest to current joint values
                station_context = station.GetMyContextFromRoot(simulator.get_context())
                q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                    station_context
                )
                q_des = kinematics_solver.find_closest_solution(Q, q_curr)

                # Step 3) Plan trajectory from current joint values to IK solution
                (
                    trajopt,
                    prog,
                    traj_plot_state,
                ) = setup_trajectory_optimization_from_q1_to_q2(
                    station=station,
                    q1=q_curr,
                    q2=q_des,
                    vel_limits=vel_limits,
                    acc_limits=acc_limits,
                    duration_constraints=(0.5, 5.0),
                    num_control_points=10,
                    duration_cost=1.0,
                    path_length_cost=1.0,
                    visualize_solving=True,
                )

                # Solve for initial guess
                traj_plot_state["rgba"] = Rgba(
                    1, 0.5, 0, 1
                )  # Set initial guess color to orange
                result = Solve(prog)

                if not result.is_success():
                    print(colored("Trajectory optimization failed!", "red"))
                    # Reset errors back
                    eef_pose_latest = RigidTransform(eef_pose_current)
                    state = State.IDLE
                    continue
                else:
                    print(colored("✓ Trajectory optimization succeeded!", "green"))

                trajectory = resolve_with_toppra(  # At this point all this is doing is time-optimizing to make the traj as fast as possible
                    station,
                    trajopt,
                    result,
                    vel_limits,
                    acc_limits,
                )
                print(
                    f"✓ TOPPRA succeeded! Trajectory duration: {trajectory.end_time():.2f}s"
                )

                eef_pose_latest = RigidTransform(
                    eef_pose_current
                )  # Update latest pose to current pose after planning new trajectory

            if new_execute_trajectory_clicks > execute_trajectory_clicks:
                eef_pose_prev = RigidTransform(eef_pose_current)
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING

        elif state == State.MOVING:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= trajectory.end_time():
                q_desired = trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Trajectory execution complete!", "green"))
                state = State.IDLE

        # Reset buttons here in case there are misclicks while the trajectory is executing
        execute_trajectory_clicks = new_execute_trajectory_clicks

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Execute Trajectory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
