# General imports
import argparse

from enum import Enum, auto
from pathlib import Path

import numpy as np

# Drake imports
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    JointSliders,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Rgba,
    Simulator,
    Solve,
)
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.primitives import FirstOrderLowPassFilter
from termcolor import colored

# Personal files
from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra
from iiwa_setup.util.traj_planning import create_traj_from_q1_to_q2
from iiwa_setup.util.visualizations import draw_sphere
from utils.hemisphere_solver import load_joint_poses_from_csv
from utils.kuka_geo_kin import KinematicsSolver


class State(Enum):
    IDLE = auto()
    MOVING = auto()
    MOVING_TO_START = auto()


def generate_spiral_points(
    center, num_spirals=2, max_radius=0.05, num_points=100, t_final=10.0
):
    # ================================================================
    # Spiral trajectory generation
    # ================================================================

    t = np.linspace(0, t_final, num_points)
    theta = np.linspace(0, num_spirals * 2 * np.pi, num_points)

    # Spiral starts at center (radius=0) and grows linearly to max_radius
    radius = (theta / theta[-1]) * max_radius

    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    z = np.zeros_like(x) + center[2]
    points = np.vstack([x, y, z])

    return points, t


def generate_hemisphere_points(center, radius, num_points=100, t_final=10.0):
    """
    Args:
        center: (x, y, z) coordinates of the hemisphere center
        radius: radius of the hemisphere
        num_points: number of points to generate on the hemisphere surface
        t_final: total time for trajectory (used to create time array for PiecewisePolynomial)
    """

    t = np.linspace(0, t_final, num_points)


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
    hemisphere_pos = np.array([0.6666666, 0.0, 0.444444])
    hemisphere_radius = 0.05
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_pos=hemisphere_pos,
            hemisphere_radius=hemisphere_radius,
            use_hardware=use_hardware,
        ),
    )

    # Load teleop sliders
    controller_plant = station.get_iiwa_controller_plant()
    teleop = builder.AddSystem(
        JointSliders(
            station.internal_meshcat,
            controller_plant,
        )
    )

    # Add connections
    builder.Connect(
        teleop.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    # Visualize internal station with Meshcat
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    # Build diagram
    diagram = builder.Build()

    # ====================================================================
    # Simulator Setup
    # ====================================================================
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    station.internal_meshcat.AddButton("Move to Top of Spiral")
    station.internal_meshcat.AddButton("Execute Trajectory")

    # Spiral parameters
    num_spirals = 2  # Number of complete rotations
    num_scan_points = 5  # Number of points where robot should stop along the spiral (including start and end)
    max_radius = 0.05  # Maximum radius in meters
    num_points = 100
    t_final = 10.0  # Total time for trajectory
    points, t = generate_spiral_points(
        hemisphere_pos,
        num_spirals=num_spirals,
        max_radius=max_radius,
        num_points=num_points,
        t_final=t_final,
    )

    # region Visualization of spiral in matplotlib (for debugging)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))

    # 3D view
    x, y, z = points
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(x, y, z, "b-", linewidth=2)
    ax1.scatter(x[0], y[0], z[0], c="g", s=100, marker="o", label="Start")
    ax1.scatter(x[-1], y[-1], z[-1], c="r", s=100, marker="x", label="End")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("Spiral Path (3D)")
    ax1.legend()
    ax1.set_aspect("equal")

    # Top-down view
    ax2 = fig.add_subplot(122)
    ax2.plot(x, y, "b-", linewidth=2)
    ax2.scatter(x[0], y[0], c="g", s=100, marker="o", label="Start")
    ax2.scatter(x[-1], y[-1], c="r", s=100, marker="x", label="End")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Spiral Path (Top View)")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True)

    plt.tight_layout()

    # Save to outputs folder
    current_dir = Path(__file__).parent.parent
    outputs_dir = current_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    spiral_path_file = outputs_dir / "spiral_path.png"
    plt.savefig(spiral_path_file)
    print(colored(f"Spiral path plot saved to {spiral_path_file}", "cyan"))
    plt.close()
    # endregion

    # region Step 1) Solve IK for desired pose
    kinematics_solver = KinematicsSolver(station)

    station_context = station.GetMyContextFromRoot(simulator.get_context())
    q_prev = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
    q_curr = None
    elbow_angle = None

    trajectory_joint_poses = []
    for i in range(num_points):
        eef_pos = points[:, i]  # Shift spiral to be around the hemisphere center
        eef_rot = np.eye(3)  # Keep end-effector orientation fixed

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
    # endregion

    # region Step 2) Break down trajectory into segments between scan points and convert to PiecewisePolynomial
    segment_length = num_points // (num_scan_points - 1)
    print(
        colored(
            f"Segment length (number of points between scan points): {segment_length}",
            "cyan",
        )
    )
    spiral_trajectories = []
    for i in range(num_scan_points - 1):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < num_scan_points - 2 else num_points

        # Create time array that starts from 0 for each segment
        num_segment_points = end_idx - start_idx
        segment_duration = t[min(end_idx - 1, len(t) - 1)] - t[start_idx]
        segment_times = np.linspace(0, segment_duration, num_segment_points)

        sub_traj = PiecewisePolynomial.FirstOrderHold(
            segment_times, trajectory_joint_poses[:, start_idx:end_idx]
        )
        spiral_trajectories.append(sub_traj)

    print(colored("Number of trajectory segments:", "cyan"), len(spiral_trajectories))
    # endregion

    # region Step 3) Plot full trajectory into Meshcat
    desired_ee_positions = points
    station.internal_meshcat.SetLine(
        "desired_spiral_path",
        desired_ee_positions,
        line_width=0.05,
        rgba=Rgba(1, 1, 1, 1),
    )
    print(colored("Spiral trajectory generated and visualized in Meshcat!", "cyan"))
    # endregion

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    state = State.IDLE

    # Button management
    num_move_to_top_clicks = 0
    num_execute_traj_clicks = 0
    current_segment = 0

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if (
            state == State.IDLE
            and station.internal_meshcat.GetButtonClicks("Execute Trajectory")
            > num_execute_traj_clicks
            and not station.internal_meshcat.GetButtonClicks("Move to Top of Spiral")
            == 0  # If already moved to top of spiral
        ):
            num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
                "Execute Trajectory"
            )

            if current_segment >= len(spiral_trajectories):
                print(colored("All trajectory segments executed", "red"))
                break

            print(colored("Executing segment of spiral trajectory", "cyan"))
            state = State.MOVING
            spiral_trajectory = spiral_trajectories[
                current_segment
            ]  # TODO: Need to execute sub-trajectories sequentially
            current_segment += 1
            trajectory_start_time = simulator.get_context().get_time()

        elif (
            state == State.IDLE
            and station.internal_meshcat.GetButtonClicks("Move to Top of Spiral")
            > num_move_to_top_clicks
        ):
            num_move_to_top_clicks = station.internal_meshcat.GetButtonClicks(
                "Move to Top of Spiral"
            )
            print(colored("Moving to top of spiral!", "cyan"))

            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )
            q_top_of_spiral = trajectory_joint_poses[
                :, 0
            ]  # Get first point of trajectory

            # Position limits
            q_lower = controller_plant.GetPositionLowerLimits()
            q_upper = controller_plant.GetPositionUpperLimits()
            print(colored("Joint limits:", "yellow"))
            for i in range(7):
                print(
                    colored(
                        f"  Joint {i+1}: [{q_lower[i]:.2f}, {q_upper[i]:.2f}]", "yellow"
                    )
                )
            print(colored(f"Current joint configuration: {q_current}", "yellow"))
            print(
                colored(
                    f"Goal joint configuration for top of spiral: {q_top_of_spiral}",
                    "yellow",
                )
            )

            start_trajectory = create_traj_from_q1_to_q2(
                station,
                q_current,
                q_top_of_spiral,
            )

            state = State.MOVING_TO_START
            trajectory_start_time = simulator.get_context().get_time()

        elif state == State.MOVING_TO_START:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= start_trajectory.end_time():
                q_desired = start_trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Trajectory execution complete!", "green"))
                state = State.IDLE

        elif state == State.MOVING:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= spiral_trajectory.end_time():
                q_desired = spiral_trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Trajectory execution complete!", "green"))
                state = State.IDLE

        # Update button counts
        num_move_to_top_clicks = station.internal_meshcat.GetButtonClicks(
            "Move to Top of Spiral"
        )
        num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
            "Execute Trajectory"
        )

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Move to Top of Spiral")
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
