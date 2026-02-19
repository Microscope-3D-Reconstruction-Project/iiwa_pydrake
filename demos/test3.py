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
    ConstantVectorSource,
    DiagramBuilder,
    JointSliders,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Rgba,
    RigidTransform,
    RotationMatrix,
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


def plot_path_with_frames(
    path_points,
    path_rots,
    hemisphere_pos,
    hemisphere_radius,
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
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw transparent hemisphere sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 25)  # Only upper hemisphere
    x_sphere = hemisphere_pos[0] + hemisphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = hemisphere_pos[1] + hemisphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = hemisphere_pos[2] + hemisphere_radius * np.outer(
        np.ones(np.size(u)), np.cos(v)
    )
    ax.plot_surface(
        x_sphere, y_sphere, z_sphere, alpha=0.2, color="cyan", edgecolor="none"
    )

    # Draw path
    ax.plot(
        path_points[0, :],
        path_points[1, :],
        path_points[2, :],
        label="Hemisphere Path",
        linewidth=2,
    )
    ax.scatter(
        hemisphere_pos[0],
        hemisphere_pos[1],
        hemisphere_pos[2],
        color="red",
        s=100,
        label="Hemisphere Center",
    )

    # Draw coordinate frames along the path (subsample for clarity)
    frame_step = max(1, len(path_rots) // num_frames)

    for i in range(0, len(path_rots), frame_step):
        pos = path_points[:, i]
        R = path_rots[i]

        # Extract and normalize each axis, then scale to desired length
        x_axis = R[:, 0]  # First column
        y_axis = R[:, 1]  # Second column
        z_axis = R[:, 2]  # Third column

        # Normalize and scale
        x_scaled = (x_axis / np.linalg.norm(x_axis)) * frame_scale
        y_scaled = (y_axis / np.linalg.norm(y_axis)) * frame_scale
        z_scaled = (z_axis / np.linalg.norm(z_axis)) * frame_scale

        # X axis (red)
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            x_scaled[0],
            x_scaled[1],
            x_scaled[2],
            color="red",
            arrow_length_ratio=0.3,
            linewidth=1.5,
        )
        # Y axis (green)
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            y_scaled[0],
            y_scaled[1],
            y_scaled[2],
            color="green",
            arrow_length_ratio=0.3,
            linewidth=1.5,
        )
        # Z axis (blue)
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            z_scaled[0],
            z_scaled[1],
            z_scaled[2],
            color="blue",
            arrow_length_ratio=0.3,
            linewidth=1.5,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Generated Path Along Hemisphere with Coordinate Frames")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_waypoints_along_hemisphere(
    center, radius, pose_curr, pose_target, num_points=100, num_spirals=2, t_final=10.0
):
    """
    Args:
        center: (x, y, z) coordinates of the hemisphere center
        radius: radius of the hemisphere
        q_curr: current joint configuration (used as initial guess for IK)
        target_pose: desired end-effector pose (NOTE: Using pose to give flexibility of IK solution
        num_points: number of points to generate along the path
        num_spirals: number of spiral loops to make around the hemisphere (default 2)
        t_final: total time for trajectory (used to create time array for PiecewisePolynomial)
    """

    # Step 1: Generate shortest path along hemisphere surface
    A = pose_curr.translation()
    B = pose_target.translation()
    path_points, t = hemisphere_slerp(
        A, B, center, radius, num_points=num_points, t_final=t_final
    )

    # Generate rotation matrices along the path using the sphere_frame function
    path_rots = []
    for i in range(num_points):
        p = path_points[:, i]
        R = sphere_frame(p, center)
        path_rots.append(R)

    return path_points, path_rots, t


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


def hemisphere_slerp(
    A, B, center, radius, num_points=100, hemisphere_axis=2, t_final=10.0
):
    """
    Interpolate along the shortest path on a hemisphere between points A and B.

    Parameters:
        A, B: np.array, shape (3,) - start and end points on the hemisphere
        center: np.array, shape (3,) - center of the sphere
        radius: float - radius of the hemisphere
        num_points: int - number of points along the path
        hemisphere_axis: int - axis index defining the hemisphere (default 2 -> z>=0)

    Returns:
        path: np.array, shape (num_points, 3) - interpolated points on hemisphere
    """
    # Shift to sphere-centered coordinates
    a = A - center
    b = B - center

    # Normalize to unit sphere
    a_hat = a / np.linalg.norm(a)
    b_hat = b / np.linalg.norm(b)

    # Compute angle between vectors
    dot = np.clip(np.dot(a_hat, b_hat), -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-6:  # points are extremely close
        path = np.tile(A, (num_points, 1))
        return path

    # Slerp interpolation
    t_vals = np.linspace(0, 1, num_points)
    path = np.zeros((num_points, 3))
    for i, t in enumerate(t_vals):
        path[i] = (
            np.sin((1 - t) * theta) * a_hat + np.sin(t * theta) * b_hat
        ) / np.sin(theta)

    # Scale and shift back to original sphere
    path = center + radius * path

    # # Enforce hemisphere constraint
    # path[:, hemisphere_axis] = np.maximum(path[:, hemisphere_axis], center[hemisphere_axis])

    # Create time array for PiecewisePolynomial
    t = np.linspace(0, t_final, num_points)

    return path.T, t


# # Example usage
# A = np.array([0.0, 0.0, 1.0])
# B = np.array([1.0, 1.0, 1.0])
# center = np.array([0.0, 0.0, 0.0])
# radius = 1.0

# path = hemisphere_slerp(A, B, center, radius, num_points=50)
# # print(path)


def sphere_frame(p, center=np.array([0.0, 0.0, 0.0])):
    """
    Compute a smooth end-effector rotation matrix at point p on a sphere.

    z-axis  -> surface normal
    x-axis  -> projected global reference direction (smooth, no twisting)
    y-axis  -> z cross x

    Parameters
    ----------
    p : array-like (3,)
        Point on the sphere.
    center : array-like (3,)
        Sphere center (default origin).

    Returns
    -------
    R : (3,3) numpy array
        Rotation matrix with columns [x, y, z]
    """

    p = np.asarray(p, dtype=float)
    center = np.asarray(center, dtype=float)

    # Surface normal
    z = p - center
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-9:
        raise ValueError("Point cannot equal sphere center.")
    z = z / z_norm

    # Global reference direction (choose something stable)
    # Here: global "down"
    g = np.array([0.0, 0.0, -1.0])

    # If z is too close to g or -g, choose alternate reference
    if abs(np.dot(z, g)) > 0.99:
        g = np.array([1.0, 0.0, 0.0])

    # Project g onto tangent plane
    x = g - np.dot(g, z) * z
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-9:
        raise ValueError("Degenerate tangent direction.")
    x = x / x_norm

    # Complete right-handed frame
    y = np.cross(z, x)

    # Ensure orthonormality (numerical cleanup)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)

    R = np.column_stack((x, y, z))

    return R


def find_target_pose_on_hemisphere(center, latitude_deg, longitude_deg, radius):
    """
    Given a hemisphere defined by its center and radius, find the target end-effector pose on the hemisphere surface corresponding to the specified latitude and longitude angles.

    Args:
        center: (x, y, z) coordinates of the hemisphere center
        latitude_deg: Latitude angle in degrees (-90 to 90)
        longitude_deg: Longitude angle in degrees (-180 to 180)
        radius: Radius of the hemisphere
    Returns:
        target_pose: A 4x4 homogeneous transformation matrix representing the desired end-effector pose
    """

    latitude_rad = np.deg2rad(latitude_deg)
    longitude_rad = np.deg2rad(longitude_deg)
    x = center[0] + radius * np.cos(latitude_rad) * np.cos(longitude_rad)
    y = center[1] + radius * np.cos(latitude_rad) * np.sin(longitude_rad)
    z = center[2] + radius * np.sin(latitude_rad)

    target_pos = np.array([x, y, z])

    target_rot = sphere_frame(target_pos, center)

    return target_rot, target_pos


def main(use_hardware: bool) -> None:
    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    # - add_model:
    #     name: sphere_obstacle
    #     file: package://iiwa_setup/sphere_obstacle.sdf
    # - add_weld:
    #     parent: worldhemisphere_radius
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
    # teleop = builder.AddSystem(
    #     JointSliders(
    #         station.internal_meshcat,
    #         controller_plant,
    #     )
    # )

    # Create dummy constant position source (using station's default position)
    default_position = station.get_iiwa_controller_plant().GetPositions(
        station.get_iiwa_controller_plant().CreateDefaultContext()
    )
    dummy = builder.AddSystem(ConstantVectorSource(default_position))
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

    # ====================================================================
    # Simulator Setup
    # ====================================================================
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    station.internal_meshcat.AddButton("Move to Top of Hemisphere")
    station.internal_meshcat.AddButton("Execute Trajectory")

    # Add custom sliders for latitude and longitude
    station.internal_meshcat.AddSlider("Latitude", -90, 90, 0.1, 90)
    station.internal_meshcat.AddSlider("Longitude", -180, 180, 0.1, 0)

    radius = 0  # Radius of hemisphere. temp, use the other one from station later

    # region Step 1) Solve IK for desired pose
    kinematics_solver = KinematicsSolver(station)

    station_context = station.GetMyContextFromRoot(simulator.get_context())

    # Calculate IK for getting to the top of the hemisphere

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    state = State.IDLE

    # Button management
    num_move_to_top_clicks = 0
    num_execute_traj_clicks = 0

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if (
            state == State.IDLE
            and station.internal_meshcat.GetButtonClicks("Execute Trajectory")
            > num_execute_traj_clicks
            and not station.internal_meshcat.GetButtonClicks(
                "Move to Top of Hemisphere"
            )
            == 0  # If already moved to top of spiral
        ):
            # Get lat and long from sliders
            latitude = station.internal_meshcat.GetSliderValue("Latitude")
            longitude = station.internal_meshcat.GetSliderValue("Longitude")
            print(
                colored(
                    f"Executing trajectory along hemisphere at lat: {latitude:.2f}, long: {longitude:.2f}",
                    "cyan",
                )
            )

            # Generate trajectory waypoints along hemisphere
            rot_des, pos_des = find_target_pose_on_hemisphere(
                hemisphere_pos, latitude, longitude, hemisphere_radius
            )
            # Convert to RigidTransform
            pose_target = RigidTransform(RotationMatrix(rot_des), pos_des)

            # Get current end-effector pose from actual robot state
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            internal_plant = station.get_internal_plant()
            internal_plant_context = station.get_internal_plant_context()
            eef_pose = internal_plant.GetFrameByName(
                "microscope_tip_link"
            ).CalcPoseInWorld(internal_plant_context)
            print(colored(f"Current end-effector pose: \n{eef_pose}", "grey"))
            print(colored(f"Target end-effector pose: \n{pose_target}", "grey"))
            path_points, path_rots, t = generate_waypoints_along_hemisphere(
                center=hemisphere_pos,
                radius=hemisphere_radius,
                pose_curr=eef_pose,
                pose_target=pose_target,
                num_points=100,
                t_final=20.0,
            )

            # Plot and save the path with coordinate frames
            plot_path_with_frames(
                path_points=path_points,
                path_rots=path_rots,
                hemisphere_pos=hemisphere_pos,
                hemisphere_radius=hemisphere_radius,
                output_path=Path(__file__).parent.parent
                / "outputs"
                / "hemisphere_path.png",
                frame_scale=0.01,
                num_frames=10,
            )
            print(
                colored(
                    "✓ Path plotted and saved to outputs/hemisphere_path.png", "green"
                )
            )

            # Solve for IK solutions along the path
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )
            trajectory_joint_poses = generate_IK_solutions_for_path(
                path_points=path_points,
                path_rots=path_rots,
                kinematics_solver=kinematics_solver,
                q_init=q_curr,
            )

            # Turn into piecewise polynomial trajectory
            trajectory = PiecewisePolynomial.FirstOrderHold(t, trajectory_joint_poses)
            num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
                "Execute Trajectory"
            )
            state = State.MOVING
            trajectory_start_time = simulator.get_context().get_time()

        elif (
            state == State.IDLE
            and station.internal_meshcat.GetButtonClicks("Move to Top of Hemisphere")
            > num_move_to_top_clicks
        ):
            num_move_to_top_clicks = station.internal_meshcat.GetButtonClicks(
                "Move to Top of Hemisphere"
            )
            print(colored("Moving to top of hemisphere!", "cyan"))

            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            # Step 1) Solve IK for top of hemisphere pose
            latitude = station.internal_meshcat.GetSliderValue("Latitude")
            longitude = station.internal_meshcat.GetSliderValue("Longitude")
            target_rot, target_pos = find_target_pose_on_hemisphere(
                hemisphere_pos, latitude, longitude, hemisphere_radius
            )

            Q = kinematics_solver.IK_for_microscope_multiple_elbows(
                target_rot,
                target_pos,
            )

            # Step 2) Find IK closest to current joint values
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )
            q_des = kinematics_solver.find_closest_solution(Q, q_curr)

            print(
                colored(
                    f"Goal joint configuration for top of hemisphere: {q_des}", "yellow"
                )
            )

            start_trajectory = create_traj_from_q1_to_q2(
                station,
                q_current,
                q_des,
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

        # Update button counts
        num_move_to_top_clicks = station.internal_meshcat.GetButtonClicks(
            "Move to Top of Hemisphere"
        )
        num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
            "Execute Trajectory"
        )

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Move to Top of Spiral")
    station.internal_meshcat.DeleteButton("Execute Trajectory")
    station.internal_meshcat.DeleteSlider("Latitude")
    station.internal_meshcat.DeleteSlider("Longitude")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
