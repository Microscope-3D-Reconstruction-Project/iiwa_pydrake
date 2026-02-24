# General imports
import argparse
import threading

from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Drake imports
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from mpl_toolkits.mplot3d import Axes3D
from pydrake.all import (
    AddFrameTriadIllustration,
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
)
from pydrake.multibody.tree import FrameIndex
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
    WAITING_FOR_NEXT_SCAN = auto()
    MOVING = auto()
    MOVING_TO_START = auto()
    COMPUTING_IK = auto()


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

    # Set axis limits based on hemisphere bounds with some padding
    ax.set_xlim(
        hemisphere_pos[0] - hemisphere_radius, hemisphere_pos[0] + hemisphere_radius
    )
    ax.set_ylim(
        hemisphere_pos[1] - hemisphere_radius, hemisphere_pos[1] + hemisphere_radius
    )
    ax.set_zlim(
        hemisphere_pos[2] - hemisphere_radius, hemisphere_pos[2] + hemisphere_radius
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


def get_pose_from_scan_point(scan_point):
    """
    Given a scan point on the hemisphere, return a desired end-effector pose (rotation and position) for that point.
    For simplicity, we can use the sphere_frame function to get a rotation matrix that aligns the end-effector z-axis with the surface normal at that point, and x/y axes tangentially.

    Args:
        scan_point: (3,) array of the scan point position on the hemisphere
    Returns:
        target_pose: RigidTransform representing the desired end-effector pose
    """

    target_rot = sphere_frame(scan_point)
    target_pos = scan_point

    return RigidTransform(RotationMatrix(target_rot), target_pos)


def generate_hemisphere_points(center, radius, num_scan_points):
    """
    Generate N approximately uniformly distributed points on a hemisphere.
    Hemisphere "top" is at (-R, 0, 0) along the x-axis.
    Points start at the top and go down.
    """
    points = []
    phi_golden = (1 + np.sqrt(5)) / 2  # golden ratio

    for k in range(num_scan_points):
        # Map k=0 to top (x=1), k=N-1 to equator (x=0)
        x_s = 1 - k / (
            num_scan_points - 1
        )  # height along hemisphere axis from top to equator
        r_xy = np.sqrt(1 - x_s**2)  # radius in yz-plane
        theta = 2 * np.pi * k / phi_golden  # golden angle

        y_s = r_xy * np.cos(theta)
        z_s = r_xy * np.sin(theta)

        # Map to world frame: top at (-R,0,0)
        x_w = -radius * x_s
        y_w = radius * y_s
        z_w = radius * z_s

        points.append([x_w, y_w, z_w] + center)

    # # plot for sanity check
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # points = np.array(points)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    # # Add hemisphere surface for visualization, make sure top is at (-R, 0, 0)
    # u = np.linspace(0, 2 * np.pi, 50)
    # v = np.linspace(0, np.pi / 2, 25)  # Only upper hemisphere
    # x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    # y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    # z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    # ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='cyan', edgecolor='none')
    # ax.set_title("Generated Hemisphere Points")
    # plt.savefig(Path(__file__).parent.parent / "outputs" / "hemisphere_points.png", dpi=150)
    # plt.close()
    # print(colored("✓ Hemisphere points generated and saved to outputs/hemisphere_points.png", "cyan"))

    return np.array(points)


def generate_waypoints_along_hemisphere(
    center, radius, pose_curr, pose_target, num_spirals=2
):
    """
    Args:
        - center: (3,) array of hemisphere center
        - radius: radius of hemisphere
        - pose_curr: RigidTransform of current end-effector pose
        - pose_target: RigidTransform of desired end-effector pose on hemisphere
    Returns:
        - path_points: (3, N) array of positions along the path
        - path_rots: List of (3, 3) rotation matrices at each point along the path
    """

    # Step 1: Generate shortest path along hemisphere surface
    A = pose_curr.translation()
    B = pose_target.translation()
    path_points, t = hemisphere_slerp(A, B, center, radius)

    # Generate rotation matrices along the path using the sphere_frame function
    path_rots = []
    num_points = path_points.shape[1]
    for i in range(num_points):
        p = path_points[:, i]
        R = sphere_frame(p, center)
        path_rots.append(R)

    print("t_final for path (slerp): ", t[-1])
    print(f"Number of time points: {len(t)}")
    print(f"Path points shape: {path_points.shape}")
    print(f"Number of rotations: {len(path_rots)}")

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


def hemisphere_slerp(A, B, center, radius, speed_factor=0.5):
    """
    Interpolate along the shortest path on a hemisphere between points A and B.

    Parameters:
        A, B: np.array, shape (3,) - start and end points on the hemisphere
        center: np.array, shape (3,) - center of the sphere
        radius: float - radius of the hemisphere
        num_points: int - number of points along the path
        hemisphere_axis: int - axis index defining the hemisphere (default 2 -> z>=0)
        speed_factor: float - factor to scale the speed of traversal along the path (default 0.5 for half speed)

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

    # Create time array for PiecewisePolynomial
    # Make t depend on the length of the arc that we are traversing for more natural timing (longer paths take more time)
    arc_length = radius * theta
    t_final = (
        arc_length * 125 / speed_factor
    )  # Scale by speed factor to make it faster or slower
    num_points = int(t_final * 500)  # 500Hz update rate seems good

    t = np.linspace(0, t_final, num_points)

    # if theta < 1e-6:  # points are extremely close
    #     path = np.tile(A, (num_points, 1))
    #     return path

    # Slerp interpolation
    t_vals = np.linspace(0, 1, num_points)
    path = np.zeros((num_points, 3))
    for i, t_val in enumerate(t_vals):
        path[i] = (
            np.sin((1 - t_val) * theta) * a_hat + np.sin(t_val * theta) * b_hat
        ) / np.sin(theta)

    # Scale and shift back to original sphere
    path = center + radius * path

    # # Enforce hemisphere constraint
    # path[:, hemisphere_axis] = np.maximum(path[:, hemisphere_axis], center[hemisphere_axis])

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
    g = np.array([0.0, 0.0, 1.0])

    # # If z is too close to g or -g, choose alternate reference
    # if abs(np.dot(z, g)) > 0.99:
    #     raise NotImplementedError("Need to implement alternate reference direction if z is close to global down. This should be rare for our hemisphere setup since the normal will never point straight down, but could be an issue if we try to use this function for points near the top of the sphere.")

    if np.dot(z, g) > 0.99:
        g = np.array([1.0, 0.0, 0.0])
    elif np.dot(z, g) < -0.99:
        g = np.array([-1.0, 0.0, 0.0])

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
    x = center[0] - radius * np.cos(latitude_rad) * np.cos(longitude_rad)
    y = center[1] - radius * np.cos(latitude_rad) * np.sin(longitude_rad)
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
    hemisphere_pos = np.array([0.6666667, 0.0, 0.344444])
    hemisphere_radius = 0.02
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_pos=hemisphere_pos,
            hemisphere_radius=hemisphere_radius,
            use_hardware=use_hardware,
        ),
    )

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

    # Get total number of frames
    plant = station.get_internal_plant()
    num_frames = plant.num_frames()

    # Iterate through all frames
    for i in range(num_frames):
        frame = plant.get_frame(FrameIndex(i))
        print(f"{i}: {frame.name()} (body: {frame.body().name()})")

    microscope_tip_frame = station.get_internal_plant().GetFrameByName(
        "microscope_tip_link"
    )
    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=station.get_internal_plant(),
        frame=microscope_tip_frame,
        length=0.1,
        radius=0.002,
        name="microscope_link",
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
    station.internal_meshcat.AddButton("Move to Start")
    station.internal_meshcat.AddButton("Execute Trajectory")

    # Add custom sliders for latitude and longitude
    station.internal_meshcat.AddSlider("Latitude", -90, 90, 0.1, 0)
    station.internal_meshcat.AddSlider("Longitude", -180, 180, 0.1, 0)

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

    # IK computation thread state
    ik_thread = None
    ik_result = {"ready": False, "trajectory": None, "trajectory_start_time": None}

    # Generate waypoints
    hemisphere_scan_points = generate_hemisphere_points(
        hemisphere_pos, hemisphere_radius, num_scan_points=30
    )
    scan_idx = 1

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if (
            state
            == State.WAITING_FOR_NEXT_SCAN
            # and station.internal_meshcat.GetButtonClicks("Execute Trajectory")
            # > num_execute_traj_clicks
        ):
            # # Get lat and long from sliders
            # latitude = station.internal_meshcat.GetSliderValue("Latitude")
            # longitude = station.internal_meshcat.GetSliderValue("Longitude")
            # print(
            #     colored(
            #         f"Executing trajectory along hemisphere at lat: {latitude:.2f}, long: {longitude:.2f}",
            #         "cyan",
            #     )
            # )

            # # Generate trajectory waypoints along hemisphere
            # rot_des, pos_des = find_target_pose_on_hemisphere(
            #     hemisphere_pos, latitude, longitude, hemisphere_radius
            # )

            pose_target = get_pose_from_scan_point(hemisphere_scan_points[scan_idx, :])

            # Get current end-effector pose from actual robot state
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            internal_plant = station.get_internal_plant()
            internal_plant_context = station.get_internal_plant_context()
            eef_pose = internal_plant.GetFrameByName(
                "microscope_tip_link"
            ).CalcPoseInWorld(internal_plant_context)
            path_points, path_rots, t = generate_waypoints_along_hemisphere(
                center=hemisphere_pos,
                radius=hemisphere_radius,
                pose_curr=eef_pose,
                pose_target=pose_target,
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

            station.internal_meshcat.SetLine(
                "desired_spiral_path",
                path_points,
                line_width=0.05,
                rgba=Rgba(1, 1, 1, 1),
            )

            # plot t values to check that they are reasonable
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(t)
            plt.title("Time array for trajectory")
            plt.xlabel("Index")
            plt.ylabel("Time (s)")
            plt.grid()
            plt.savefig(
                Path(__file__).parent.parent / "outputs" / "time_array.png", dpi=150
            )
            plt.close()

            # Solve for IK solutions in background thread
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            print(colored("Starting IK computation in background...", "yellow"))

            def compute_ik_async():
                trajectory_joint_poses = generate_IK_solutions_for_path(
                    path_points=path_points,
                    path_rots=path_rots,
                    kinematics_solver=kinematics_solver,
                    q_init=q_curr,
                )

                print(f"Trajectory joint poses shape: {trajectory_joint_poses.shape}")
                print(f"Time array length: {len(t)}")
                print(f"Time array: start={t[0]}, end={t[-1]}")
                print(f"Current robot position: {q_curr}")
                print(f"First trajectory position: {trajectory_joint_poses[:, 0]}")
                print(
                    f"Position error at start: {np.linalg.norm(q_curr - trajectory_joint_poses[:, 0])}"
                )

                # Turn into piecewise polynomial trajectory
                traj = PiecewisePolynomial.FirstOrderHold(t, trajectory_joint_poses)
                print(
                    f"Trajectory start_time: {traj.start_time()}, end_time: {traj.end_time()}"
                )

                # Store results
                ik_result["trajectory"] = traj
                ik_result["ready"] = True
                print(colored("✓ IK computation complete!", "green"))

            ik_result["ready"] = False
            ik_thread = threading.Thread(target=compute_ik_async, daemon=True)
            ik_thread.start()

            num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
                "Execute Trajectory"
            )
            state = State.COMPUTING_IK

        elif (
            state == State.IDLE
            and station.internal_meshcat.GetButtonClicks("Move to Start") > 0
        ):
            print(colored("Moving to start", "cyan"))

            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            # Step 1) Solve IK for Start
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

            print(colored(f"Goal joint configuration for Start: {q_des}", "yellow"))

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
                state = State.WAITING_FOR_NEXT_SCAN

        elif state == State.COMPUTING_IK:
            # Wait for IK computation to complete
            if ik_result["ready"]:
                trajectory = ik_result["trajectory"]
                trajectory_start_time = simulator.get_context().get_time()
                print(
                    f"Simulator time when starting trajectory: {trajectory_start_time}"
                )
                print(colored("Starting trajectory execution!", "cyan"))
                scan_idx += 1

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
                state = State.WAITING_FOR_NEXT_SCAN

        # Update button counts
        num_move_to_top_clicks = station.internal_meshcat.GetButtonClicks(
            "Move to Start"
        )
        num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
            "Execute Trajectory"
        )

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Move to Start")
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
