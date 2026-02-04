import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    AddFrameTriadIllustration,
    ApplySimulatorConfig,
    BsplineBasis,
    BsplineTrajectory,
    CoulombFriction,
    DiagramBuilder,
    InverseKinematics,
    JointSliders,
    KinematicTrajectoryOptimization,
    KnotVectorType,
    Meshcat,
    MeshcatVisualizer,
    MinimumDistanceLowerBoundConstraint,
    PiecewisePolynomial,
    Rgba,
    RigidTransform,
    SceneGraphCollisionChecker,
    Simulator,
    Solve,
    SpatialInertia,
    Sphere,
    UnitInertia,
)
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.primitives import FirstOrderLowPassFilter

from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra
from iiwa_setup.util.traj_planning import (
    add_collision_constraints_to_trajectory,
    compute_simple_traj_from_q1_to_q2,
    resolve_with_toppra,
    setup_trajectory_optimization_from_q1_to_q2,
)
from iiwa_setup.util.visualizations import draw_sphere

# Personal files
from scripts.hemisphere_solver import (
    SphereScorer,
    find_best_hemisphere_center,
    generate_hemisphere_joint_poses,
)
from scripts.kuka_geo_kin import KinematicsSolver


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

    # Load all values I use later
    internal_station = station.internal_station
    internal_plant = station.get_internal_plant()
    controller_plant = station.get_iiwa_controller_plant()

    # Load teleop sliders
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
    station.internal_meshcat.AddButton("Plan Trajectory")
    station.internal_meshcat.AddButton("Move to Goal")

    # ====================================================================
    # Compute all joint poses for sphere scanning
    # ====================================================================
    # Solve example IK
    # draw_sphere(
    #     station.internal_meshcat,
    #     "target_sphere",
    #     position=hemisphere_pos,
    #     radius=hemisphere_radius,
    # )

    kinematics_solver = KinematicsSolver(station)
    _, path_joint_poses = generate_hemisphere_joint_poses(
        station=station,
        center=hemisphere_pos,
        radius=hemisphere_radius,
        num_poses=30,
        num_rotations_per_pose=7,
        num_elbow_positions=10,
        kinematics_solver=kinematics_solver,
    )

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    move_clicks = 0
    plan_clicks = 0
    trajectory = None
    execute_trajectory = False
    trajectory_start_time = 0.0
    path_idx = 0
    vel_limits = np.full(7, 1.0)  # rad/s
    acc_limits = np.full(7, 1.0)  # rad/s^2
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        new_move_clicks = station.internal_meshcat.GetButtonClicks("Move to Goal")
        new_plan_clicks = station.internal_meshcat.GetButtonClicks("Plan Trajectory")
        if new_plan_clicks > plan_clicks:
            plan_clicks = new_plan_clicks
            print("Planning trajectory...")

            if path_idx >= len(path_joint_poses) - 1:
                print("Completed all joint poses for hemisphere scanning.")
                continue

            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            trajopt, prog = setup_trajectory_optimization_from_q1_to_q2(
                station=station,
                q1=q_current,
                q2=path_joint_poses[path_idx],
                duration_constraints=(0.5, 5.0),
                num_control_points=10,
                duration_cost=1.0,
                path_length_cost=1.0,
                visualize_solving=True,
            )

            # Solve for initial guess
            result = Solve(prog)
            if not result.is_success():
                print("Trajectory optimization failed, even without collisions!")
                print(result.get_solver_id().name())
            trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))

            trajopt = add_collision_constraints_to_trajectory(
                station,
                trajopt,
            )

            # Solve for trajectory with collision avoidance
            result = Solve(prog)
            if not result.is_success():
                print("Trajectory optimization failed")
                print(result.get_solver_id().name())
                continue

            print("Trajectory optimization succeeded!")

            trajectory = resolve_with_toppra(
                station,
                trajopt,
                result,
                vel_limits,
                acc_limits,
            )

            print(
                f"✓ TOPPRA succeeded! Trajectory duration: {trajectory.end_time():.2f}s"
            )

            path_idx += 1

        # If we have a trajectory, execute it
        if new_move_clicks > move_clicks:  # Triggered when Move to Goal is pressed
            move_clicks = new_move_clicks
            if trajectory is None:
                print("No trajectory planned yet!")
            else:
                print("Executing trajectory...")
                execute_trajectory = True
                trajectory_start_time = simulator.get_context().get_time()

        if execute_trajectory:
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
                print("✓ Trajectory execution complete!")
                trajectory = None
                execute_trajectory = False

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Plan Trajectory")
    station.internal_meshcat.DeleteButton("Move to Goal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
