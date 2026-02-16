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
    KinematicTrajectoryOptimization,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Simulator,
    Solve,
    TrajectorySource,
)
from pydrake.systems.drawing import plot_system_graphviz
from termcolor import colored

# Personal files
from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra
from iiwa_setup.util.traj_planning import (
    add_collision_constraints_to_trajectory,
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

    # Load teleop sliders
    teleop = builder.AddSystem(
        JointSliders(
            station.internal_meshcat,
            controller_plant,
        )
    )

    # Make constant trajectory from t=0 to t=10 (any long duration)
    controller_context = controller_plant.CreateDefaultContext()
    q_current = controller_plant.GetPositions(controller_context)
    times = [0.0, 10.0]
    knots = np.column_stack([q_current, q_current])

    traj = PiecewisePolynomial.ZeroOrderHold(times, knots)
    traj_source = builder.AddSystem(TrajectorySource(traj))

    # Add connections
    builder.Connect(
        traj_source.get_output_port(),
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

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    vel_limits = np.full(7, 0.3)  # rad/s
    acc_limits = np.full(7, 0.3)  # rad/s^2
    state = State.IDLE

    # Create trajectory
    station_context = station.GetMyContextFromRoot(simulator.get_context())
    q_current = station.GetOutputPort("iiwa.position_measured").Eval(station_context)

    teleop_context = teleop.GetMyContextFromRoot(simulator.get_context())
    q_prev = teleop.get_output_port().Eval(teleop_context)

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if state == State.IDLE:
            # Check if teleop sliders have changed
            teleop_context = teleop.GetMyContextFromRoot(simulator.get_context())
            q_current = teleop.get_output_port().Eval(teleop_context)
            if (q_current != q_prev).any():
                print(
                    colored(
                        "Teleop sliders changed, re-planning trajectory...", "yellow"
                    )
                )
                trajopt, prog, _ = setup_trajectory_optimization_from_q1_to_q2(
                    station=station,
                    q1=q_prev,
                    q2=q_current,
                    vel_limits=vel_limits,
                    acc_limits=acc_limits,
                    duration_constraints=(0.5, 5.0),
                    num_control_points=10,
                    duration_cost=1.0,
                    path_length_cost=1.0,
                    visualize_solving=True,
                )
                result = Solve(prog)

                traj = resolve_with_toppra(  # At this point all this is doing is time-optimizing to make the traj as fast as possible
                    station,
                    trajopt,
                    result,
                    vel_limits,
                    acc_limits,
                )

                traj_source.UpdateTrajectory(traj)
                q_prev = q_current
                # state = State.MOVING
                # trajectory_start_time = simulator.get_context().get_time()
                traj_source.UpdateTrajectory(traj)
                print(colored("Trajectory planned, executing...", "green"))
        # elif (state == State.MOVING):
        #     current_time = simulator.get_context().get_time()
        #     traj_time = current_time - trajectory_start_time

        #     if traj_time <= traj.end_time():
        #         q_desired = traj.value(traj_time)
        #         station_context = station.GetMyMutableContextFromRoot(
        #             simulator.get_mutable_context()
        #         )
        #         station.GetInputPort("iiwa.position").FixValue(
        #             station_context, q_desired
        #         )
        #     else:
        #         print(colored("✓ Trajectory execution complete!", "green"))
        #         state = State.IDLE

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
