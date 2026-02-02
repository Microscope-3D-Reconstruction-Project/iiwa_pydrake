import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
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

# Personal files
from scripts.hemisphere_solver import generate_hemisphere_joint_poses


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
        IiwaHardwareStationDiagram(scenario=scenario, use_hardware=use_hardware),
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
    station.internal_meshcat.AddButton("Plan Trajectory")
    station.internal_meshcat.AddButton("Move to Goal")

    # ====================================================================
    # Compute all joint poses for sphere scanning
    # ====================================================================

    test = generate_hemisphere_joint_poses(
        station=station,
        center=[0.7, 0.0, 0.8],
        radius=0.15,
        num_poses=30,
        num_rotations_per_pose=7,
        num_elbow_positions=10,
    )

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
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
