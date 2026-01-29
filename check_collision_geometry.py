#!/usr/bin/env python3
"""Check if collision geometries are properly loaded."""

from manipulation.station import LoadScenario
from pydrake.all import DiagramBuilder
from iiwa_setup.iiwa import IiwaHardwareStationDiagram

scenario_data = """
directives:
- add_directives:
    file: package://iiwa_setup/iiwa14.dmd.yaml
- add_model:
    name: sphere_obstacle
    file: package://iiwa_setup/sphere_obstacle.sdf
- add_weld:
    parent: world
    child: sphere_obstacle::sphere_body
    X_PC:
        translation: [0.5, 0, 0.5]
plant_config:
    time_step: 0.005
    contact_model: "hydroelastic"
    discrete_contact_approximation: "sap"
model_drivers:
    iiwa: !IiwaDriver
        lcm_bus: "default"
        control_mode: position_only
lcm_buses:
    default:
        lcm_url: ""
"""

builder = DiagramBuilder()
scenario = LoadScenario(data=scenario_data)
station = builder.AddNamedSystem(
    "station",
    IiwaHardwareStationDiagram(
        scenario=scenario, has_wsg=False, use_hardware=False
    ),
)

# Get the external plant (the one that's actually simulated)
external_station = station.GetSubsystemByName("external_station")
plant = external_station.GetSubsystemByName("plant")

print("\n=== COLLISION GEOMETRY CHECK ===")
print(f"Total bodies in plant: {plant.num_bodies()}")
print(f"\nBodies with collision geometry:")
from pydrake.multibody.tree import ModelInstanceIndex
for model_idx in range(plant.num_model_instances()):
    model_instance = ModelInstanceIndex(model_idx)
    for body_idx in plant.GetBodyIndices(model_instance):
        body = plant.get_body(body_idx)
        collision_geoms = plant.GetCollisionGeometriesForBody(body)
        if len(collision_geoms) > 0:
            print(f"  {body.name()}: {len(collision_geoms)} collision geometries")

# Check sphere specifically
print("\n=== SPHERE OBSTACLE ===")
try:
    sphere_body = plant.GetBodyByName("sphere_body")
    sphere_geoms = plant.GetCollisionGeometriesForBody(sphere_body)
    print(f"Sphere body collision geometries: {len(sphere_geoms)}")
except:
    print("ERROR: sphere_body not found!")

# Check iiwa links
print("\n=== IIWA LINKS ===")
iiwa_instance = plant.GetModelInstanceByName("iiwa")
for body_idx in plant.GetBodyIndices(iiwa_instance):
    body = plant.get_body(body_idx)
    collision_geoms = plant.GetCollisionGeometriesForBody(body)
    if len(collision_geoms) > 0:
        print(f"  {body.name()}: {len(collision_geoms)} collision geometries")

print("\n=== CONTACT PAIRS ===")
scene_graph = external_station.GetSubsystemByName("scene_graph")
inspector = scene_graph.model_inspector()
print(f"Total geometries registered: {inspector.num_geometries()}")
