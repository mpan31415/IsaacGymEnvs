import trimesh
import numpy as np

# asset_obj_files = [
#     '../assets/urdf/kuka_allegro_description/meshes/touchsensor/collision/touch_sensor_base.obj',
#     '../assets/urdf/kuka_allegro_description/meshes/touchsensor/collision/touch_sensor_base.obj',
#     '../assets/urdf/kuka_allegro_description/meshes/touchsensor/collision/touch_sensor_base.obj',
#     '../assets/urdf/kuka_allegro_description/meshes/touchsensor/collision/touch_sensor_thumb_base.obj',
# ]

# asset_obj_files = [
#     '../assets/urdf/franka_description_tmr/allegro/meshes/allegro/link_3.0.obj',
#     '../assets/urdf/franka_description_tmr/allegro/meshes/allegro/link_3.0.obj',
#     '../assets/urdf/franka_description_tmr/allegro/meshes/allegro/link_3.0.obj',
#     '../assets/urdf/franka_description_tmr/allegro/meshes/allegro/link_15.0.obj',
# ]

asset_obj_files = [
    '../assets/urdf/franka_description_tmr/allegro/meshes/digit_centered/digit2_sensor_base_tip.obj',
    '../assets/urdf/franka_description_tmr/allegro/meshes/digit_centered/digit2_sensor_base_tip.obj',
    '../assets/urdf/franka_description_tmr/allegro/meshes/digit_centered/digit2_sensor_base_tip.obj',
    '../assets/urdf/franka_description_tmr/allegro/meshes/digit_centered/digit2_sensor_base_tip.obj',
]


fingertip_offsets = []

for obj_file in asset_obj_files:

    mesh = trimesh.load(obj_file)

    # Axis-aligned bounding box: min and max corners
    bounds = mesh.bounds  # shape (2, 3)
    print("bounds:")
    print(bounds)
    min_bound = bounds[0]
    max_bound = bounds[1]

    # Compute contact point — typically the tip is at the **maximum X** (outward direction)
    # Adjust this if your finger extends in a different direction (e.g., -Z, etc.)
    # This assumes X+ is the direction the fingertip points
    tip_offset = max_bound.copy()
    tip_offset[1] = (min_bound[1] + max_bound[1]) / 2  # center in Y
    tip_offset[2] = (min_bound[2] + max_bound[2]) / 2  # center in Z

    fingertip_offsets.append(tip_offset)

fingertip_offsets = np.array(fingertip_offsets, dtype=np.float32)

print("Computed fingertip offsets:")
print(fingertip_offsets)
