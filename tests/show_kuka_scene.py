import math
import numpy as np
from isaacgym import gymapi, gymutil


def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))

def object_start_pose(arms_y_ofs: float, table_pose_dy: float, table_pose_dz: float):
    object_start_pose = gymapi.Transform()
    object_start_pose.p = gymapi.Vec3()
    object_start_pose.p.x = 0.0

    pose_dy, pose_dz = table_pose_dy, table_pose_dz + 0.25

    object_start_pose.p.y = arms_y_ofs + pose_dy
    object_start_pose.p.z = pose_dz

    return object_start_pose


###################################### DEFINE ASSETS ######################################
robot_asset_file = "urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
robot_asset_flip_visual = False

table_asset_file = "urdf/table_wide.urdf"
object_asset_file = "urdf/objects/cube_multicolor.urdf"

# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},])


###################################### INITIALIZE SIM ######################################
# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 30.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# set up axis, define gravity vector, and add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()


###################################### LOAD ASSETS ######################################
asset_root = "../assets"

############## load robot ##############
robot_asset_options = gymapi.AssetOptions()
robot_asset_options.fix_base_link = True
robot_asset_options.flip_visual_attachments = False
robot_asset_options.collapse_fixed_joints = True
robot_asset_options.disable_gravity = True
robot_asset_options.thickness = 0.001
robot_asset_options.angular_damping = 0.01
robot_asset_options.linear_damping = 0.01

print("Loading robot asset '%s' from '%s'" % (robot_asset_file, asset_root))
robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, robot_asset_options)

# get arrays of robot DOF names, properties
robot_dof_names = gym.get_asset_dof_names(robot_asset)
robot_dof_props = gym.get_asset_dof_properties(robot_asset)
num_robot_dofs = gym.get_asset_dof_count(robot_asset)

# create an array of DOF states that will be used to update the actors
robot_dof_states = np.zeros(num_robot_dofs, dtype=gymapi.DofState.dtype)

# get the position slice of the DOF state array (in-place)
robot_dof_positions = robot_dof_states['pos']

# set the default positions for the kuka arms
kuka_home_pos = [-1.571, 1.571, -0.000, 1.6, -0.000, 1.485, 2.358]
robot_dof_positions[0:7] = kuka_home_pos

############## load table ##############
table_asset_options = gymapi.AssetOptions()
table_asset_options.disable_gravity = False
table_asset_options.fix_base_link = True
table_asset = gym.load_asset(sim, asset_root, table_asset_file, table_asset_options)

############## load object ##############
object_asset_options = gymapi.AssetOptions()
object_asset = gym.load_asset(sim, asset_root, object_asset_file, object_asset_options)


###################################### CREATE ENVS ######################################
# set up the env grid
num_envs = 16
num_per_row = 4
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
cam_target = gymapi.Vec3(5.0, 5.0, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # define robot1 pose
    robot1_base_pose = gymapi.Transform()
    robot1_base_pose.p = gymapi.Vec3(*get_axis_params(0.0, axis_idx=2)) + gymapi.Vec3(-1.1, 0.0, 0.0)
    robot1_base_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi / 2)

    # add robot1 actor
    actor1_handle = gym.create_actor(env, robot_asset, robot1_base_pose, f"robot_actor{i}", i, 1)
    actor_handles.append(actor1_handle)
    gym.set_actor_dof_states(env, actor1_handle, robot_dof_states, gymapi.STATE_ALL)

    # define robot2 pose
    robot2_base_pose = gymapi.Transform()
    robot2_base_pose.p = gymapi.Vec3(*get_axis_params(0.0, axis_idx=2)) + gymapi.Vec3(1.1, 0.0, 0.0)
    robot2_base_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi / 2)

    # add robot2 actor
    actor2_handle = gym.create_actor(env, robot_asset, robot2_base_pose, f"robot_actor{i}", i, 1)
    actor_handles.append(actor2_handle)
    gym.set_actor_dof_states(env, actor2_handle, robot_dof_states, gymapi.STATE_ALL)

    # define table pose
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, 0.38)
    table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # add table actor
    table_handle = gym.create_actor(env, table_asset, table_pose, f"table_object{i}", i, 0, 0)

    # define object pose
    object_pose = object_start_pose(0.0, 0.0, 0.38)
    object_handle = gym.create_actor(env, object_asset, object_pose, f"object{i}", i, 0, 0)



###################################### RUN SIM + VIEWER ######################################
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # clone actor state in all of the environments
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[2*i], robot_dof_states, gymapi.STATE_POS)
        gym.set_actor_dof_states(envs[i], actor_handles[2*i+1], robot_dof_states, gymapi.STATE_POS)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)


print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
