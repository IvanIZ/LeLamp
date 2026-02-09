"""
LeLamp Demo - showcasing FK, IK, and motion control
This script demonstrates the complete kinematics functionality
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from test_trajectory import generate_circle_trajectory
from lelamp_utils import (
    LeLampKinematics,
    move_to_pose,
    move_to_ee_position,
    hold_position,
    convert_to_dictionary, 
    convert_to_array,
    track_trajectory,
)

# Load model
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Initialize kinematics
kinematics = LeLampKinematics(model, data)

X_RANGE = [-0.34, 0.34]

# Y_RANGE = [-0.34, 0.34]
Y_RANGE = [0, 0.34]

# Z_RANGE = [0.17, 0.5]
Z_RANGE = [0.2, 0.5]

# ee_pose = (array([-0.11748194,  0.03565924,  0.28509967]), array([[ 0.95196127,  0.20270676,  0.2295206 ],
#        [-0.30513375,  0.56490378,  0.76666623],
#        [ 0.02575138, -0.79987104,  0.5996192 ]]))

# initial joint configurations in degrees
initial_config = {'joint_1': np.float64(1.0633575843184032e-08), 
                  'joint_2': np.float64(-0.407033976554832), 
                  'joint_3': np.float64(2.2523879423745203), 
                  'joint_4': np.float64(-0.09802881551910159), 
                  'joint_5': np.float64(-0.14702632655583533)}


desired_config1 = {'joint_1': 0, 
                  'joint_2': 0, 
                  'joint_3': 0, 
                  'joint_4': 0, 
                  'joint_5': 0}


desired_config2 = {'joint_1': 0, 
                  'joint_2': 20, 
                  'joint_3': -40, 
                  'joint_4': 0, 
                  'joint_5': 0}


desired_config3 = {'joint_1': 0, 
                  'joint_2': 20, 
                  'joint_3': 0, 
                  'joint_4': 0, 
                  'joint_5': 0}

# Initialize
mujoco.mj_forward(model, data)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    

    # while viewer.is_running():
    #     mujoco.mj_step(model, data)
    #     viewer.sync()

    #     # print("joint angles: ", convert_to_dictionary(kinematics.get_joint_angles()))
    #     print("get ee position: ", kinematics.get_ee_position())
    # #     print("get ee rotation: ", kinematics.get_ee_pose())

    # move_to_pose(model, data, viewer, desired_config1, duration=5)
    # print("config 1 results:")
    # print("get ee position: ", kinematics.get_ee_position())
    # print("forward kinematics: ", kinematics.forward_kinematics(convert_to_array(desired_config1))[0])
    # print("forward kinematics gt: ", kinematics.forward_kinematics_gt(convert_to_array(desired_config1))[0])

    # move_to_pose(model, data, viewer, desired_config2, duration=5)
    # print("config 2 results:")
    # print("get ee position: ", kinematics.get_ee_position())
    # print("forward kinematics: ", kinematics.forward_kinematics(convert_to_array(desired_config2))[0])
    # print("forward kinematics gt: ", kinematics.forward_kinematics_gt(convert_to_array(desired_config2))[0])

    # # kinematics.fix_base(position=np.array([0, 0, 0]))
    # move_to_pose(model, data, viewer, desired_config3, duration=5)
    # print("config 3 results:")
    # print("get ee position: ", kinematics.get_ee_position())
    # print("forward kinematics: ", kinematics.forward_kinematics(convert_to_array(desired_config3))[0])

    # print("TEST IK ======================================")
    # print("initial pose: ", kinematics.get_ee_pose())
    # move_to_ee_position(model, data, viewer, target_position=np.array([0.2, 0.2, 0.25]), duration=5)
    # print("final pose1: ", kinematics.get_ee_pose())

    # move_to_ee_position(model, data, viewer, target_position=np.array([0.2, -0.2, 0.25]), duration=5)
    # print("final pose2: ", kinematics.get_ee_pose())


    # Reference position: EE at zero angles
    ref_pos, _ = kinematics.forward_kinematics(np.zeros(5))
    print(f"\nZero-angle EE position: {ref_pos}")

    # ------------------------------------------------------------------
    # Circle parameters
    # ------------------------------------------------------------------
    radius = 0.04          # 4 cm
    n_points = 60          # 60 waypoints around the circle
    # plane = "xy"           # horizontal circle
    plane = "yz"           # horizontal circle
    # Center at the zero-angle EE position (circle in XY around it)
    center = ref_pos.copy()

    print(f"Circle center: {center}")
    print(f"Circle radius: {radius} m")
    print(f"Circle plane:  {plane}")
    print(f"Waypoints:     {n_points} (+ 1 closing point)")

    trajectory = generate_circle_trajectory(
        center, radius, n_points, plane=plane, closed=True
    )

    # add 2 more circles just for fun
    trajectory = trajectory + generate_circle_trajectory(
        center, radius, n_points, plane=plane, closed=True
    ) + generate_circle_trajectory(
        center, radius, n_points, plane=plane, closed=True
    )



    # ------------------------------------------------------------------
    # Track trajectory (headless — no viewer)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Tracking trajectory (headless)...")
    print("-" * 70)

    results = track_trajectory(
        model, data, viewer=viewer,
        trajectory=trajectory,
        duration_per_segment=0.5,    # 0.5s per segment → ~30s total
        position_only=True,
        ik_position_tolerance=1e-5,  # tight IK tolerance
        max_ik_iterations=1000,
        approach_duration=2.0,       # generous approach time
        settle_steps=0,              # continuous motion (no stops)
    )

