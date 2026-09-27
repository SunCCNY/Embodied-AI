"""Unitree R1 constants.

PD gains (stiffness/damping) and effort limits below are NOT derived from a
published R1 motor spec sheet the way mjlab's G1 constants are (rotor inertia +
gear ratio -> armature -> natural-frequency-based gains). Unitree has not
published that level of detail for R1's actuators in this project. Instead:

- stiffness/damping come from this project's own validated real-robot
  deployment PD gains (`r1_walk/build_xml.py`, sourced from
  `r1_stand_sim2sim.py`, which mirrors the actual Unitree DDS controller).
- effort_limit comes from the official Unitree R1 MJCF
  (`unitree_mujoco/unitree_robots/r1/R1_C++.xml`), which uses raw torque
  <motor> actuators with explicit ctrlrange per joint.
- armature is left at the MJCF's own uniform joint default (0.01, see
  `xmls/r1.xml`'s top-level <default><joint .../>) rather than a
  per-motor-class guess, since no real rotor-inertia data exists for R1 here.

This is "adapted from real numbers, not derived from a spec sheet" -- same
honesty standard as this project's GMR retargeting scale factors.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

R1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_r1" / "xmls" / "r1.xml"
)
assert R1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(R1_XML))


##
# Actuator config.
##
# Gains are the real deployment PD values from r1_stand_sim2sim.py (via
# r1_walk/build_xml.py's LEG_ACTS / ARM_WAIST_ACTS), not a per-motor-class
# derivation. Effort limits are Unitree's own R1_C++.xml ctrlrange values.

R1_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint"),
  stiffness=150.0,
  damping=8.0,
  effort_limit=88.0,
)
R1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=200.0,
  damping=10.0,
  effort_limit=139.0,
)
R1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=60.0,
  damping=4.0,
  effort_limit=50.0,
)
R1_ACTUATOR_WAIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_roll_joint",),
  stiffness=80.0,
  damping=5.0,
  effort_limit=50.0,
)
R1_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=80.0,
  damping=5.0,
  effort_limit=88.0,
)
R1_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"),
  stiffness=60.0,
  damping=4.0,
  effort_limit=25.0,
)
R1_ACTUATOR_ARM_DISTAL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"),
  stiffness=40.0,
  damping=3.0,
  effort_limit=25.0,
)

##
# Keyframe config.
##
# Real, previously-validated stand pose: DEFAULT_LEG + ARM_WAIST_HOLD from
# r1_walk/r1_walk_env.py (matches r1_stand_sim2sim.py's STAND_POS). Base
# height 0.734 m is the corrected standing height from the Goal-1 sim2sim
# work (the raw keyframe height of 0.76 left the feet ~3.9cm off the floor).

STAND_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.734),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_ankle_roll_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_joint": 0.87,
    ".*_wrist_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##
# R1's foot geoms are named `(left|right)_foot[1-7]_collision` -- identical
# naming convention to G1's, even though their MJCF <default> class is
# "foot_capsule" rather than "collision" (name-based matching below doesn't
# care about MJCF class, only the runtime geom name).

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

##
# Final config.
##

R1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    R1_ACTUATOR_HIP,
    R1_ACTUATOR_KNEE,
    R1_ACTUATOR_ANKLE,
    R1_ACTUATOR_WAIST_ROLL,
    R1_ACTUATOR_WAIST_YAW,
    R1_ACTUATOR_SHOULDER,
    R1_ACTUATOR_ARM_DISTAL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_r1_robot_cfg() -> EntityCfg:
  """Get a fresh R1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=STAND_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=R1_ARTICULATION,
  )


R1_ACTION_SCALE: dict[str, float] = {}
for a in R1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    R1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_r1_robot_cfg())

  viewer.launch(robot.spec.compile())
