"""Unitree R1 articulation for Isaac Lab; numbers copied from mjlab's r1_constants.py / xmls/r1.xml so PhysX and MuJoCo see the same robot."""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ARMATURE = 0.01  # r1.xml <default><joint armature="0.01">
PASSIVE_DAMPING = 0.05  # r1.xml <default><joint damping="0.05"> adds to the PD damping in MuJoCo

# (stiffness, damping, effort) per joint group, from r1_constants.py
_HIP = (150.0, 8.0, 88.0)
_KNEE = (200.0, 10.0, 139.0)
_ANKLE = (60.0, 4.0, 50.0)
_WAIST_ROLL = (80.0, 5.0, 50.0)
_WAIST_YAW = (80.0, 5.0, 88.0)
_SHOULDER = (60.0, 4.0, 25.0)
_DISTAL = (40.0, 3.0, 25.0)


def _act(exprs, g):
    kp, kd, eff = g
    return ImplicitActuatorCfg(
        joint_names_expr=exprs,
        stiffness=kp,
        damping=kd + PASSIVE_DAMPING,
        effort_limit_sim=eff,
        velocity_limit_sim=50.0,
        armature=ARMATURE,
    )


R1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/r1/r1.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.734),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
            ".*_elbow_joint": 0.87,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": _act([".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint"], _HIP),
        "knee": _act([".*_knee_joint"], _KNEE),
        "ankle": _act([".*_ankle_pitch_joint", ".*_ankle_roll_joint"], _ANKLE),
        "waist_roll": _act(["waist_roll_joint"], _WAIST_ROLL),
        "waist_yaw": _act(["waist_yaw_joint"], _WAIST_YAW),
        "shoulder": _act([".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"], _SHOULDER),
        "distal": _act([".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"], _DISTAL),
    },
)

R1_ACTION_SCALE = {}
for a in R1_CFG.actuators.values():
    for n in a.joint_names_expr:
        R1_ACTION_SCALE[n] = 0.25 * a.effort_limit_sim / a.stiffness
