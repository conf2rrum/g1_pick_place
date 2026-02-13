# 파일명: g1_env.py
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, Articulation, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg, RewardTermCfg, SceneEntityCfg, ObservationGroupCfg
import isaaclab.envs.mdp as mdp
import torch
from isaaclab_assets import G1_CFG

# ------------------------------------------------------------------------------
# 커스텀 함수들 (관측, 보상)
# ------------------------------------------------------------------------------

# 1. 관측(Observations) 정의
def object_pos_rel(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    """
    로봇의 루트(골반/베이스) 기준으로 물체의 상대 위치를 계산
    """
    # 1. Scene에서 로봇과 물체 객체를 가져옴
    robot: Articulation = env.scene[robot_cfg.name]
    object: Articulation = env.scene[object_cfg.name]

    # 2. 로봇과 물체의 월드 좌표 가져오기
    # root_pos_w: (num_envs, 3) 크기의 텐서 (x, y, z)
    robot_pos = robot.data.root_pos_w
    object_pos = object.data.root_pos_w

    # 3. 뺄셈으로 상대 위치 계산 (물체 - 로봇)
    # 결과: 로봇에서 봤을 때 물체가 어디에 있는지 (벡터)
    return object_pos - robot_pos

def goal_pos_rel(env, robot_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg):
    """
    로봇의 루트 기준으로 목표점의 상대 위치를 계산
    """
    robot: Articulation = env.scene[robot_cfg.name]
    goal: Articulation = env.scene[goal_cfg.name]

    return goal.data.root_pos_w - robot.data.root_pos_w

def reward_hand_reach_object(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, hand_body_name: str):
    robot: Articulation = env.scene[robot_cfg.name]
    object = env.scene[object_cfg.name]

    # 해당 이름의 바디 인덱스 찾기
    ids = robot.find_bodies(hand_body_name)[0][0]

    hand_pos = robot.data.body_pos_w[:, ids]
    object_pos = object.data.root_pos_w

    # 거리 계산
    distance = torch.norm(hand_pos - object_pos, dim=-1)

    return -distance

def reward_pick_and_place(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg, hand_body_name: str):
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]

    # 1. 데이터 가져오기
    ids = robot.find_bodies(hand_body_name)[0][0]
    hand_pos = robot.data.body_pos_w[:, ids]
    object_pos = object.data.root_pos_w
    goal_pos = goal.data.root_pos_w

    # 2. 거리 계산
    # 손 <-> 물체 거리
    d_hand_obj = torch.norm(hand_pos - object_pos, dim=1)
    # 물체 <-> 목표 거리
    d_obj_goal = torch.norm(object_pos - goal_pos, dim=1)

    # 3. 상태 판단
    # 물체의 높이가 0.6m 이상이면 "들었다"고 판단
    is_lifted = object_pos[:, 2] > 0.65

    # 4. 보상 계산 로직
    # 기본 점수: 손이 물체에 가까이 갈수록 점수
    reward = 1.0 / (1.0 + d_hand_obj**2)

    # [핵심] 물체를 들어 올렸다면? -> 목표 지점으로 가는 것에 점수를 줌
    # 들어 올린 상태에서는 손-물체 거리는 이미 가까우니 무시하고, 물체-목표 거리에 집중
    reward = torch.where(
        is_lifted,
        2.0 + (1.0 / (1.0 + d_obj_goal**2)),    # 들었으면 보너스 2점 + 목표 접근 점수
        reward  # 안들었으면 그냥 접근 점수만
    )

    return reward

def reward_object_out_of_bounds(env, object_cfg: SceneEntityCfg, x_limits: list, y_limits: list, z_limits: list):
    object: RigidObject = env.scene[object_cfg.name]
    pos = object.data.root_pos_w

    # 각 축(x, y, z)별로 범위를 벗어났는지 체크
    is_out_x = (pos[:, 0] < x_limits[0]) | (pos[:, 0] > x_limits[1])
    is_out_y = (pos[:, 1] < y_limits[0]) | (pos[:, 1] > y_limits[1])
    is_out_z = (pos[:, 2] < z_limits[0]) | (pos[:, 2] > z_limits[1])

    is_out = is_out_x | is_out_y | is_out_z

    # 벗어났으면 1.0, 아니면 0.0 반환
    # (나중에 weight에 -1.0을 곱해서 벌점으로 만듦)
    return is_out.float()


# ------------------------------------------------------------------------------
# 1. 씬(Scene) 정의
# ------------------------------------------------------------------------------
@configclass
class G1PickPlaceSceneCfg(InteractiveSceneCfg):
    # (1) 바닥
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # (2) 조명
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # (3) 책상
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        # 크기: 가로 0.6m, 세로 1m, 높이 0.7m
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1, 0.7), 
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.25),  # 로봇 앞(x=0.8)에 배치 (높이는 0.5의 절반인 0.25)
        ),
    )

    # (4) G1 로봇 [수정됨: 허리 고정 모드]
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.articulation_props.fix_root_link = True
    from isaaclab.actuators import ImplicitActuatorCfg
    robot.actuators = {
        "strong_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=100.0,
            damping=5.0,
        ),
    }
    robot.init_state.pos = (0.0, 0.0, 0.73)

    # (5) 집을 물체 (빨간 큐브)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),

            # 1. 시각적 색상
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),

            # 2. 강체 속성 (이게 있어야 움직이는 물체로 인식)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),

            # 3. 충돌 속성
            collision_props=sim_utils.CollisionPropertiesCfg(),

            # 4. 질량 설정
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),

            # 5. 물리 재질 (마찰력)
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.2, 0.63)),
    )

    # (6) 목표 지점
    goal_maker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.8),
            # [중요] 물리 속성을 부여해야 .data를 쓸 수 있습니다.
            # kinematic_enabled=True로 설정해서 중력을 무시하고 고정시킵니다.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            # 충돌 속성(collision_props)은 넣지 않습니다. (로봇이 목표물을 뚫고 지나가야 하므로)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0, 0.63)),
    )

# ------------------------------------------------------------------------------
# 2. 행동(Actions) 정의 - 로봇을 어떻게 움직일 것인가?
# ------------------------------------------------------------------------------
# ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_pitch_joint', 'right_elbow_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_elbow_roll_joint', 'right_elbow_roll_joint', 'left_five_joint', 'left_three_joint', 'left_zero_joint', 'right_five_joint', 'right_three_joint', 'right_zero_joint', 'left_six_joint', 'left_four_joint', 'left_one_joint', 'right_six_joint', 'right_four_joint', 'right_one_joint', 'left_two_joint', 'right_two_joint']
@configclass
class ActionCfg:
    # [설명]
    # 1. torso: 허리를 움직여 팔이 닿는 범위를 넓혀줌
    # 2. left_shoulder/elbow: 팔을 목표 위치로 이동시킴
    # 3. left_zero~six: 손가락을 움직여 물체를 잡음

    joint_pos = mdp.JointPositionActionCfg(
        asset_name = "robot",
        joint_names = [
            "torso_joint",
            ".*left_shoulder.*",
            ".*left_elbow.*",
            ".*left_.*(zero|one|two|three|four|five|six)_joint"
        ],
        scale = 0.5,
        use_default_offset = True,
    )



@configclass
class PolicyObsCfg(ObservationGroupCfg):
    joint_pos = mdp.ObservationTermCfg(func=mdp.joint_pos_rel)
    joint_vel = mdp.ObservationTermCfg(func=mdp.joint_vel_rel)
    
    object_pos = mdp.ObservationTermCfg(
        func=object_pos_rel,
        params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")}
    )
    
    goal_pos = mdp.ObservationTermCfg(
        func=goal_pos_rel,
        params={"robot_cfg": SceneEntityCfg("robot"), "goal_cfg": SceneEntityCfg("goal_maker")}
    )

@configclass
class ObservationsCfg:
    # policy 변수에 위에서 만든 클래스의 '인스턴스()'를 할당
    policy: PolicyObsCfg = PolicyObsCfg()           
        

# ------------------------------------------------------------------------------
# 4. 환경(Environment) 통합
# ------------------------------------------------------------------------------
@configclass
class G1PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    # 위에서 만든 설정 조립
    scene=G1PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    observations=ObservationsCfg()
    actions=ActionCfg()

    # 물리 엔진 업데이트 설정
    sim = sim_utils.SimulationCfg(dt=0.01, render_interval=4, device="cuda:0")
    decimation = 4  # 0.04초마다 제어(25Hz)
    episode_length_s = 5.0   # 에피소드 길이 (초 단위)

    # 보상 설정
    @configclass
    class RewardsCfg:
        pick_place = RewardTermCfg(
            func=reward_pick_and_place,
            weight=1.0,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
                "goal_cfg": SceneEntityCfg("goal_maker"),
                "hand_body_name": "left_palm_link"
            }
        )

        # [추가] 물체를 떨어뜨리거나 너무 멀리 날리면 벌점
        object_far = RewardTermCfg(
            func=reward_object_out_of_bounds,
            weight=-1.0, # 벌점 (1.0 * -1.0 = -1점)
            params={
                "object_cfg": SceneEntityCfg("object"),
                "x_limits": [0.0, 1.0],     # 책상 앞뒤 범위
                "y_limits": [-0.5, 0.5],    # 책상 좌우 범위
                "z_limits": [0.0, 1.5]      # 높이 범위
            }
        )
       

    rewards = RewardsCfg()

    # 종료 조건 (최소한 시간 초과는 있어야 함)
    @configclass
    class TerminationsCfg:
        time_out = TerminationTermCfg(func=mdp.time_out)

    terminations = TerminationsCfg()

