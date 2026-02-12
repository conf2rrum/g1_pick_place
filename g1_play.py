import argparse
import os
import torch
import sys

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Play G1 Robot")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")
parser.add_argument("--checkpoint", type=str, default="logs/rsl_rl/g1_pick_place/model_1499.pt", help="Path to the saved model .pt file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, Articulation, RigidObject
from isaaclab.managers import TerminationTermCfg, RewardTermCfg, SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab_assets import G1_CFG
from g1_env import G1PickPlaceEnvCfg


# ==============================================================================
# 1. 커스텀 함수 (학습 때와 동일)
# ==============================================================================
def object_pos_rel(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w - robot.data.root_pos_w

def goal_pos_rel(env, robot_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    return goal.data.root_pos_w - robot.data.root_pos_w

def reward_hand_reach_object(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, hand_body_name: str):
    return 0.0 # 테스트 때는 보상이 필요 없으므로 0 반환 (에러 방지용)

# ==============================================================================
# 2. 환경 설정 (학습 때와 동일)
# ==============================================================================
@configclass
class G1PickPlaceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.articulation_props.fix_root_link = True
    from isaaclab.actuators import ImplicitActuatorCfg
    robot.actuators = {
        "body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=5.0),
    }
    robot.init_state.pos = (0.0, 0.0, 0.73)

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)), rigid_props=sim_utils.RigidBodyPropertiesCfg(), collision_props=sim_utils.CollisionPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=0.1)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.55)), # 수정된 거리 적용
    )

    goal_maker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.SphereCfg(radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.3), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.25, 0.55)),
    )

@configclass
class ActionCfg:
    # 학습 때와 똑같은 Joint 설정을 써야 합니다.
    joint_names = [".*torso.*", ".*shoulder.*", ".*elbow.*", ".*_(zero|one|two|three|four|five|six)_joint"]
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=joint_names, scale=0.5, use_default_offset=True)

@configclass
class PolicyObsCfg(ObservationGroupCfg):
    joint_pos = mdp.ObservationTermCfg(func=mdp.joint_pos_rel)
    joint_vel = mdp.ObservationTermCfg(func=mdp.joint_vel_rel)
    object_pos = mdp.ObservationTermCfg(func=object_pos_rel, params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")})
    goal_pos = mdp.ObservationTermCfg(func=goal_pos_rel, params={"robot_cfg": SceneEntityCfg("robot"), "goal_cfg": SceneEntityCfg("goal_maker")})

@configclass
class ObservationsCfg:
    policy: PolicyObsCfg = PolicyObsCfg()


# ==============================================================================
# 3. Runner 설정 (네트워크 구조 불러오기용)
# ==============================================================================
runner_cfg = {
    "seed": 42,
    "runner_class_name": "OnPolicyRunner",
    "num_steps_per_env": 24,
    "max_iterations": 1500,
    "save_interval": 50,
    "experiment_name": "g1_pick_place",
    "run_name": "play_run",
    "logger": "tensorboard",
    "resume": True,     # [중요] 불러오기 모드
    "load_run": -1,
    "load_checkpoint": -1,
    "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    }
}

# ==============================================================================
# 4. 메인 실행 (Play Loop)
# ==============================================================================
def main():
    # 1. 환경 생성
    env_cfg = G1PickPlaceEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # 2. 러너 생성 및 모델 로드
    log_dir = os.path.join("logs", "rsl_rl", "play_test") # 임시 로그 경로
    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=str(env.device))
    
    # [핵심] 저장된 모델 불러오기
    if args_cli.checkpoint is None:
        print("[ERROR] --checkpoint 인자로 모델 경로(.pt)를 입력해야 합니다!")
        return
        
    print(f"[INFO] 모델 로딩 중: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    
    # 3. 추론용 정책(Policy) 가져오기
    policy = runner.get_inference_policy(device=str(env.device))

    # 4. 카메라 설정
    if env.unwrapped.sim.has_gui():
        env.unwrapped.sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.5])

    # 5. 실행 루프 (무한 반복)
    print("[INFO] 시뮬레이션 시작! (Ctrl+C로 종료)")
    obs, _ = env.reset()
    
    while simulation_app.is_running():
        # (1) 모델이 행동 결정 (Inference)
        with torch.no_grad(): # 학습 안 하니까 기울기 계산 끔
            actions = policy(obs)
            
        # (2) 환경에 행동 적용
        obs, _, _, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()