import argparse
import os
import torch
import sys


# [1] AppLauncher: 시뮬레이터 실행 준비
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Train G1 Robot")
parser.add_argument("--num_envs", type=int, default=60, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# [2] Import
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.runners import OnPolicyRunner
from g1_env import G1PickPlaceEnvCfg

# ------------------------------------------------------------------------------
# 3. PPO 알고리즘 설정 (클래스 대신 딕셔너리로 변경!)
# ------------------------------------------------------------------------------
# rsl_rl이 원하는 딕셔너리 구조를 그대로 만듭니다.
runner_cfg = {
    "seed": 42,
    "runner_class_name": "OnPolicyRunner",
    "num_steps_per_env": 24,
    "max_iterations": 1500,
    "save_interval": 50,
    "experiment_name": "g1_pick_place",
    "run_name": "test_run",
    "logger": "tensorboard",
    "resume": False,
    "load_run": -1,
    "load_checkpoint": -1,

    "obs_groups": {
        "policy": ["policy"], 
        "critic": ["policy"],
    },
    
    # 정책 네트워크 설정
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "activation": "elu",
    },

    # 알고리즘 설정
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

# ------------------------------------------------------------------------------
# 4. 메인 학습 루프
# ------------------------------------------------------------------------------
def main():
    # (1) 환경 설정 로드
    env_cfg = G1PickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # (2) 환경 생성
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # [디버깅] 관측 그룹이 제대로 잡혔는지 확인
    print(f"\n[DEBUG] Env Observation Keys: {env.observation_manager.group_obs_dim.keys()}")
    
    # (3) RSL-RL용 래퍼
    env = RslRlVecEnvWrapper(env)

    # (4) PPO 러너 생성
    log_dir = os.path.join("logs", "rsl_rl", runner_cfg["experiment_name"])
    
    # [수정됨] 
    # 1. runner_cfg는 이제 딕셔너리이므로 .to_dict() 필요 없음
    # 2. device=str(env.device)로 문자열 변환
    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=str(env.device))

    # [추가] 학습 시작 전에 카메라 위치를 잡아줍니다.
    # 래퍼(Wrapper)가 씌워져 있어서 .unwrapped를 통해 원본 환경에 접근해야 합니다.
    if env.unwrapped.sim.has_gui():
        env.unwrapped.sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.5])

    # (5) 학습 시작
    print(f"[INFO] 학습 시작! 로그 경로: {log_dir}")
    runner.learn(num_learning_iterations=runner_cfg["max_iterations"], init_at_random_ep_len=True)

    # (6) 종료
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()