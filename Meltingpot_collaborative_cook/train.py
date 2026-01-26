import gc
import os
import warnings

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.air.integrations.wandb import WandbLoggerCallback

import torch
import torch.nn as nn
import gymnasium as gym
import supersuit as ss
import imageio.v2 as imageio

from shimmy import MeltingPotCompatibilityV0  # shimmy[meltingpot] 필요


# 경고 메시지 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 평가 지표 누락 시 에러 무시
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

MAX_CYCLES = 500

# Melting Pot collaborative cooking (2인) 기본 variant
SUBSTRATE_NAME = "collaborative_cooking__circuit"


class CoopCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        length = episode.length
        success = 1.0 if length >= MAX_CYCLES - 1 else 0.0
        episode.custom_metrics["success"] = success


class ObsExtractWrapper:
    """
    Melting Pot(PettingZoo) 관측이 Dict일 때, CNN에 넣을 이미지(Box) 하나만 뽑아내는 래퍼.
    - 우선순위 키: RGB, rgb, pixels, observation, image
    - 없으면 Dict 안에서 "이미지처럼 생긴 Box(차원 2~3)"를 첫 번째로 선택
    """

    def __init__(self, env, prefer_keys=("RGB", "rgb", "pixels", "observation", "image")):
        self.env = env
        self.prefer_keys = prefer_keys
        self._selected_key = None

        # 기본 속성 위임
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.possible_agents = getattr(env, "possible_agents", [])
        self.agents = getattr(env, "agents", [])

    def _pick_key_from_space(self, space: gym.Space):
        if not isinstance(space, gym.spaces.Dict):
            return None

        for k in self.prefer_keys:
            if k in space.spaces:
                return k

        # fallback: 이미지처럼 보이는 Box 탐색
        for k, sp in space.spaces.items():
            if isinstance(sp, gym.spaces.Box):
                # (H,W) or (H,W,C) 같은 경우를 우선
                if hasattr(sp, "shape") and sp.shape is not None and len(sp.shape) in (2, 3):
                    return k

        # 그래도 없으면 그냥 첫 키
        return next(iter(space.spaces.keys()))

    def _ensure_selected_key(self, agent_id: str):
        if self._selected_key is not None:
            return
        sp = self.env.observation_space(agent_id)
        self._selected_key = self._pick_key_from_space(sp)

    def _extract_obs(self, agent_id: str, obs):
        if isinstance(obs, dict):
            self._ensure_selected_key(agent_id)
            return obs[self._selected_key]
        return obs

    def observation_space(self, agent):
        sp = self.env.observation_space(agent)
        if isinstance(sp, gym.spaces.Dict):
            key = self._pick_key_from_space(sp)
            return sp.spaces[key]
        return sp

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = getattr(self.env, "agents", [])
        obs = {aid: self._extract_obs(aid, o) for aid, o in obs.items()}
        return obs, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = getattr(self.env, "agents", [])
        obs = {aid: self._extract_obs(aid, o) for aid, o in obs.items()}
        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def state(self):
        # shimmy wrapper에 state()가 있으면 그대로 사용
        if hasattr(self.env, "state"):
            return self.env.state()
        return None


def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    substrate_name: str,
    max_cycles: int,
    every_n_steps: int = 4,
    max_frames: int = 200,
    fps: int = 30,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Melting Pot -> PettingZoo ParallelEnv
    env = MeltingPotCompatibilityV0(
        substrate_name=substrate_name,
        max_cycles=max_cycles,
        render_mode="rgb_array",
    )

    # 관측을 이미지(Box) 하나로 축약
    env = ObsExtractWrapper(env)

    # (이전 코드와 동일) 리사이즈
    env = ss.resize_v1(env, x_size=84, y_size=168)

    frames = []
    try:
        obs, infos = env.reset()
        step_i = 0

        fr0 = env.render()
        if fr0 is not None:
            frames.append(fr0)

        terminations = {a: False for a in obs.keys()}
        truncations = {a: False for a in obs.keys()}

        while True:
            if all(
                terminations.get(a, False) or truncations.get(a, False)
                for a in obs.keys()
            ):
                break

            actions = {}
            for agent_id, agent_obs in obs.items():
                action = algorithm.compute_single_action(
                    agent_obs,
                    policy_id="shared_policy",
                    explore=False,
                )
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames:
                    break
                fr = env.render()
                if fr is not None:
                    frames.append(fr)

            step_i += 1

        if frames:
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"[GIF] saved: {out_path} frames={len(frames)}")
        else:
            print(f"[GIF] skipped (no frames): {out_path}")

    finally:
        try:
            env.close()
            gc.collect()
        except Exception:
            pass


class GifCallbacks(CoopCallbacks):
    """
    evaluation 5번마다 GIF 1개 저장 (기존 코드 흐름 그대로)
    """

    def __init__(
        self,
        out_dir: str,
        substrate_name: str,
        every_n_evals: int = 5,
        max_cycles: int = 500,
        every_n_steps: int = 2,
        max_frames: int = 400,
        fps: int = 30,
        filename_prefix: str = "eval5",
    ):
        super().__init__()
        self.out_dir = out_dir
        self.substrate_name = substrate_name
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.every_n_steps = every_n_steps
        self.max_frames = max_frames
        self.fps = fps
        self.filename_prefix = filename_prefix

        self.eval_count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_result(self, *, algorithm, result, **kwargs):
        training_iter = int(result.get("training_iteration", 0))

        if "evaluation" not in result:
            return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0:
            return

        out_path = os.path.join(
            self.out_dir,
            f"{self.filename_prefix}_{self.eval_count:04d}_iter{training_iter:06d}.gif",
        )

        rollout_and_save_gif(
            algorithm=algorithm,
            out_path=out_path,
            substrate_name=self.substrate_name,
            max_cycles=self.max_cycles,
            every_n_steps=self.every_n_steps,
            max_frames=self.max_frames,
            fps=self.fps,
        )


class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.agents = self.par_env.possible_agents
        self._agent_ids = set(self.agents)

    def reset(self, *, seed=None, options=None):
        obs, info = self.par_env.reset(seed=seed, options=options)
        return obs, info


def env_creator(config=None):
    config = config or {}
    substrate_name = config.get("substrate_name", SUBSTRATE_NAME)

    env = MeltingPotCompatibilityV0(
        substrate_name=substrate_name,
        max_cycles=MAX_CYCLES,
        render_mode="rgb_array",
    )

    env = ObsExtractWrapper(env)

    # (이전 코드와 동일) 리사이즈
    env = ss.resize_v1(env, x_size=84, y_size=168)

    return FixedParallelPettingZooEnv(env)


class CustomCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        shape = obs_space.shape
        if len(shape) == 3:
            input_h, input_w, input_channels = shape
        elif len(shape) == 2:
            input_h, input_w = shape
            input_channels = 1
        else:
            raise ValueError(f"Unsupported observation shape: {shape}")

        print(f"DEBUG: CNN Init - Input Shape: ({input_h}, {input_w}, {input_channels})")

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        dummy_input = torch.zeros(1, input_channels, input_h, input_w)
        with torch.no_grad():
            dummy_out = self.conv_layers(dummy_input)
            self.flatten_dim = dummy_out.numel()

        print(f"DEBUG: CNN Flatten Dim calculated: {self.flatten_dim}")

        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(512, num_outputs)
        self.value_head = nn.Linear(512, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()

        if x.max() > 10.0:
            x = x / 255.0

        if x.dim() == 3:
            x = x.unsqueeze(-1)

        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = self.fc_net(x)

        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out


ModelCatalog.register_custom_model("custom_cnn", CustomCNN)


if __name__ == "__main__":
    ray.init()

    env_name = "meltingpot_collaborative_cooking_shared_ppo"
    register_env(env_name, lambda cfg: env_creator(cfg))

    # 공간 추출 (이전 코드 흐름 그대로)
    tmp_env = env_creator({"substrate_name": SUBSTRATE_NAME})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    # Shared policy
    policies = {
        "shared_policy": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "shared_policy"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")

    experiment_name = "PPO_meltingpot_collaborative_cooking_shared_CNN_5e-5_resize_mirrorOFF"

    gif_save_path = os.path.join(local_log_dir, experiment_name, "gifs")

    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(
            env=env_name,
            env_config={"substrate_name": SUBSTRATE_NAME},
            clip_actions=True,
            disable_env_checking=True,
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=8,
            rollout_fragment_length=256,
            compress_observations=True,
        )
        .training(
            model={"custom_model": "custom_cnn"},
            train_batch_size=8 * 256,
            sgd_minibatch_size=256,
            num_sgd_iter=8,
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .callbacks(lambda: GifCallbacks(
            out_dir=gif_save_path,
            substrate_name=SUBSTRATE_NAME,
            every_n_evals=5,
            max_cycles=MAX_CYCLES,
        ))
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .evaluation(
            evaluation_interval=100,
            evaluation_num_episodes=25,
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")
    print(f"### Substrate: {SUBSTRATE_NAME} ###")

    tune.run(
        "PPO",
        name=experiment_name,
        stop={"timesteps_total": 10_000_000},
        local_dir=local_log_dir,
        metric="evaluation/custom_metrics/success_mean",
        mode="max",
        keep_checkpoints_num=2,
        checkpoint_score_attr="evaluation/custom_metrics/success_mean",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        config=config.to_dict(),
        callbacks=[
            WandbLoggerCallback(
                project="meltingpot_collaborative_cooking",
                group="ppo_experiments",
                job_type="training",
                name=experiment_name,
                log_config=True,
            )
        ],
    )
