import argparse
import copy
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/normalize.py
class RunningMeanStd(nn.Module):
    def __init__(self, epsilon=1e-4, shape=()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

    def update(self, x):
        x = torch.as_tensor(x, dtype=torch.float64).to(self.mean.device)
        batch_mean = torch.mean(x, dim=0).to(self.mean.device)
        batch_var = torch.var(x, dim=0, unbiased=False).to(self.mean.device)
        batch_count = x.shape[0]

        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

        self.enable = True
        self.freeze = False

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        if not self.freeze:
            self.obs_rms.update(obs)
        if self.enable:
            return (obs - self.obs_rms.mean.cpu().numpy()) / np.sqrt(self.obs_rms.var.cpu().numpy() + self.epsilon)
        return obs


class NormalizeReward(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

        self.enable = True
        self.freeze = False

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma * (1 - terminateds) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def normalize(self, rews):
        if not self.freeze:
            self.return_rms.update(self.returns)
        if self.enable:
            return rews / np.sqrt(self.return_rms.var.cpu().numpy() + self.epsilon)
        return rews


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="SRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument("--teacher-policy-hf-repo", type=str, default=None,
        help="the huggingface repo of the teacher policy")
    parser.add_argument("--teacher-env-num-ep", type=int, default=10,
        help="the number of episodes to be stored in the teacher environment buffer.")
    parser.add_argument("--teacher-best-env-num-ep", type=int, default=10,
        help="select number of best episodes to use")
    parser.add_argument("--teacher-env-save-frequency", type=int, default=10,
        help="the frequency of save")
    parser.add_argument("--snapshot-timesteps", type=int, default=100000,
        help="snap env timesteps of the experiments")
    parser.add_argument("--n-clusters", type=int, default=6,
        help="KMeans n_clusters")
    parser.add_argument("--truncate-step", type=int, default=100,
        help="the truncate step of the snapshot env")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on

    if args.teacher_policy_hf_repo is None:
        policy_dict = {
            "Hopper-v4": "sdpkjc/Hopper-v4-ppo_fix_continuous_action-seed3",
            "Walker2d-v4": "sdpkjc/Walker2d-v4-ppo_fix_continuous_action-seed4",
            "HalfCheetah-v4": "sdpkjc/HalfCheetah-v4-ppo_fix_continuous_action-seed1",
            "Ant-v4": "sdpkjc/Ant-v4-ppo_fix_continuous_action-seed2",
            "Swimmer-v4": "sdpkjc/Swimmer-v4-ppo_fix_continuous_action-seed1",
            "Humanoid-v4": "sdpkjc/Humanoid-v4-ppo_fix_continuous_action-seed4",
        }
        if args.env_id in policy_dict.keys():
            args.teacher_policy_hf_repo = policy_dict[args.env_id]
        else:
            args.teacher_policy_hf_repo = f"sdpkjc/{args.env_id}-ppo_fix_continuous_action-seed1"
        print(f"Use default policy: {args.teacher_policy_hf_repo}")

    return args


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    obs_rms=None,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, obs_rms)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns, episodic_lengths = [], []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
                episodic_lengths += [info["episode"]["l"]]
        obs = next_obs

    return episodic_returns, episodic_lengths


def env_deepcopy(obj):
    import types

    import shimmy

    if isinstance(obj, type):
        return obj
    if isinstance(obj, types.FunctionType):
        return obj
    if not hasattr(obj, "__dict__"):
        return copy.deepcopy(obj)

    cls = obj.__class__
    result = cls.__new__(cls)

    if hasattr(shimmy, "atari_env") and isinstance(obj, shimmy.atari_env.AtariEnv):
        result = copy.deepcopy(obj)
        result.restore_state(obj.clone_state())
        return result

    for k, v in obj.__dict__.items():
        if isinstance(v, (list, dict, set)):
            if isinstance(v, list):
                setattr(result, k, [env_deepcopy(i) for i in v])
            elif isinstance(v, dict):
                setattr(result, k, {key: env_deepcopy(value) for key, value in v.items()})
            elif isinstance(v, set):
                setattr(result, k, {env_deepcopy(i) for i in v})
        elif hasattr(v, "__dict__"):
            setattr(result, k, env_deepcopy(v))
        else:
            setattr(result, k, copy.deepcopy(v))

    return result


class SaveObservationAndInfo(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.obs_ = None
        self.info_ = None

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_, self.info_ = copy.deepcopy(obs), copy.deepcopy(info)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.obs_, self.info_ = copy.deepcopy(obs), copy.deepcopy(info)
        return obs, info

    def get_data(self):
        return copy.deepcopy(self.obs_), copy.deepcopy(self.info_)


class SnapshotEnv(gym.Env):
    run_steps = 0

    def __init__(
        self,
        env_deq_deq,
        render_mode=None,
        truncate_step=1000,
        snapshot_timesteps=100000,
        teacher_best_env_num_ep=1,
        n_clusters=10,
    ):
        self.observation_space = env_deq_deq[0][0][1].observation_space
        self.action_space = env_deq_deq[0][0][1].action_space
        self.returns = np.zeros(1)
        self.env_deq_deq = env_deq_deq
        self.render_mode = render_mode
        self.current_env_idx, self.current_env = None, None
        self.truncate_step = truncate_step
        self.snapshot_timesteps = snapshot_timesteps
        self.n_clusters = n_clusters

        self.env_deq_deq = sorted(self.env_deq_deq, key=lambda x: x[-1][1].episode_returns[0], reverse=True)
        for env_deq in self.env_deq_deq:
            print(env_deq[-1][1].episode_lengths[0], env_deq[-1][1].episode_returns[0])

        arr = []
        for i, env_deq in enumerate(self.env_deq_deq):
            if i < teacher_best_env_num_ep:
                arr.extend(list(env_deq))
                print("teacher_best_env_num_ep", i, env_deq[-1][1].episode_lengths[0], env_deq[-1][1].episode_returns[0])

        arr_f = np.array([f for f, obj in arr]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=min(self.n_clusters, len(arr)), random_state=0, n_init="auto").fit(arr_f)
        self.clustered_data = [[] for _ in range(self.n_clusters)]
        for idx, label in enumerate(kmeans.labels_):
            self.clustered_data[label].append((arr[idx][0], arr[idx][1]))
        self.clustered_data = [cluster for cluster in self.clustered_data if len(cluster) > 0]
        self.clustered_data = sorted(self.clustered_data, key=lambda x: len(x), reverse=True)

        for i, cluster in enumerate(self.clustered_data):
            print(
                f"Cluster {i}: {len(cluster)}/{len(arr)} {np.mean([val for val, env in cluster])} {np.mean([env.episode_returns[0] for val, env in cluster])}"
            )
            print(sorted(val for val, env in cluster))
            print("----------------------------")
            print()
        self.clustered_data = [[env for val, env in cluster] for cluster in self.clustered_data]

    def step(self, action):
        self.__class__.run_steps += 1
        return self.current_env.step(action)

    def reset(self, **kwargs):
        # special case for atari env EpisodicLifeEnv
        if self.current_env is None or not hasattr(self.current_env, "was_real_done") or self.current_env.was_real_done:
            self.current_env = env_deepcopy(random.choice(random.choice(self.clustered_data)))
            print("start_step:", self.current_env.episode_lengths[0])
            print(self.__class__.run_steps, self.snapshot_timesteps)
            self.current_env_point = self.current_env
            while hasattr(self.current_env_point, "env"):
                if isinstance(self.current_env_point, gym.wrappers.TimeLimit):
                    self.current_env_point._max_episode_steps = self.current_env_point._elapsed_steps + self.truncate_step
                    break
                self.current_env_point = self.current_env_point.env
            else:
                raise RuntimeError("can't find TimeLimit")

            while hasattr(self.current_env, "env"):
                if isinstance(self.current_env, gym.wrappers.ClipAction):
                    break
                self.current_env = self.current_env.env
            else:
                raise RuntimeError("can't find ClipAction")

            print(
                "RESET ENV -> episode_length:",
                self.current_env.episode_lengths,
                "| episode_return:",
                self.current_env.episode_returns,
            )
        self.observation, self.info = self.current_env.get_data()
        return self.observation, self.info

    def render(self):
        return self.current_env.render(mode=self.render_mode)

    def close(self):
        return


def make_snapshot_env(
    teacher_env_deq_deq,
    truncate_step,
    snapshot_timesteps,
    teacher_best_env_num_ep,
    n_clusters,
    gamma,
    obs_rms=None,
    return_rms=None,
):
    def thunk():
        env = SnapshotEnv(
            teacher_env_deq_deq,
            truncate_step=truncate_step,
            snapshot_timesteps=snapshot_timesteps,
            teacher_best_env_num_ep=teacher_best_env_num_ep,
            n_clusters=n_clusters,
        )
        env = NormalizeObservation(env)
        if obs_rms is not None:
            env.obs_rms = copy.deepcopy(obs_rms)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = NormalizeReward(env, gamma=gamma)
        if return_rms is not None:
            env.return_rms = copy.deepcopy(return_rms)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def make_teacher_env(env_id, idx, capture_video, run_name, obs_rms=None):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = SaveObservationAndInfo(env)
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservation(env)
        if obs_rms is not None:
            env.obs_rms = copy.deepcopy(obs_rms)
        env.freeze = True
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk


def make_student_env(env_id, idx, capture_video, run_name, gamma, obs_rms=None, return_rms=None):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservation(env)
        if obs_rms is not None:
            env.obs_rms = copy.deepcopy(obs_rms)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = NormalizeReward(env, gamma=gamma)
        if return_rms is not None:
            env.return_rms = copy.deepcopy(return_rms)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def make_eval_env(env_id, idx, capture_video, run_name, obs_rms=None):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservation(env)
        if obs_rms is not None:
            env.obs_rms = copy.deepcopy(obs_rms)
        env.freeze = True
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk


def get_rms(env):
    obs_rms, return_rms = None, None
    env_point = env
    while hasattr(env_point, "env"):
        if isinstance(env_point, NormalizeObservation):
            obs_rms = copy.deepcopy(env_point.obs_rms)
            break
        env_point = env_point.env
    else:
        print("can't find NormalizeReward")

    env_point = env
    while hasattr(env_point, "env"):
        if isinstance(env_point, NormalizeReward):
            return_rms = copy.deepcopy(env_point.return_rms)
            break
        env_point = env_point.env
    else:
        print("can't find NormalizeReward")

    return obs_rms, return_rms


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.obs_rms = RunningMeanStd(shape=envs.single_observation_space.shape)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_teacher_env(args.env_id, i, args.capture_video, run_name) for i in range(1)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    teacher_model_path = hf_hub_download(
        repo_id=args.teacher_policy_hf_repo, filename="ppo_fix_continuous_action.cleanrl_model"
    )
    teacher_model = Agent(envs).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    teacher_model.eval()
    envs = gym.vector.SyncVectorEnv(
        [make_teacher_env(args.env_id, i, args.capture_video, run_name, teacher_model.obs_rms) for i in range(1)]
    )

    envs.single_observation_space.dtype = np.float32

    # collect teacher envs for args.teacher_steps
    teacher_env_deq_deq = deque(maxlen=args.teacher_env_num_ep)
    ep_env_deq = deque()

    teacher_step = 0
    obs, infos = envs.reset(seed=args.seed)
    with torch.no_grad():
        _, _, _, q_val = teacher_model.get_action_and_value(torch.Tensor(obs).to(device))
    q_val = float(q_val)
    ep_env_deq.append((q_val, env_deepcopy(envs.envs[0])))
    ep_cnt = 0
    while teacher_env_deq_deq.maxlen > len(teacher_env_deq_deq):
        with torch.no_grad():
            actions, _, _, _ = teacher_model.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                ep_cnt += 1
            teacher_env_deq_deq.append(ep_env_deq)
            ep_env_deq = deque()
        obs = next_obs
        if teacher_step % args.teacher_env_save_frequency == 0:
            print(
                f"SAVE ENV {teacher_step} {ep_cnt}/{args.teacher_env_num_ep} -> episode_length:",
                envs.envs[0].episode_lengths,
                "| episode_return:",
                envs.envs[0].episode_returns,
            )
            with torch.no_grad():
                _, _, _, q_val = teacher_model.get_action_and_value(torch.Tensor(obs).to(device))
            q_val = float(q_val)
            ep_env_deq.append((q_val, env_deepcopy(envs.envs[0])))
        teacher_step += 1

    obs_rms, return_rms = None, None
    envs = gym.vector.SyncVectorEnv(
        [
            make_snapshot_env(
                teacher_env_deq_deq,
                truncate_step=args.truncate_step,
                snapshot_timesteps=args.snapshot_timesteps,
                teacher_best_env_num_ep=args.teacher_best_env_num_ep,
                n_clusters=args.n_clusters,
                gamma=args.gamma,
                obs_rms=obs_rms,
                return_rms=return_rms,
            )
            for i in range(args.num_envs)
        ]
    )
    envs.single_observation_space.dtype = np.float32

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        if global_step >= args.snapshot_timesteps and "SnapshotEnv" in str(envs.envs[0]):
            obs_rms, return_rms = get_rms(envs.envs[0])
            envs = gym.vector.SyncVectorEnv(
                [
                    make_student_env(args.env_id, i, args.capture_video, run_name, args.gamma, obs_rms, return_rms)
                    for i in range(args.num_envs)
                ]
            )
            envs.single_observation_space.dtype = np.float32
            next_obs, _ = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # https://github.com/DLR-RM/stable-baselines3/pull/658
            for idx, trunc in enumerate(truncated):
                if trunc and not terminated[idx]:
                    real_next_obs = infos["final_observation"][idx]
                    with torch.no_grad():
                        terminal_value = agent.get_value(torch.Tensor(real_next_obs).to(device)).reshape(1, -1)[0][0]
                    rewards[step][idx] += args.gamma * terminal_value

            if global_step % (5000 // args.num_envs * args.num_envs) == 0:
                model_path = f"runs/{run_name}/{args.exp_name}-{global_step}.cleanrl_model"
                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")

                obs_rms, return_rms = get_rms(envs.envs[0])
                episodic_returns, episodic_lengths = evaluate(
                    model_path,
                    make_eval_env,
                    args.env_id,
                    eval_episodes=3,
                    run_name=f"{run_name}-eval",
                    Model=Agent,
                    device=device,
                    capture_video=False,
                    obs_rms=obs_rms,
                )

                print(episodic_returns, episodic_lengths)
                writer.add_scalar("charts/eval/episodic_return", np.mean(episodic_returns), global_step)
                writer.add_scalar("charts/eval/episodic_length", np.mean(episodic_lengths), global_step)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                if "SnapshotEnv" in str(envs.envs[0]):
                    print(
                        f"snapshot_env  global_step={global_step}, episodic_length={info['episode']['l']}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar("charts/snapshot_env/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/snapshot_env/episodic_length", info["episode"]["l"], global_step)
                else:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
