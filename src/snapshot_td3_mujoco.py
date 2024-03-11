import argparse
import copy
import os
import random
import time
from collections import defaultdict, deque
from distutils.util import strtobool
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")

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
    parser.add_argument("--drop-snap-replay", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, will be drop replay data in snapshot env period")
    args = parser.parse_args()
    # fmt: on

    if args.teacher_policy_hf_repo is None:
        policy_dict = {
            "Hopper-v4": "sdpkjc/Hopper-v4-td3_continuous_action-seed3",
            "Walker2d-v4": "cleanrl/Walker2d-v4-td3_continuous_action-seed1",
            "HalfCheetah-v4": "cleanrl/HalfCheetah-v4-td3_continuous_action-seed1",
            "Ant-v4": "sdpkjc/Ant-v4-td3_continuous_action-seed4",
            "Swimmer-v4": "sdpkjc/Swimmer-v4-td3_continuous_action-seed4",
            "Humanoid-v4": "sdpkjc/Humanoid-v4-td3_continuous_action-seed3",
        }
        if args.env_id in policy_dict.keys():
            args.teacher_policy_hf_repo = policy_dict[args.env_id]
        else:
            args.teacher_policy_hf_repo = f"cleanrl/{args.env_id}-td3_continuous_action-seed1"
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
    exploration_noise: float = 0.1,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    actor = Model[0](envs).to(device)
    qf1 = Model[1](envs).to(device)
    qf2 = Model[1](envs).to(device)
    actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf1.load_state_dict(qf1_params)
    qf2.load_state_dict(qf2_params)
    qf1.eval()
    qf2.eval()
    # note: qf1 and qf2 are not used in this script

    obs, _ = envs.reset()
    episodic_returns, episodic_lengths = [], []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * exploration_noise)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        next_obs, _, _, _, infos = envs.step(actions)
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


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = SaveObservationAndInfo(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    teacher_model_path = hf_hub_download(repo_id=args.teacher_policy_hf_repo, filename="td3_continuous_action.cleanrl_model")
    teacher_model = Actor(envs).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device)[0])
    teacher_model.eval()
    teacher_model_qf = QNetwork(envs).to(device)
    teacher_model_qf.load_state_dict(torch.load(teacher_model_path, map_location=device)[1])
    teacher_model_qf.eval()

    envs.single_observation_space.dtype = np.float32

    # collect teacher envs for args.teacher_steps
    teacher_env_deq_deq = deque(maxlen=args.teacher_env_num_ep)
    ep_env_deq = deque()

    teacher_step = 0
    obs, infos = envs.reset(seed=args.seed)
    with torch.no_grad():
        q_val = float(
            teacher_model_qf(
                torch.Tensor(obs).to(device), torch.Tensor(teacher_model(torch.Tensor(obs).to(device))).to(device)
            ).view(-1)[0]
        )
    ep_env_deq.append((q_val, env_deepcopy(envs.envs[0])))
    ep_cnt = 0
    while teacher_env_deq_deq.maxlen > len(teacher_env_deq_deq):
        with torch.no_grad():
            actions = teacher_model(torch.Tensor(obs).to(device))
            # actions += torch.normal(0, actor.action_scale * args.exploration_noise)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
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
                q_val = float(
                    teacher_model_qf(
                        torch.Tensor(obs).to(device), torch.Tensor(teacher_model(torch.Tensor(obs).to(device))).to(device)
                    ).view(-1)[0]
                )
            ep_env_deq.append((q_val, env_deepcopy(envs.envs[0])))
        teacher_step += 1

    envs = gym.vector.SyncVectorEnv(
        [
            lambda: SnapshotEnv(
                teacher_env_deq_deq,
                truncate_step=args.truncate_step,
                snapshot_timesteps=args.snapshot_timesteps,
                teacher_best_env_num_ep=args.teacher_best_env_num_ep,
                n_clusters=args.n_clusters,
            )
        ]
    )
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step >= args.snapshot_timesteps and "SnapshotEnv" in str(envs.envs[0]):
            envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
            envs.single_observation_space.dtype = np.float32
            obs, _ = envs.reset(seed=args.seed)
            if args.drop_snap_replay:
                rb = ReplayBuffer(
                    args.buffer_size,
                    envs.single_observation_space,
                    envs.single_action_space,
                    device,
                    optimize_memory_usage=False,
                    handle_timeout_termination=False,
                )
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
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
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncateds):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminateds, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if global_step % 5000 == 0:
            # evaluate the student model
            model_path = f"runs/{run_name}/{args.exp_name}-{global_step}.cleanrl_model"
            torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
            print(f"model saved to {model_path}")

            episodic_returns, episodic_lengths = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=3,
                run_name=f"{run_name}-eval",
                Model=(Actor, QNetwork),
                device=device,
                capture_video=False,
                exploration_noise=0.0,
            )

            print(episodic_returns, episodic_lengths)
            writer.add_scalar("charts/eval/episodic_return", np.mean(episodic_returns), global_step)
            writer.add_scalar("charts/eval/episodic_length", np.mean(episodic_lengths), global_step)

    envs.close()
    writer.close()
