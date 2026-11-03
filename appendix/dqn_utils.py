"""
Deep Q-network (DQN) on the cart-pole swing-up, with and without the safety filter of `safety_filter.py`.

The learner is the DQN of the RWTH course "Reinforcement Learning and Learning-based Control" (DSME), with which the
results of the tutorial paper were produced: one environment, epsilon-greedy exploration with a linearly decaying
epsilon, a replay buffer of 100k transitions, a mini-batch of 128 every 4 steps, the mean squared TD error, and a
target network that is overwritten with the Q-network every 500 steps. The original ran on gymnasium and the
stable-baselines3 replay buffer; this version needs only numpy and torch and, for the same seed and PyTorch version
(2.7.1), reproduces those runs number for number (same random draws, same arithmetic). Other PyTorch versions round
the gradient steps slightly differently, so the runs drift apart once learning starts.

Safe DQN is the same learner on `SafetyWrapper(CartPoleSwingUp())`: the environment executes the filtered force and
shapes the reward, while the replay buffer stores the action the agent intended.

    from appendix.dqn_utils import train
    run = train(seed=1, safe=True)       # about 3 minutes on one CPU core (DQN without the filter: about 1.5)
"""
import copy
import multiprocessing as mp
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from appendix.safety_filter import DEFAULT as SAFETY, CartPoleSwingUp, SafetyWrapper

__all__ = ["HYPERPARAMS", "q_network", "ReplayBuffer", "make_env", "evaluate", "train", "train_all", "save_runs",
           "load_runs", "load_policy", "learning_curves", "summary", "greedy_rollout"]

HYPERPARAMS = dict(
    total_steps=350_000,            # environment interactions
    learning_rate=2.5e-4,
    gamma=0.99,
    buffer_size=100_000,
    batch_size=128,
    train_frequency=4,              # one gradient step every 4 interactions
    target_network_frequency=500,   # the target network is overwritten every 500 interactions
    start_learning=10_000,          # interactions collected before the first gradient step
    start_e=1.0, end_e=0.05,        # epsilon decays linearly ...
    exploration_fraction=0.3,       # ... over the first 30 % of the interactions
)
EVAL_EVERY_STEPS = 1000             # the greedy policy is evaluated every 1000 interactions,
EVAL_EVERY_EPISODES = 25            # every 25 training episodes,
EVAL_EPISODES = 10                  # on 10 episodes of the plain task (no bounds, no filter)


def q_network(obs_dim=5, num_actions=3):
    """The Q-network: an MLP from the observation to one Q-value per force."""
    return nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, num_actions))


def greedy_action(q, obs):
    with torch.no_grad():
        return int(q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item())


def linear_schedule(start_e, end_e, duration, t):
    return max((end_e - start_e) / duration * t + start_e, end_e)


class ReplayBuffer:
    """Ring buffer of transitions (s, a, s', r, done); `sample` draws a uniform mini-batch with numpy's global RNG."""

    def __init__(self, size, obs_dim=5):
        self.size, self.pos, self.full = size, 0, False
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, 1), dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.pos], self.next_obs[self.pos], self.actions[self.pos] = obs, next_obs, action
        self.rewards[self.pos], self.dones[self.pos] = reward, done
        self.pos = (self.pos + 1) % self.size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size if self.full else self.pos, size=batch_size)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.actions[idx]), torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.rewards[idx]), torch.from_numpy(self.dones[idx]))


def make_env(safe=False, safety_cfg=None):
    """The plain swing-up, or the swing-up with the safety filter and the CBF-RL reward shaping."""
    env = CartPoleSwingUp()
    return SafetyWrapper(env, safety_cfg) if safe else env


def evaluate(q, env, seed, episodes=EVAL_EPISODES):
    """Returns of `episodes` consecutive greedy episodes, the first one started from `seed`."""
    returns, ret = [], np.float32(0.0)
    obs, _ = env.reset(seed=seed)
    while len(returns) < episodes:
        obs, reward, terminated, truncated, _ = env.step(greedy_action(q, obs))
        ret = np.float32(ret + np.float32(reward))         # accumulated in single precision
        if terminated or truncated:
            returns.append(ret)
            ret = np.float32(0.0)
            obs, _ = env.reset()
    return np.array(returns, dtype=np.float32)


def train(seed, safe=False, snapshot_steps=(), safety_cfg=None, verbose=True, **hyperparams):
    """Trains one agent. Returns a dict with the evaluation points `steps`, the evaluation `returns`
    (steps x EVAL_EPISODES), the Q-network weights at `snapshot_steps` (`policies`), and how often the cart left
    the interval [x_min, x_max] of the safety filter during training: the number of training `episodes`, the number
    of episodes in which the cart left the interval (`violations`) and the number of interactions after which it was
    outside (`steps_outside`). Both arms are checked against the same interval, although only the safe arm knows it."""
    hp = {**HYPERPARAMS, **hyperparams}
    torch.set_num_threads(1)               # a tiny network: more threads only cost synchronisation
    random.seed(seed)                      # epsilon coin
    np.random.seed(seed)                   # replay-buffer sampling
    torch.manual_seed(seed)                # network initialisation
    action_rng = np.random.default_rng(seed)   # random actions

    env, eval_env = make_env(safe, safety_cfg), CartPoleSwingUp()
    env.reset(seed=seed)                   # the environment was seeded once when it was created,
    obs, _ = env.reset()                   # and training starts from its next initial state
    q = q_network()
    target = copy.deepcopy(q)
    optimizer = torch.optim.Adam(q.parameters(), lr=hp["learning_rate"])
    buffer = ReplayBuffer(hp["buffer_size"], obs_dim=len(obs))

    steps, returns, policies = [], [], {}
    episodes, last_eval_episode, t0 = 0, None, time.time()
    bounds = {**SAFETY, **(safety_cfg or {})}
    violations = steps_outside = 0
    outside_in_episode = False

    def evaluate_now(step):
        steps.append(step)
        returns.append(evaluate(q, eval_env, seed))
        if step in snapshot_steps:
            policies[step] = copy.deepcopy(q.state_dict())

    total = hp["total_steps"]
    for step in range(1, total + 1):
        epsilon = linear_schedule(hp["start_e"], hp["end_e"], hp["exploration_fraction"] * total, step - 1)
        action = int(action_rng.integers(env.num_actions)) if random.random() < epsilon else greedy_action(q, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if not bounds["x_min"] <= env.state[0] <= bounds["x_max"]:
            steps_outside += 1
            outside_in_episode = True
        if terminated or truncated:
            episodes += 1
            violations += outside_in_episode
            outside_in_episode = False
            start_obs, _ = env.reset()
        else:
            start_obs = next_obs

        due_episode = episodes % EVAL_EVERY_EPISODES == 0 and episodes != last_eval_episode
        if due_episode or (step % EVAL_EVERY_STEPS == 0 and step != total):
            last_eval_episode = episodes if due_episode else last_eval_episode
            evaluate_now(step)

        buffer.add(obs, next_obs, action, reward, terminated)   # the intended action, the true successor
        obs = start_obs

        if step > hp["start_learning"]:
            if step % hp["train_frequency"] == 0:
                s, a, s_next, r, d = buffer.sample(hp["batch_size"])
                with torch.no_grad():
                    td_target = r + hp["gamma"] * target(s_next).max(dim=1).values * (1 - d)
                q_sa = q(s).gather(1, a).squeeze()
                loss = F.mse_loss(td_target, q_sa)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if step % hp["target_network_frequency"] == 0:
                target.load_state_dict(q.state_dict())

    if steps[-1] != total:
        evaluate_now(total)
    if verbose:
        print(f"{'Safe DQN' if safe else 'DQN'}, seed {seed}: {total:,} interactions, {episodes} episodes in "
              f"{(time.time() - t0) / 60:.1f} min, best evaluation return {max(r.mean() for r in returns):.1f}, "
              f"cart outside [{bounds['x_min']}, {bounds['x_max']}] in {violations} episode(s)", flush=True)
    return dict(steps=np.array(steps), returns=np.stack(returns), policies=policies,
                episodes=episodes, violations=violations, steps_outside=steps_outside)


def _train_job(args):
    seed, safe, snapshot_steps = args
    return (("safe_dqn" if safe else "dqn"), seed), train(seed, safe, snapshot_steps)


def train_all(seeds=range(1, 11), arms=("dqn", "safe_dqn"), snapshot_steps=(), processes=None):
    """Trains every (arm, seed) in its own process; each run is independent, so the results do not change."""
    jobs = [(s, arm == "safe_dqn", tuple(snapshot_steps)) for arm in arms for s in seeds]
    with mp.get_context("spawn").Pool(processes or min(len(jobs), mp.cpu_count())) as pool:
        return dict(pool.map(_train_job, jobs))


# ----------------------------------------------------------------------------- results
RUN_FIELDS = ("steps", "returns", "episodes", "violations", "steps_outside")


def save_runs(path, runs):
    np.savez_compressed(path, **{f"{arm}_seed{seed}_{field}": np.asarray(run[field])
                                 for (arm, seed), run in runs.items() for field in RUN_FIELDS})


def load_runs(path):
    """{(arm, seed): dict(steps, returns, episodes, violations, steps_outside)} from a file written by `save_runs`."""
    z, runs = np.load(path), {}
    for key in z.files:
        match = re.fullmatch(r"(\w+?)_seed(\d+)_(\w+)", key)
        runs.setdefault((match[1], int(match[2])), {})[match[3]] = z[key]
    return runs


def load_policy(path):
    q = q_network()
    q.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return q.eval()


def learning_curves(runs, arm, num=200):
    """Mean and standard deviation over the seeds of the mean evaluation return, on a common grid of interactions."""
    curves = [(r["steps"], r["returns"].astype(np.float64).mean(axis=1)) for (a, _), r in runs.items() if a == arm]
    grid = np.linspace(max(s[0] for s, _ in curves), min(s[-1] for s, _ in curves), num)
    stacked = np.stack([np.interp(grid, s, v) for s, v in curves])
    return grid, stacked.mean(axis=0), stacked.std(axis=0), len(curves)


def summary(runs, threshold=350.0):
    """Per arm: interactions until the mean evaluation return first reaches `threshold`, the area under the curve
    (normalised by the number of interactions), the best and the final return; mean +- std over the seeds."""
    rows = {}
    for arm in ("dqn", "safe_dqn"):
        per = []
        for (a, _), r in runs.items():
            if a != arm:
                continue
            s, m = r["steps"].astype(np.float64), r["returns"].astype(np.float64).mean(axis=1)
            hit = np.flatnonzero(m >= threshold)
            per.append([s[hit[0]] if len(hit) else np.nan, np.sum((m[1:] + m[:-1]) / 2 * np.diff(s)) / s[-1],
                        m.max(), m[-1]])
        per = np.array(per)
        rows[arm] = dict(seeds=len(per), reached=int(np.isfinite(per[:, 0]).sum()),
                         **{k: (np.nanmean(per[:, i]), np.nanstd(per[:, i], ddof=1))
                            for i, k in enumerate(("steps_to_threshold", "auc", "best", "final"))})
    return rows


def greedy_rollout(q, seed):
    """One greedy episode on the plain task: the pole-tip path (before every step) and the return."""
    env = CartPoleSwingUp()
    obs, _ = env.reset(seed=seed)
    pole_length = 2 * env.length                 # `length` is half the pole
    tip_x, tip_y, total, done = [], [], 0.0, False
    while not done:
        x, _, theta, _ = env.state
        tip_x.append(x + pole_length * np.sin(theta))
        tip_y.append(pole_length * np.cos(theta))
        obs, reward, terminated, truncated, _ = env.step(greedy_action(q, obs))
        total += float(reward)
        done = terminated or truncated
    return np.array(tip_x), np.array(tip_y), total, pole_length
