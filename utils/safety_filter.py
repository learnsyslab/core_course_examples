"""
Safety filter for the cart-pole swing-up: the maximal control-invariant set of the cart-position constraint,
computed offline with Hamilton-Jacobi reachability, enforced online by a check-and-replace filter, and the CBF-RL
reward shaping that lets a learner see what the filter does.

Ported from the `safety/` package of the CDC tutorial safe-RL code, without gymnasium: `CartPoleSwingUp` below is
a stand-alone copy of the `CartPoleSwingUp-v0` environment used there (gymnasium's CartPole dynamics, the pole
starting downward, a dense reward, a 500-step limit), so only numpy is needed at run time.

    from utils.safety_filter import CartPoleSwingUp, SafetyWrapper
    env = SafetyWrapper(CartPoleSwingUp())            # loads the precomputed set from utils/safe_sets/
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)   # executes a safe force, shaped reward

The constraint is x_min <= x <= x_max on the cart position (default -1 <= x <= 2.4); the pole is not constrained.
Per step the filter
  1. predicts the successor state of every available force with the cart-pole's own Euler step,
  2. keeps the forces whose successor stays inside the invariant set, i.e. V(successor) >= eps,
  3. executes the intended force if it is among them, otherwise the one with the largest V.
The executed force is therefore always one of the discrete forces. Where the grid has no value (the pole spinning
faster than the grid covers) the filter falls back to a high-order CBF condition on the same bounds.

The reward gets the two graded terms of CBF-RL (Yang et al., arXiv:2510.14959), computed from the intended force
and the executed one, both exactly zero when the intent is safe:

    r_cbf = w_v * clip(min(V(s'_intent) - eps, 0) / V_max, -1, 0)  +  w_d * (exp(-(F_intent - F_safe)^2 / sigma^2) - 1)

and leaving the interval ends the episode with a penalty of `wall_penalty`.

The invariant set (a 41 x 41 x 48 x 41 grid over x, x_dot, theta, theta_dot) ships in `safe_sets/`. Recomputing it,
e.g. for other bounds, needs `pip install "jax[cpu]" hj-reachability` and takes about 12 minutes on a CPU.
`python utils/safety_filter.py` runs the invariance checks.
"""
import hashlib
import json
import os
import time

import numpy as np

SAFE_SET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "safe_sets")


# ----------------------------------------------------------------------------- the cart-pole swing-up
# the parameters of gymnasium's CartPole-v1
PARAMS = dict(gravity=9.8, masscart=1.0, masspole=0.1, length=0.5, tau=0.02, force_mag=10.0, x_threshold=2.4)


def cart_pole_dynamics(state, force, params=PARAMS):
    """One forward-Euler step of gym's CartPole dynamics for a (N, 4) state array and (N,) forces.
    Identical to `CartPoleSwingUp.step`."""
    g, mc, mp, l, tau = params["gravity"], params["masscart"], params["masspole"], params["length"], params["tau"]
    total, pml = mc + mp, mp * l
    x, x_dot, theta, theta_dot = state.T
    cos, sin = np.cos(theta), np.sin(theta)
    temp = (force + pml * theta_dot ** 2 * sin) / total
    thetaacc = (g * sin - cos * temp) / (l * (4.0 / 3.0 - mp * cos ** 2 / total))
    xacc = temp - pml * thetaacc * cos / total
    return np.stack([x + tau * x_dot, x_dot + tau * xacc, theta + tau * theta_dot, theta_dot + tau * thetaacc], axis=1)


class CartPoleSwingUp:
    """
    Cart-pole in which the pole starts hanging down and has to be swung up and balanced, with gymnasium's
    `reset`/`step` interface. It reuses the dynamics of the standard CartPole, but

    - the pole starts hanging down (angle pi) with a small random perturbation;
    - the reward is dense: how upright the pole is, times how close the cart is to the middle of the track, times
      how slowly the pole rotates. A balanced pole in the middle of the track yields 1 per step, and the reward
      stays inside [0, 1];
    - a falling pole does not end the episode, only leaving the track (|x| > x_threshold) does, and an episode is
      truncated after `max_episode_steps` steps.

    The observation is (x, x_dot, cos(theta), sin(theta), theta_dot), and the action picks one of `num_actions`
    equally spaced forces between -force_mag and +force_mag.
    """

    def __init__(self, num_actions=3, max_episode_steps=500, params=PARAMS):
        self.params = dict(params)
        self.gravity, self.masscart, self.masspole = params["gravity"], params["masscart"], params["masspole"]
        self.length, self.tau = params["length"], params["tau"]
        self.force_mag, self.x_threshold = params["force_mag"], params["x_threshold"]
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        self.num_actions = num_actions
        self.forces = np.linspace(-self.force_mag, self.force_mag, num_actions)
        self.max_episode_steps = max_episode_steps
        self.np_random = np.random.default_rng()
        self.state = None
        self.elapsed_steps = 0

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def reset(self, *, seed=None):
        if seed is not None:                               # the same generator gymnasium seeds
            self.np_random = np.random.default_rng(seed)
        hanging_down = np.array([0.0, 0.0, np.pi, 0.0])
        self.state = hanging_down + self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.elapsed_steps = 0
        return self._get_obs(), {}

    def step(self, action):
        assert 0 <= int(action) < self.num_actions, f"{action!r} invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.forces[action]

        # same dynamics as the standard CartPole environment
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)

        # dense reward: upright (1 upright, 0 hanging down) x centered (1 in the middle, 0 at the border)
        # x slow (1 at rest, 1/2 for a fast spinning pole)
        upright = (1.0 + np.cos(theta)) / 2.0
        centered = 1.0 - (x / self.x_threshold) ** 2
        slow = (1.0 + 10.0 ** (-(theta_dot / 5.0) ** 2)) / 2.0
        reward = upright * centered * slow

        # only leaving the track ends the episode, a falling pole does not
        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.max_episode_steps
        return self._get_obs(), reward, terminated, truncated, {}


# ----------------------------------------------------------------------------- the invariant set (offline)
# With the constraint function l(s) = min(x - x_min, x_max - x) (or x - x_min if x_max is None) the value function
# of the "avoid" problem, V(s) = max over control sequences min over time l(s(t)), solves the Hamilton-Jacobi-Isaacs
# PDE, and its zero super-level set {V >= 0} is the maximal control-invariant subset of {l >= 0}: from every state in
# it some admissible force keeps the cart inside the interval forever. The Hamiltonian is linear in the force, so
# the optimal safe control is bang-bang and the set computed for the continuous interval [-F_max, F_max] is the same
# as for the three discrete forces.

def default_config(x_min=-1.0, x_max=2.4, x_grid_max=2.4):
    """`x_max=None` enforces the lower bound only; `x_grid_max` is where the grid ends (the track edge)."""
    return dict(x_min=x_min, x_max=x_max, x_grid_max=x_grid_max,
                x_pad=0.3,                   # the grid extends this far beyond the bounds
                v_max=5.0, w_max=15.0,       # |x_dot| and |theta_dot| covered by the grid
                shape=(41, 41, 48, 41),      # (x, x_dot, theta, theta_dot)
                force_max=10.0, disturbance=0.0, accuracy="high",
                horizon=6.0, chunk=1.0, tol=1e-3)


def safe_set_path(cfg):
    """File of the set for a configuration: its name is a hash of the configuration."""
    return os.path.join(SAFE_SET_DIR, f"hj_safe_set_{hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]}.npz")


class SafeSet:
    """The value function on the 4-D grid, with multilinear interpolation (theta is periodic)."""

    def __init__(self, axes, values, meta=None):
        self.axes = [np.linspace(float(a[0]), float(a[-1]), len(a), dtype=np.float64) for a in axes]
        self.values = np.asarray(values, dtype=np.float64)
        self.meta = meta or {}
        self.lo = np.array([a[0] for a in self.axes])
        self.dx = np.array([a[1] - a[0] for a in self.axes])
        self.shape = np.array(self.values.shape)

    def save(self, path):
        np.savez_compressed(path, values=self.values, meta=json.dumps(self.meta),
                            **{f"axis{i}": a for i, a in enumerate(self.axes)})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls([z[f"axis{i}"] for i in range(4)], z["values"], json.loads(str(z["meta"])))

    def safe_fraction(self):
        return float((self.values >= 0).mean())

    def interpolate(self, states, outside_value=-1.0):
        """V at `states` (N, 4). Points outside the grid in x, x_dot or theta_dot get `outside_value`."""
        s = np.asarray(states, dtype=np.float64).copy()
        n_th = self.shape[2]
        s[:, 2] = self.lo[2] + np.mod(s[:, 2] - self.lo[2], n_th * self.dx[2])      # wrap the angle
        pos = (s - self.lo) / self.dx
        inside = np.ones(len(s), dtype=bool)
        for d in (0, 1, 3):
            inside &= (pos[:, d] >= -1e-6) & (pos[:, d] <= self.shape[d] - 1 + 1e-6)
            pos[:, d] = np.clip(pos[:, d], 0.0, self.shape[d] - 1)
        idx0 = np.floor(pos).astype(np.int64)
        for d in (0, 1, 3):
            idx0[:, d] = np.clip(idx0[:, d], 0, self.shape[d] - 2)
        idx0[:, 2] = np.mod(idx0[:, 2], n_th)
        frac = np.clip(pos - idx0, 0.0, 1.0)
        frac[:, 2] = np.mod(pos[:, 2] - np.floor(pos[:, 2]), 1.0)
        out = np.zeros(len(s))
        for c0 in (0, 1):
            for c1 in (0, 1):
                for c2 in (0, 1):
                    for c3 in (0, 1):
                        w = ((frac[:, 0] if c0 else 1 - frac[:, 0]) * (frac[:, 1] if c1 else 1 - frac[:, 1]) *
                             (frac[:, 2] if c2 else 1 - frac[:, 2]) * (frac[:, 3] if c3 else 1 - frac[:, 3]))
                        out += w * self.values[idx0[:, 0] + c0, idx0[:, 1] + c1,
                                               np.mod(idx0[:, 2] + c2, n_th), idx0[:, 3] + c3]
        return np.where(inside, out, outside_value)


def _hj_dynamics(params, cfg):
    import jax.numpy as jnp
    import hj_reachability as hj

    g, m_pole = params["gravity"], params["masspole"]
    mu = params["masscart"] + params["masspole"]
    l = params["length"]
    ml = m_pole * l

    class CartPoleHJ(hj.ControlAndDisturbanceAffineDynamics):
        """dx/dt = f(s) + G_u(s) F + G_d(s) d, the control-affine form of gym's CartPole dynamics."""

        def __init__(self):
            f = cfg["force_max"]
            d = cfg["disturbance"]
            super().__init__("max", "min", hj.sets.Box(jnp.array([-f]), jnp.array([f])),
                             hj.sets.Box(jnp.array([-d, -d]), jnp.array([d, d])))

        def open_loop_dynamics(self, state, time):
            _, x_dot, theta, theta_dot = state
            sin, cos = jnp.sin(theta), jnp.cos(theta)
            D = l * (4.0 / 3.0 - m_pole * cos ** 2 / mu)
            f_th = (g * sin - cos * (ml * theta_dot ** 2 * sin) / mu) / D
            f_x = ml * (theta_dot ** 2 * sin - f_th * cos) / mu
            return jnp.array([x_dot, f_x, theta_dot, f_th])

        def control_jacobian(self, state, time):
            cos = jnp.cos(state[2])
            D = l * (4.0 / 3.0 - m_pole * cos ** 2 / mu)
            g_th = (-cos / mu) / D
            return jnp.array([[0.0], [(1.0 - ml * g_th * cos) / mu], [0.0], [g_th]])

        def disturbance_jacobian(self, state, time):
            return jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])

    return CartPoleHJ()


_LOADED = {}


def compute_safe_set(params=PARAMS, cfg=None, verbose=True) -> SafeSet:
    """Loads the set of `cfg` from `safe_sets/` if it is there, otherwise solves the PDE to convergence (JAX and
    hj_reachability needed) and saves it. Within a process a set is loaded only once."""
    cfg = {**default_config(), **(cfg or {})}
    path = safe_set_path(cfg)
    if path in _LOADED:
        return _LOADED[path]
    if os.path.isfile(path):
        s = _LOADED[path] = SafeSet.load(path)
        if verbose:
            print(f"loaded {path} (safe fraction {s.safe_fraction():.3f})")
        return s

    try:
        import jax
        import jax.numpy as jnp
        import hj_reachability as hj
    except ImportError as e:
        raise ImportError(f"no precomputed set at {path}; computing one needs "
                          f"`pip install \"jax[cpu]\" hj-reachability`") from e

    grid_max = cfg["x_max"] if cfg["x_max"] is not None else cfg["x_grid_max"]
    lo = np.array([cfg["x_min"] - cfg["x_pad"], -cfg["v_max"], -np.pi, -cfg["w_max"]])
    hi = np.array([grid_max + cfg["x_pad"], cfg["v_max"], np.pi, cfg["w_max"]])
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array(lo), jnp.array(hi)), tuple(cfg["shape"]), periodic_dims=2)
    x = grid.states[..., 0]
    l = x - cfg["x_min"] if cfg["x_max"] is None else jnp.minimum(x - cfg["x_min"], cfg["x_max"] - x)
    settings = hj.SolverSettings.with_accuracy(cfg["accuracy"],
                                               hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    dynamics = _hj_dynamics(params, cfg)
    values, t, t0, history = l, 0.0, time.time(), []
    while t > -cfg["horizon"] + 1e-9:
        new = jax.block_until_ready(hj.step(settings, dynamics, grid, t, values, t - cfg["chunk"], progress_bar=False))
        diff = float(jnp.max(jnp.abs(new - values)))
        values, t = new, t - cfg["chunk"]
        history.append(dict(t=-t, max_change=diff, safe_fraction=float((values >= 0).mean()), seconds=time.time() - t0))
        if verbose:
            print(f"  horizon {-t:4.1f} s: max change {diff:.2e}, safe fraction {history[-1]['safe_fraction']:.4f}")
        if diff < cfg["tol"]:
            break
    s = SafeSet([np.asarray(c) for c in grid.coordinate_vectors], np.asarray(values),
                dict(cfg=cfg, history=history, converged=bool(diff < cfg["tol"]), horizon_used=-t))
    os.makedirs(SAFE_SET_DIR, exist_ok=True)
    s.save(path)
    _LOADED[path] = s
    if verbose:
        print(f"saved {path}; converged={s.meta['converged']} after {-t:.1f} s, safe fraction {s.safe_fraction():.3f}")
    return s


# ----------------------------------------------------------------------------- the filter (online)
# The tuned configuration. `x_min`/`x_max` reach both the invariant set and the CBF fallback, so both enforce the
# same interval; `x_max=None` enforces the lower bound only.
DEFAULT = dict(
    x_min=-1.0, x_max=2.4,
    wall_penalty=1.0,        # subtracted once from the reward of the step that leaves the interval
    w_v=2.0, w_d=2.0, sigma=10.0,   # CBF-RL reward shaping (violation magnitude, distance to the safe force)
    eps=0.05,                # a force is allowed if V(successor) >= eps
    k1=8.0, k2=8.0, margin=0.05,    # high-order CBF fallback for states the grid does not cover
    grid_shape=(41, 41, 48, 41),
)


class _CBFFallback:
    """High-order CBF on the cart position, used only where the grid has no value: with
    h = x - x_min - margin and psi = x_dot + k1 h the condition psi_dot + k2 psi >= 0 is one linear
    inequality in the force (and mirrored for the upper bound)."""

    def __init__(self, params, x_min, x_max, k1, k2, margin):
        self.g, self.m_pole = params["gravity"], params["masspole"]
        self.mu = params["masscart"] + params["masspole"]
        self.l = params["length"]
        self.ml = self.m_pole * self.l
        self.x_min, self.x_max, self.k1, self.k2, self.margin = x_min, x_max, k1, k2, margin

    def affine_x(self, state):
        """x_ddot = f_x + g_x F, with g_x > 0 for every pole angle."""
        theta, theta_dot = state[:, 2], state[:, 3]
        sin, cos = np.sin(theta), np.cos(theta)
        D = self.l * (4.0 / 3.0 - self.m_pole * cos ** 2 / self.mu)
        f_th = (self.g * sin - cos * (self.ml * theta_dot ** 2 * sin) / self.mu) / D
        g_th = (-cos / self.mu) / D
        return self.ml * (theta_dot ** 2 * sin - f_th * cos) / self.mu, (1.0 - self.ml * g_th * cos) / self.mu

    def margins(self, state, forces):
        x, x_dot = state[:, 0], state[:, 1]
        f_x, g_x = self.affine_x(state)
        F = np.asarray(forces)[None, :]
        out = np.full((len(x), len(forces)), np.inf)
        if self.x_min is not None:
            h = x - self.x_min - self.margin
            out = np.minimum(out, g_x[:, None] * F - (-f_x - (self.k1 + self.k2) * x_dot - self.k1 * self.k2 * h)[:, None])
        if self.x_max is not None:
            h = self.x_max - self.margin - x
            out = np.minimum(out, -g_x[:, None] * F - (f_x + (self.k1 + self.k2) * x_dot - self.k1 * self.k2 * h)[:, None])
        return out


class SafetyFilter:
    """Check-and-replace filter over the discrete forces, plus the CBF-RL reward shaping."""

    def __init__(self, safe_set, forces, params=PARAMS, cfg=None):
        self.cfg = {**DEFAULT, **(cfg or {})}
        self.set = safe_set
        self.forces = np.asarray(forces, dtype=float)
        self.params = params
        self.v_scale = float(max(safe_set.values.max(), 1e-6))
        self.fallback = _CBFFallback(params, self.cfg["x_min"], self.cfg["x_max"],
                                     self.cfg["k1"], self.cfg["k2"], self.cfg["margin"])
        self.off_grid = np.zeros(0, dtype=bool)

    def filter(self, state, intent):
        """state (N, 4), intent (N,) -> executed actions, intervened, infeasible, V(state), margin of the intent."""
        state = np.atleast_2d(np.asarray(state, dtype=float))
        intent = np.atleast_1d(np.asarray(intent, dtype=np.int64))
        n, eps = len(intent), self.cfg["eps"]
        succ = np.stack([cart_pole_dynamics(state, np.full(n, F), self.params) for F in self.forces], axis=1)
        v_next = self.set.interpolate(succ.reshape(-1, 4), outside_value=np.nan).reshape(n, len(self.forces))
        off_grid = np.isnan(v_next).all(axis=1)
        self.off_grid = off_grid
        v_next = np.where(np.isnan(v_next), -1.0, v_next)
        margins = v_next - eps
        if off_grid.any():                                   # no value here: fall back to the CBF condition
            margins[off_grid] = self.fallback.margins(state[off_grid], self.forces)
        margin_intent = margins[np.arange(n), intent]
        unsafe = margin_intent < 0.0
        # the safest available force. Where the value function is flat several forces share the same margin and
        # the first of them is taken; preferring the one closest to the intent instead would return the intent
        # itself (its distance is zero) and the filter would stop correcting exactly where it has to
        safest = margins.argmax(axis=1)
        executed = np.where(unsafe, safest, intent)
        return (executed, unsafe & (executed != intent), (v_next.max(axis=1) < 0.0) & ~off_grid,
                self.set.interpolate(state), margin_intent)

    def penalty(self, intent, executed, margin_intent):
        """The two CBF-RL reward terms, zero unless the intended force was unsafe."""
        violation = np.clip(np.minimum(margin_intent, 0.0) / self.v_scale, -1.0, 0.0)
        distance = np.exp(-(self.forces[intent] - self.forces[executed]) ** 2 / self.cfg["sigma"] ** 2) - 1.0
        return self.cfg["w_v"] * violation + self.cfg["w_d"] * distance


def make_filter(env=None, cfg=None, verbose=False):
    """The filter for a `CartPoleSwingUp` (or its default parameters and three forces), with its invariant set."""
    cfg = {**DEFAULT, **(cfg or {})}
    params = env.params if env is not None else PARAMS
    forces = env.forces if env is not None else np.linspace(-params["force_mag"], params["force_mag"], 3)
    hj_cfg = default_config(x_min=cfg["x_min"], x_max=cfg["x_max"], x_grid_max=params["x_threshold"])
    hj_cfg["shape"] = tuple(cfg["grid_shape"])
    return SafetyFilter(compute_safe_set(params, hj_cfg, verbose=verbose), forces, params, cfg)


class SafetyWrapper:
    """Filters the action, adds the CBF-RL penalty and ends the episode outside the interval.

    The environment executes the safe force while the learner should store its own intended action, which keeps an
    off-policy update on the filtered MDP consistent. Adds to `info`: cbf_intervened, cbf_infeasible, cbf_off_grid,
    cbf_V, cbf_violated. `filter_actions=False` keeps the reward shaping but executes the intended action (a
    reward-only ablation), `shape_reward=False` filters without changing the reward."""

    def __init__(self, env, cfg=None, safety_filter=None, filter_actions=True, shape_reward=True):
        self.env = env
        self.cfg = {**DEFAULT, **(cfg or {})}
        self.filter = safety_filter if safety_filter is not None else make_filter(env, self.cfg)
        self.filter_actions, self.shape_reward = filter_actions, shape_reward
        self.forces = np.asarray(env.forces, dtype=float)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        state = np.asarray(self.env.state, dtype=float)[None]
        intent = np.array([int(action)])
        safe, intervened, infeasible, V, margin = self.filter.filter(state, intent)
        obs, reward, terminated, truncated, info = self.env.step(int(safe[0] if self.filter_actions else intent[0]))
        if self.shape_reward:
            reward = float(reward) + float(self.filter.penalty(intent, safe, margin)[0])
        x = float(self.env.state[0])
        violated = ((self.cfg["x_min"] is not None and x < self.cfg["x_min"]) or
                    (self.cfg["x_max"] is not None and x > self.cfg["x_max"]))
        if violated:
            reward -= self.cfg["wall_penalty"]
            terminated = True
        info = dict(info)
        info.update(cbf_intervened=bool(intervened[0]) and self.filter_actions, cbf_infeasible=bool(infeasible[0]),
                    cbf_off_grid=bool(self.filter.off_grid[0]), cbf_V=float(V[0]), cbf_violated=bool(violated))
        return obs, reward, terminated, truncated, info


# ----------------------------------------------------------------------------- verification
def check_invariance(cfg=None, steps=3125, num_envs=32, seed=0, mode="random"):
    """Rollouts from random states inside the set, with the filter: it must never leave the set or the interval.
    `mode`: "random" intents or "left"/"right" for an adversarial policy that always pushes one way."""
    cfg = {**DEFAULT, **(cfg or {})}
    f = make_filter(cfg=cfg)
    rng = np.random.default_rng(seed)
    x_hi = cfg["x_max"] if cfg["x_max"] is not None else 2.4
    lo = np.array([a[0] for a in f.set.axes]); hi = np.array([a[-1] for a in f.set.axes])
    state = np.zeros((num_envs, 4))
    exits = violations = infeasible = off_grid = 0
    min_V, min_x, max_x = np.inf, np.inf, -np.inf
    for t in range(steps):
        if t % 500 == 0:                                   # restart inside the set
            c = np.column_stack([rng.uniform(cfg["x_min"], x_hi, 4000), rng.uniform(-3, 3, 4000),
                                 rng.uniform(-np.pi, np.pi, 4000), rng.uniform(-6, 6, 4000)])
            state = c[f.set.interpolate(c) >= 0.2][:num_envs]
        intent = (rng.integers(0, len(f.forces), num_envs) if mode == "random"
                  else np.full(num_envs, 0 if mode == "left" else len(f.forces) - 1))
        executed, _, infeas, _, _ = f.filter(state, intent)
        state = cart_pole_dynamics(state, f.forces[executed], f.params)
        V = f.set.interpolate(state)
        # states the grid does not cover (the pole spinning faster than w_max) have no value; they are reported
        # separately instead of counting as an exit from the set
        covered = np.all((state[:, [1, 3]] >= lo[[1, 3]]) & (state[:, [1, 3]] <= hi[[1, 3]]), axis=1)
        off_grid += int((~covered).sum())
        exits += int((V[covered] < 0).sum()); infeasible += int(infeas.sum())
        violations += int((state[:, 0] < cfg["x_min"]).sum())
        if cfg["x_max"] is not None:
            violations += int((state[:, 0] > cfg["x_max"]).sum())
        min_V = min(min_V, float(V[covered].min()) if covered.any() else min_V); min_x = min(min_x, float(state[:, 0].min())); max_x = max(max_x, float(state[:, 0].max()))
    return dict(transitions=steps * num_envs, safe_set_exits=exits, bound_violations=violations,
                infeasible=infeasible, off_grid=off_grid, min_V=round(min_V, 3),
                min_x=round(min_x, 3), max_x=round(max_x, 3))


if __name__ == "__main__":
    for mode in ("random", "right", "left"):
        print(f"{mode:>6s} intents:", check_invariance(mode=mode))
