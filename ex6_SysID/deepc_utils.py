import numpy as np
import osqp
from scipy import sparse

from utils.env import Env, Dynamics
from utils.controller import check_input_constraints


'''------Implementation of DeePC (Chapter 6.4)------'''


def _as_time_major(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("Data must be a 1D or 2D time-major array.")
    return data


def _block_hankel(data: np.ndarray, depth: int) -> np.ndarray:
    """Build a block Hankel matrix from time-major data with shape (T, dim)."""
    data = _as_time_major(data)
    if depth < 1:
        raise ValueError("Hankel depth must be at least 1.")

    T, dim = data.shape
    n_cols = T - depth + 1
    if n_cols <= 0:
        raise ValueError(f"Not enough samples ({T}) for Hankel depth {depth}.")

    return np.column_stack([
        data[i:i + depth].reshape(depth * dim)
        for i in range(n_cols)
    ])


def persistent_excitation_rank(
        u_data: np.ndarray,
        order: int,
        input_offset: np.ndarray = None
    ):
    """Return (rank, required_rank) of the input Hankel matrix H_order(u)."""
    u_data = _as_time_major(u_data)
    if input_offset is not None:
        u_data = u_data - np.asarray(input_offset, dtype=float).reshape(1, -1)

    H = _block_hankel(u_data, order)
    return np.linalg.matrix_rank(H), H.shape[0]




def reconstruct_trajectory(
        u_data: np.ndarray,
        y_data: np.ndarray,
        u_test: np.ndarray,
        y_test: np.ndarray,
        length: int = None,
        input_offset: np.ndarray = None,
        output_offset: np.ndarray = None
    ):
    """
    Reconstruct one finite input-output trajectory from Hankel columns.

    This helper is used to demonstrate Willems' fundamental lemma.  y_data may
    contain either T or T+1 samples for T input samples; when it contains the
    extra final sample, the first T outputs are used for the aligned trajectory.

    Returns:
        g, u_reconstructed, y_reconstructed, relative_error
    """
    u_data = _as_time_major(u_data)
    y_data = _as_time_major(y_data)
    u_test = _as_time_major(u_test)
    y_test = _as_time_major(y_test)

    if length is None:
        length = u_test.shape[0]
    if u_test.shape[0] < length or y_test.shape[0] < length:
        raise ValueError("Test trajectory is shorter than the requested length.")

    # Align input/output trajectories at the same time indices.
    if y_data.shape[0] == u_data.shape[0] + 1:
        y_data_aligned = y_data[:-1]
    elif y_data.shape[0] == u_data.shape[0]:
        y_data_aligned = y_data
    else:
        raise ValueError("y_data must contain T or T+1 samples for T input samples.")

    if input_offset is None:
        input_offset = np.zeros(u_data.shape[1])
    if output_offset is None:
        output_offset = np.zeros(y_data.shape[1])

    input_offset = np.asarray(input_offset, dtype=float).reshape(1, -1)
    output_offset = np.asarray(output_offset, dtype=float).reshape(1, -1)

    u_dev = u_data - input_offset
    y_dev = y_data_aligned - output_offset
    u_test_dev = u_test[:length] - input_offset
    y_test_dev = y_test[:length] - output_offset

    Hu = _block_hankel(u_dev, length)
    Hy = _block_hankel(y_dev, length)
    n_cols = min(Hu.shape[1], Hy.shape[1])
    W = np.vstack([Hu[:, :n_cols], Hy[:, :n_cols]])
    w_test = np.concatenate([u_test_dev.reshape(-1), y_test_dev.reshape(-1)])

    g, *_ = np.linalg.lstsq(W, w_test, rcond=None)
    w_reconstructed = W @ g

    nu = u_data.shape[1]
    ny = y_data.shape[1]
    n_u = length * nu
    u_rec = w_reconstructed[:n_u].reshape(length, nu) + input_offset
    y_rec = w_reconstructed[n_u:].reshape(length, ny) + output_offset

    denom = max(np.linalg.norm(w_test), 1e-12)
    relative_error = np.linalg.norm(w_reconstructed - w_test) / denom
    return g, u_rec, y_rec, relative_error




def collect_deepc_data(
        env: Env,
        dynamics: Dynamics,
        freq: float,
        n_samples: int,
        excitation_amplitude: float = 0.8,
        initial_state: np.ndarray = None,
        seed: int = 0,
        output_indices=None
    ):
    """
    Collect one open-loop trajectory for DeePC.

    The excitation is centered around the equilibrium input of env.target_state.
    By default the full state is returned as the output.  `output_indices` can be
    used for output-only examples, e.g. output_indices=[0] for position only.

    Returns:
        u_data: shape (n_samples, nu)
        y_data: shape (n_samples + 1, ny)
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / freq

    state = np.array(
        env.target_state if initial_state is None else initial_state,
        dtype=float
    ).copy()

    u_eq = np.atleast_1d(
        dynamics.get_equilibrium_input(env.target_state)
    ).astype(float)

    # Random continuous excitation is persistently exciting with probability one
    # when enough data are collected. Center it around the equilibrium input.
    u_data = u_eq + rng.uniform(
        -excitation_amplitude,
        excitation_amplitude,
        size=(n_samples, dynamics.dim_inputs)
    )

    if env.input_lbs is not None:
        u_data = np.maximum(u_data, np.asarray(env.input_lbs))
    if env.input_ubs is not None:
        u_data = np.minimum(u_data, np.asarray(env.input_ubs))

    x_data = np.zeros((n_samples + 1, dynamics.dim_states))
    x_data[0] = state

    for k in range(n_samples):
        state = dynamics.one_step_forward(state, u_data[k], dt)
        x_data[k + 1] = state

    if output_indices is None:
        y_data = x_data
    else:
        y_data = x_data[:, np.asarray(output_indices, dtype=int)]

    return u_data, y_data


def collect_deepc_data_closed_loop(
        env: Env,
        dynamics: Dynamics,
        freq: float,
        n_samples: int,
        feedback_gain: np.ndarray,
        excitation_amplitude: float = 0.6,
        initial_state: np.ndarray = None,
        seed: int = 0,
        output_indices=None
    ):
    """
    Collect one closed-loop trajectory for DeePC around env.target_state.

    Open-loop excitation (collect_deepc_data) is fine on a flat, marginally
    stable plant, but on a non-flat terrain a constant input offset makes the
    car roll away and the data leave the region of interest.  Here a simple
    stabilizing feedback keeps the car near the target while random excitation
    is added on top:

        u_k = u_eq - K (x_k - x_target) + e_k,   e_k ~ U(-a, a).

    Because e_k is random, the applied input remains persistently exciting
    (closed-loop identification).  The result is a *local* dataset, i.e. the
    Hankel matrices describe the plant's behavior near the target only.

    Returns:
        u_data: shape (n_samples, nu)
        y_data: shape (n_samples + 1, ny)
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / freq
    K = np.asarray(feedback_gain, dtype=float).reshape(dynamics.dim_inputs, dynamics.dim_states)
    target = np.asarray(env.target_state, dtype=float)

    state = np.array(target if initial_state is None else initial_state, dtype=float).copy()
    u_eq = np.atleast_1d(dynamics.get_equilibrium_input(target)).astype(float)

    u_data = np.zeros((n_samples, dynamics.dim_inputs))
    x_data = np.zeros((n_samples + 1, dynamics.dim_states))
    x_data[0] = state

    for k in range(n_samples):
        u = u_eq - K @ (state - target) + rng.uniform(
            -excitation_amplitude, excitation_amplitude, size=dynamics.dim_inputs
        )
        if env.input_lbs is not None:
            u = np.maximum(u, np.asarray(env.input_lbs))
        if env.input_ubs is not None:
            u = np.minimum(u, np.asarray(env.input_ubs))
        u_data[k] = u
        state = dynamics.one_step_forward(state, u, dt)
        x_data[k + 1] = state

    if output_indices is None:
        y_data = x_data
    else:
        y_data = x_data[:, np.asarray(output_indices, dtype=int)]

    return u_data, y_data


class NonlinearMPCOracle:
    """
    Reference nonlinear MPC with the *true* plant model (tutorial oracle only).

    Multiple-shooting transcription with an RK4 step of the exact dynamics,
    solved with CasADi/ipopt and warm-started by shifting the previous
    solution.  It exists to show what is achievable when the nonlinear model
    is known; DeePC and the linear MPC never see this information.
    """

    def __init__(
            self,
            env: Env,
            dynamics: Dynamics,
            Q: np.ndarray,
            R: np.ndarray,
            Qf: np.ndarray,
            freq: float,
            N: int,
            name: str = 'NMPC_oracle',
            type: str = 'NMPC',
            verbose: bool = False
        ) -> None:
        import casadi as ca

        self.env = env
        self.dynamics = dynamics
        self.freq = freq
        self.dt = 1.0 / freq
        self.N = int(N)
        self.name = name
        self.type = type
        self.verbose = verbose

        self.target_state = np.asarray(env.target_state, dtype=float)
        self.u_eq = float(np.atleast_1d(dynamics.get_equilibrium_input(self.target_state))[0])
        f = dynamics.dynamics_function
        dt = self.dt

        def rk4(x, u):
            k1 = f(x, u)
            k2 = f(x + dt / 2 * k1, u)
            k3 = f(x + dt / 2 * k2, u)
            k4 = f(x + dt * k3, u)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        nx, nu = dynamics.dim_states, dynamics.dim_inputs
        opti = ca.Opti()
        X = opti.variable(nx, self.N + 1)
        U = opti.variable(nu, self.N)
        x0 = opti.parameter(nx)
        cost = 0
        for i in range(self.N):
            opti.subject_to(X[:, i + 1] == rk4(X[:, i], U[:, i]))
            dx = X[:, i] - self.target_state
            du = U[:, i] - self.u_eq
            cost += ca.mtimes([dx.T, ca.DM(Q), dx]) + ca.mtimes([du.T, ca.DM(R), du])
            if env.input_lbs is not None:
                opti.subject_to(U[:, i] >= env.input_lbs)
            if env.input_ubs is not None:
                opti.subject_to(U[:, i] <= env.input_ubs)
            if env.state_lbs is not None:
                opti.subject_to(X[:, i + 1] >= ca.DM(np.asarray(env.state_lbs, dtype=float)))
            if env.state_ubs is not None:
                opti.subject_to(X[:, i + 1] <= ca.DM(np.asarray(env.state_ubs, dtype=float)))
        dx = X[:, self.N] - self.target_state
        cost += ca.mtimes([dx.T, ca.DM(Qf), dx])
        opti.subject_to(X[:, 0] == x0)
        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {"print_time": False},
            {"print_level": 0, "max_iter": 500, "sb": "yes"},
        )
        self._opti, self._X, self._U, self._x0 = opti, X, U, x0
        self._previous = None

    def setup(self) -> None:
        pass

    def compute_action(self, current_state: np.ndarray, current_time=None):
        """Return (u_0, x_prediction, u_prediction) like the MPC controllers."""
        self._opti.set_value(self._x0, np.asarray(current_state, dtype=float))
        if self._previous is not None:
            self._opti.set_initial(self._X, self._previous[0])
            self._opti.set_initial(self._U, self._previous[1])
        try:
            sol = self._opti.solve()
        except RuntimeError as exc:
            raise RuntimeError(f"NMPC oracle solve failed: {exc}") from exc
        X = np.array(sol.value(self._X)).reshape(self.dynamics.dim_states, self.N + 1)
        U = np.array(sol.value(self._U)).reshape(self.dynamics.dim_inputs, self.N)
        # Warm start for the next step: shift the plan by one sample.
        self._previous = (
            np.hstack([X[:, 1:], X[:, -1:]]),
            np.hstack([U[:, 1:], U[:, -1:]]),
        )
        return U[:, 0].copy(), X.T.copy(), U.T.copy()


class DeePCController:
    """
    Minimal deterministic / regularized DeePC controller.

    Deterministic defaults:
    - hard matching of the past output trajectory,
    - explicit matching of the current measured output,
    - OSQP QP in the single DeePC coefficient g.

    Optional arguments are intentionally small and support the tutorial demos:
    - output_indices: which state coordinates are measured. The Chapter 6.4
      notebook uses position-only output, output_indices=[0] (y_data with one
      column and scalar Q, Qf). Without it the full state is the output;
    - enforce_current_output=False: textbook output-feedback DeePC where the
      future first output is determined only by past I/O;
    - lambda_y>0: soften output-history matching for noisy/nonlinear cases;
    - enforce_pe_check=False: build the controller even if the offline input is
      not persistently exciting (for demonstrations of what then goes wrong).
    """

    def __init__(
            self,
            env: Env,
            dynamics: Dynamics,
            u_data: np.ndarray,
            y_data: np.ndarray,
            Q: np.ndarray,
            R: np.ndarray,
            Qf: np.ndarray,
            freq: float,
            N: int,
            T_ini: int = 4,
            lambda_g: float = 1e-6,
            history_initialization: str = 'equilibrium',
            output_indices=None,
            enforce_current_output: bool = True,
            lambda_y: float = None,
            velocity_bounds: tuple = None,
            name: str = 'DeePC',
            type: str = 'MPC',
            verbose: bool = False,
            enforce_pe_check: bool = True,
        ) -> None:

        self.env = env
        self.enforce_pe_check = enforce_pe_check
        self.dynamics = dynamics
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.Qf = np.asarray(Qf, dtype=float)

        self.freq = freq
        self.dt = 1.0 / freq
        self.N = N
        self.T_ini = T_ini
        self.lambda_g = lambda_g
        self.lambda_y = lambda_y
        self.history_initialization = history_initialization
        self.enforce_current_output = enforce_current_output
        # Optional bounds on the finite-difference velocity (y_{i+1} - y_i) / dt of a
        # single (position) output, so that a speed limit can be imposed although the
        # velocity itself is never measured.
        self.velocity_bounds = velocity_bounds

        self.name = name
        self.type = type
        self.verbose = verbose

        self.dim_states = dynamics.dim_states
        self.dim_inputs = dynamics.dim_inputs

        self.target_state = np.asarray(env.target_state, dtype=float)
        self.equilibrium_input = np.atleast_1d(
            dynamics.get_equilibrium_input(self.target_state)
        ).astype(float)

        self.u_data = _as_time_major(u_data)
        self.y_data = _as_time_major(y_data)
        self.dim_outputs = self.y_data.shape[1]

        if output_indices is None:
            if self.dim_outputs != self.dim_states:
                raise ValueError(
                    "When y_data is not full-state, provide output_indices "
                    "(e.g. output_indices=[0] for position-only output)."
                )
            self.output_indices = np.arange(self.dim_states)
        else:
            self.output_indices = np.asarray(output_indices, dtype=int).reshape(-1)
            if len(self.output_indices) != self.dim_outputs:
                raise ValueError("output_indices must match the number of y_data columns.")

        self.target_output = self.target_state[self.output_indices]

        if self.Q.shape != (self.dim_outputs, self.dim_outputs):
            raise ValueError("Q must match the output dimension used by DeePC.")
        if self.Qf.shape != (self.dim_outputs, self.dim_outputs):
            raise ValueError("Qf must match the output dimension used by DeePC.")
        if self.R.shape != (self.dim_inputs, self.dim_inputs):
            raise ValueError("R must match the input dimension.")

        if self.y_data.shape[0] != self.u_data.shape[0] + 1:
            raise ValueError(
                "For DeePC, y_data must contain one more sample than u_data "
                "(y_0, ..., y_T versus u_0, ..., u_{T-1})."
            )

        self.u_history = None
        self.y_history = None
        self.last_output = None
        self.last_input = None
        self.initialized = False

        self.Up = None
        self.Yp = None
        self.Uf = None
        self.Yf = None
        self.solver = None

        self.setup()

    def _measure_output(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1)
        return state[self.output_indices]

    def initialize_history(
            self,
            current_state: np.ndarray,
            mode: str = None,
            u_history: np.ndarray = None,
            y_history: np.ndarray = None
        ) -> None:
        """
        Initialize the past trajectory used by DeePC in one place.

        Default synthetic initialization:
        - mode='equilibrium': repeat current output and equilibrium input;
        - mode='zero': repeat current output and zero input.

        For experiments or a real system with measured history, pass both
        `u_history` and `y_history`; these must each contain T_ini samples.
        """
        current_output = self._measure_output(current_state)

        if (u_history is None) != (y_history is None):
            raise ValueError("Provide both u_history and y_history, or neither.")

        if u_history is not None:
            u_history = _as_time_major(u_history)
            y_history = _as_time_major(y_history)
            if u_history.shape != (self.T_ini, self.dim_inputs):
                raise ValueError("u_history must have shape (T_ini, dim_inputs).")
            if y_history.shape != (self.T_ini, self.dim_outputs):
                raise ValueError("y_history must have shape (T_ini, dim_outputs).")
            self.u_history = u_history.copy()
            self.y_history = y_history.copy()
        else:
            mode = self.history_initialization if mode is None else mode
            if mode == 'equilibrium':
                u_init = self.equilibrium_input
            elif mode == 'zero':
                u_init = np.zeros(self.dim_inputs)
            else:
                raise ValueError("history initialization must be 'equilibrium' or 'zero'.")

            self.u_history = np.tile(u_init, (self.T_ini, 1))
            self.y_history = np.tile(current_output, (self.T_ini, 1))

        self.last_output = current_output.copy()
        self.last_input = None
        self.initialized = True

    def setup(self) -> None:
        """Build the Hankel matrices and the constant part of the OSQP problem."""
        nu = self.dim_inputs
        ny = self.dim_outputs

        # Work in deviation coordinates around the target equilibrium/output.
        u_dev = self.u_data - self.equilibrium_input
        y_dev = self.y_data - self.target_output

        # All matrices share the same trajectory columns. y_data has one extra
        # sample, which lets Yf contain y_0, ..., y_N like the MPC prediction.
        n_cols = self.u_data.shape[0] - self.T_ini - self.N + 1
        if n_cols <= 0:
            raise ValueError("Offline dataset is too short for T_ini + N.")

        Hu = _block_hankel(u_dev, self.T_ini + self.N)[:, :n_cols]
        Hy = _block_hankel(y_dev, self.T_ini + self.N + 1)[:, :n_cols]

        self.Up = Hu[:self.T_ini * nu]
        self.Uf = Hu[self.T_ini * nu:]
        self.Yp = Hy[:self.T_ini * ny]
        self.Yf = Hy[self.T_ini * ny:]

        # Two rank conditions on the offline data, both reported for the notebooks.
        #
        # (a) Persistency of excitation of order T_ini + N + n_x (Willems et al., 2005):
        #     rank H_{T_ini+N+n_x}(u^d) = n_u (T_ini + N + n_x), i.e. full row rank. A condition
        #     on the input alone, sufficient for the Hankel columns to span the behaviour.
        pe_order = self.T_ini + self.N + self.dim_states
        H_pe = _block_hankel(u_dev, pe_order)
        pe_rank = np.linalg.matrix_rank(H_pe)
        self.pe_rank, self.pe_required_rank = int(pe_rank), int(H_pe.shape[0])
        self.pe_columns = int(H_pe.shape[1])

        # (b) Generalized persistency of excitation (Markovsky and Doerfler, 2023):
        #     rank [H_{T_ini+N}(u^d); H_{T_ini+N}(y^d)] = n_u (T_ini + N) + n_x. A condition on the
        #     measured input-output data, necessary AND sufficient for the same span. Every column
        #     of the stacked Hankel is a trajectory of the system, so the rank can never exceed
        #     this value; it only has to reach it.
        H_gpe = np.vstack([self.Up, self.Uf, self.Yp, self.Yf])
        self.gpe_rank = int(np.linalg.matrix_rank(H_gpe))
        self.gpe_required_rank = int(nu * (self.T_ini + self.N) + self.dim_states)
        self.gpe_columns = int(H_gpe.shape[1])
        if pe_rank < H_pe.shape[0] and self.enforce_pe_check:
            raise ValueError(
                f"Offline input is not persistently exciting enough: "
                f"rank(H_{pe_order})={pe_rank}, expected {H_pe.shape[0]}."
            )

        # Cost on y_0 ... y_N and u_0 ... u_{N-1}.
        Q_bar = sparse.block_diag(
            [self.Q] * self.N + [self.Qf],
            format='csc'
        )
        R_bar = sparse.block_diag(
            [self.R] * self.N,
            format='csc'
        )

        H = (
            self.Yf.T @ Q_bar @ self.Yf
            + self.Uf.T @ R_bar @ self.Uf
            + self.lambda_g * np.eye(n_cols)
        )

        # In robust/noisy mode, output-history matching is a quadratic penalty
        # instead of a hard equality.  Optionally include the current output.
        self._Y_history_fit = self.Yp
        if self.enforce_current_output:
            self._Y_history_fit = np.vstack([self.Yp, self.Yf[:ny]])

        if self.lambda_y is not None and self.lambda_y > 0.0:
            H = H + self.lambda_y * (self._Y_history_fit.T @ self._Y_history_fit)

        # OSQP solves 1/2 g' P g + q' g.
        P = sparse.csc_matrix(np.triu(2.0 * np.asarray(H)))
        q = np.zeros(n_cols)

        # Hard equalities.  Input history always remains exact.  Output history
        # is either hard (deterministic mode) or moved to the cost (robust mode).
        hard_blocks = [self.Up]
        self._hard_output_history = not (self.lambda_y is not None and self.lambda_y > 0.0)
        if self._hard_output_history:
            hard_blocks.append(self.Yp)
            if self.enforce_current_output:
                hard_blocks.append(self.Yf[:ny])

        n_eq = sum(block.shape[0] for block in hard_blocks)

        # Future input/output constraints are unchanged between the two modes.
        vel_blocks = []
        if self.velocity_bounds is not None:
            if ny != 1:
                raise ValueError("velocity_bounds requires a single (position) output.")
            D = (np.eye(self.N + 1, k=1) - np.eye(self.N + 1))[:-1] / self.dt   # (N, N+1) forward differences
            vel_blocks = [D @ self.Yf]
        A = sparse.csc_matrix(np.vstack(hard_blocks + [self.Uf, self.Yf] + vel_blocks))
        l = np.zeros(A.shape[0])
        u = np.zeros(A.shape[0])

        # Future input bounds in deviation coordinates.
        idx = n_eq
        if self.env.input_lbs is None:
            u_lb = np.full(nu, -np.inf)
        else:
            u_lb = np.broadcast_to(np.asarray(self.env.input_lbs, dtype=float), (nu,))
        if self.env.input_ubs is None:
            u_ub = np.full(nu, np.inf)
        else:
            u_ub = np.broadcast_to(np.asarray(self.env.input_ubs, dtype=float), (nu,))

        l[idx:idx + self.N * nu] = np.tile(u_lb - self.equilibrium_input, self.N)
        u[idx:idx + self.N * nu] = np.tile(u_ub - self.equilibrium_input, self.N)
        idx += self.N * nu

        # Output bounds. For full-state output these are exactly state bounds;
        # for partial outputs select only the measured/output coordinates.
        if self.env.state_lbs is None:
            y_lb = np.full(ny, -np.inf)
        else:
            y_lb = np.asarray(self.env.state_lbs, dtype=float)[self.output_indices]
        if self.env.state_ubs is None:
            y_ub = np.full(ny, np.inf)
        else:
            y_ub = np.asarray(self.env.state_ubs, dtype=float)[self.output_indices]

        l[idx:idx + (self.N + 1) * ny] = np.tile(y_lb - self.target_output, self.N + 1)
        u[idx:idx + (self.N + 1) * ny] = np.tile(y_ub - self.target_output, self.N + 1)
        idx += (self.N + 1) * ny

        # Velocity bounds on the output differences (offsets cancel in the difference).
        if vel_blocks:
            l[idx:] = self.velocity_bounds[0]
            u[idx:] = self.velocity_bounds[1]

        self._l_template = l
        self._u_template = u
        self._n_eq = n_eq
        self._q_template = q

        self.solver = osqp.OSQP()
        self.solver.setup(
            P=P,
            q=q,
            A=A,
            l=l,
            u=u,
            verbose=self.verbose,
            warm_start=True,
            polish=True,
        )

        if self.verbose:
            mode = "soft output history" if not self._hard_output_history else "hard output history"
            print(
                f"DeePC setup completed: {n_cols} Hankel columns, "
                f"PE rank {pe_rank}/{H_pe.shape[0]}, {mode}."
            )

    @check_input_constraints
    def compute_action(
            self,
            current_state: np.ndarray,
            current_time: float = None
        ):
        """Solve the DeePC QP and return (u_0, y_prediction, u_prediction)."""
        current_state = np.asarray(current_state, dtype=float).reshape(-1)
        current_output = self._measure_output(current_state)

        if not self.initialized:
            self.initialize_history(current_state)
        elif self.last_input is not None:
            # Append the previous measured output/input pair to the past window.
            self.u_history = np.vstack([self.u_history[1:], self.last_input])
            self.y_history = np.vstack([self.y_history[1:], self.last_output])

        u_ini = (self.u_history - self.equilibrium_input).reshape(-1)
        y_ini = (self.y_history - self.target_output).reshape(-1)
        y_current = current_output - self.target_output

        l = self._l_template.copy()
        u = self._u_template.copy()
        q = self._q_template.copy()

        if self._hard_output_history:
            eq_parts = [u_ini, y_ini]
            if self.enforce_current_output:
                eq_parts.append(y_current)
            b_eq = np.concatenate(eq_parts)
        else:
            # Only past input is hard; measured outputs enter the soft cost.
            b_eq = u_ini
            y_fit = y_ini
            if self.enforce_current_output:
                y_fit = np.concatenate([y_fit, y_current])
            q = q - 2.0 * self.lambda_y * (self._Y_history_fit.T @ y_fit)

        l[:self._n_eq] = b_eq
        u[:self._n_eq] = b_eq
        self.solver.update(q=q, l=l, u=u)

        result = self.solver.solve()
        if result.info.status_val not in (1, 2):
            raise RuntimeError(f"DeePC OSQP solve failed: {result.info.status}")

        g_opt = result.x
        u_pred = (self.Uf @ g_opt).reshape(self.N, self.dim_inputs)
        y_pred = (self.Yf @ g_opt).reshape(self.N + 1, self.dim_outputs)

        # Diagnostics for the tutorial notebooks: the behavioral coefficient,
        # the pure OSQP solve time, and the output-history fit residual
        # ||Y_fit g - y_meas|| (identically ~0 in the hard-constrained mode).
        self.last_g = g_opt.copy()
        self.last_solve_time = float(result.info.solve_time)
        self.last_run_time = float(result.info.run_time)
        y_fit_meas = y_ini if not self.enforce_current_output else np.concatenate([y_ini, y_current])
        self.last_output_residual = float(np.linalg.norm(self._Y_history_fit @ g_opt - y_fit_meas))

        # Transform predictions back to physical coordinates.
        u_pred = u_pred + self.equilibrium_input
        y_pred = y_pred + self.target_output
        # Round-off can leave the first input a hair outside the bounds it satisfies
        # in the QP; clip it so that the simulator does not warn about it.
        if self.env.input_lbs is not None:
            u_pred = np.maximum(u_pred, np.asarray(self.env.input_lbs, dtype=float))
        if self.env.input_ubs is not None:
            u_pred = np.minimum(u_pred, np.asarray(self.env.input_ubs, dtype=float))
        u_optimal = u_pred[0].copy()

        # Store the output/input pair for the next receding-horizon update.
        self.last_output = current_output.copy()
        self.last_input = u_optimal.copy()

        if self.verbose:
            print(f"Optimal control action: {u_optimal}")
            print(f"y_pred: {y_pred}")
            print(f"u_pred: {u_pred}")

        return u_optimal, y_pred, u_pred


"""------Model-based predictive control in ARX form (oracle / indirect baseline)------"""


def true_arx_double_integrator(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold ARX model of the flat Mountain Car, p_ddot = u.

        p_{k+1} = 2 p_k - p_{k-1} + (dt^2 / 2) (u_k + u_{k-1})

    Returns (a, b) with a = [a_1, a_2], b = [b_1, b_2] in
        y_{k+1} = a_1 y_k + a_2 y_{k-1} + b_1 u_k + b_2 u_{k-1}.
    """
    return np.array([2.0, -1.0]), np.array([0.5 * dt**2, 0.5 * dt**2])


def identify_arx_least_squares(
        u_data: np.ndarray,
        y_data: np.ndarray,
        order: int = 2,
        input_offset: np.ndarray = None,
        output_offset: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
    """Ordinary least squares fit of a SISO ARX(order, order) model.

        y_{k+1} = sum_i a_i y_{k+1-i} + sum_i b_i u_{k+1-i},   i = 1..order

    in deviation coordinates.  This is the textbook *indirect* route: first a
    model, then a controller.  With output noise the regressors are noisy too
    (errors in variables), so the estimate is biased; nothing is done about
    that here on purpose.

    Returns (a, b, one_step_rms_residual).
    """
    u = _as_time_major(u_data)[:, 0]
    y = _as_time_major(y_data)[:, 0]
    if input_offset is not None:
        u = u - float(np.asarray(input_offset).reshape(-1)[0])
    if output_offset is not None:
        y = y - float(np.asarray(output_offset).reshape(-1)[0])
    T = u.shape[0]
    y = y[:T + 1] if y.shape[0] >= T + 1 else y

    rows, targets = [], []
    for k in range(order - 1, min(T, y.shape[0] - 1)):
        rows.append(np.concatenate([[y[k - i] for i in range(order)], [u[k - i] for i in range(order)]]))
        targets.append(y[k + 1])
    Phi = np.asarray(rows)
    Y = np.asarray(targets)
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    residual = float(np.sqrt(np.mean((Phi @ theta - Y)**2)))
    return theta[:order], theta[order:], residual


class ARXPredictiveController:
    """
    Linear predictive controller built from an ARX model, on the same OSQP
    backend, with the same cost, constraints and receding history as the
    DeePC controller.  Used as
      - the *oracle* (true ARX model of the flat plant),
      - the *indirect* data-driven controller (ARX identified from data),
      - a fair computational baseline (decision variable u_f, dimension N).

    The predictor is exact in deviation coordinates:
        y_f = M_y y_hist + M_u u_hist + Gamma u_f,
    where the history holds the last `order` outputs (including the current
    one) and the last `order - 1` inputs.  Output = position only.
    """

    def __init__(
            self,
            env: Env,
            dynamics: Dynamics,
            a: np.ndarray,
            b: np.ndarray,
            Q: np.ndarray,
            R: np.ndarray,
            Qf: np.ndarray,
            freq: float,
            N: int,
            output_indices=(0,),
            velocity_bounds: tuple = None,
            name: str = 'ARX-MPC',
            type: str = 'MPC',
            verbose: bool = False,
            tolerance: float = 1e-6,
        ) -> None:
        self.env = env
        self.dynamics = dynamics
        self.a = np.asarray(a, dtype=float).reshape(-1)
        self.b = np.asarray(b, dtype=float).reshape(-1)
        self.order = self.a.shape[0]
        if self.b.shape[0] != self.order:
            raise ValueError("a and b must have the same length (ARX order).")
        self.Q = np.asarray(Q, dtype=float).reshape(1, 1)
        self.R = np.asarray(R, dtype=float).reshape(1, 1)
        self.Qf = np.asarray(Qf, dtype=float).reshape(1, 1)
        self.freq = freq
        self.dt = 1.0 / freq
        self.N = int(N)
        self.name, self.type, self.verbose = name, type, verbose
        self.tolerance = float(tolerance)
        self.velocity_bounds = velocity_bounds   # bounds on (y_{i+1} - y_i) / dt, see DeePCController
        self.output_indices = np.asarray(output_indices, dtype=int).reshape(-1)
        self.dim_inputs = dynamics.dim_inputs
        self.target_state = np.asarray(env.target_state, dtype=float)
        self.target_output = float(self.target_state[self.output_indices][0])
        self.equilibrium_input = float(np.atleast_1d(dynamics.get_equilibrium_input(self.target_state))[0])

        self.y_history = None   # last `order` outputs (deviation), newest last
        self.u_history = None   # last `order - 1` inputs (deviation), newest last
        self.last_input = None
        self.initialized = False
        self.last_solve_time = None
        self.last_run_time = None
        self.setup()

    # -- lifted predictor -------------------------------------------------
    def _state_space(self):
        """Observable realization with state s_k = [y_k, ..., y_{k-n+1}, u_{k-1}, ..., u_{k-n+1}]."""
        n = self.order
        ns = n + (n - 1)
        A = np.zeros((ns, ns)); B = np.zeros((ns, 1))
        A[0, :n] = self.a
        A[0, n:] = self.b[1:]
        B[0, 0] = self.b[0]
        for i in range(1, n):
            A[i, i - 1] = 1.0
        if n > 1:
            B[n, 0] = 1.0
            for i in range(n + 1, ns):
                A[i, i - 1] = 1.0
        C = np.zeros((1, ns)); C[0, 0] = 1.0
        return A, B, C

    def setup(self) -> None:
        A, B, C = self._state_space()
        ns = A.shape[0]
        N = self.N
        # y_{1..N} = Phi s_0 + Gamma u_{0..N-1}
        Phi = np.zeros((N, ns)); Gamma = np.zeros((N, N))
        Ak = np.eye(ns)
        powers = [Ak]
        for k in range(1, N + 1):
            Ak = A @ Ak
            powers.append(Ak)
        for i in range(1, N + 1):
            Phi[i - 1] = (C @ powers[i]).ravel()
            for j in range(i):
                Gamma[i - 1, j] = float(C @ powers[i - 1 - j] @ B)
        self.Phi, self.Gamma = Phi, Gamma
        q_diag = np.array([self.Q[0, 0]] * (N - 1) + [self.Qf[0, 0]])
        self._Qbar = np.diag(q_diag)
        H = Gamma.T @ self._Qbar @ Gamma + self.R[0, 0] * np.eye(N)
        self._P = sparse.csc_matrix(np.triu(2.0 * H))
        # constraints: u bounds (identity rows), y bounds (Gamma rows) and, optionally,
        # velocity bounds on the differences of [y_0, y_1, ..., y_N]
        blocks = [np.eye(N), Gamma]
        if self.velocity_bounds is not None:
            D_full = (np.eye(N + 1, k=1) - np.eye(N + 1))[:-1] / self.dt      # (N, N+1)
            self._d0v, self._Dv = D_full[:, 0], D_full[:, 1:]                 # y_0 column, y_1..y_N block
            blocks.append(self._Dv @ Gamma)
        self._A = sparse.csc_matrix(np.vstack(blocks))
        u_lb = -np.inf if self.env.input_lbs is None else float(np.broadcast_to(self.env.input_lbs, (1,))[0])
        u_ub = np.inf if self.env.input_ubs is None else float(np.broadcast_to(self.env.input_ubs, (1,))[0])
        y_lb = -np.inf if self.env.state_lbs is None else float(np.asarray(self.env.state_lbs)[self.output_indices][0])
        y_ub = np.inf if self.env.state_ubs is None else float(np.asarray(self.env.state_ubs)[self.output_indices][0])
        self._l = np.concatenate([np.full(N, u_lb - self.equilibrium_input), np.full(N, y_lb - self.target_output)])
        self._u = np.concatenate([np.full(N, u_ub - self.equilibrium_input), np.full(N, y_ub - self.target_output)])
        if self.velocity_bounds is not None:
            self._l = np.concatenate([self._l, np.full(N, float(self.velocity_bounds[0]))])
            self._u = np.concatenate([self._u, np.full(N, float(self.velocity_bounds[1]))])
        self.solver = osqp.OSQP()
        self.solver.setup(P=self._P, q=np.zeros(N), A=self._A, l=self._l, u=self._u,
                          verbose=self.verbose, warm_start=True, polish=True,
                          eps_abs=self.tolerance, eps_rel=self.tolerance)

    # -- history handling (mirrors DeePCController) ---------------------------
    def _measure_output(self, state) -> float:
        return float(np.asarray(state, dtype=float).reshape(-1)[self.output_indices][0])

    def initialize_history(self, current_state, u_history=None, y_history=None) -> None:
        y0 = self._measure_output(current_state) - self.target_output
        n = self.order
        if y_history is not None:
            y_hist = np.asarray(y_history, dtype=float).reshape(-1) - self.target_output
            u_hist = np.asarray(u_history, dtype=float).reshape(-1) - self.equilibrium_input
            self.y_history = np.concatenate([y_hist, [y0]])[-n:]
            self.u_history = u_hist[-(n - 1):] if n > 1 else np.zeros(0)
        else:
            self.y_history = np.full(n, y0)
            self.u_history = np.zeros(n - 1)
        self.last_input = None
        self.initialized = True

    @check_input_constraints
    def compute_action(self, current_state, current_time=None):
        y0 = self._measure_output(current_state) - self.target_output
        if not self.initialized:
            self.initialize_history(current_state)
        elif self.last_input is not None:
            self.y_history = np.concatenate([self.y_history[1:], [y0]])
            if self.order > 1:
                self.u_history = np.concatenate([self.u_history[1:], [self.last_input - self.equilibrium_input]])
        s0 = np.concatenate([self.y_history[::-1], self.u_history[::-1]])   # newest first
        y_free = self.Phi @ s0
        q = 2.0 * (self.Gamma.T @ self._Qbar @ y_free)
        N = self.N
        l = self._l.copy(); u = self._u.copy()
        l[N:2 * N] -= y_free; u[N:2 * N] -= y_free
        if self.velocity_bounds is not None:
            shift = self._d0v * y0 + self._Dv @ y_free
            l[2 * N:] -= shift; u[2 * N:] -= shift
        self.solver.update(q=q, l=l, u=u)
        result = self.solver.solve()
        if result.info.status_val not in (1, 2):
            raise RuntimeError(f"ARX-MPC OSQP solve failed: {result.info.status}")
        self.last_solve_time = float(result.info.solve_time)
        self.last_run_time = float(result.info.run_time)
        u_f = result.x
        y_pred = np.concatenate([[y0], y_free + self.Gamma @ u_f]) + self.target_output
        u_pred = (u_f + self.equilibrium_input).reshape(N, 1)
        u_optimal = np.array([u_pred[0, 0]])
        self.last_input = float(u_optimal[0])
        return u_optimal, y_pred.reshape(N + 1, 1), u_pred
