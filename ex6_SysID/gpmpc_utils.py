"""Stochastic MPC with a learned probabilistic terrain model (Chapters 6.3 and 6.4)."""
import numpy as np
import casadi as ca
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from ex2_LQR.lqr_utils import LQRController


class GPMPCController(LQRController):
    '''
    Stochastic MPC using a learned probabilistic model.

    At each step the controller
      1. rolls out the predicted mean trajectory using the learned mean dynamics,
      2. propagates the state covariance along that trajectory,
             Sigma_{i+1} = A_i Sigma_i A_i^T + Sigma^w(mu_i, u_i),
         where A_i is the discrete-time Jacobian of the learned dynamics, and
      3. tightens the state constraints by beta standard deviations,
             lb + beta*sqrt(diag(Sigma_i))  <=  x_i  <=  ub - beta*sqrt(diag(Sigma_i)),
         which is the deterministic reformulation of a chance constraint under a
         Gaussian assumption (beta = 2 corresponds to roughly 95% per constraint).

    Setting beta = 0 recovers a certainty-equivalent MPC that uses the learned mean
    and ignores its uncertainty.
    '''

    def __init__(self, env, dynamics, Q, R, Qf, N, freq, beta=2.0,
                 Sigma_w=None, name='GPMPC', type='MPC', verbose=False):
        self.Qf = Qf
        self.N = N
        self.beta = beta
        super().__init__(env, dynamics, Q, R, freq, name, type, verbose)
        if self.env.dh_cov is None:
            raise ValueError("GP-MPC needs env.dh_cov, the variance of the terrain slope.")
        self.dynamics.build_stochastic_model(self.dt)
        self.Sigma_w = Sigma_w if Sigma_w is not None else np.zeros((self.dim_states, self.dim_states))
        self.Sigma_x_log = []
        self.n_saturated = 0
        self.K_ancillary = self._ancillary_gain()

    def setup(self):
        model = AcadosModel()
        model.name = self.name
        model.x = self.dynamics.states
        model.u = self.dynamics.inputs
        model.f_expl_expr = ca.vertcat(self.dynamics.dynamics_function(self.dynamics.states, self.dynamics.inputs))
        model.f_impl_expr = None

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.N * self.dt
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 100

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = np.block([[self.Q, np.zeros((self.dim_states, self.dim_inputs))],
                               [np.zeros((self.dim_inputs, self.dim_states)), self.R]])
        ocp.cost.W_e = self.Qf
        ocp.cost.Vx = np.block([[np.eye(self.dim_states)], [np.zeros((self.dim_inputs, self.dim_states))]])
        ocp.cost.Vu = np.block([[np.zeros((self.dim_states, self.dim_inputs))], [np.eye(self.dim_inputs)]])
        ocp.cost.Vx_e = np.eye(self.dim_states)
        ocp.cost.yref = np.zeros(self.dim_states + self.dim_inputs)
        ocp.cost.yref_e = np.zeros(self.dim_states)

        # State constraints are softened so that an over-tightened problem stays
        # feasible and reports a violation instead of failing to solve.
        ocp.constraints.idxbx_0 = np.arange(self.dim_states)
        ocp.constraints.idxbx = np.arange(self.dim_states)
        ocp.constraints.idxbx_e = np.arange(self.dim_states)
        ocp.constraints.idxbu = np.arange(self.dim_inputs)
        ocp.constraints.lbx_0 = np.array(self.env.state_lbs)
        ocp.constraints.ubx_0 = np.array(self.env.state_ubs)
        ocp.constraints.lbx = np.array(self.env.state_lbs)
        ocp.constraints.ubx = np.array(self.env.state_ubs)
        ocp.constraints.lbx_e = np.array(self.env.state_lbs)
        ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        ocp.constraints.lbu = np.array(self.env.input_lbs)
        ocp.constraints.ubu = np.array(self.env.input_ubs)

        ocp.constraints.idxsbx = np.arange(self.dim_states)
        ocp.constraints.idxsbx_e = np.arange(self.dim_states)
        ns, ns_e = self.dim_states, self.dim_states
        ocp.cost.Zl = 1e4 * np.ones(ns);   ocp.cost.Zu = 1e4 * np.ones(ns)
        ocp.cost.zl = 1e2 * np.ones(ns);   ocp.cost.zu = 1e2 * np.ones(ns)
        ocp.cost.Zl_e = 1e4 * np.ones(ns_e); ocp.cost.Zu_e = 1e4 * np.ones(ns_e)
        ocp.cost.zl_e = 1e2 * np.ones(ns_e); ocp.cost.zu_e = 1e2 * np.ones(ns_e)

        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file=f"{self.name}.json", verbose=False)

    def _ancillary_gain(self):
        '''
        LQR gain used only inside the uncertainty propagation.

        The open-loop dynamics of the car are unstable on the bumps: linearizing on a
        crest gives eigenvalues of roughly +-5.6, so over a 2 s horizon an open-loop
        propagation A_i Sigma_i A_i^T amplifies the covariance by four orders of
        magnitude and the predicted uncertainty is meaningless. The predicted state is
        not actually open loop, however, since the controller will keep acting. We
        therefore propagate under an ancillary feedback u = K x, which is the standard
        device in tube- and covariance-based MPC, and keeps Sigma bounded.
        '''
        A_d, B_d = self.dynamics.get_linearized_AB_discrete(self.env.target_state,
                                                            np.zeros(self.dim_inputs), self.dt)
        P = scipy.linalg.solve_discrete_are(A_d, B_d, self.Q, self.R)
        return -np.linalg.solve(self.R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)

    def propagate_uncertainty(self, mu_x_seq, mu_u_seq):
        '''Sigma_{i+1} = (A_i + B_i K) Sigma_i (A_i + B_i K)^T + Sigma^w(mu_i, u_i).'''
        Sigma_seq = [np.zeros((self.dim_states, self.dim_states))]
        for i in range(self.N):
            A_d, B_d = self.dynamics.get_linearized_AB_discrete(mu_x_seq[i], mu_u_seq[i], self.dt)
            A_cl = A_d + B_d @ self.K_ancillary
            Sigma_w_i = np.array(self.dynamics.dynamics_variance_function_disc(mu_x_seq[i], mu_u_seq[i]))
            Sigma_seq.append(A_cl @ Sigma_seq[i] @ A_cl.T + Sigma_w_i + self.Sigma_w)
        return Sigma_seq

    # The tightened box is never allowed to shrink below this fraction of the original
    # width. Saturating means the requested confidence level demands more margin than
    # the constraint set contains, which is a modelling outcome worth knowing about,
    # so occurrences are counted rather than silently clipped.
    MIN_WIDTH_FRACTION = 0.2

    def tighten_state_constraints(self, Sigma_x):
        half_width = 0.5 * (self.env.state_ubs - self.env.state_lbs)
        margin = self.beta * np.sqrt(np.maximum(np.diag(Sigma_x), 0.0))
        max_margin = (1.0 - self.MIN_WIDTH_FRACTION) * half_width
        self.n_saturated += int(np.any(margin > max_margin))
        margin = np.minimum(margin, max_margin)
        return self.env.state_lbs + margin, self.env.state_ubs - margin

    def compute_action(self, current_state, current_time):
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)

        # Linearization points for the covariance propagation. The previous solution,
        # shifted by one step, is a converged and feasible trajectory, so it is a far
        # better reference than re-integrating the learned dynamics under a stale input
        # sequence: the open-loop mountain-car dynamics are unstable on the bumps, and
        # an inaccurate rollout makes A_i Sigma_i A_i^T diverge over a 2 s horizon.
        if hasattr(self, "last_x_pred"):
            mu_x_seq = [np.asarray(current_state, dtype=float)] + \
                       [self.last_x_pred[min(i + 1, self.N)] for i in range(1, self.N + 1)]
            mu_u_seq = [self.last_u_pred[min(i + 1, self.N - 1)] for i in range(self.N)]
        else:
            mu_x_seq, mu_u_seq = [np.asarray(current_state, dtype=float)], []
            for i in range(self.N):
                mu_u = np.zeros(self.dim_inputs)
                mu_x_seq.append(self.dynamics.one_step_forward(mu_x_seq[-1], mu_u, self.dt))
                mu_u_seq.append(mu_u)

        Sigma_seq = self.propagate_uncertainty(mu_x_seq, mu_u_seq)
        self.Sigma_x_log.append(np.array([np.diag(S) for S in Sigma_seq]))

        ref = np.concatenate((self.env.target_state, np.zeros(self.dim_inputs)))
        for i in range(1, self.N):
            self.solver.set(i, "yref", ref)
            lb, ub = self.tighten_state_constraints(Sigma_seq[i])
            self.solver.set(i, "lbx", lb); self.solver.set(i, "ubx", ub)
        self.solver.set(self.N, "yref", self.env.target_state)
        lb, ub = self.tighten_state_constraints(Sigma_seq[self.N])
        self.solver.set(self.N, "lbx", lb); self.solver.set(self.N, "ubx", ub)

        self.solver.solve()
        u_opt = self.solver.get(0, "u")
        x_pred = np.array([self.solver.get(i, "x") for i in range(self.N + 1)])
        u_pred = np.array([self.solver.get(i, "u") for i in range(self.N)])
        self.last_u_pred, self.last_x_pred = u_pred.copy(), x_pred.copy()
        return u_opt, x_pred, u_pred
