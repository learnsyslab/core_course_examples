import numpy as np
import casadi as ca
import scipy.linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import List, Callable, Union, Optional, Tuple, Any
from scipy.spatial import ConvexHull, QhullError, cKDTree




class Env:
    def __init__(
            self, 
            case: int, 
            init_state: np.ndarray, 
            target_state: np.ndarray,
            symbolic_h_mean_ext_case: Callable[[ca.Function], ca.Function] = None, 
            symbolic_h_mean_ext: Callable[[ca.Function], ca.Function] = None, 
            symbolic_h_cov_ext: Callable[[ca.Function], ca.Function] = None, 
            symbolic_dh_cov_ext: Callable[[ca.Function], ca.Function] = None,
            symbolic_theta_ext: Callable[[ca.Function], ca.Function] = None,
            param: float = None,
            state_lbs: np.ndarray = None, 
            state_ubs: np.ndarray = None, 
            input_lbs: float = None, 
            input_ubs: float = None,
            disturbance_lbs: np.ndarray = None, 
            disturbance_ubs: np.ndarray = None
        ) -> None:

        self.case = case  # 1, 2, 3, 4
        self.param = param  # terrain parameter, used in case 2, 3, 4

        self.initial_position = init_state[0]
        self.initial_velocity = init_state[1]
        self.init_state = np.array([self.initial_position, self.initial_velocity])

        self.target_position = target_state[0]
        self.target_velocity = target_state[1]
        self.target_state = np.array([self.target_position, self.target_velocity])

        self.state_lbs = state_lbs
        if state_lbs is not None:
            self.pos_lbs = state_lbs[0]
            self.vel_lbs = state_lbs[1]
        self.state_ubs = state_ubs
        if state_ubs is not None:
            self.pos_ubs = state_ubs[0]
            self.vel_ubs = state_ubs[1]

        self.input_lbs = input_lbs
        self.input_ubs = input_ubs

        self.disturbance_lbs = disturbance_lbs
        self.disturbance_ubs = disturbance_ubs

        # Define argument p as CasADi symbolic parameters
        p = ca.SX.sym("p") # p: horizontal displacement

        # Set functions h(p) based on symbolic parameters or given expression
        if symbolic_h_mean_ext_case:
            self.h = ca.Function("h", [p], [symbolic_h_mean_ext_case(p, case)]) # function of height h w.r.t. p
        if symbolic_h_mean_ext:
            self.h = ca.Function("h", [p], [symbolic_h_mean_ext(p)]) # function of height h w.r.t. p
        if not (symbolic_h_mean_ext or symbolic_h_mean_ext_case):
            def symbolic_h_mean_ext_case(p, case):
                if case == 1:  # zero slope
                    h = 0
                elif case == 2: # constant slope
                    param = self.param if self.param is not None else 18
                    h = (ca.pi * p) / param
                elif case == 3: # varying slope (small disturbance)
                    param = self.param if self.param is not None else 0.005
                    h = param * ca.cos(18 * p)
                elif case == 4: # varying slope (underactated case)
                    condition_left = p <= -ca.pi/2
                    condition_right = p >= ca.pi/6
                    h_center = ca.sin(3 * p)
                    h_flat = 1
                    h = ca.if_else(condition_left, h_flat, ca.if_else(condition_right, h_flat, h_center))
                return h
            self.h = ca.Function("h", [p], [symbolic_h_mean_ext_case(p, case)]) # function of height h w.r.t. p
        
        # Set functions h_cov(p) based on given expression
        if symbolic_h_cov_ext:
            self.h_cov = symbolic_h_cov_ext
        else:
            self.h_cov = None

        # Set the variance of the *slope*, Var[h'(p)], based on given expression.
        # The dynamics depend on h only through the inclination theta(p) = atan(h'(p)),
        # so this - and not the variance of h itself - is what propagates into the
        # process noise covariance. Note that Var[h'(p)] cannot be recovered from the
        # marginal variance Var[h(p)]: it is the second mixed derivative of the joint
        # covariance, d^2/dp dp' Cov[h(p), h(p')] evaluated at p' = p, and must
        # therefore be supplied by the model that produced h_cov.
        if symbolic_dh_cov_ext:
            self.dh_cov = symbolic_dh_cov_ext
        else:
            self.dh_cov = None
        
        # Set functions theta(p) based on symbolic parameters or given expression
        if not symbolic_theta_ext:
            def symbolic_theta_ext(h_func):
                h = h_func(p) 
                dh_dp = ca.jacobian(h, p)
                theta = ca.atan(dh_dp)
                return ca.Function("theta", [p], [theta])
        self.theta = symbolic_theta_ext(self.h) # function of inclination angle theta w.r.t. p
        
        # Generate grid mesh along p to visualize the curve of slope and inclination angle
        if self.state_lbs is not None:
            lbs_position = self.state_lbs[0]
        else:
            lbs_position = self.initial_position-0.2

        if self.state_ubs is not None:
            ubs_position = self.state_ubs[0]
        else:
            ubs_position = self.target_position+0.2

        self.p_vals_disp = np.linspace(lbs_position, ubs_position, 200)
    
    def show_slope(self) -> None:
        
        # Calculate values of h and theta on grid mesh
        h_vals = [float(self.h(p)) for p in self.p_vals_disp]
        theta_vals = [float(self.theta(p)) for p in self.p_vals_disp]
        
        # Calculate teh value of h for initial state and terminal state
        initial_h = float(self.h(self.initial_position).full().flatten()[0])
        target_h = float(self.h(self.target_position).full().flatten()[0])
        
        # Display curve theta(p) (left), h(p) (right)
        _, ax = plt.subplots(1, 2, figsize=(12, 3))

        # h(p)
        ax[0].plot(self.p_vals_disp, h_vals, label="h(p)", color='green')
        ax[0].set_xlabel("p")
        ax[0].set_ylabel("h")
        ax[0].set_ylim(-1.4, 1.4)  
        ax[0].set_title("h(p)")
        ax[0].legend()

        # theta(p)
        ax[1].plot(self.p_vals_disp, theta_vals, label=r"$\theta(p)$", color='blue')
        ax[1].set_xlabel("p")
        ax[1].set_ylabel(r"$\theta$")
        ax[1].set_ylim(-1.5, 1.5)  
        ax[1].set_title(r"$\theta$($p$)")
        ax[1].legend()

        plt.show()

    def test_env(self) -> None:
        
        # Calculate values of h and theta on grid mesh
        h_vals = [float(self.h(p)) for p in self.p_vals_disp]
        theta_vals = [float(self.theta(p)) for p in self.p_vals_disp]
        
        # Calculate teh value of h for initial state and terminal state
        initial_h = float(self.h(self.initial_position).full().flatten()[0])
        target_h = float(self.h(self.target_position).full().flatten()[0])
        
        # Display curve theta(p) (left), h(p) (right)
        _, ax = plt.subplots(1, 2, figsize=(12, 3))

        # h(p)
        ax[0].plot(self.p_vals_disp, h_vals, label="h(p)", color='green')
        ax[0].scatter([self.initial_position], [initial_h], color="blue", label="Start")
        #ax[0].scatter([self.target_position], [target_h], color="orange", label="Target")
        ax[0].plot(self.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')
        ax[0].set_xlabel("p")
        ax[0].set_ylabel("h")
        ax[0].set_ylim(-1.4, 1.4)  
        ax[0].set_title("h(p)")
        ax[0].legend()

        # theta(p)
        ax[1].plot(self.p_vals_disp, theta_vals, label=r"$\theta(p)$", color='blue')
        ax[1].set_xlabel("p")
        ax[1].set_ylabel(r"$\theta$")
        ax[1].set_ylim(-1.5, 1.5)  
        ax[1].set_title(r"$\theta$($p$)")
        ax[1].legend()

        plt.show()


class Dynamics:
    def __init__(
            self, 
            env: Env,
            state_names: List[str] = None,
            input_names: List[str] = None,
            setup_dynamics: Callable[[ca.Function], Callable[[ca.SX, ca.SX], ca.SX]] = None
        ) -> None:

        # Initialize system dynmaics if not given
        if state_names is None and input_names is None:
            state_names = ["p", "v"]
            input_names = ["u"]

        # Define state and input as CasADi symbolic parameters
        self.states = ca.vertcat(*[ca.SX.sym(name) for name in state_names])
        self.inputs = ca.vertcat(*[ca.SX.sym(name) for name in input_names])

        self.dim_states = self.states.shape[0]
        self.dim_inputs = self.inputs.shape[0]

        self.env = env

        # Initialize system dynmaics if not given
        if setup_dynamics is None:

            def setup_dynamics(theta_function):

                p = ca.SX.sym("p")
                v = ca.SX.sym("v")
                u = ca.SX.sym("u")
                Gravity = 9.81

                theta = theta_function(p)
        
                # Expression of dynamics
                dpdt = v
                dvdt = u * ca.cos(theta) - Gravity * ca.sin(theta) * ca.cos(theta)
                
                state = ca.vertcat(p, v)
                input = ca.vertcat(u)
                rhs = ca.vertcat(dpdt, dvdt)

                return ca.Function("dynamics_function", [state, input], [rhs])
            
        self.dynamics_function = setup_dynamics(self.env.theta)

        # Initialize Jacobians
        self.A_c_func = None
        self.B_c_func = None

    def linearization(
        self,
        current_state: np.ndarray, 
        current_input: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute symbolic Jacobians A(x,u) & B(x,u) and state / input matrix A & B of the system dynamics.
        """

        f = self.dynamics_function(self.states, self.inputs)
        A_c_sym = ca.jacobian(f, self.states)
        B_c_sym = ca.jacobian(f, self.inputs)

        self.A_c_func = ca.Function("A_func", [self.states, self.inputs], [A_c_sym])
        self.B_c_func = ca.Function("B_func", [self.states, self.inputs], [B_c_sym])

        A_c = np.array(self.A_c_func(current_state, current_input))
        B_c = np.array(self.B_c_func(current_state, current_input))

        return A_c, B_c
    
    def discretization(
        self,
        A_c: np.ndarray,
        B_c: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize continuous-time linear system x_dot = A_c x + B_c u
        using matrix exponential method (Zero-Order Hold).

        Returns:
            A_d, B_d: Discrete-time system matrices
        """

        # Construct augmented matrix
        aug_matrix = np.zeros((self.dim_states + self.dim_inputs, self.dim_states + self.dim_inputs))
        aug_matrix[:self.dim_states, :self.dim_states] = A_c
        aug_matrix[:self.dim_states, self.dim_states:] = B_c

        # Compute matrix exponential
        exp_matrix = scipy.linalg.expm(aug_matrix * dt)

        # Extract A_d and B_d
        A_d = exp_matrix[:self.dim_states, :self.dim_states]
        B_d = exp_matrix[:self.dim_states, self.dim_states:]

        return A_d, B_d

    def get_linearized_AB_discrete(
            self, 
            current_state: np.ndarray, 
            current_input: np.ndarray, 
            dt: float
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''use current state, and current input to calculate the linearized state transfer matrix A_lin and input matrix B_lin'''
        
        # Linearize the system dynamics
        A_c, B_c = self.linearization(current_state, current_input)

        # Discretize the system dynamics
        A_d, B_d = self.discretization(A_c, B_c, dt)
        
        # Check controllability of the discretized system
        controllability_matrix = np.hstack([np.linalg.matrix_power(A_d, i) @ B_d for i in range(self.dim_states)])
        rank = np.linalg.matrix_rank(controllability_matrix)

        if rank < self.dim_states:
            raise ValueError(f"System (A, B) is not controllable，rank of controllability matrix is {rank}, while the dimension of state space is {self.dim_states}.")
        
        return A_d, B_d
    
    def get_linearized_AB_discrete_sym(
            self, 
            dt: float
        ) -> Tuple[ca.Function, ca.Function]:
        """
        Return CasADi-wrapped functions A_d(x,u), B_d(x,u) by computing
        ZOH-discretized matrices using scipy.linalg.expm (safe, plugin-free).
        """

        # MX symbolic variables
        x = ca.MX.sym("x", self.dim_states)
        u = ca.MX.sym("u", self.dim_inputs)

        f = self.dynamics_function(x, u)

        A_c = ca.jacobian(f, x)
        B_c = ca.jacobian(f, u)

        A_func = ca.Function("A_c_func", [x, u], [A_c])
        B_func = ca.Function("B_c_func", [x, u], [B_c])

        def compute_discrete_matrices(x_val, u_val):
            A_val = np.array(A_func(x_val, u_val))
            B_val = np.array(B_func(x_val, u_val))

            aug = np.zeros((self.dim_states + self.dim_inputs, self.dim_states + self.dim_inputs))
            aug[:self.dim_states, :self.dim_states] = A_val
            aug[:self.dim_states, self.dim_states:] = B_val

            expm_matrix = scipy.linalg.expm(aug * dt)
            A_d_val = expm_matrix[:self.dim_states, :self.dim_states]
            B_d_val = expm_matrix[:self.dim_states, self.dim_states:]

            return A_d_val, B_d_val

        # CasADi external function wrapping NumPy+SciPy code
        A_d_func = ca.external("A_d_func", lambda x_, u_: compute_discrete_matrices(x_, u_)[0])
        B_d_func = ca.external("B_d_func", lambda x_, u_: compute_discrete_matrices(x_, u_)[1])

        return A_d_func, B_d_func

    def get_equilibrium_input(
            self, 
            state: np.ndarray
        ) -> float:
        
        '''
        u = ca.SX.sym("u")
        state_sym = ca.vertcat(ca.SX.sym("p"), ca.SX.sym("v"))
        
        # Define equilibrium_condition: dv/dt = 0
        dynamics_output = self.dynamics_function(state_sym, ca.vertcat(u))
        dvdt = dynamics_output[1] # extract v_dot

        # Substitue the value into symbolic variable to get equilibrium equation
        equilibrium_condition = ca.substitute(dvdt, state_sym, ca.vertcat(p, v))

        # Solve equilibrium equation to get u_eq
        u_eq = ca.solve(equilibrium_condition, u)
        '''

        p, v = state

        theta = self.env.theta(p)

        u_eq = 9.81 * np.sin(theta)

        return float(u_eq)
    
    def one_step_forward(
            self, 
            current_state: np.ndarray, 
            current_input: np.ndarray, 
            dt: float
        ) -> np.ndarray:

        '''use current state, current input, and time difference to calculate next state'''
        
        t_span = [0, dt]
        sim_dynamics = lambda t, state: self.dynamics_function(state, [current_input]).full().flatten()

        solution = solve_ivp(
            sim_dynamics,
            t_span,
            current_state,
            method='RK45',
            t_eval=[dt]
        )
        
        next_state = np.array(solution.y)[:, -1]

        #print(next_state)

        if self.env.disturbance_lbs is not None and self.env.disturbance_lbs is not None:
            next_state += np.random.uniform(self.env.disturbance_lbs, self.env.disturbance_ubs) * dt

        return next_state
    
    def build_stochastic_model(
            self, 
            dt: float
        ) -> None:
        """
        Build CasADi expressions for continuous-time and discrete-time
        process noise covariance matrices based on self.env.h and self.env.h_cov.

        Stores:
        - self.dynamics_variance_function_cont: continuous-time Σ^w(x,u)
        - self.dynamics_variance_function_disc: discrete-time Σ^w(x,u)
        """

        # Get symbolic variables
        p = self.states[0]
        v = self.states[1]
        u = self.inputs[0]
        x = ca.vertcat(p, v)

        Gravity = 9.81

        # Mean slope by automatic differentiation of h(p)
        mu_h = self.env.h(p)
        mu_dh = ca.gradient(mu_h, p)

        # Variance of the slope Var[h'(p)]. When the learned model supplies it
        # (symbolic_dh_cov_ext, see Chapter 6.3) it is used directly: for a model that
        # is linear in the parameters this is phi'(p)^T Sigma phi'(p), for a GP it is
        # the second mixed derivative of the posterior covariance. Otherwise fall back
        # to the original approximation of Chapter 6.2, the derivative of the marginal
        # variance Var[h(p)], which is kept so that the earlier notebooks run unchanged.
        if getattr(self.env, "dh_cov", None) is not None:
            sigma_dh = self.env.dh_cov(p)
        else:
            sigma_h2 = self.env.h_cov(p)
            sigma_dh = ca.fabs(ca.gradient(sigma_h2, p))  # safety clamp to non-negative

        # Mean of trig terms
        denom = ca.sqrt(1 + mu_dh**2)
        mu_cos = 1 / denom
        mu_sin = mu_dh / denom

        # Variance propagation
        dcos_dz = -mu_dh / (1 + mu_dh**2)**(3/2)
        dsin_dz = 1 / (1 + mu_dh**2)**(3/2)
        dprod_dz = mu_cos * dsin_dz + mu_sin * dcos_dz

        sigma_cos2 = (dcos_dz**2) * sigma_dh
        sigma_sincos2 = (dprod_dz**2) * sigma_dh

        sigma_v_cont = Gravity**2 * sigma_sincos2 + u**2 * sigma_cos2
        sigma_vec_cont = ca.vertcat(0, sigma_v_cont)
        Sigma_w_cont = ca.diag(sigma_vec_cont)
        Sigma_w_disc = dt**2 * Sigma_w_cont

        # Store CasADi functions
        self.dynamics_variance_function_cont = ca.Function(
            "Sigma_w_cont", [self.states, self.inputs], [Sigma_w_cont]
        )
        self.dynamics_variance_function_disc = ca.Function(
            "Sigma_w_disc", [self.states, self.inputs], [Sigma_w_disc]
        )
    







def linspace_include_target(
        target: np.ndarray, 
        lb: np.ndarray, 
        ub: np.ndarray, 
        num: int
    ) -> np.ndarray:
    """
    Generate a linspace array that includes the target value.
    """

    assert target >= lb and target <= ub, "Target is out of bounds"
    # Create the original linspace
    linspace_array = np.linspace(lb, ub, num)

    # Find the closest index to target
    closest_idx = np.argmin(np.abs(linspace_array - target))

    # Calculate the difference and shift the entire array
    difference = target - linspace_array[closest_idx]
    linspace_array += difference

    return linspace_array


def plot_discrete_state_space(
        pos_partitions: np.ndarray, 
        vel_partitions: np.ndarray, 
        target_state: np.ndarray = None
    ) -> None:

    """
    Visualize the discrete state space.
    """

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    pos_grid, vel_grid = np.meshgrid(pos_partitions, vel_partitions)
    ax.plot(pos_grid.flatten(), vel_grid.flatten(), 'o', label='Discrete States')
    if target_state is not None:
        ax.plot(target_state[0], target_state[1], 'r*', label='Target State')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.legend()
    plt.show()


class Env_rl_d(Env):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics = None,
            num_states: np.array = np.array([20, 20]),
            num_actions: int = 10,
            dt: float = 0.1,
            use_nn: bool = True,
            num_samples: int = 100,
            state_noise_levels: np.array = np.array([1e-2, 1e-3]),
            input_noise_levels: np.array = np.array([1e-3]),
            build_stochastic_mdp = True
        ) -> None:

        self.env = env
        
        super().__init__(self.env.case, self.env.init_state, self.env.target_state, 
                         state_lbs=self.env.state_lbs, state_ubs=self.env.state_ubs, 
                         input_lbs=self.env.input_lbs, input_ubs=self.env.input_ubs)

        self.num_pos = num_states[0]
        self.num_vel = num_states[1]
        self.num_acc = num_actions

        if self.state_lbs is None or self.state_ubs is None or self.input_lbs is None or self.input_ubs is None:
            raise ValueError("Constraints on states and input must been fully specified!")

        # Partitions over state space and input space
        # self.pos_partitions = np.linspace(self.pos_lbs, self.pos_ubs, self.num_pos)
        # self.vel_partitions = np.linspace(self.vel_lbs, self.vel_ubs, self.num_vel)
        # self.acc_partitions = np.linspace(self.input_lbs, self.input_ubs, self.num_acc)
        self.pos_partitions = linspace_include_target(self.env.target_state[0], self.pos_lbs, self.pos_ubs, self.num_pos)
        self.vel_partitions = linspace_include_target(self.env.target_state[1], self.vel_lbs, self.vel_ubs, self.num_vel)
        self.acc_partitions = linspace_include_target(0.0, self.input_lbs, self.input_ubs, self.num_acc)

        # Create kdtrees for state space and input space
        self.kdtree_pos = cKDTree(self.pos_partitions.reshape(-1, 1))
        self.kdtree_vel = cKDTree(self.vel_partitions.reshape(-1, 1))
        self.kdtree_acc = cKDTree(self.acc_partitions.reshape(-1, 1))

        # Look up closest state index for the target state
        self.target_state_index = self.nearest_state_index_lookup_kdtree(self.env.target_state)

        # Generate state space (position, velocity combinations)
        POS, VEL = np.meshgrid(self.pos_partitions, self.vel_partitions)
        self.state_space = np.vstack((POS.ravel(), VEL.ravel()))  # Shape: (2, num_states)
        self.num_states = self.state_space.shape[1]

        # Define action space (acceleration)
        self.input_space = self.acc_partitions  # Shape: (1, num_actions)
        self.num_actions = len(self.input_space)
        
        # State propagation
        self.dynamics = dynamics
        self.dt = dt

        # Define transition probability matrix and reward matrix
        self.T = [None] * self.num_actions  # Shape: list (1, num_actions) -> array (num_states, num_states)
        self.R = [None] * self.num_actions  # Shape: list (1, num_actions) -> array (num_states, num_states)
        
        # Build MDP 
        if build_stochastic_mdp: 
            self.use_nn = use_nn
            if self.use_nn:
                self.num_samples = num_samples
                self.state_noise_levels = state_noise_levels
                self.input_noise_levels = input_noise_levels
            else:
                self.num_samples = 1
                self.state_noise_levels = np.array([0.0, 0.0])
                self.input_noise_levels = np.array([0.0])
            self.build_stochastic_mdp()

    def one_step_forward(
            self, 
            cur_state: np.ndarray, 
            cur_input: float
        ) -> Tuple[np.ndarray, float]:
        
        # Check whether current state and input is within the state space and input space
        cur_pos = max(min(cur_state[0], self.pos_ubs), self.pos_lbs)
        cur_vel = max(min(cur_state[1], self.vel_ubs), self.vel_lbs)
        cur_state = np.array([cur_pos, cur_vel])
        cur_input = max(min(cur_input, self.input_ubs), self.input_lbs)
        
        # Propagate the state
        next_state = self.dynamics.one_step_forward(cur_state, cur_input, self.dt)  

        # Check whether next state is within the state space
        next_pos = max(min(next_state[0], self.pos_ubs), self.pos_lbs)
        next_vel = max(min(next_state[1], self.vel_ubs), self.vel_lbs)
        next_state_filtered = np.array([next_pos, next_vel])
        
        # Get reward
        next_state_index = self.nearest_state_index_lookup_kdtree(next_state)
        if next_state_index == self.target_state_index:
            reward = 10
        elif np.any(next_state != next_state_filtered):
            reward = -10
        else:
            reward = -1
        
        return next_state, reward
    
    def nearest_state_index_lookup(
            self, 
            state: np.ndarray
        ) -> int:
        """
        Find the nearest state index in the discrete state space for a given state.
        """
        distances = np.linalg.norm(self.state_space.T - np.array(state), axis=1)
        return np.argmin(distances)
    
    def nearest_state_index_lookup_kdtree(
            self, 
            state: np.ndarray
        ) -> int:
        """
        Find the nearest state index in the discrete state space for a given state.
        """
        nearest_pos_idx = self.kdtree_pos.query(state[0])[1]
        nearest_vel_idx = self.kdtree_vel.query(state[1])[1]

        return nearest_pos_idx + nearest_vel_idx * self.num_pos
    
    def build_stochastic_mdp(
            self, 
            verbose=False
        ) -> None:
        """
        Construct transition probability (T) and reward (R) matrices for the MDP.
        """
        # Iterate over all states
        for state_index in range(self.num_states):
            cur_state = self.state_space[:, state_index]
            print(f"Building model... state {state_index + 1}/{self.num_states}")

            # Apply each possible action
            for action_index in range(self.num_actions):
                action = self.input_space[action_index]

                if verbose:
                    print(f"cur_state: {cur_state}")
                    print(f"action: {action}")

                if self.use_nn:
                    probs, nodes, rewards = self.nearest_neighbor_approach(cur_state, action, verbose)

                else:
                    probs, nodes, rewards = self.linear_interpolation_approach(cur_state, action, verbose)

                # Initialize T and R matrices if not already done
                if self.T[action_index] is None:
                    self.T[action_index] = np.zeros((self.num_states, self.num_states))
                if self.R[action_index] is None:
                    self.R[action_index] = np.zeros((self.num_states, self.num_states))

                # Update transition and reward matrices
                for i, node in enumerate(nodes):
                    # node_index = self.nearest_state_index_lookup(node)
                    node_index = self.nearest_state_index_lookup_kdtree(node)
                    
                    self.T[action_index][state_index, node_index] += probs[i] / self.num_samples
                    self.R[action_index][state_index, node_index] += rewards[i] / self.num_samples

    def nearest_neighbor_approach(
            self, 
            cur_state: np.ndarray, 
            action: float, 
            verbose=False
        ) -> Tuple[List[float], List[np.ndarray], List[float]]:

        probs = [1.0] * self.num_samples
        nodes = []
        rewards = []
        for i in range(self.num_samples):
            # Sample noise
            state_noise = np.random.normal(0, 1.0, size=cur_state.shape) * self.state_noise_levels
            input_noise = np.random.normal(0, 1.0, size=1) * self.input_noise_levels

            if verbose:
                print(f"state_noise: {state_noise}")
                print(f"input_noise: {input_noise}")

            # Propagate forward to get next state and reward
            next_state, reward = self.one_step_forward(cur_state + state_noise, action + input_noise)

            if verbose:
                print(f"next_state: {next_state}")

            nodes.append(next_state)
            rewards.append(reward)

        return probs, nodes, rewards
    
    def linear_interpolation_approach(
            self, 
            cur_state: np.ndarray, 
            action: float, 
            verbose=False
        ) -> Tuple[List[float], List[np.ndarray], List[float]]:

        # Propagate forward to get next state and reward
        next_state, reward = self.one_step_forward(cur_state, action)

        if verbose:
            print(f"next_state: {next_state}")

        pos_indices = self.kdtree_pos.query(next_state[0], k=2)[1]
        vel_indices = self.kdtree_vel.query(next_state[1], k=2)[1]

        pos_bounds = [self.pos_partitions[pos_indices[0]], self.pos_partitions[pos_indices[1]]]
        vel_bounds = [self.vel_partitions[vel_indices[0]], self.vel_partitions[vel_indices[1]]]

        if verbose:
            print(f"pos_bounds: {pos_bounds}")
            print(f"vel_bounds: {vel_bounds}")

        # Normalize next state within bounds
        x_norm = (next_state[0] - min(pos_bounds)) / (max(pos_bounds) - min(pos_bounds))
        y_norm = (next_state[1] - min(vel_bounds)) / (max(vel_bounds) - min(vel_bounds))

        if verbose:
            print(f"x_norm: {x_norm}")
            print(f"y_norm: {y_norm}")

        # Calculate bilinear interpolation probabilities
        probs = [
            (1 - x_norm) * (1 - y_norm),  # bottom-left
            x_norm * (1 - y_norm),        # bottom-right
            x_norm * y_norm,              # top-right
            (1 - x_norm) * y_norm         # top-left
        ]

        # Four vertices of the enclosing box
        nodes = [
            [min(pos_bounds), min(vel_bounds)],  # bottom-left
            [max(pos_bounds), min(vel_bounds)],  # bottom-right
            [max(pos_bounds), max(vel_bounds)],  # top-right
            [min(pos_bounds), max(vel_bounds)]   # top-left
        ]

        rewards = [reward] * len(nodes)

        return probs, nodes, rewards

    def plot_t(self):

        fig, ax = plt.subplots(2, self.num_actions, figsize=(15, 5))

        for i in range(self.num_actions):
            ax[0, i].imshow(self.T[i], cmap='hot', interpolation='nearest')
            ax[0, i].set_title(f"Action {i}")
            ax[0, i].set_xlabel("Next State")
            ax[0, i].set_ylabel("Current State")
        
        for i in range(self.num_actions):
            ax[1, i].imshow(self.R[i], cmap='hot', interpolation='nearest')
            ax[1, i].set_title(f"Action {i}")
            ax[1, i].set_xlabel("Next State")
            ax[1, i].set_ylabel("Current State")

        plt.tight_layout()
        plt.show()
    

class Env_rl_c(Env):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics = None,
            dt: float = 0.1
        ) -> None:

        self.env = env
        
        super().__init__(self.env.case, self.env.init_state, self.env.target_state, 
                         state_lbs=self.env.state_lbs, state_ubs=self.env.state_ubs, 
                         input_lbs=self.env.input_lbs, input_ubs=self.env.input_ubs)

        if self.state_lbs is None or self.state_ubs is None or self.input_lbs is None or self.input_ubs is None:
            raise ValueError("Constraints on states and input must been fully specified!")
        
        # self.pos_lbs, self.pos_ubs
        # self.vel_lbs, self.vel_ubs
        # self.input_lbs, self.input_ubs
        
        # State propagation
        self.dynamics = dynamics
        self.dt = dt

    def one_step_forward(
        self, 
        cur_state: np.ndarray, 
        cur_input: np.ndarray
    ) -> Tuple[bool, np.ndarray, float]:

        # Propagate the state
        next_state = self.dynamics.one_step_forward(cur_state, cur_input, self.dt)  
        next_state_raw = next_state

        # Check whether next state is within the state space
        next_pos = max(min(next_state[0], self.pos_ubs), self.pos_lbs)
        next_vel = max(min(next_state[1], self.vel_ubs), self.vel_lbs)
        next_state = np.array([next_pos, next_vel])
        
        # Get reward
        if np.linalg.norm(next_state[0]-self.env.target_state[0])<5e-2:
            done = True
            reward = 10.0
        elif np.any(next_state_raw != next_state):
            done = True
            reward = -10.0
        else:
            done = False
            reward = -1 # np.exp( - 1.0 * np.linalg.norm(next_pos-self.env.target_state[0]))

        return done, next_state, reward

