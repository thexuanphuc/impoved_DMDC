import casadi as ca
import numpy as np

def mpc_controller_lotka(a, b, c, d, x_ini, horizon=20, dt=0.1, nx=2, nu=1):

    x0 = x_ini.reshape(-1, 1)
    # Define symbolic variables
    x = ca.MX.sym('x', nx, horizon + 1)  # States over horizon
    u = ca.MX.sym('u', nu, horizon)      # Inputs over horizon

    # Define weighting matrices
    Q = 10.0 * np.eye(nx)  # State cost matrix
    R = 0.01 * np.eye(nu)  # Input cost matrix

    # Reference state (equilibrium point: [-a/b, -d/c])
    x_ref = ca.vertcat(1, 3)
    # Initialize cost and constraints
    cost = 0.0
    constraints = []

    # Initial condition
    constraints.append(x[0, 0] - x0[0])  # Constraint for x[0]
    constraints.append(x[1, 0] - x0[1])  # Constraint for x[1]

    # Formulate the MPC problem
    for t in range(horizon):
        # Nonlinear dynamics
        f1 = a * x[0, t] - b * x[0, t] * x[1, t]
        f2 = c * x[0, t] * x[1, t] - d * x[1, t] + u[:, t]
        
        # Euler discretization: x_{k+1} = x_k + dt * f
        next_state = x[:, t] + dt * ca.vertcat(f1, f2)
        
        # Add dynamics constraint
        constraints.append(x[0, t+1] - next_state[0])  # Constraint for x[0]
        constraints.append(x[1, t+1] - next_state[1])  # Constraint for x[1]
                
        # Cost: quadratic state and input penalties        
        state_error = x[:, t] - x_ref

        cost += ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([u[:, t].T, R, u[:, t]])


    # Create NLP problem
    nlp = {'x': ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1)), 
           'f': cost, 
           'g': ca.vertcat(*constraints)}

    # NLP solver options
    opts = {'ipopt.print_level': 1, 'print_time': 0}
    
    # Create and solve NLP
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    x0_flat = np.zeros(nx * (horizon + 1) + nu * horizon).reshape(-1,)  # Initial guess

    # bound for decision variables
    lbx = -ca.inf * np.ones(nx * (horizon + 1) + nu * horizon)
    ubx = ca.inf * np.ones(nx * (horizon + 1) + nu * horizon)

    # Input bounds: ||u||_inf <= 6.0
    u_idx = nx * (horizon + 1)
    lbx[u_idx:] = -12.0
    ubx[u_idx:] = 12.0
    
    # equality constraints bounds
    lbg = np.zeros(len(constraints))
    ubg = np.zeros(len(constraints))
    
    # Solve the NLP
    sol = solver(x0=x0_flat, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    
    # Extract the first control input
    u_opt = sol['x'][nx * (horizon + 1):].full().reshape(nu, horizon)

    if(u_opt is None):
        print("can not find the solution, return 0s")
    return u_opt[:, 0] if u_opt is not None else np.zeros(nu)

