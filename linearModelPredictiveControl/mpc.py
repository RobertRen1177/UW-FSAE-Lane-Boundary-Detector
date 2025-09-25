"""
Decoupled lateral MPC (linear) in CasADi
----------------------------------------
States  (path/frame): x = [e_y, e_psi, v_y, r]^T
  e_y   : lateral error [m]
  e_psi : heading error [rad]
  v_y   : lateral velocity in body frame [m/s]
  r     : yaw rate [rad/s]

Input: u = delta (front steering angle) [rad]
Exogenous (known preview): w = kappa_ref (road curvature over horizon) [1/m]
Assumption (decoupled): v_x(t) is known/measured/planned and treated as a parameter.

Continuous-time linear model (small-slip linear tire):
  dot(e_y)   = v_y + v_x * e_psi
  dot(e_psi) = r - v_x * kappa_ref
  dot(v_y)   = -(Cf+Cr)/(m*v_x) * v_y + (-(a*Cf - b*Cr)/(m*v_x) - v_x) * r + (Cf/m) * delta
  dot(r)     = -(a*Cf - b*Cr)/(Iz*v_x) * v_y - (a^2*Cf + b^2*Cr)/(Iz*v_x) * r + (a*Cf/Iz) * delta

Discretization: forward Euler (Ad = I + Ac*Ts; Bd = Bc*Ts; Ed = Ec*Ts)
- For small Ts this is adequate. Swap in exact c2d if desired.

Cost (quadratic): sum_k x_k^T Q x_k + u_k^T R u_k + (Δu_k)^T Rdu (Δu_k)
Constraints: u_min ≤ u_k ≤ u_max, |Δu_k| ≤ du_max
(Optional) state bounds on e_y and r are also shown.
"""

from typing import Tuple
import numpy as np
import casadi as ca

# -----------------------
# Parameters (edit these)
# -----------------------
params = {
    'm' : 230.0,      # mass [kg]
    'Iz': 110.0,      # yaw inertia [kg*m^2]
    'a' : 0.90,       # cg->front axle [m]
    'b' : 0.90,       # cg->rear axle [m]
    'Cf': 45000.0,    # front cornering stiffness [N/rad]
    'Cr': 50000.0,    # rear cornering stiffness [N/rad]
}

Ts = 0.02          # sample time [s]
N  = 25            # horizon steps

# weights (tune)
Q  = np.diag([4.0, 2.0, 0.2, 0.5])   # ey, epsi, vy, r
R  = np.array([[1e-3]])              # on u
Rdu= np.array([[2e-2]])              # on du

# input and state limits (tune to your hardware)
U_MIN, U_MAX     = np.deg2rad(-35.0), np.deg2rad(35.0)
DU_MAX           = np.deg2rad(300.0) * Ts   # steering rate limit per step
EY_MAX, R_MAX    = 2.0, np.deg2rad(80.0)

# ------------------------------------
# Discrete-time matrices as functions
# ------------------------------------
'''
Continuous-time linear model (small-slip linear tire):
  dot(e_y)   = v_y + v_x * e_psi
  dot(e_psi) = r - v_x * kappa_ref
  dot(v_y)   = -(Cf+Cr)/(m*v_x) * v_y + (-(a*Cf - b*Cr)/(m*v_x) - v_x) * r + (Cf/m) * delta
  dot(r)     = -(a*Cf - b*Cr)/(Iz*v_x) * v_y - (a^2*Cf + b^2*Cr)/(Iz*v_x) * r + (a*Cf/Iz) * delta
'''

def lateral_discrete_mats(vx: float, p=params, Ts: float = Ts) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Ad, Bd, Ed) for given forward speed vx."""
    m, Iz, a, b, Cf, Cr = p['m'], p['Iz'], p['a'], p['b'], p['Cf'], p['Cr']
    # Avoid division by zero
    vx = float(max(vx, 0.5))

    # Continuous-time A, B, E (w = kappa_ref)
    Ac = np.zeros((4,4))
    Ac[0,1] = vx
    Ac[0,2] = 1.0
    Ac[1,3] = 1.0
    Ac[2,2] = -(Cf + Cr) / (m * vx)
    Ac[2,3] = - (a*Cf - b*Cr) / (m * vx) - vx
    Ac[3,2] = - (a*Cf - b*Cr) / (Iz * vx)
    Ac[3,3] = - (a*a*Cf + b*b*Cr) / (Iz * vx)

    Bc = np.zeros((4,1))
    Bc[2,0] = Cf / m
    Bc[3,0] = a * Cf / Iz

    Ec = np.zeros((4,1))
    Ec[1,0] = -vx

    # Discretize (forward Euler). Replace with exact c2d if desired.
    Ad = np.eye(4) + Ts * Ac
    Bd = Ts * Bc
    Ed = Ts * Ec
    return Ad, Bd, Ed

def build_prediction_mats_numeric(Ad, Bd, Ed, N):
    nx = Ad.shape[0]
    nu = Bd.shape[1]
    nw = Ed.shape[1]

    Sx = np.zeros((nx*N, nx))
    Su = np.zeros((nx*N, nu*N))
    Sw = np.zeros((nx*N, nw*N))

    A_pows = [np.eye(nx)]
    for i in range(1, N+1):
        A_pows.append(Ad @ A_pows[-1])  # A^i

    # X_i = A^i x0 + sum_{j=0}^{i-1} A^{i-1-j} B u_j + sum_{j=0}^{i-1} A^{i-1-j} E w_j
    for i in range(1, N+1):
        Sx[(i-1)*nx:i*nx, :] = A_pows[i]
        for j in range(i):
            Aij = A_pows[i-1-j]
            Su[(i-1)*nx:i*nx, j*nu:(j+1)*nu] = Aij @ Bd
            Sw[(i-1)*nx:i*nx, j*nw:(j+1)*nw] = Aij @ Ed
    return Sx, Su, Sw

# -----------------------
# MPC QP builder
# -----------------------

def build_mpc_qp(Ad, Bd, Ed, N, Q, R, Rdu, x0, w_seq,
                 u_prev,
                 u_min=U_MIN, u_max=U_MAX,
                 du_max=DU_MAX,
                 ey_max=EY_MAX, r_max=R_MAX):

    nx, nu, nw = 4, 1, 1
    Qb = ca.diag(ca.repmat(ca.DM(Q).diag(), N, 1))   # block-diag Q
    Rb = ca.diag(ca.repmat(ca.DM(R).diag(), N, 1))   # block-diag R
    # We'll implement Δu via auxiliary var v = Δu and constraint u_k = u_{k-1} + sum_{i<=k} v_i

    # Stacked prediction matrices (numpy then to DM)
    Sx, Su, Sw = build_prediction_mats_numeric(Ad, Bd, Ed, N)
    Sx, Su, Sw = ca.DM(Sx), ca.DM(Su), ca.DM(Sw)

    X = Sx @ ca.DM(x0)
    W = ca.DM(np.array(w_seq).reshape(-1,1))  # shape (N*1, 1)

    # Decision variables: U (N), V (N) with U being absolute steering, V being Δu per step
    U = ca.SX.sym('U', N, 1)
    V = ca.SX.sym('V', N, 1)

    # Enforce U from rate V: U_k = u_prev + sum_{i=0..k} V_i
    # Build lower-triangular summation matrix T so that U = u_prev*1 + T V
    T = np.tril(np.ones((N, N)))
    U_from_V = ca.DM(T) @ V + u_prev

    # Dynamics: X = Sx x0 + Su U + Sw W
    X_pred = X + Su @ U + Sw @ W

    # Cost: X'QX + U'RU + V'Rdu V
    cost = ca.mtimes([X_pred.T, Qb, X_pred]) \
         + ca.mtimes([U.T, Rb, U]) \
         + ca.mtimes([V.T, ca.diag(ca.repmat(ca.DM(Rdu).diag(), N, 1)), V])

    # Constraints
    g = []
    lbg = []
    ubg = []

    # Input limits: u_min <= U_k <= u_max (using equality U == U_from_V)
    g.append(U - U_from_V)  # enforce equality
    lbg += [0.0]*N
    ubg += [0.0]*N

    g.append(U)
    lbg += [u_min]*N
    ubg += [u_max]*N

    # Rate limits: |V_k| <= du_max
    g.append(V)
    lbg += [-du_max]*N
    ubg += [ du_max]*N

    # Soft state bounds (ey and r) — here as hard bounds via linear constraints on X blocks
    # X layout per step: [ey, epsi, vy, r]
    Ey = []
    Rr = []
    for k in range(N):
        ey_k = X_pred[k*nx + 0]
        r_k  = X_pred[k*nx + 3]
        Ey.append(ey_k)
        Rr.append(r_k)
    Ey = ca.vcat(Ey)
    Rr = ca.vcat(Rr)
    g.append(Ey)
    lbg += [-ey_max]*N
    ubg += [ ey_max]*N
    g.append(Rr)
    lbg += [-r_max]*N
    ubg += [ r_max]*N

    g = ca.vcat(g)

    # Pack variables
    z = ca.vcat([U, V])

    qp = {
        'h': cost,
        'g': g,
        'x': z
    }

    # Bounds on decision vars: none besides via g, but we can add simple bounds if desired
    # Solve with qpoases (dense QP). You can switch to OSQP by changing the solver name and options.
    solver = ca.qpsol('solver', 'qpoases', qp, {'printLevel': 'low'})
    sol = solver(lbg=ca.DM(lbg), ubg=ca.DM(ubg))

    z_opt = np.array(sol['x']).flatten()
    U_opt = z_opt[:N]
    V_opt = z_opt[N:]

    # Return first input and the whole sequence
    return U_opt[0], U_opt, V_opt, X_pred.full().reshape(N,4)

# -----------------------
# Example usage (dummy)
# -----------------------
if __name__ == "__main__":
    # Example current state and preview
    x0 = np.array([0.3, 0.05, 0.0, 0.0])  # ey=0.3m, epsi=0.05rad
    u_prev = np.array([0.0])              # last applied steering

    # Speed (constant over horizon here). You can schedule per-step if needed.
    vx = 12.0  # m/s (~43 km/h)

    # Reference curvature over horizon (e.g., from path/spline). Here: gentle left turn.
    kappa_seq = 1.0/60.0 * np.ones((N,))  # 60 m radius constant

    Ad, Bd, Ed = lateral_discrete_mats(vx)
    u0, U_seq, V_seq, X_seq = build_mpc_qp(Ad, Bd, Ed, N, Q, R, Rdu, x0, kappa_seq, u_prev)

    print("u0 (rad):", u0)
    print("u0 (deg):", np.rad2deg(u0))
    # X_seq contains predicted states per step
