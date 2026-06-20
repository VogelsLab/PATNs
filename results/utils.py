import numpy as np
import scipy.linalg
from scipy.special import expit 

# =====================================================================
# Lyapunov-based initial conditions
# =====================================================================
def init_cond(J):
    """
    Compute amplifying and non-amplifying initial conditions via the Lyapunov equation.

    Parameters
    ----------
    J : np.ndarray, shape (N, N)
        Connectivity matrix with eigenvalues satisfying Re(λ) < 1
        (i.e. the raw output, before subtracting I).

    Returns
    -------
    a : np.ndarray, shape (N, N)
        Eigenvectors of Q sorted by descending eigenvalue.
    """
    E = np.transpose(J - np.eye(J.shape[0]))
    F = -2 * np.eye(J.shape[0])
    Q = scipy.linalg.solve_continuous_lyapunov(E, F)
    e, a = np.linalg.eig(Q)
    idx = e.argsort()[::-1]
    e = e[idx]
    a = a[:, idx]
    return a

# =====================================================================
# Linear dynamics amplification (2x2 only)
# =====================================================================
def J_dynamics(J, ic, dt=0.1, T=100, tau=1):
    """
    Simulate linear dynamics x' = A x from initial condition ic, where A = J - I.

    Parameters
    ----------
    J : np.ndarray, shape (N, N)
        Connectivity matrix (eigenvalues Re(λ) < 1).
    ic : np.ndarray, shape (N,)
        Initial condition vector.
    dt : float
        Time step (ms).
    T : float
        Total simulation time (ms).
    tau : float
        Time constant.

    Returns
    -------
    rates : np.ndarray, shape (T/dt, N)
        Trajectory.
    proxy_ampl : float
        Max norm minus initial norm — proxy for transient amplification.
    matrix_type : str
        'complex' if eigenvalues are complex, 'real' otherwise.
    norm : np.ndarray
        Norm of the trajectory at each time step.
    """
    A = J - np.eye(J.shape[0])
    evalues, evectors = np.linalg.eig(A)
    matrix_type = 'complex' if np.imag(evalues)[0] != 0 else 'real'

    no_points = int(T / dt)
    tvec = np.linspace(0, T, no_points)

    c = np.linalg.inv(evectors).dot(ic)
    rE = (np.exp(1/tau * evalues[0] * tvec) * evectors[0, 0] * c[0]
        + np.exp(1/tau * evalues[1] * tvec) * evectors[0, 1] * c[1])
    rI = (np.exp(1/tau * evalues[0] * tvec) * evectors[1, 0] * c[0]
        + np.exp(1/tau * evalues[1] * tvec) * evectors[1, 1] * c[1])

    rates = np.concatenate([rE.reshape(1, -1), rI.reshape(1, -1)], axis=0)
    norm = np.linalg.norm(rates, axis=0)
    proxy_ampl = np.max(norm) - norm[0]

    return rates.T, proxy_ampl, matrix_type, norm
    
# =====================================================================
# Nonlinear dynamics — W rescaling and simulation
# =====================================================================
def W_from_J(J, a, b, c):
    """
    Compute W = D^(-1) @ J where D is a diagonal matrix with entries:
        D_ii = (c_i * a_i * exp(a_i * b_i)) / (exp(a_i * b_i) + 1)^2
 
    Parameters
    ----------
    J : np.ndarray, shape (N, N)
    a, b, c : np.ndarray, shape (N,)
        Elementwise sigmoid parameters.
 
    Returns
    -------
    W : np.ndarray, shape (N, N)
    D : np.ndarray, shape (N, N)  — the diagonal scaling matrix
    """
    assert J.shape[0] == J.shape[1], "J must be square"
    assert len(a) == J.shape[0] and len(b) == J.shape[0] and len(c) == J.shape[0], \
        "Parameter arrays must match J dimension"
    exp_term = np.exp(a * b)
    D_diag = (c * a * exp_term) / (exp_term + 1)**2
    D_inv = np.diag(1.0 / D_diag)
    return D_inv @ J, np.diag(D_diag)
 
 
def f(h, a, b, c):
    """
    Shifted sigmoid nonlinearity:
        f_i(h_i) = c_i / (1 + exp(-a_i*(h_i - b_i))) - c_i / (1 + exp(a_i * b_i))
 
    Parameters
    ----------
    h : np.ndarray, shape (N,)
    a, b, c : np.ndarray, shape (N,)
 
    Returns
    -------
    output : np.ndarray, shape (N,)
    """
    term1 = c * expit(a * (h - b))
    term2 = c / (1 + np.exp(a * b))
    return term1 - term2
 
 
def dxdt(x, W, external_input, tauE, tauI, a, b, c, bias):
    """
    Compute dx/dt for a 2-population (E/I) network.
 
    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Current state.
    W : np.ndarray, shape (N, N)
        Connectivity matrix (output of W_from_J).
    external_input : np.ndarray, shape (N,)
    tauE, tauI : float
        Time constants for excitatory and inhibitory populations.
    a, b, c : np.ndarray, shape (N,)
        Sigmoid parameters.
    bias : np.ndarray, shape (N,)
 
    Returns
    -------
    dxdt : np.ndarray, shape (N,)
    """
    quint = len(x) // 5
    aE, bE, cE = a[:4*quint], b[:4*quint], c[:4*quint]
    aI, bI, cI = a[4*quint:], b[4*quint:], c[4*quint:]
 
    preE = W[:4*quint, :] @ x + bias[:4*quint]
    preI = W[4*quint:, :] @ x + bias[4*quint:]
 
    dxE = (-x[:4*quint] + f(preE, aE, bE, cE)) * (1.0 / tauE) + external_input[:4*quint]
    dxI = (-x[4*quint:] + f(preI, aI, bI, cI)) * (1.0 / tauI) + external_input[4*quint:]
    return np.concatenate((dxE, dxI))
 
 
def rk4_step(x, W, external_input, tauE, tauI, dt, a, b, c, bias):
    """Single RK4 integration step."""
    k1 = dxdt(x,                 W, external_input, tauE, tauI, a, b, c, bias)
    k2 = dxdt(x + 0.5*dt*k1,    W, external_input, tauE, tauI, a, b, c, bias)
    k3 = dxdt(x + 0.5*dt*k2,    W, external_input, tauE, tauI, a, b, c, bias)
    k4 = dxdt(x + dt*k3,        W, external_input, tauE, tauI, a, b, c, bias)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
 
 
def solve_rk4_springBox(init_condition, W, tauE, tauI, T, dt, a, b, c, bias):
    """
    Simulate nonlinear network dynamics from a given initial condition with no external input.
 
    Parameters
    ----------
    init_condition : np.ndarray, shape (N,)
    W : np.ndarray, shape (N, N)
        Connectivity matrix (output of W_from_J).
    tauE, tauI : float
    T : float
        Total simulation time.
    dt : float
        Time step.
    a, b, c : np.ndarray, shape (N,)
    bias : np.ndarray, shape (N,)
 
    Returns
    -------
    tvec : np.ndarray, shape (T/dt + 1,)
    r : np.ndarray, shape (T/dt + 1, N)
        Trajectory, one row per time point.
    """
    tvec = np.arange(0, T + dt, dt)
    r = np.zeros((len(init_condition), len(tvec)))
    r[:, 0] = init_condition
    external_input = np.zeros(W.shape[0])
 
    for i in range(1, len(tvec)):
        r[:, i] = rk4_step(r[:, i-1], W, external_input, tauE, tauI, dt, a, b, c, bias)
 
    return tvec, r.T
