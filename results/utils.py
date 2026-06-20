import numpy as np
import scipy.linalg
from scipy.special import expit 
from sklearn.decomposition import PCA
from numpy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import entropy

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
    
# =====================================================================
# Orbit and fixed-point analysis
# =====================================================================
def nearest_phase_idx(x, r_orbit):
    """
    Return the index j* such that ||x - r_orbit[:, j*]|| is minimal.

    Parameters
    ----------
    x      : np.ndarray, shape (N,)
    r_orbit: np.ndarray, shape (N, T_orbit)

    Returns
    -------
    j : int
    """
    d2 = np.sum((r_orbit - x[:, None])**2, axis=0)
    return int(np.argmin(d2))


def compute_dominant_freq(time, x, s, num_peaks=2):
    """
    Compute dominant frequencies of each neuron's trajectory via FFT.

    Parameters
    ----------
    time     : np.ndarray, shape (T,)
    x        : np.ndarray, shape (T, N)
    s        : np.ndarray, shape (N,)  — baseline to subtract before FFT
    num_peaks: int — number of dominant frequencies to extract per neuron

    Returns
    -------
    dominant_freqs    : list of length N, each entry shape (num_peaks,)
    dominant_periods  : list of length N, each entry shape (num_peaks,)
    dominant_phases   : list of length N, each entry shape (num_peaks,)
    peak_indices      : list of length N, each entry shape (num_peaks,)
    positive_freqs    : np.ndarray — the positive frequency bins
    fft_amplitudes    : list of length N, each entry shape (T//2,)
    fft_phases        : list of length N, each entry shape (T//2,)
    """
    freqs = np.fft.fftfreq(len(time), d=time[1] - time[0])
    positive_freqs = freqs[freqs > 0]

    dominant_freqs, dominant_periods, dominant_phases = [], [], []
    peak_indices, fft_amplitudes, fft_phases = [], [], []

    for i in range(x.shape[1]):
        variable_fft = np.fft.fft(x[:, i] - s[i])
        amplitudes = np.abs(variable_fft[freqs > 0])
        phases     = np.angle(variable_fft[freqs > 0], deg=True)

        top_idx = np.argsort(amplitudes)[::-1][:num_peaks]

        dominant_freqs.append(positive_freqs[top_idx])
        dominant_periods.append(1 / positive_freqs[top_idx])
        dominant_phases.append(phases[top_idx])
        peak_indices.append(top_idx)
        fft_amplitudes.append(amplitudes)
        fft_phases.append(phases)

    return dominant_freqs, dominant_periods, dominant_phases, peak_indices, positive_freqs, fft_amplitudes, fft_phases


def precompute_fixed_point(init_condition, W, tauE, tauI, T, dt,
                           a, b, c, s, bias, tail_avg_steps=500):
    """
    Estimate the fixed point by running unforced dynamics and averaging the tail.

    Parameters
    ----------
    init_condition  : np.ndarray, shape (N,)
    tail_avg_steps  : int — number of final time steps to average over

    Returns
    -------
    r_fp    : np.ndarray, shape (N,) — estimated fixed point
    r       : np.ndarray, shape (T/dt+1, N) — full trajectory
    """
    t, r = solve_rk4_springBox(init_condition, W, tauE, tauI, T, dt, a, b, c, bias)
    r_fp = np.mean(r[-tail_avg_steps:, :], axis=0)
    return r_fp, r


def precompute_orbit(ic_orbit, W, tauE, tauI, T, dt,
                     a, b, c, s, bias, no_freqs, wash):
    """
    Run unforced dynamics from ic_orbit and extract one full period of the
    limit cycle, along with its unit tangent vector.

    Parameters
    ----------
    ic_orbit : np.ndarray, shape (N,)
    no_freqs : int   — number of frequency peaks to extract
    wash     : float — transient duration (s) to discard before FFT

    Returns
    -------
    orbit_info : dict with keys:
        r_orbit    : np.ndarray, shape (N, n0)   — one period of the orbit
        t_orbit    : np.ndarray, shape (n0,)     — corresponding time vector
        t_hat      : np.ndarray, shape (N, n0)   — unit tangent along orbit
        n0         : int                         — number of steps per period
        tvec       : np.ndarray                  — full time vector
        r_springBox: np.ndarray, shape (N, T/dt+1) — full trajectory
    """
    from matplotlib import pyplot as plt

    tvec = np.arange(0, T + dt, dt)
    r_springBox = np.zeros((ic_orbit.shape[0], len(tvec)))
    r_springBox[:, 0] = ic_orbit
    external_input = np.zeros(W.shape[0])

    for i in range(1, len(tvec)):
        r_springBox[:, i] = rk4_step(
            r_springBox[:, i-1], W, external_input, tauE, tauI, dt, a, b, c, bias)

    _, dominant_periods, _, _, _, _, _ = compute_dominant_freq(
        tvec[int(wash/dt):],
        r_springBox.T[int(wash/dt):, :],
        s,
        num_peaks=no_freqs
    )
    dominant_period = dominant_periods[0][0]
    n0 = int(np.round(dominant_period / dt))

    r_orbit = r_springBox[:, (int(T/dt) - n0):int(T/dt)].copy()
    t_orbit = tvec[(int(T/dt) - n0):int(T/dt)].copy()

    plt.plot(r_orbit.T)
    plt.show()

    v_orbit = np.gradient(r_orbit, t_orbit, axis=1)
    t_hat = v_orbit / (np.linalg.norm(v_orbit, axis=0, keepdims=True) + 1e-12)

    return dict(r_orbit=r_orbit, t_orbit=t_orbit, t_hat=t_hat,
                n0=n0, tvec=tvec, r_springBox=r_springBox)


def solve_rk4_controlled(ic_orbit, ic_transient, W, cues, tauE, tauI, T, dt,
                         a, b, c, s, bias, r_fp, orbit_info,
                         sigma_ou=0.0, tau_c=0.01, t_start_noise=0.0, rng=None):
    """
    Simulate controlled dynamics with stop/go/hold/resume cue inputs and
    optional Ornstein-Uhlenbeck noise.

    Parameters
    ----------
    ic_orbit     : np.ndarray, shape (N,) — initial condition for the orbit
    ic_transient : np.ndarray, shape (N,) — initial condition for transient kick
    W            : np.ndarray, shape (N, N)
    cues         : dict with keys 'stop', 'go', 'transient', 'hold_on', 'resume'
                   each a list [enabled, t_start, duration, ...]
    r_fp         : np.ndarray, shape (N,) — fixed point estimate
    orbit_info   : dict from precompute_orbit
    sigma_ou     : float — OU noise amplitude (0 = no noise)
    tau_c        : float — OU correlation time constant
    t_start_noise: float — time at which noise injection begins
    rng          : np.random.Generator or None

    Returns
    -------
    tvec : np.ndarray, shape (T/dt+1,)
    r    : np.ndarray, shape (T/dt+1, N)
    exts : list (reserved for external input logging)
    """
    if rng is None:
        rng = np.random.default_rng()

    tvec    = orbit_info["tvec"]
    r_orbit = orbit_info["r_orbit"]
    t_hat   = orbit_info["t_hat"]

    d = ic_orbit.shape[0]
    r = np.zeros((d, len(tvec))) + s[:, None]
    r[:, 0] = s

    quint = d // 5
    tau_vec = np.concatenate([
        float(tauE) * np.ones(4 * quint),
        float(tauI) * np.ones(d - 4 * quint)
    ])

    n = np.zeros(d, dtype=float)
    inv_tau_c = 1.0 / float(tau_c)
    if sigma_ou > 0:
        s_n = float(sigma_ou) * np.sqrt(float(tau_c) / 2.0)
        n = rng.normal(0.0, s_n, size=d)

    exts = []

    for i in range(1, len(tvec)):
        external_input = np.zeros(W.shape[0])

        # kick onto orbit at t=1
        if np.isclose(tvec[i-1], 1, atol=0.5 * dt):
            external_input = (1.0 / dt) * ic_orbit

        # STOP: clamp state toward baseline
        if cues['stop'][0]:
            in_stop1 = cues['stop'][1] <= tvec[i-1] <= cues['stop'][1] + cues['stop'][2]
            in_stop2 = cues['stop'][3] <= tvec[i-1] <= cues['stop'][3] + cues['stop'][4]
            if in_stop1 or in_stop2:
                external_input = -10 * (r[:, i-1] - s)

        # TRANSIENT: kick onto transient IC
        if cues['transient'][0] and np.isclose(tvec[i-1], cues['transient'][1], atol=0.5*dt):
            external_input = (1 / dt) * ic_transient

        # HOLD ON: clamp state toward fixed point
        if cues['hold_on'][0]:
            if cues['hold_on'][1] <= tvec[i-1] <= cues['hold_on'][1] + cues['hold_on'][2]:
                external_input = 10 * (r_fp - r[:, i-1])

        # GO: kick back onto orbit
        if cues['go'][0] and np.isclose(tvec[i-1], cues['go'][1], atol=0.5*dt):
            external_input = (1 / dt) * ic_orbit

        # RESUME: phase-matched push back to orbit
        if cues['resume'][0]:
            if cues['resume'][1] <= tvec[i-1] <= cues['resume'][1] + cues['resume'][2]:
                j = nearest_phase_idx(r[:, i-1], r_orbit)
                r_star  = r_orbit[:, j]
                tauhat  = t_hat[:, j]
                e       = r_star - r[:, i-1]
                e_tan   = np.dot(e, tauhat) * tauhat
                e_norm  = e - e_tan
                external_input += 10 * 0.6 * e_norm

        # RK4 step
        r_det = rk4_step(r[:, i-1], W, external_input, tauE, tauI, dt, a, b, c, bias)

        # OU noise injection
        t_now = tvec[i]
        if (t_now >= t_start_noise) and (sigma_ou > 0):
            n = n + (-inv_tau_c * n) * dt + float(sigma_ou) * np.sqrt(dt) * rng.normal(size=d)
            r[:, i] = r_det + dt * (n / tau_vec)
        else:
            r[:, i] = r_det

    return tvec, r.T, exts
    
# =====================================================================
# Noisy springBox simulation (OU noise)
# =====================================================================
def solve_rk4_noisy_springBox(init_condition, W, tauE, tauI, T, dt,
                               a, b, c, bias,
                               sigma_ic=0.0, sigma_ou=0.0,
                               tau_c=0.01, t_start_noise=None, rng=None):
    """
    Unforced dynamics with optional IC perturbation and Ornstein-Uhlenbeck noise.

    Parameters
    ----------
    sigma_ic      : float — std of Gaussian noise added to initial condition
    sigma_ou      : float — OU diffusion coefficient
    tau_c         : float — OU correlation time constant
    t_start_noise : float or None — time at which OU noise begins (None = never)

    Returns
    -------
    tvec : np.ndarray, shape (T/dt+1,)
    r    : np.ndarray, shape (T/dt+1, N)
    """
    if rng is None:
        rng = np.random.default_rng()
    tvec = np.arange(0, T + dt, dt)
    d = len(init_condition)
    r = np.zeros((d, len(tvec)))
    r[:, 0] = init_condition + rng.normal(0.0, sigma_ic, size=d) if sigma_ic > 0 else init_condition

    external_input = np.zeros(W.shape[0])
    if t_start_noise is None:
        t_start_noise = np.inf

    n = np.zeros(d, dtype=float)
    inv_tau_c = 1.0 / float(tau_c)
    quint = d // 5
    tau_vec = np.concatenate([
        float(tauE) * np.ones(4 * quint),
        float(tauI) * np.ones(d - 4 * quint)
    ])
    ou_started = False

    for i in range(1, len(tvec)):
        t = tvec[i]
        r_det = rk4_step(r[:, i-1], W, external_input, tauE, tauI, dt, a, b, c, bias)
        if (t >= t_start_noise) and (sigma_ou > 0):
            if not ou_started:
                s_n = float(sigma_ou) * np.sqrt(float(tau_c) / 2.0)
                n = rng.normal(0.0, s_n, size=d)
                ou_started = True
            n = n + (-inv_tau_c * n) * dt + float(sigma_ou) * np.sqrt(dt) * rng.normal(size=d)
            r[:, i] = r_det + dt * (n / tau_vec)
        else:
            n[:] = 0.0
            ou_started = False
            r[:, i] = r_det

    return tvec, r.T

# =====================================================================
# Linear dynamics — numerical RK4 (arbitrary N, E/I split)
# =====================================================================
def J_dynamics_numerical(J, dt=0.1, T=100, tauE=1, tauI=1, x=None):
    """
    Simulate linear dynamics x' = (J - I) x using RK4, with separate
    time constants for E and I populations (first half E, second half I).

    Parameters
    ----------
    J  : np.ndarray, shape (N, N)
        Connectivity matrix (eigenvalues Re(λ) < 1).
    dt : float
    T  : float
    tauE, tauI : float
    x  : np.ndarray, shape (N,) or None
        Initial condition. If None, uses the most amplifying IC from init_cond(J).

    Returns
    -------
    r          : np.ndarray, shape (T/dt, N)
    proxy_ampl : float — max norm / initial norm
    """
    no_points = int(T / dt)
    tvec = np.linspace(0, T, no_points)
    A = J - np.eye(J.shape[0])
    N = A.shape[0]
    half = N // 2

    if x is None:
        x = init_cond(J)[:, 0].copy()
    x = np.array(x, dtype=float, copy=True)

    r = np.zeros((N, no_points))
    r[:, 0] = x

    for t in range(no_points - 1):
        def _deriv(v):
            Av = A @ v
            return np.concatenate([Av[:half] / tauE, Av[half:] / tauI])

        k1 = _deriv(x)
        k2 = _deriv(x + 0.5 * dt * k1)
        k3 = _deriv(x + 0.5 * dt * k2)
        k4 = _deriv(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        r[:, t+1] = x

    norm = np.linalg.norm(r, axis=0)
    proxy_ampl = np.max(norm) / norm[0]
    return r.T, proxy_ampl

def classify_behaviour(r, fft_tstart, fft_tend, dt, n_pcs=5):
    #Check norm of activity tail
    norm_r = np.sqrt(np.sum(r[:,:]**2,axis=1)) #across time
    activity_tail = r[-200:,:]
    norm_tail = np.sum(np.sqrt(np.sum(activity_tail**2,axis=1)))
    
    l, N = activity_tail.shape
    metrics={}
    if not np.all(np.isfinite(r)):
        return 'unclassified', metrics

    if norm_tail<0.01*np.sqrt(N)*l:
        behaviour = '0_fp'
        
    elif norm_tail>=0.01*np.sqrt(N)*l:
        
        #Check average of norm of trajectory for first and second half of trajectory
        # Split the trajectory into two halves
        third = len(norm_r) // 3
        first_tri_avg = np.mean(norm_r[:2*third])
        second_tri_avg = np.mean(norm_r[2*third:])
    
        # Calculate percentage decrease between the two averages
        percentage_decrease = (first_tri_avg - second_tri_avg) / first_tri_avg * 100
    
        if percentage_decrease>20:
            behaviour = '0_fp'
        
        
        else:
            #Check norm of activity tail with 0 mean
            norm_tail_0mean = np.sum(np.sqrt(np.sum((activity_tail-np.mean(activity_tail,axis=0))**2,axis=1)))
            if norm_tail_0mean<0.01*np.sqrt(N)*l:
                behaviour = 'new_fp'
            else:

                if N==2:
                    #signals = r[int(fft_tstart/dt):int(fft_tend/dt),:]
                    behaviour = 'periodic'
                    return behaviour, metrics 
                else:
                    # PCA
                    pca = PCA(n_components=min(n_pcs, N))
                    signals = pca.fit_transform(r[int(fft_tstart/dt):int(fft_tend/dt),:])  # shape (T, n_pcs)

                peak_ratios = []
                spectral_entropies = []
                valid_dims = 0

                global_max_power = 0
                for i in range(signals.shape[1]):
                    signal = signals[:, i]
                    freqs = fftfreq(len(signal), d=dt)
                    power = np.abs(fft(signal)[freqs>0])**2 + 1e-12
                    global_max_power = max(global_max_power, np.max(power))

                for i in range(signals.shape[1]):
                    signal = signals[:, i]
                    freqs = fftfreq(len(signal), d=dt)
                    power = np.abs(fft(signal)[freqs>0])**2 + 1e-12
                    norm_power = power / np.sum(power)

                    # Peaks
                    peaks, properties = find_peaks(power, height=np.max(global_max_power) * 0.01)
                    #print(peaks)
                    if len(peaks) == 0:
                        continue  # Skip flat / fixed-point dimensions

                    valid_dims += 1
                    peak_powers = properties['peak_heights'] if 'peak_heights' in properties else np.array([])
                    top_power = np.sum(np.sort(peak_powers)[-5:]) if len(peak_powers) > 0 else 0
                    peak_ratio = top_power / np.sum(power)
                    peak_ratios.append(peak_ratio)

                    # Spectral entropy
                    spec_entropy = entropy(norm_power, base=2)
                    spectral_entropies.append(spec_entropy)

                    #print(["peak_ratio:",peak_ratio], ["entropy:", spec_entropy])

                avg_peak_ratio = np.mean(peak_ratios)
                avg_entropy = np.mean(spectral_entropies)

                if avg_peak_ratio > 0.8 and avg_entropy < 1.5:
                    behaviour = 'periodic'
                elif avg_peak_ratio > 0.4 and avg_entropy < 2.3:
                    behaviour = 'quasi-periodic'
                elif avg_peak_ratio < 0.4 and avg_entropy < 2:
                    behaviour = 'quasi-periodic'
                elif avg_entropy > 2 and avg_peak_ratio < 0.4:
                    behaviour = 'chaotic'
                elif avg_entropy > 2.3:
                    behaviour = 'chaotic'       
                else:
                    behaviour = 'uncertain'

                metrics = {
                    'avg_peak_ratio': avg_peak_ratio,
                    'avg_entropy': avg_entropy,
                    'classification': behaviour,
                    'valid_dimensions': valid_dims
                }

    return behaviour, metrics
