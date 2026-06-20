import numpy as np
import pickle
import matplotlib.pyplot as plt


OSC = {"periodic", "quasi-periodic"}


# =====================================================================
# Data loading
# =====================================================================
def load_fig2_data(data_root='../../data/fig2'):
    """
    Load connectivity matrices, LP solutions, eigenvalues, and behavioural
    labels for all three sigma conditions (non-normal and normal).

    Parameters
    ----------
    data_root : str — path to the fig2 data directory

    Returns
    -------
    matrices0, matrices1, matrices2 : lists of np.ndarray — non-normal W matrices
    matrices0_normal, matrices1_normal, matrices2_normal : lists of np.ndarray
    L0, L1, L2 : lists of np.ndarray — prescribed eigenvalues per matrix
    behaviours0, behaviours1, behaviours2 : lists of label lists (non-normal)
    behaviours_normal0, behaviours_normal1, behaviours_normal2 : lists of label lists
    """
    sigmas = ['sigma02', 'sigma05', 'sigma1']
    data, data_normal = [], []
    for s in sigmas:
        with open(f'{data_root}/{s}/Ws.pkl', 'rb') as f:
            data.append(pickle.load(f))
        with open(f'{data_root}/{s}/Ws_normal.pkl', 'rb') as f:
            data_normal.append(pickle.load(f))

    solutions  = [[d for d in ds if not np.all(d.get('W') == 0)] for ds in data]
    matrices   = [[d['W']        for d in s]  for s in solutions]
    lambdas    = [[d['Lambda']   for d in s]  for s in solutions]
    matrices_n = [[d['W_normal'] for d in ds] for ds in data_normal]

    beh, beh_n = [], []
    for i in range(3):
        loaded = np.load(f'{data_root}/behaviours/behaviours{i}.npy', allow_pickle=True)
        beh.append(loaded.tolist())
        loaded = np.load(f'{data_root}/behaviours/behaviours_normal{i}.npy', allow_pickle=True)
        beh_n.append(loaded.tolist())

    return (*matrices, *matrices_n, *lambdas, *beh, *beh_n)


# =====================================================================
# Behaviour classification
# =====================================================================
def split_by_required_states(behaviours_list):
    """
    Partition matrices by whether their behavioural repertoire contains
    oscillations (periodic or quasi-periodic), a new fixed point, and the
    zero fixed point — split further by the presence of chaos.

    Parameters
    ----------
    behaviours_list : list of length n_mats
        Each element is a list of strings from:
        {'periodic', 'quasi-periodic', 'chaotic', 'new_fp', '0_fp'}

    Returns
    -------
    idx_with_chaos    : list of int — indices of qualifying matrices with chaos
    idx_without_chaos : list of int — indices of qualifying matrices without chaos
    """
    idx_with_chaos, idx_without_chaos = [], []
    for i, labels in enumerate(behaviours_list):
        s = set(labels)
        if s & OSC and 'new_fp' in s and '0_fp' in s:
            (idx_with_chaos if 'chaotic' in s else idx_without_chaos).append(i)
    return idx_with_chaos, idx_without_chaos


def classify_all(matrices0, matrices1, matrices2,
                 behaviours0, behaviours1, behaviours2):
    """
    Apply split_by_required_states to all three sigma conditions and
    return the filtered matrix and behaviour lists.

    Returns
    -------
    For each sigma (0, 1, 2):
        mat_with_chaos, mat_without_chaos,
        beh_with_chaos, beh_without_chaos
    Packaged as a list of 3 dicts, one per sigma.
    """
    result = []
    for mats, behs in zip([matrices0, matrices1, matrices2],
                          [behaviours0, behaviours1, behaviours2]):
        idx_chaos, idx_no_chaos = split_by_required_states(behs)
        result.append(dict(
            mat_with_chaos    = [mats[i] for i in idx_chaos],
            mat_without_chaos = [mats[i] for i in idx_no_chaos],
            beh_with_chaos    = [behs[i] for i in idx_chaos],
            beh_without_chaos = [behs[i] for i in idx_no_chaos],
            idx_with_chaos    = idx_chaos,
            idx_without_chaos = idx_no_chaos,
            n_with_chaos      = len(idx_chaos),
            n_without_chaos   = len(idx_no_chaos),
        ))
    return result
    
def recurrent_EI_currents(r_traj, W, nE, bias=None, include_bias=False):
    """
    Compute excitatory and inhibitory recurrent currents at each time step.

    Parameters
    ----------
    r_traj       : np.ndarray, shape (T, N)
    W            : np.ndarray, shape (N, N)
    nE           : int — number of excitatory neurons (first nE columns)
    include_bias : bool — add bias to excitatory current if True

    Returns
    -------
    I_exc : np.ndarray, shape (T, N)
    I_inh : np.ndarray, shape (T, N)
    """
    rE = r_traj[:, :nE]
    rI = r_traj[:, nE:]
    I_exc = rE @ W[:, :nE].T
    I_inh = rI @ W[:, nE:].T
    if include_bias:
        if bias is None:
            raise ValueError("include_bias=True requires bias.")
        I_exc = I_exc + bias[None, :]
    return I_exc, I_inh


def global_ratio_per_neuron(I_E, I_I, eps=1e-12, denom_mode="sum"):
    """
    Compute per-neuron E/I balance ratio averaged over time.

    Parameters
    ----------
    denom_mode : str
        'sum' — |IE|+|II| (cancellation efficiency)
        'I'   — |II| (net vs inhibitory)
        'E'   — |IE| (net vs excitatory)

    Returns
    -------
    ratio : np.ndarray, shape (N,)
    """
    Ires = I_E + I_I
    num = np.mean(np.abs(Ires), axis=0)
    if denom_mode == "sum":
        den = np.mean(np.abs(I_E) + np.abs(I_I), axis=0)
    elif denom_mode == "I":
        den = np.mean(np.abs(I_I), axis=0)
    elif denom_mode == "E":
        den = np.mean(np.abs(I_E), axis=0)
    else:
        raise ValueError("denom_mode must be 'sum', 'I', or 'E'")
    return num / (den + eps)


def sigma_ou_from_s_n(s_n, tau_c):
    """Convert stationary OU std s_n to diffusion coefficient sigma_ou."""
    return np.sqrt(2.0) * float(s_n) / np.sqrt(float(tau_c))


def plot_EI_trace_one_ic(tvec, I_exc, I_inh, k, ylim, start, end, dt=0.005):
    """
    Plot excitatory, inhibitory, and net current traces for one neuron.

    Parameters
    ----------
    k     : int   — neuron index
    ylim  : list  — [ymin, ymax]
    start, end : int — time step indices for the plot window
    """
    fig, ax = plt.subplots(figsize=(0.8, 0.5), dpi=500)
    ax.hlines(0, dt*start, dt*end, 'gray', lw=0.4, linestyle='--')
    ax.plot(tvec[start:end],  np.abs(I_exc[start:end, k]),              lw=0.5, color='firebrick',      alpha=0.8)
    ax.plot(tvec[start:end], -np.abs(I_inh[start:end, k]),              lw=0.5, color='cornflowerblue', alpha=0.7)
    ax.plot(tvec[start:end],  I_exc[start:end, k] + I_inh[start:end, k], 'k',  lw=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, pad=1, length=2, labelsize=5)
    ax.set_xticks([dt*start, dt*end]); ax.set_xticklabels(['', ''])
    ax.set_xlim([dt*start, dt*end])
    ax.set_yticks([ylim[0], 0, ylim[1]]); ax.set_yticklabels(['', '', ''])
    ax.set_ylim(ylim)
    plt.show()
    return fig, ax


    
   
