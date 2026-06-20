import numpy as np

def create_miniSOCs(max_wt, seed, max_solns):
    """
    Generate random 2x2 Dalean, stable matrices.
    
    Parameters
    ----------
    max_wt : float
        Maximum absolute weight value.
    seed : int
        Random seed for reproducibility.
    max_solns : int
        Number of random candidates to sample before filtering.
    
    Returns
    -------
    np.ndarray of shape (N, 4)
        Each row [a, b, c, d] is a flattened 2x2 Dale-compliant stable matrix,
        with b <= 0, d <= 0 (inhibitory entries).
    """
    np.random.seed(seed)
    all = np.random.rand(int(max_solns), 4)
    all[:, 1] *= -1
    all[:, 3] *= -1
    all *= max_wt

    cnd1 = np.where(all[:, 0] + all[:, 3] < 2)
    all1 = all[cnd1]
    cnd2 = np.where(((all1[:, 0]*all1[:, 3])
                     - (all1[:, 1]*all1[:, 2])
                     - (all1[:, 0] + all1[:, 3])) > -1)
    all2 = all1[cnd2]
    cnd3 = np.where(((all2[:, 0]*all2[:, 3])
                     - (all2[:, 1]*all2[:, 2])
                     - (all2[:, 0] + all2[:, 3])
                     - (0.25*(all2[:, 1]-all2[:, 2])**2)) < -1)
    all3 = all2[cnd3]

    return all3
