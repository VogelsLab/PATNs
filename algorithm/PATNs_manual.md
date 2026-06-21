# PATNs — parameter guide and algorithm overview

## Algorithm

PATNs (Periodic and Transient Networks) constructs a Dale-compliant recurrent connectivity matrix $W$ with a prescribed eigenspectrum. The core decomposition is

$$W = Q(\Lambda + U)Q^\top$$

where $Q$ is an orthogonal matrix, $\Lambda$ is a real block-diagonal Schur matrix encoding the target eigenvalues, and $U$ is a strictly upper-triangular matrix that carries the non-normality. Dale's law (excitatory/inhibitory sign constraints on columns of $W$) translates into a set of linear inequalities in the entries of $U$, which are solved via linear programming (LP). The LP objective minimises a weighted $\ell_1$ norm of $W$, where the weights can be set independently for three groups of entries via `l1_weights = (wd, we, wi)`: the diagonal entries (subject to the constraint that $\text{tr}(W) = \sum_i \lambda_i$, which is fixed, but the weights can be redistributed across neurons within this constraint), the excitatory off-diagonal entries, and the inhibitory off-diagonal entries.

This notebook uses an **improved version** of the algorithm described in the paper. The core steps are identical; what has changed is the construction of $Q$.

---

## Structure of Q — blocks and global modes

In the original algorithm $Q$ was a an orthogonal matrix with: 

- **Column 0 — uniform mode** (`'U'`): the vector $\mathbf{1}/\sqrt{N}$, shared across all neurons. Its eigenvalue is `lam0`.

Now $Q$ is built to have additional structure:

- **Columns 1 … P−1 — coarse modes** (`'C'`): one per block (block 0 has no coarse column — its slot is absorbed by the uniform). These live in the span of block-mean vectors and mediate coupling between blocks through the global subspace. Their eigenvalues are set via `coarse_vals` (length P−1).
- **Remaining columns — localized modes** (`'L'`): for each block $m$, there are `block_sizes[m] − 1` localized columns that are nonzero only on the neurons of that block. Their eigenvalues are sampled as described below.

The placement order of coarse vs. localized columns is controlled by `coarse_placement` (`'beginning'`, `'end'`, or `'per_block'`). All downstream indexing (eigenvalue assignment, LP mask) reads from the same `make_layout` function, so the order is consistent throughout.

### Functional subpopulations

Grouping neurons into blocks creates **functional subpopulations**. The blocks are specified by `block_sizes` (a list of integers summing to $N$, or a scalar for equal-sized blocks) and populated by `random_block_assignment`, which distributes inhibitory neurons proportionally (at least one per block).

Subpopulations are always coupled through the shared uniform and coarse modes. Beyond that, you can add **direct cross-block couplings** in the Schur basis via:

| Parameter | Effect |
|-----------|--------|
| `target_cross` | List of explicit block-index pairs `(a, b)` to connect directly |
| `random_cross` | Integer — number of randomly chosen cross-block couplings to add |
| `cross_reward` | Scales the LP incentive for cross-block entries (larger → stronger cross-block weights) |
| `maxw_cross` | Separate weight bound for targeted cross-block entries (defaults to `maxw`) |

The `coupling` parameter controls which Schur upper-triangular entries are free (non-zero bounds) vs. locked to zero:

- `'within'` — only within-block localized couplings are free
- `'between_via_global'` *(default)* — within-block, plus any coupling that touches a global/coarse column
- `'between_via_cross'` — within-block, plus direct localized cross-block couplings
- `'between_via_global+cross'` — all of the above

---

## Eigenvalue configuration

### Global and coarse modes
- `lam0` — eigenvalue of the uniform mode; usually set to `0.0`.
- `coarse_vals` — array of length P−1, one eigenvalue per coarse mode. Setting `coarse_vals` to zero maximises non-normality by forcing the coarse eigenvectors to be more aligned, making the eigenvector matrix more ill-conditioned. Sampling them from a distribution instead reduces non-normality.

### Localized modes — the "spear" configuration

Each block of size $s_m$ contributes $s_m - 1$ localized eigenvalues, split into:
- `n_real_per_block[m]` real eigenvalues
- $(s_m - 1 - \texttt{n\_real})~/~2$ complex-conjugate pairs (so the remainder must be even)

All localized real parts are drawn from a truncated normal with mean `mu_r` and std `sigma_r`. The real eigenvalues are sampled from a distribution shifted by `real_spear = 0.02` relative to the complex ones, placing them slightly closer to the imaginary axis.

**To reliably obtain self-sustained oscillations, use a "spear" configuration**: set `n_real_per_block ≥ 1` for every block. The real eigenvalue sits just ahead of the pack of complex eigenvalues in real part (the tip of the spear), acting as a slow integrating mode that accumulates activity and drives the network into oscillation through the interaction of non-normality and the nonlinearity. The complex eigenvalues follow close behind, providing the oscillatory structure. This mechanism relies on the nonlinearity — the linearised system is stable by construction.

```python
n_real_per_block = [2] * 10   # 2 real eigenvalues per block — recommended starting point
```

### Connectivity strength

- `maxw` — upper bound on the absolute value of all LP-free entries in $U$. This is the primary dial for recurrence strength. **Recommended range: 2.5 – 5**, depending on other parameters. Too weak and the network decays; too strong and it saturates or becomes chaotic.

### LP objective weights

`l1_weights = (wd, we, wi)` sets relative weights on diagonal, excitatory off-diagonal, and inhibitory off-diagonal entries in the $\ell_1$ objective:

```python
l1_weights = (1, 0, 0)   # penalise only diagonal entries
```
---

## Workflow

```
find_solution(...)
    ├── random_block_assignment   # assign neurons to blocks, distribute inhibitory cells
    ├── sample_Q                  # build structured orthogonal basis
    ├── sample_eigenvalues        # draw eigenvalues, assemble Lambda_Schur
    ├── compute_M                 # encode W = Q(Lambda+U)Q^T as a linear system in u
    ├── calculate_u               # solve LP for upper-triangular entries of U
    └── create_W                  # reconstruct W = Q(Lambda+U)Q^T, plot U
```

`find_solution` returns a dict with:
- `W` — the $N \times N$ connectivity matrix
- `Q` — the orthogonal Schur basis
- `Lambda` — the prescribed complex eigenvalues
- `big_u` — the upper-triangular non-normality matrix $U$
- `info` — block structure metadata (`blocks`, `sizes`, `P`, `coarse_placement`)
- `sol`, `status` — raw LP solution and solver message

---

## Dynamics

The network is simulated using a nonlinear rate model (spring-box / sigmoid transfer function, RK4 integration). Initial conditions are generated from `init_cond(J)`, which seeds perturbations along the leading directions of the linearised connectivity $J$.

**Full-network dynamics** — one initial condition applied to all $N$ neurons simultaneously:

```python
initial_conditions = init_cond(J)
# columns of initial_conditions are candidate ICs for the full network
t, r = solve_rk4_springBox(0.5 * initial_conditions[:, i], W, ...)
```

**Block-localised dynamics** — initial condition restricted to neurons within a single subpopulation, with all other neurons starting at rest. This probes whether a subpopulation can independently sustain oscillations, and how the activity then spreads:

```python
block_ics = block_initial_conditions_from_init_cond(result, scale=0.5)
# block_ics[m] has shape (N, n_ics): nonzero only at neurons of block m
t, r = solve_rk4_springBox(block_ics[m][:, i], W, ...)
```
