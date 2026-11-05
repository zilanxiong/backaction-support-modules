# backaction-support-modules

Support code for Jupyter-notebook studies of the quantum Fisher information
(QFI) of optical probe states under radiation-pressure back-action. This is the
lean subset of `zilanxiong/ligo-backaction-qfi`: the modules a notebook needs,
their tests, and nothing else (no sweep scripts, results or figures).

It builds on the group's `quantum_sensing` package (`shruti-jm/quantum-sensing`),
which is **not** included here and must be installed in the same environment
(see *Set up*). `quantum_sensing.rp_kappa(Omega)` is the kappa_ba(Omega) used throughout.

| path | what it is |
|---|---|
| `backaction_qfi/probes.py` | fixed probes at a given cutoff and photon number: `coherent`, `squeezed`, `cat`, `squeezed_cat`, `fock`, `make_probe`; `mean_photon_number`, `var_x` |
| `backaction_qfi/optimised.py` | `load_probe(state_type, loss_config, eta, pn, N_target, N_basis)`: the group's optimised states from `data/optimised_states/`, rotated by a quarter turn onto the GW signal (`epsilon_a`) |
| `backaction_qfi/solver.py` | the package channel with tight ODE tolerances (`rp_channel`, `rp_qfi`) and the lossless pure-state track (`rp_qfi_pure`) |
| `backaction_qfi/frame.py` | the channel in the frame moving with the shear: exact at any kappa with ~200 levels. `rp_qfi_unsheared_exact(rho, kappa_ba=..., eta_in=..., pn_in=..., eta_ch=..., eta_out=..., N_basis=...)` is the primary QFI routine (derivative via the sensitivity equation, no finite-difference step) |
| `backaction_qfi/gaussian.py` | covariance-matrix track, exact for Gaussian probes: `gaussian_qfi`, optimal-angle (FD) squeezing, SQL, strain conversion |
| `backaction_qfi/labframe.py` | the package channel integrated **in the lab frame** on NumPy (`device="cpu"`) or CuPy (`device="gpu"`): the Python counterpart of the group's Julia CUDA extension. Same three stages, same parameter conversions, same QFI formula; the derivative is integrated alongside the state (no finite difference) with a DOP853 integrator that applies `a`, `a†` as shifts, so no `N²×N²` generator is formed. `rp_qfi_labframe(psi, kappa_ba=..., eta_in=..., pn_in=..., eta_ch=..., eta_out=..., device="gpu")`. Lab-frame cutoffs grow with kappa (hundreds of levels at kappa ~ 1–3, over a thousand at 10 Hz); on a GPU that is minutes per point |
| `backaction_qfi/convergence.py` | Fock-cutoff ladder (`converged_qfi`, `converge`) and the truncation alarm (`tail_population`) |
| `data/optimised_states/` | the group's optimised probes (`states.h5.gz`, unpacked on first use) and the orientation notes |
| `tests/` | package import smoke tests, the convergence suite, the Gaussian benchmark suite, the optimised-state loader checks |

## Set up

In the environment that already has `quantum_sensing` (for example the one with
`ligo-backaction-qfi` installed), from this folder:

```bash
pip install -r requirements.txt
pytest                      # a few minutes; passing means both packages are importable
python -m jupyter lab       # the default kernel is this environment
```

From scratch, `conda env create -f environment.yml` makes the environment, and
`quantum_sensing` is then installed by one of the two lines in `requirements.txt`'s comment.

## In a notebook

```python
import numpy as np, qutip as qt
from quantum_sensing import rp_kappa
from backaction_qfi import make_probe, load_probe, rp_qfi_unsheared_exact, converged_qfi

kappa = rp_kappa(2 * np.pi * 30.0)                     # kappa_ba at 30 Hz (~1.05)
psi = make_probe("squeezed", 120, 5)                    # p-squeezed vacuum, <n> = 5, 120 levels
rho = psi * psi.dag()
F = rp_qfi_unsheared_exact(rho, kappa_ba=kappa, eta_in=0.95, pn_in=0.2,
                           eta_ch=0.99, eta_out=0.90, N_basis=120)
opt, entry = load_probe("fock_sup", "loss_ch", 0.85, 0.2, N_target=5, N_basis=120)  # ket, stored metadata
```

Conventions: hbar = 1, x = (a + a^dagger)/sqrt(2), unit stage duration, GW signal on
`epsilon_a` (generator x). Lossless QFI = 8 Var(x), independent of kappa.
