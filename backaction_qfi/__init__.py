"""Support modules for QFI studies under radiation-pressure back-action.

Built on top of the group's ``quantum_sensing`` package (vendored unmodified):
``probes`` (fixed probe states), ``optimised`` (the group's optimised states,
rotated onto the GW signal), ``solver`` (the package channel with tight ODE
tolerances), ``frame`` (moving-frame solver, exact at any kappa, with the
exact-derivative QFI), ``gaussian`` (covariance-matrix track), ``convergence``
(Fock-cutoff ladder).  kappa_ba(Omega) itself is ``quantum_sensing.rp_kappa``.
"""

from .solver import SOLVER_OPTIONS, tight_solver, rp_channel, rp_qfi, rp_qfi_pure
from .probes import (coherent, squeezed, cat_even, cat, squeezed_cat, fock,
                     mean_photon_number, var_x, FAMILIES, make_probe)
from .gaussian import gaussian_channel, gaussian_qfi, homodyne_noise_spectrum, fd_squeeze_angle, strain_noise_from_qfi
from .optimised import load_probe, quarter_turn, rotate
from .frame import unsheared_channel, rp_qfi_unsheared, rp_qfi_unsheared_exact
from .convergence import (
    ConvergenceResult, converge, converged_qfi, tail_population, suggested_cutoff,
)

__all__ = [
    "SOLVER_OPTIONS", "tight_solver", "rp_channel", "rp_qfi", "rp_qfi_pure",
    "coherent", "squeezed", "cat_even", "cat", "squeezed_cat", "fock", "mean_photon_number", "var_x",
    "FAMILIES", "make_probe",
    "ConvergenceResult", "converge", "converged_qfi", "tail_population", "suggested_cutoff",
    "gaussian_channel", "gaussian_qfi", "homodyne_noise_spectrum", "fd_squeeze_angle", "strain_noise_from_qfi",
    "load_probe", "quarter_turn", "rotate", "unsheared_channel", "rp_qfi_unsheared", "rp_qfi_unsheared_exact",
]
