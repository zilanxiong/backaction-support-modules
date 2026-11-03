"""Probe states at a given Fock cutoff.

Every builder takes the cutoff ``N`` first, so ``lambda N: squeezed(N, 2.0)``
is a valid ``state_builder`` for :func:`backaction_qfi.convergence.converged_qfi`.

Quadrature convention (matches ``quantum_sensing.dynamics``):
``x = (a + a†)/√2``, ``p = -i(a - a†)/√2``, vacuum ``Var(x) = Var(p) = 1/2``.
The GW-frame signal ``epsilon_a (a + a†)`` has generator ``√2 x``, so for a
pure probe through a lossless channel the QFI is ``8 Var(x)``: the useful
squeezing for this signal is *p-squeezed* (``x`` anti-squeezed).
"""

import numpy as np
import qutip as qt
from scipy.optimize import brentq

__all__ = ["coherent", "squeezed", "cat_even", "cat", "squeezed_cat", "fock",
           "mean_photon_number", "var_x", "FAMILIES", "make_probe"]


def coherent(N, nbar):
    """Coherent state with mean photon number ``nbar`` (real amplitude)."""
    return qt.coherent(N, np.sqrt(nbar))


def squeezed(N, nbar, squeeze_p=True):
    """Squeezed vacuum with mean photon number ``nbar = sinh²r``.

    ``squeeze_p=True`` (default) squeezes ``p`` and anti-squeezes ``x``: the
    optimal fixed-``nbar`` Gaussian probe for the ``epsilon_a`` signal.
    """
    r = np.arcsinh(np.sqrt(nbar))
    # qutip.squeeze(N, z) = exp((z* a² - z a†²)/2): real z > 0 squeezes x.
    z = -r if squeeze_p else r
    return (qt.squeeze(N, z) * qt.fock(N, 0)).unit()


def cat_even(N, alpha):
    """Even cat ``|α⟩ + |-α⟩`` (normalised); mean photon number ``|α|² tanh|α|²``."""
    return (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()


def _solve_alpha(ket_of_alpha, N, nbar):
    """Real alpha > 0 such that ``ket_of_alpha(alpha)`` has mean photon number ``nbar``
    (root-finding on the numerically evaluated ⟨n⟩, so the cutoff is accounted for)."""
    f = lambda a: mean_photon_number(ket_of_alpha(a)) - nbar
    lo, hi = 1e-6, 1.0
    if f(lo) > 0:
        return lo
    while f(hi) < 0:
        hi *= 1.5
        if hi > np.sqrt(N):
            raise ValueError(f"cannot reach nbar={nbar} at cutoff N={N}")
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-13)


def cat(N, nbar):
    """Even cat with mean photon number ``nbar`` (real alpha: peaks separated along ``x``)."""
    if nbar <= 0:
        return qt.fock(N, 0)
    return cat_even(N, _solve_alpha(lambda a: cat_even(N, a), N, nbar))


def squeezed_cat(N, nbar, squeeze_fraction=0.5):
    """Squeezed cat ``S(-r)(|α⟩ + |-α⟩)`` with ``sinh² r = squeeze_fraction · nbar`` and
    ``alpha`` solved so the total mean photon number is ``nbar``.  The squeezing
    anti-squeezes ``x`` (same direction as :func:`squeezed`).  Same convention as
    the earlier study."""
    if not 0.0 <= squeeze_fraction < 1.0:
        raise ValueError("squeeze_fraction must lie in [0, 1)")
    if nbar <= 0:
        return qt.fock(N, 0)
    r = np.arcsinh(np.sqrt(squeeze_fraction * nbar))
    S = qt.squeeze(N, -r)
    ket = lambda a: (S * cat_even(N, a)).unit()
    return ket(_solve_alpha(ket, N, nbar))


def fock(N, n):
    """Number state ``|n⟩``."""
    return qt.fock(N, n)


def _fock_at_nbar(N, nbar):
    if abs(nbar - round(nbar)) > 1e-9:
        raise ValueError("a Fock probe needs an integer photon number")
    return qt.fock(N, int(round(nbar)))


#: Probe families by name: ``FAMILIES[name](N, nbar)`` -> ket.
FAMILIES = {
    "coherent": coherent,
    "squeezed": lambda N, nbar: squeezed(N, nbar),
    "cat": cat,
    "squeezed_cat": lambda N, nbar: squeezed_cat(N, nbar, 0.5),
    "fock": _fock_at_nbar,
}

LABELS = {
    "coherent": "coherent", "squeezed": "squeezed vacuum", "cat": "even cat",
    "squeezed_cat": "squeezed cat (½ budget in squeezing)", "fock": "Fock",
}


def make_probe(name, N, nbar):
    """Build a probe by family name at cutoff ``N`` and mean photon number ``nbar``."""
    try:
        return FAMILIES[name](N, nbar)
    except KeyError:
        raise ValueError(f"unknown family {name!r}; choose from {sorted(FAMILIES)}") from None


def mean_photon_number(state):
    N = state.shape[0]
    return float(np.real(qt.expect(qt.num(N), state)))


def var_x(state):
    """Variance of ``x = (a + a†)/√2`` (vacuum: 1/2)."""
    N = state.shape[0]
    a = qt.destroy(N)
    x = (a + a.dag()) / np.sqrt(2)
    return float(np.real(qt.expect(x * x, state) - qt.expect(x, state) ** 2))
