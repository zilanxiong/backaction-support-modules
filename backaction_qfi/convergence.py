"""Automated Fock-cutoff convergence.

Radiation pressure is a shear: it stretches the ``p`` quadrature by about
``kappa_ba`` and inflates the output photon number by roughly
``(1 + kappa_ba²)(nbar + 1)``.  A cutoff that is adequate for the *input*
state, including the package default ``N_basis=20``, can be badly inadequate
for the *output*, and non-Gaussian probes are the worst offenders.  Every
user-facing back-action number should be produced through :func:`converge`
or :func:`converged_qfi`.
"""

from dataclasses import dataclass, field

import numpy as np

from .solver import rp_qfi

__all__ = ["ConvergenceResult", "converge", "converged_qfi", "tail_population",
           "suggested_cutoff"]


@dataclass
class ConvergenceResult:
    """Outcome of a cutoff ladder.  ``converged=False`` is a hard warning."""

    value: float
    cutoff: int
    converged: bool
    rel_change: float
    history: list = field(default_factory=list)   # [(cutoff, value), ...]

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        flag = "converged" if self.converged else "NOT CONVERGED"
        return (f"ConvergenceResult(value={self.value:.8g}, cutoff={self.cutoff}, "
                f"rel_change={self.rel_change:.2g}, {flag})")


def default_ladder(start=20, stop=600, factor=1.4):
    """Geometric ladder of cutoffs from ``start`` up to and including ``stop``."""
    ladder, n = [], int(start)
    while n < stop:
        ladder.append(n)
        n = int(np.ceil(n * factor))
    ladder.append(int(stop))
    return ladder


def converge(fn, cutoffs=None, rtol=1e-3, atol=0.0, n_stable=1):
    """Evaluate ``fn(N_basis)`` on an increasing ladder until it stops moving.

    Convergence at rung ``k`` means ``|v_k - v_{k-1}| <= atol + rtol |v_k|``
    for ``n_stable`` consecutive rungs.  If the ladder runs out first the
    result carries ``converged=False`` and the last value.
    """
    cutoffs = list(cutoffs) if cutoffs is not None else default_ladder()
    if len(cutoffs) < 2:
        raise ValueError("need at least two cutoffs to judge convergence")
    history, stable, rel = [], 0, np.inf
    prev = None
    for N in cutoffs:
        v = float(fn(int(N)))
        history.append((int(N), v))
        if prev is not None:
            diff = abs(v - prev)
            rel = diff / abs(v) if v != 0 else diff
            stable = stable + 1 if diff <= atol + rtol * abs(v) else 0
            if stable >= n_stable:
                return ConvergenceResult(v, int(N), True, rel, history)
        prev = v
    return ConvergenceResult(prev, int(cutoffs[-1]), False, rel, history)


def converged_qfi(state_builder, cutoffs=None, rtol=1e-3, n_stable=1,
                  param_type="epsilon_a", param_value=0.0, **channel_kwargs):
    """Convergence-checked QFI through the back-action channel.

    ``state_builder(N)`` returns the probe at cutoff ``N``; ``channel_kwargs``
    (``kappa_ba``, ``eta_in``, ``eta_ch``, ``eta_out``, ...) go to
    :func:`backaction_qfi.solver.rp_channel`.
    """
    def value(N):
        return rp_qfi(state_builder(N), param_type=param_type,
                      param_value=param_value, N_basis=N, **channel_kwargs)
    return converge(value, cutoffs=cutoffs, rtol=rtol, n_stable=n_stable)


def tail_population(state, n_levels=5):
    """Population in the top ``n_levels`` Fock levels: a cheap truncation alarm."""
    rho = state.full()
    if state.isket:
        rho = rho @ rho.conj().T
    return float(np.real(np.trace(rho[-n_levels:, -n_levels:])))


def suggested_cutoff(nbar, kappa_ba, safety=8.0):
    """Rule-of-thumb first rung: ``safety (1 + kappa_ba²)(nbar + 1)``.

    A starting point for :func:`converge`, never a substitute for it.
    """
    return int(np.ceil(safety * (1.0 + float(kappa_ba) ** 2) * (float(nbar) + 1.0)))
