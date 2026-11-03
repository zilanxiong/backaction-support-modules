"""Exact Fock track in the frame that moves with the shear (shear interaction picture).

The sensing-stage shear ``U_BA(t) = exp(-i (κ/2) x² t)`` is a signal-independent
unitary.  Writing ``ρ(t) = U_BA(t) ρ_I(t) U_BA(t)†``, the master equation for
``ρ_I`` has

* Hamiltonian ``ε_a (a + a†)`` only — the signal generator ``x`` commutes with ``x²``;
* a *time-dependent* loss operator ``√γ a_H(t)`` with
  ``a_H(t) = U_BA(t)† a U_BA(t) = a - i (κ t / 2)(a + a†)``  (since ``p → p - κ t x``);
* output loss (stage 3, after the full shear) with the constant operator
  ``√γ_out a_H(1)``.

Nothing stretches ``ρ_I``: the only effect of the shear is to reshape the noise
that loss injects, and for a few per cent loss that is a small addition to the
input state's own extent.  So the cutoff stays near the input's (~100-300) for
any ``κ``, where the plain density-matrix track would need ~1000 at ``κ ≈ 10``.

The final ``U_BA(1)`` is independent of ``ε_a`` and drops out of the QFI, so
``QFI(ρ_out) = QFI(ρ_I(1))``.  Input loss and input phase noise act before the
shear and are unchanged.  Channel/output *dephasing* is not supported.
"""

import numpy as np
import qutip

from quantum_sensing.conversions import loss_to_kappa, phirms_to_chi
from quantum_sensing.sld import calculate_qfi

from .solver import SOLVER_OPTIONS

__all__ = ["unsheared_channel", "rp_qfi_unsheared", "rp_qfi_unsheared_exact", "qfi_from_rho_drho"]


def unsheared_channel(*, kappa_ba=0.0, epsilon_a=0.0, eta_in=1.0, pn_in=0.0, eta_ch=1.0,
                      eta_out=1.0, N_basis=20, rho=None, **_ignored):
    """ρ_I(1): the package's three-stage channel, in the frame moving with the shear.

    Same keyword names as ``quantum_sensing.get_state_single_mode_rp`` so it can be
    handed to ``quantum_sensing.sld.calculate_qfi`` as the ``dynamics``.
    """
    N = int(N_basis)
    a = qutip.destroy(N)
    n = a.dag() * a
    D = a + a.dag()
    opts = dict(SOLVER_OPTIONS)
    rho = qutip.ket2dm(rho) if rho.isket else rho

    # stage 1: input loss + input dephasing (before the shear: unchanged)
    c1 = []
    if eta_in < 1.0:
        c1.append(np.sqrt(loss_to_kappa(1 - eta_in)) * a)
    if pn_in > 0.0:
        c1.append(np.sqrt(phirms_to_chi(pn_in)) * n)
    if c1:
        rho = qutip.mesolve(0 * n, rho, [0.0, 1.0], c1, options=opts).states[-1]

    # stage 2: signal Hamiltonian; channel loss through the time-dependent a_H(t)
    H = epsilon_a * D
    c2 = []
    if eta_ch < 1.0:
        g = np.sqrt(loss_to_kappa(1 - eta_ch))
        c2.append(qutip.QobjEvo([[g * a, lambda t: 1.0], [-1j * g * (kappa_ba / 2) * D, lambda t: t]]))
    rho = qutip.mesolve(H, rho, [0.0, 1.0], c2, options=opts).states[-1]

    # stage 3: output loss after the full shear: constant a_H(1)
    if eta_out < 1.0:
        g = np.sqrt(loss_to_kappa(1 - eta_out))
        L = g * (a - 1j * (kappa_ba / 2) * D)
        rho = qutip.mesolve(0 * n, rho, [0.0, 1.0], [L], options=opts).states[-1]
    return rho


def rp_qfi_unsheared(rho, *, param_value=0.0, **channel_kwargs):
    """QFI for ``epsilon_a`` from the unsheared-frame channel (same signature as ``rp_qfi``).

    The input-noise stage does not depend on the signal, so it is run once and the
    finite-difference evaluations start from its output.
    """
    N = int(channel_kwargs.pop("N_basis", rho.shape[0]))
    eta_in, pn_in = channel_kwargs.pop("eta_in", 1.0), channel_kwargs.pop("pn_in", 0.0)
    rho1 = unsheared_channel(rho=rho, N_basis=N, eta_in=eta_in, pn_in=pn_in, kappa_ba=0.0)   # stage 1 only
    return float(calculate_qfi(unsheared_channel, param_value=param_value, param_type="epsilon_a",
                               rho=rho1, N_basis=N, **channel_kwargs))


# --------------------------------------------------------------------------------------
# Exact derivative: integrate (ρ, ∂ρ/∂ε) together (sensitivity equation), no finite step
# --------------------------------------------------------------------------------------

def qfi_from_rho_drho(rho, drho, eig_tol=1e-8):
    """QFI from a density matrix and its exact parameter derivative (same formula and
    eigenvalue cutoff as ``quantum_sensing.sld.calculate_qfi``)."""
    lam, vec = np.linalg.eigh(rho)
    d = vec.conj().T @ drho @ vec
    denom = lam[:, None] + lam[None, :]
    mask = denom > eig_tol
    return float(np.sum(2 * np.abs(d[mask]) ** 2 / denom[mask]))


def _superops(N, kappa_ba, eta_ch, eta_out):
    """Sparse superoperators of the shear frame: stage-2 Lindbladian coefficients of
    1, t, t² (channel loss through a_H(t) = a - i(κt/2)(a+a†)), the signal generator
    G = -i[a+a†, ·], and the stage-3 Lindbladian (output loss through a_H(1))."""
    a = qutip.destroy(N)
    D = a + a.dag()
    G = -1j * (qutip.spre(D) - qutip.spost(D))
    zero = qutip.spre(0 * a)
    L0 = L1 = L2 = zero
    if eta_ch < 1.0:
        g = np.sqrt(loss_to_kappa(1 - eta_ch))
        c0, c1 = g * a, -1j * g * (kappa_ba / 2) * D          # a_H(t) = (c0 + t c1)/g
        L0 = qutip.lindblad_dissipator(c0)
        L1 = qutip.lindblad_dissipator(c0, c1) + qutip.lindblad_dissipator(c1, c0)
        L2 = qutip.lindblad_dissipator(c1)
    L3 = zero
    if eta_out < 1.0:
        g = np.sqrt(loss_to_kappa(1 - eta_out))
        L3 = qutip.lindblad_dissipator(g * (a - 1j * (kappa_ba / 2) * D))
    return [x.data.as_scipy().tocsr() for x in (L0, L1, L2, G, L3)]


def rp_qfi_unsheared_exact(rho, *, kappa_ba=0.0, eta_in=1.0, pn_in=0.0, eta_ch=1.0, eta_out=1.0,
                           N_basis=None, rtol=1e-10, atol=1e-12, eig_tol=1e-8):
    """QFI for ``epsilon_a`` at ``epsilon_a = 0`` with the derivative taken exactly.

    Stage 1 (input noise) as in :func:`unsheared_channel`.  Stage 2 integrates the pair
    ``(ρ, σ = ∂ρ/∂ε)`` under ``dρ/dt = L(t)ρ``, ``dσ/dt = L(t)σ + G ρ`` with
    ``G ρ = -i[a + a†, ρ]`` (the signal Hamiltonian is ``ε (a + a†)`` and the operating
    point is ``ε = 0``, so ``L(t)`` carries no Hamiltonian).  Stage 3 propagates both
    under the output-loss Lindbladian.  ``scipy.integrate.solve_ivp`` (DOP853) with the
    given tolerances; no finite-difference step anywhere.
    """
    from scipy.integrate import solve_ivp
    N = int(N_basis or rho.shape[0])
    if rho.shape[0] != N:
        v = np.zeros(N, dtype=complex); n = min(N, rho.shape[0]); v[:n] = rho.full().ravel()[:n]
        rho = qutip.Qobj(v).unit()
    rho1 = unsheared_channel(rho=rho, N_basis=N, eta_in=eta_in, pn_in=pn_in, kappa_ba=0.0)
    L0, L1, L2, G, L3 = _superops(N, kappa_ba, eta_ch, eta_out)
    n2 = N * N
    y0 = np.concatenate([rho1.full().reshape(-1, order="F"), np.zeros(n2, dtype=complex)])

    def rhs2(t, y):
        r, s = y[:n2], y[n2:]
        Lt = lambda v: L0 @ v + t * (L1 @ v) + t * t * (L2 @ v)
        return np.concatenate([Lt(r), Lt(s) + G @ r])

    sol = solve_ivp(rhs2, (0.0, 1.0), y0, method="DOP853", rtol=rtol, atol=atol)
    y = sol.y[:, -1]
    if eta_out < 1.0:
        def rhs3(t, y):
            return np.concatenate([L3 @ y[:n2], L3 @ y[n2:]])
        y = solve_ivp(rhs3, (0.0, 1.0), y, method="DOP853", rtol=rtol, atol=atol).y[:, -1]
    rho_out = y[:n2].reshape(N, N, order="F")
    drho = y[n2:].reshape(N, N, order="F")
    rho_out = 0.5 * (rho_out + rho_out.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    return qfi_from_rho_drho(rho_out, drho, eig_tol=eig_tol)
