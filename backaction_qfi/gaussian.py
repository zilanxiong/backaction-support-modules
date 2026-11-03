r"""Gaussian (covariance-matrix) track: exact for Gaussian probes, closed-form fast.

A parallel implementation of the package's single-mode back-action channel at
the level of first and second moments.  It is exact whenever the probe is
Gaussian and the channel is Gaussian (loss, shear, displacement), and it is
used (i) as an analytic cross-check of the Fock-space track and (ii) to
reproduce the textbook free-mass curves: the standard quantum limit (SQL),
variational readout and frequency-dependent (FD) squeezing.

Conventions (identical to ``quantum_sensing.dynamics``)
-------------------------------------------------------
``x = (a + a†)/√2``, ``p = -i(a - a†)/√2``, ``[x, p] = i``, vacuum ``V = ½·1``.
Phase-space vector ``(x, p)``.  Over the package's unit sensing time:

* back-action  ``H_BA = (kappa_ba/2) x²``       →  ``p → p - kappa_ba·x``   (shear ``S``)
* signal       ``H = epsilon_a (a + a†) = √2 ε_a x``  →  ``p → p - √2 ε_a``
* signal       ``H = i ε_p (a† - a) = √2 ε_p p``     →  ``x → x + √2 ε_p``
* loss with transmissivity ``η``: ``d → √η d``,  ``V → η V + (1-η) V_vac``

The QFI for a parameter that enters only the mean is
``F = (∂d)ᵀ V⁻¹ (∂d)``, so the vacuum gives ``F_{ε_a} = 2 · (1/2)⁻¹ = 4``,
matching ``backaction_qfi.rp_qfi``.

Relation to KLMTV (PRD 65, 022002): their two-photon quadratures are
``a_1 = √2 x``, ``a_2 = √2 p`` and the lossless input–output relation is
``b_1 = a_1``, ``b_2 = a_2 - κ a_1 + √(2κ) h/h_SQL``.  Hence the signal
displacement is ``√2 ε_a = √κ h/h_SQL`` and the strain Cramér–Rao bound is
``σ_h²/h_SQL² = 2 / (κ F_{ε_a})`` (:func:`strain_noise_from_qfi`).
"""

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "V_VAC", "GaussianState", "vacuum", "coherent_gaussian", "squeezed_gaussian",
    "squeezed_at_fixed_nbar", "shear_matrix", "rotation_matrix", "squeeze_matrix",
    "shear_to_squeeze_rotation", "apply_symplectic", "apply_loss",
    "gaussian_channel", "gaussian_qfi", "optimal_squeeze_angle",
    "fd_squeeze_angle", "homodyne_noise_spectrum", "strain_noise_from_qfi", "h_sql",
]

V_VAC = 0.5 * np.eye(2)
SQRT2 = np.sqrt(2.0)


# --- states ------------------------------------------------------------------

@dataclass
class GaussianState:
    """First and second moments, plus ``dmean = ∂(mean)/∂(parameter)``."""

    mean: np.ndarray
    cov: np.ndarray
    dmean: np.ndarray = None

    def __post_init__(self):
        self.mean = np.asarray(self.mean, float).reshape(2)
        self.cov = np.asarray(self.cov, float).reshape(2, 2)
        self.dmean = np.zeros(2) if self.dmean is None else np.asarray(self.dmean, float).reshape(2)

    def copy(self):
        return replace(self, mean=self.mean.copy(), cov=self.cov.copy(), dmean=self.dmean.copy())

    @property
    def nbar(self):
        """Mean photon number ``½(tr V - 1) + ½|d|²``."""
        return 0.5 * (np.trace(self.cov) - 1.0) + 0.5 * float(self.mean @ self.mean)

    @property
    def var_x(self):
        return float(self.cov[0, 0])


def vacuum():
    return GaussianState(np.zeros(2), V_VAC.copy())


def coherent_gaussian(alpha):
    """Coherent state ``|α⟩``: mean ``(√2 Re α, √2 Im α)``, vacuum covariance."""
    return GaussianState(SQRT2 * np.array([np.real(alpha), np.imag(alpha)]), V_VAC.copy())


def squeezed_gaussian(r, theta=0.0):
    """Squeezed vacuum with ``Var(x) = e^{2r}/2`` at ``theta = 0``, rotated by ``theta``.

    ``r > 0`` anti-squeezes ``x`` (squeezes ``p``), which raises the QFI for
    the ``epsilon_a`` signal.  Same state as ``backaction_qfi.probes.squeezed``.
    """
    v = 0.5 * np.diag([np.exp(2 * r), np.exp(-2 * r)])
    R = rotation_matrix(theta)
    return GaussianState(np.zeros(2), R @ v @ R.T)


def squeezed_at_fixed_nbar(nbar, theta=0.0):
    """Squeezed vacuum with ``sinh² r = nbar``, rotated by ``theta``."""
    return squeezed_gaussian(np.arcsinh(np.sqrt(max(nbar, 0.0))), theta)


# --- symplectic building blocks ------------------------------------------------

def shear_matrix(kappa):
    """Ponderomotive shear ``[[1, 0], [-κ, 1]]``: ``p → p - κ x``."""
    return np.array([[1.0, 0.0], [-float(kappa), 1.0]])


def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def squeeze_matrix(r):
    return np.diag([np.exp(r), np.exp(-r)])


def shear_to_squeeze_rotation(kappa):
    """``S(κ) = R(θ_post) diag(e^r, e^-r) R(θ_pre)`` with ``e^r - e^-r = κ``.

    The "squeezing plus rotation" form of the ponderomotive response.
    Returns ``(r, theta_pre, theta_post)``.
    """
    u, sv, vh = np.linalg.svd(shear_matrix(kappa))
    if np.linalg.det(u) < 0:          # make both factors proper rotations
        u[:, 1] *= -1
        vh[1, :] *= -1
    return float(np.log(sv[0])), float(np.arctan2(vh[1, 0], vh[0, 0])), float(np.arctan2(u[1, 0], u[0, 0]))


def apply_symplectic(state, S):
    return GaussianState(S @ state.mean, S @ state.cov @ S.T, S @ state.dmean)


def apply_loss(state, eta):
    """Beam-splitter loss with power transmissivity ``eta``."""
    eta = float(eta)
    return GaussianState(np.sqrt(eta) * state.mean,
                         eta * state.cov + (1.0 - eta) * V_VAC,
                         np.sqrt(eta) * state.dmean)


# --- the channel -----------------------------------------------------------------

def _sensing_stage(state, kappa_ba, epsilon_a, epsilon_p, eta_ch, param):
    """Stage 2 of the package channel, exactly: shear + signal + *concurrent* loss.

    Heisenberg/Lindblad moment equations with drift ``M = A - (γ/2)·1``,
    ``A = [[0,0],[-κ,0]]`` (nilpotent), ``γ = -ln η_ch``, constant force
    ``b = (√2 ε_p, -√2 ε_a)``, diffusion ``γ V_vac``, over unit time:

        d(t) = e^{M} d(0) + ∫₀¹ e^{M(1-s)} b ds
        V(1) = e^{M} V(0) e^{Mᵀ} + γ ∫₀¹ e^{Ms} V_vac e^{Mᵀ s} ds

    with ``e^{Mt} = e^{-γt/2} (1 + A t)``.  All integrals are closed-form.
    """
    kappa = float(kappa_ba)
    A = np.array([[0.0, 0.0], [-kappa, 0.0]])
    b = SQRT2 * np.array([epsilon_p, -epsilon_a])
    db = SQRT2 * (np.array([0.0, -1.0]) if param == "epsilon_a" else np.array([1.0, 0.0]))
    gamma = -np.log(float(eta_ch))
    I = np.eye(2)

    if gamma == 0.0:
        E = I + A                          # e^{M}
        J0, J1 = 1.0, 0.5                  # ∫₀¹ e^{-γu/2} du, ∫₀¹ u e^{-γu/2} du
        K0, K1, K2 = 1.0, 0.5, 1.0 / 3.0   # ∫₀¹ e^{-γs}{1, s, s²} ds
    else:
        h = gamma / 2.0
        E = np.exp(-h) * (I + A)
        J0 = (1 - np.exp(-h)) / h
        J1 = (1 - np.exp(-h) * (1 + h)) / h**2
        K0 = (1 - np.exp(-gamma)) / gamma
        K1 = (1 - np.exp(-gamma) * (1 + gamma)) / gamma**2
        K2 = (2 - np.exp(-gamma) * (2 + 2 * gamma + gamma**2)) / gamma**3

    # ∫₀¹ e^{M(1-s)} b ds = ∫₀¹ e^{Mu} b du = (J0·1 + J1·A) b
    force = J0 * I + J1 * A
    # γ ∫₀¹ e^{Ms} V_vac e^{Mᵀs} ds, V_vac = ½·1:
    #   = (γ/2) ∫ e^{-γs} (1 + sA)(1 + sAᵀ) ds = (γ/2)[K0·1 + K1 (A + Aᵀ) + K2 A Aᵀ]
    diffusion = 0.0 * I if gamma == 0.0 else 0.5 * gamma * (K0 * I + K1 * (A + A.T) + K2 * (A @ A.T))

    return GaussianState(E @ state.mean + force @ b,
                         E @ state.cov @ E.T + diffusion,
                         E @ state.dmean + force @ db)


def gaussian_channel(state, *, kappa_ba=0.0, epsilon_a=0.0, epsilon_p=0.0,
                     eta_in=1.0, eta_ch=1.0, eta_out=1.0, param="epsilon_a"):
    """Gaussian mirror of ``quantum_sensing.get_state_single_mode_rp`` (``Delta=0``,
    ``t_final=1``, no dephasing/phase noise, which are non-Gaussian).

    Pipeline: input loss → [shear + signal + concurrent channel loss] → output loss.
    ``dmean`` accumulates the derivative with respect to ``param``.
    """
    if param not in ("epsilon_a", "epsilon_p"):
        raise ValueError("param must be 'epsilon_a' or 'epsilon_p'")
    s = state.copy()
    if eta_in != 1.0:
        s = apply_loss(s, eta_in)
    s = _sensing_stage(s, kappa_ba, epsilon_a, epsilon_p, eta_ch, param)
    if eta_out != 1.0:
        s = apply_loss(s, eta_out)
    return s


def gaussian_qfi(state, **channel_kwargs):
    """QFI ``(∂d)ᵀ V⁻¹ (∂d)`` of the channel output for ``param`` (default ``epsilon_a``).

    Directly comparable with ``backaction_qfi.rp_qfi`` for a Gaussian probe.
    """
    out = gaussian_channel(state, **channel_kwargs)
    return float(out.dmean @ np.linalg.solve(out.cov, out.dmean))


def optimal_squeeze_angle(nbar, n_grid=721, **channel_kwargs):
    """Squeeze angle maximising the QFI at fixed ``nbar``: ``(theta, qfi, var_x)``.

    Without loss the optimum is ``theta = 0`` (maximal anti-squeezing of ``x``);
    with detection loss it rotates away and the optimal ``Var(x)`` is finite.
    """
    thetas = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_grid)
    qfis = np.array([gaussian_qfi(squeezed_at_fixed_nbar(nbar, t), **channel_kwargs) for t in thetas])
    j = int(np.argmax(qfis))
    return float(thetas[j]), float(qfis[j]), squeezed_at_fixed_nbar(nbar, thetas[j]).var_x


# --- textbook homodyne curves (KLMTV free mass) -----------------------------------

def fd_squeeze_angle(kappa):
    """Squeeze angle realising frequency-dependent (filter-cavity) squeezing for
    phase-quadrature readout: ``θ(κ) = arctan2(1, -κ) - π/2``."""
    return np.arctan2(1.0, -np.asarray(kappa, float)) - 0.5 * np.pi


def homodyne_noise_spectrum(kappa, zeta=0.0, r=0.0, squeeze_angle=0.0, eta=1.0):
    r"""Strain noise ``S_h / h_SQL²`` of a homodyne readout at quadrature ``zeta``.

    ``b_ζ = b_1 sin ζ + b_2 cos ζ`` with ``b_1 = a_1``,
    ``b_2 = a_2 - κ a_1 + √(2κ) h/h_SQL``, squeezed input ``(r, squeeze_angle)``
    and detection efficiency ``eta``.

    * ``zeta = 0, r = 0``: ``(1 + κ²)/(2κ)``, minimum 1 at ``κ = 1``: the SQL.
    * ``tan ζ = κ`` (variational readout): ``1/(2κ)``, the vacuum Cramér–Rao bound.
    * ``squeeze_angle = fd_squeeze_angle(κ)``: ``e^{-2r}(1 + κ²)/(2κ)``.
    """
    kappa, zeta, r, th, eta = np.broadcast_arrays(*(np.asarray(v, float) for v in (kappa, zeta, r, squeeze_angle, eta)))
    ep, em = np.exp(2 * r), np.exp(-2 * r)
    c, s = np.cos(th), np.sin(th)
    v11, v22, v12 = ep * c**2 + em * s**2, ep * s**2 + em * c**2, (ep - em) * s * c
    t1, t2 = np.sin(zeta) - kappa * np.cos(zeta), np.cos(zeta)
    noise = eta * (t1**2 * v11 + 2 * t1 * t2 * v12 + t2**2 * v22) + (1 - eta) * (t1**2 + t2**2)
    return noise / (eta * 2.0 * kappa * np.cos(zeta) ** 2)


def strain_noise_from_qfi(qfi_epsilon_a, kappa):
    """Quantum Cramér–Rao bound ``σ_h²/h_SQL² = 2/(κ F_{ε_a})`` from the ``epsilon_a`` QFI."""
    return 2.0 / (np.asarray(kappa, float) * np.asarray(qfi_epsilon_a, float))


_HBAR = 1.054571817e-34


def h_sql(Omega, *, m=40.0, L=3995.0):
    """Free-mass SQL strain ``h_SQL = √(8ħ/(m Ω² L²))`` (KLMTV Eq. 2; ``m`` = mass of
    each mirror, matching ``quantum_sensing.rp_power_ratio``)."""
    return np.sqrt(8.0 * _HBAR / (m * np.asarray(Omega, float) ** 2 * L**2))
