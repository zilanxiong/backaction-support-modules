"""Gaussian (covariance-matrix) track: textbook free-mass curves and exact
cross-checks of the Fock-space track.

SQL, variational readout and frequency-dependent squeezing are closed forms
(KLMTV, PRD 65, 022002).  The Gaussian channel is then compared, number for
number, with the package's density-matrix channel for Gaussian probes.
"""

import numpy as np
import pytest

import backaction_qfi as b
import backaction_qfi.gaussian as g

KAPPAS = np.array([0.1, 0.5, 1.0, 2.0, 5.0])


# --- shear algebra ---------------------------------------------------------------

@pytest.mark.parametrize("kappa", KAPPAS)
def test_shear_decomposes_into_rotation_and_squeezing(kappa):
    r, pre, post = g.shear_to_squeeze_rotation(kappa)
    rebuilt = g.rotation_matrix(post) @ g.squeeze_matrix(r) @ g.rotation_matrix(pre)
    assert np.allclose(rebuilt, g.shear_matrix(kappa), atol=1e-12)
    assert np.exp(r) - np.exp(-r) == pytest.approx(kappa, rel=1e-12)   # e^r - e^-r = kappa
    assert np.linalg.det(g.shear_matrix(kappa)) == pytest.approx(1.0)


# --- SQL, variational readout, FD squeezing ------------------------------------------

def test_sql_curve():
    """Phase-quadrature readout of vacuum: (1 + k²)/(2k), minimum 1 at k = 1."""
    assert np.allclose(g.homodyne_noise_spectrum(KAPPAS), (1 + KAPPAS**2) / (2 * KAPPAS))
    grid = np.unique(np.concatenate([np.linspace(0.05, 20, 4001), [1.0]]))
    assert np.min(g.homodyne_noise_spectrum(grid)) == pytest.approx(1.0, rel=1e-12)
    assert g.homodyne_noise_spectrum(1.0) == pytest.approx(1.0)


def test_variational_readout_reaches_the_vacuum_cramer_rao_bound():
    """tan(zeta) = kappa cancels back-action: 1/(2k), equal to the QCRB from F = 4."""
    assert np.allclose(g.homodyne_noise_spectrum(KAPPAS, zeta=np.arctan(KAPPAS)), 1 / (2 * KAPPAS))
    qfi = np.array([g.gaussian_qfi(g.vacuum(), kappa_ba=k) for k in KAPPAS])
    assert np.allclose(g.strain_noise_from_qfi(qfi, KAPPAS), 1 / (2 * KAPPAS))


@pytest.mark.parametrize("r", [0.0, 0.5, 1.15])
def test_frequency_dependent_squeezing_scales_the_sql_curve(r):
    s = g.homodyne_noise_spectrum(KAPPAS, r=r, squeeze_angle=g.fd_squeeze_angle(KAPPAS))
    assert np.allclose(s, np.exp(-2 * r) * (1 + KAPPAS**2) / (2 * KAPPAS))


def test_frozen_squeeze_angle_cannot_beat_fd_squeezing():
    r = 1.0
    fd = g.homodyne_noise_spectrum(KAPPAS, r=r, squeeze_angle=g.fd_squeeze_angle(KAPPAS))
    for angle in np.linspace(-np.pi, np.pi, 41):
        assert np.all(g.homodyne_noise_spectrum(KAPPAS, r=r, squeeze_angle=angle) >= fd - 1e-12)


def test_h_sql_closed_form_falls_as_one_over_omega():
    Om = 2 * np.pi * np.array([10.0, 100.0, 1000.0])
    h = g.h_sql(Om)
    assert np.allclose(h * Om, h[0] * Om[0])
    assert 1e-24 < g.h_sql(2 * np.pi * 100.0) < 1e-22       # aLIGO-scale number


# --- analytic QFI results ---------------------------------------------------------

def test_vacuum_qfi_is_four():
    assert g.gaussian_qfi(g.vacuum()) == pytest.approx(4.0)


@pytest.mark.parametrize("kappa", KAPPAS)
@pytest.mark.parametrize("eta", [1.0, 0.9, 0.5])
def test_vacuum_backaction_then_detection_loss(kappa, eta):
    """F = 4 eta / (1 + eta (1 - eta) k²)."""
    assert g.gaussian_qfi(g.vacuum(), kappa_ba=kappa, eta_out=eta) == pytest.approx(
        4 * eta / (1 + eta * (1 - eta) * kappa**2), rel=1e-12)


@pytest.mark.parametrize("kappa", KAPPAS)
def test_backaction_alone_does_not_change_the_qfi(kappa):
    st = g.squeezed_gaussian(0.8, theta=0.3)
    assert g.gaussian_qfi(st, kappa_ba=kappa) == pytest.approx(g.gaussian_qfi(st), rel=1e-12)
    assert g.gaussian_qfi(st) == pytest.approx(8 * st.var_x, rel=1e-12)      # pure: 8 Var(x)


@pytest.mark.parametrize("r", [0.0, 0.7, 1.4])
@pytest.mark.parametrize("kappa", [0.0, 1.0, 3.0])
def test_injection_loss_is_blind_to_backaction(r, kappa):
    """F = 4 / (eta e^{-2r} + 1 - eta), independent of kappa (and < 8 Var x: mixed state)."""
    eta = 0.7
    got = g.gaussian_qfi(g.squeezed_gaussian(r), kappa_ba=kappa, eta_in=eta)
    assert got == pytest.approx(4 / (eta * np.exp(-2 * r) + 1 - eta), rel=1e-12)


def test_detection_loss_is_the_one_that_couples_to_backaction():
    st = g.squeezed_gaussian(1.0)
    inj = [g.gaussian_qfi(st, kappa_ba=k, eta_in=0.8) for k in KAPPAS]
    det = [g.gaussian_qfi(st, kappa_ba=k, eta_out=0.8) for k in KAPPAS]
    assert np.allclose(inj, inj[0], rtol=1e-12)
    assert np.all(np.diff(det) < 0)


def test_optimal_squeeze_angle_is_zero_without_loss_and_rotates_with_it():
    th0, q0, _ = g.optimal_squeeze_angle(2.0, kappa_ba=1.0)
    assert th0 == pytest.approx(0.0, abs=1e-9)
    assert q0 == pytest.approx(8 * g.squeezed_at_fixed_nbar(2.0).var_x, rel=1e-9)
    th1, q1, vx1 = g.optimal_squeeze_angle(2.0, kappa_ba=1.0, eta_out=0.8)
    assert abs(th1) > 0.05 and q1 < q0
    assert vx1 < g.squeezed_at_fixed_nbar(2.0).var_x


# --- cross-check against the Fock track ------------------------------------------------

@pytest.mark.parametrize("probe", ["vacuum", "coherent", "squeezed"])
@pytest.mark.parametrize("kappa, eta_in, eta_ch, eta_out", [
    (0.0, 1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0, 0.9),
    (2.0, 0.8, 1.0, 1.0),
    (1.0, 1.0, 0.8, 1.0),      # loss *during* the shear: the closed-form moment integrals
    (1.5, 0.9, 0.9, 0.9),
])
def test_fock_track_matches_gaussian_track(probe, kappa, eta_in, eta_ch, eta_out):
    N = 70
    nbar = 1.0
    if probe == "vacuum":
        fock, gauss = b.fock(N, 0), g.vacuum()
    elif probe == "coherent":
        fock, gauss = b.coherent(N, nbar), g.coherent_gaussian(np.sqrt(nbar))
    else:
        fock, gauss = b.squeezed(N, nbar), g.squeezed_at_fixed_nbar(nbar)
    kw = dict(kappa_ba=kappa, eta_in=eta_in, eta_ch=eta_ch, eta_out=eta_out)
    assert b.rp_qfi(fock, **kw) == pytest.approx(g.gaussian_qfi(gauss, **kw), rel=2e-4)


def test_fock_track_matches_gaussian_track_for_the_non_commuting_signal():
    # epsilon_p displaces x, which the shear then feeds into p: the concurrent
    # (I + A/2) correction in the moment equations is exercised here.  The
    # derivative probes the anti-squeezed direction, so this needs a larger
    # cutoff than the epsilon_a cases (Fock -> Gaussian: 8e-4 at N=70, 2e-5 at
    # N=100, 4e-8 at N=140).
    N = 120
    fock, gauss = b.squeezed(N, 1.0), g.squeezed_at_fixed_nbar(1.0)
    kw = dict(kappa_ba=1.5, eta_out=0.9, param_type="epsilon_p")
    got = b.rp_qfi(fock, **kw)
    kw_g = dict(kappa_ba=1.5, eta_out=0.9, param="epsilon_p")
    assert got == pytest.approx(g.gaussian_qfi(gauss, **kw_g), rel=2e-4)


def test_gaussian_state_photon_number_matches_fock_probe():
    assert g.squeezed_at_fixed_nbar(2.0).nbar == pytest.approx(2.0)
    assert g.coherent_gaussian(np.sqrt(3.0)).nbar == pytest.approx(3.0)
    assert g.squeezed_at_fixed_nbar(2.0).var_x == pytest.approx(b.var_x(b.squeezed(120, 2.0)), rel=1e-3)
