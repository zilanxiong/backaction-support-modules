"""The group's optimised states load, reconstruct, and are oriented for epsilon_a."""

import numpy as np
import pytest
import qutip as qt

import backaction_qfi as b
import backaction_qfi.optimised as opt
from quantum_sensing import calculate_qfi, get_state_single_mode


def test_data_is_present_and_indexed():
    assert (opt.DATA_DIR / "states.h5.gz").exists()     # what the repo carries (2.5 MB)
    assert (opt.DATA_DIR / "states.h5").exists()        # unpacked on first use (29 MB, gitignored)
    assert (opt.DATA_DIR / "master_results.csv").exists()
    entries = opt.list_entries("fock_sup", "loss_ch", 0.9, 0.0)
    assert len(entries) > 10
    assert {"qfi", "N_target", "N_basis", "c_vals", "n_vals"} <= set(entries[0])


def test_missing_slice_raises():
    with pytest.raises(LookupError):
        opt.best_entry("fock_sup", "loss_ch", 0.9, 0.0, N_target=123.0)


def test_rotation_is_unitary_and_quarter_turn_swaps_x_and_p():
    psi = b.squeezed(60, 2.0)                   # large Var(x), small Var(p)
    rot = opt.quarter_turn(psi)
    assert rot.norm() == pytest.approx(1.0)
    assert b.mean_photon_number(rot) == pytest.approx(b.mean_photon_number(psi))
    N = 60
    a = qt.destroy(N)
    p = -1j * (a - a.dag()) / np.sqrt(2)
    var_p = lambda s: float(np.real(qt.expect(p * p, s) - qt.expect(p, s) ** 2))
    assert b.var_x(rot) == pytest.approx(var_p(psi), rel=1e-9)
    assert var_p(rot) == pytest.approx(b.var_x(psi), rel=1e-9)
    assert opt.rotate(psi, 0.0) == psi


@pytest.mark.parametrize("state_type, loss_config, eta, pn", [
    ("fock_sup", "loss_ch", 0.9, 0.0),
    ("fock_sup", "loss_out", 0.8, 0.0),
    ("sqz_vac", "loss_ch", 0.9, 0.0),
    ("fock_sup", "loss_ch", 0.9, 0.2),
])
def test_stored_states_were_optimised_for_epsilon_p_and_rotate_onto_epsilon_a(state_type, loss_config, eta, pn):
    # As stored, the QFI reproduces for epsilon_p; after the quarter turn it
    # reproduces for the GW signal epsilon_a.  Without back-action the channel
    # is rotation-symmetric, so the two numbers must coincide.
    as_stored, entry = opt.load_probe(state_type, loss_config, eta, pn, 5.0, oriented_for="epsilon_p")
    oriented, _ = opt.load_probe(state_type, loss_config, eta, pn, 5.0)
    N = as_stored.shape[0]
    kw = dict(opt.channel_kwargs(entry), N_basis=N)
    f_p = float(calculate_qfi(get_state_single_mode, param_type="epsilon_p", rho=as_stored, **kw))
    f_a = float(calculate_qfi(get_state_single_mode, param_type="epsilon_a", rho=oriented, **kw))
    assert f_p == pytest.approx(entry["qfi"], rel=1e-3)
    assert f_a == pytest.approx(f_p, rel=1e-9)


@pytest.mark.parametrize("state_type, loss_config, expected_rel", [
    ("fock_sup", "loss_ch", 1e-4),
    ("sqz_vac", "loss_ch", 2e-2),
    ("sqz_coh", "loss_in", 2e-2),
])
def test_reconstructed_photon_number_against_stored(state_type, loss_config, expected_rel):
    # Fock superpositions reproduce the stored N_actual exactly.  Both squeezed
    # families reconstruct ~1.7-1.8 % *above* it, independent of cutoff, while
    # their stored QFI still reproduces to 1e-4: a Julia-vs-Python bookkeeping
    # difference in the squeezed-state reconstruction (see data README).  The
    # tolerance documents the known offset; tighten it once that is resolved.
    psi, entry = opt.load_probe(state_type, loss_config, 0.9, 0.0, 5.0, N_basis=200)
    assert b.mean_photon_number(psi) == pytest.approx(entry["N_actual"], rel=expected_rel)


def test_unrotated_state_is_a_poor_probe_for_the_gw_signal():
    as_stored, entry = opt.load_probe("sqz_vac", "loss_ch", 0.9, 0.0, 5.0, oriented_for="epsilon_p")
    kw = dict(opt.channel_kwargs(entry), N_basis=as_stored.shape[0])
    f_wrong = float(calculate_qfi(get_state_single_mode, param_type="epsilon_a", rho=as_stored, **kw))
    assert f_wrong < 0.05 * entry["qfi"]


def test_oriented_probe_goes_through_the_backaction_channel():
    # The whole point: an optimised, oriented probe through rp_channel with
    # back-action on.  Without loss the QFI must be 8 Var(x) regardless of kappa.
    psi, entry = opt.load_probe("fock_sup", "loss_ch", 0.9, 0.0, 5.0, N_basis=160)
    lossless = b.rp_qfi(psi, kappa_ba=1.0)
    assert lossless == pytest.approx(8 * b.var_x(psi), rel=1e-4)
    with_loss = b.rp_qfi(psi, kappa_ba=1.0, eta_out=0.9)
    assert 0 < with_loss < lossless
