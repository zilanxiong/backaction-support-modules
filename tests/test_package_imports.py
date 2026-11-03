"""Smoke test: the group's quantum_sensing package (installed separately, see
requirements.txt) is importable and its public API is reachable."""
import numpy as np
import quantum_sensing as qs


def test_public_api_present():
    for name in ("calculate_qfi", "calculate_cfi", "get_state_single_mode",
                 "get_state_single_mode_rp", "rp_kappa", "ba_to_g", "gkp"):
        assert hasattr(qs, name), name


def test_rp_kappa_anchor():
    # KLMTV anchor documented in quantum_sensing.conversions: rp_kappa(gamma) == rp_power_ratio()
    gamma = 2 * np.pi * 450.0
    assert np.isclose(qs.rp_kappa(gamma, gamma=gamma), qs.rp_power_ratio(gamma=gamma))
