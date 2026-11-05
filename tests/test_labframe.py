"""The lab-frame NumPy/CuPy port of the package channel (backaction_qfi.labframe).

Fast checks (~1 min on CPU): the port reproduces the package itself point by point,
the lossless closed form, and the exact Gaussian-track value as the cutoff grows.
The GPU path is the same code on CuPy; it is exercised only when CuPy is installed.
"""
import types

import numpy as np
import pytest
import qutip as qt

import quantum_sensing as qs
import quantum_sensing.dynamics as dyn
import backaction_qfi as b
from backaction_qfi.labframe import rp_qfi_labframe, LabFrameChannel, array_module
from backaction_qfi.gaussian import squeezed_at_fixed_nbar, gaussian_qfi

KAPPA_31HZ = 0.949          # kappa_ba at 31.6 Hz: within the package's cutoff budget at N = 120


@pytest.fixture(scope="module")
def package_qfi():
    """The package's own channel + QFI, with solver options injected (it passes none)."""
    opts = dict(atol=1e-12, rtol=1e-10, nsteps=1_000_000)

    class _QT(types.ModuleType):
        def __getattr__(self, name):
            return getattr(qt, name)

        def mesolve(self, *a, **kw):
            kw.setdefault("options", opts)
            return qt.mesolve(*a, **kw)

    saved, dyn.qt = dyn.qt, _QT("qutip")
    yield lambda psi, N, **ch: qs.calculate_qfi(dynamics=qs.get_state_single_mode_rp, param_type="epsilon_a",
                                                 rho=psi, N_basis=N, **ch)
    dyn.qt = saved


@pytest.mark.parametrize("probe, channel", [
    ("squeezed", dict(eta_ch=0.95)),
    ("cat", dict(eta_ch=0.95, pn_in=0.2)),
    ("coherent", dict(eta_in=0.9, eta_out=0.9)),
    ("fock", dict(eta_in=0.95, pn_in=0.2, eta_ch=0.99, eta_out=0.9)),
])
def test_port_matches_the_package_point_by_point(package_qfi, probe, channel):
    psi = b.make_probe(probe, 120, 5)
    lab = rp_qfi_labframe(psi, kappa_ba=KAPPA_31HZ, **channel)
    ref = package_qfi(psi, 120, kappa_ba=KAPPA_31HZ, **channel)
    assert lab == pytest.approx(ref, rel=1e-7)


@pytest.mark.parametrize("probe, kappa", [("squeezed", 2.0), ("cat", 2.0)])
def test_lossless_qfi_is_eight_var_x(probe, kappa):
    psi = b.make_probe(probe, 160, 5)
    assert rp_qfi_labframe(psi, kappa_ba=kappa) == pytest.approx(8 * b.var_x(psi), rel=1e-7)


def test_converges_to_the_exact_gaussian_value_with_cutoff():
    channel = dict(kappa_ba=1.0, eta_ch=0.95, eta_out=0.9)
    exact = gaussian_qfi(squeezed_at_fixed_nbar(5.0), **channel)
    errs = [abs(rp_qfi_labframe(b.make_probe("squeezed", N, 5), **channel) / exact - 1) for N in (160, 240)]
    assert errs[1] < errs[0]                     # the ladder is converging
    assert errs[1] < 5e-5                        # and already close at N = 240 (3e-7 at N = 320)


def test_sensitivity_source_acts_only_in_the_sensing_stage():
    # Without shear a coherent probe stays coherent: QFI 4, and loss after the signal scales
    # the displacement by sqrt(eta_out), so the QFI by eta_out exactly. The derivative must
    # ride through stage 3 without a source term (a source there gave 4.1x this value).
    psi = b.make_probe("coherent", 60, 2)
    assert rp_qfi_labframe(psi) == pytest.approx(4.0, rel=1e-8)
    assert rp_qfi_labframe(psi, eta_out=0.8) == pytest.approx(3.2, rel=1e-7)


def test_device_switch():
    assert array_module("cpu") is np
    with pytest.raises(ValueError):
        array_module("tpu")
    cupy = pytest.importorskip("cupy")
    psi = b.make_probe("cat", 60, 2)
    assert rp_qfi_labframe(psi, kappa_ba=0.5, eta_ch=0.9, device="gpu") == pytest.approx(
        rp_qfi_labframe(psi, kappa_ba=0.5, eta_ch=0.9), rel=1e-8)
    # the case where cuSOLVER's eigh returned NaN on an H200 (coherent probe, 1 kHz): host fallback
    psi = b.make_probe("coherent", 160, 5)
    assert rp_qfi_labframe(psi, kappa_ba=1.6e-4, device="gpu") == pytest.approx(4.0, rel=1e-8)
