"""Trial of qutip-cuquantum as the GPU backend for the UNCHANGED quantum_sensing channel.

With the backend switched on, qs.get_state_single_mode_rp and qs.calculate_qfi run as
written: the operators the package builds land on the GPU (cuDensityMat), mesolve uses
QuTiP's Verner-7 integrator on the GPU state, and the QFI eigendecomposition runs on the
host after the state is pulled back.  The package passes no solver settings, so the same
proxy as elsewhere injects tolerances and the step budget.

Run on a GPU node:  sbatch --qos=debug --time=00:30:00 scripts/cuquantum_check.slurm
"""
import sys, time, types
import numpy as np
import qutip as qt
import quantum_sensing as qs
import quantum_sensing.dynamics as dyn
from backaction_qfi import make_probe, var_x

# --- solver settings for the package's mesolve calls (it passes none) ---------------
OPTS = dict(atol=1e-12, rtol=1e-10, nsteps=1_000_000)

class _QT(types.ModuleType):
    def __getattr__(self, name):
        return getattr(qt, name)

    def mesolve(self, *a, **kw):
        kw.setdefault("options", {}).update({k: v for k, v in OPTS.items() if k not in kw["options"]})
        return qt.mesolve(*a, **kw)

dyn.qt = _QT("qutip")


def package_qfi(psi, N, **ch):
    return qs.calculate_qfi(dynamics=qs.get_state_single_mode_rp, param_type="epsilon_a", rho=psi, N_basis=N, **ch)


k31 = float(qs.rp_kappa(2 * np.pi * 31.6))
cases = [("squeezed", dict(eta_ch=0.95)), ("coherent", {}), ("cat", dict(eta_ch=0.95, pn_in=0.2)),
         ("fock", dict(eta_in=0.95, pn_in=0.2, eta_ch=0.99, eta_out=0.9))]

# --- CPU reference first (standard QuTiP) ------------------------------------------
ref = {}
for probe, ch in cases:
    psi = make_probe(probe, 200, 5); t0 = time.perf_counter()
    ref[probe] = package_qfi(psi, 200, kappa_ba=k31, **ch)
    print(f"[cpu] {probe:9s} {ch}  QFI = {ref[probe]:.7f}   {time.perf_counter() - t0:.1f}s", flush=True)

# --- switch the backend on -----------------------------------------------------------
try:
    import qutip_cuquantum
    from cuquantum.densitymat import WorkStream
except Exception as err:
    sys.exit(f"qutip-cuquantum / cuquantum not importable: {err!r}")
qutip_cuquantum.set_as_default(WorkStream())
print("backend on: default_dtype =", qt.settings.core["default_dtype"],
      "| MESolver method =", qt.MESolver.solver_options.get("method") if hasattr(qt.MESolver, "solver_options") else "?", flush=True)

# what does the package's channel return now?
psi = make_probe("coherent", 200, 5)
out = qs.get_state_single_mode_rp(rho=psi, kappa_ba=k31, N_basis=200, eta_ch=0.95)
print("channel output dtype with backend on:", out.dtype.__name__, flush=True)

for probe, ch in cases:
    psi = make_probe(probe, 200, 5); t0 = time.perf_counter()
    F = package_qfi(psi, 200, kappa_ba=k31, **ch)
    print(f"[gpu] {probe:9s} {ch}  QFI = {F:.7f}   rel. diff vs cpu {F / ref[probe] - 1:+.2e}   {time.perf_counter() - t0:.1f}s", flush=True)

# --- timing at 10 Hz with channel loss, two cutoffs ----------------------------------
k10 = float(qs.rp_kappa(2 * np.pi * 10.0))
for N in (1000, 2000):
    psi = make_probe("squeezed", 400, 5); t0 = time.perf_counter()
    try:
        F = package_qfi(psi, N, kappa_ba=k10, eta_ch=0.95)
        print(f"[gpu] squeezed 10 Hz eta_ch=0.95 N={N}: QFI = {F:.5f}   {time.perf_counter() - t0:.0f}s  "
              f"(converged value ~2.40; lab-frame port gave 2.09 at 1332 and 2.24 at 1865)", flush=True)
    except Exception as err:
        print(f"[gpu] squeezed 10 Hz N={N}: FAILED after {time.perf_counter() - t0:.0f}s: {err!r}", flush=True)
