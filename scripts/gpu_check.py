"""GPU vs CPU diagnostic for backaction_qfi.labframe: run the same points on both devices
and print every intermediate that could differ. Run on a GPU node:
    sbatch --qos=debug --time=00:30:00 scripts/gpu_check.slurm
"""
import time, numpy as np, cupy, qutip as qt
import quantum_sensing as qs
from backaction_qfi import make_probe
from backaction_qfi.labframe import LabFrameChannel, qfi_from_rho_drho, rp_qfi_labframe

print("cupy", cupy.__version__, "| device:", cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(), flush=True)
k1000 = float(qs.rp_kappa(2 * np.pi * 1000.0))
cases = [("coherent", k1000, {}), ("squeezed", k1000, {}), ("coherent", 0.95, {}), ("cat", 0.5, dict(eta_ch=0.9)),
         ("fock", k1000, {})]
for probe, kappa, ch in cases:
    psi = make_probe(probe, 160, 5)
    out = {}
    for dev in ("cpu", "gpu"):
        chan = LabFrameChannel(160, device=dev)
        t0 = time.perf_counter()
        rho, sig = chan.run(psi, kappa_ba=kappa, **ch)
        xp = chan.xp
        vals = xp.linalg.eigvalsh(rho)
        F = qfi_from_rho_drho(rho, sig, xp=xp)
        get = (lambda a: a.get()) if dev == "gpu" else (lambda a: a)
        out[dev] = dict(F=F, tr=float(get(xp.trace(rho)).real), sig_norm=float(get(xp.linalg.norm(sig))),
                        rho_norm=float(get(xp.linalg.norm(rho))), lam_max=float(get(vals).max()),
                        lam_min=float(get(vals).min()), steps=chan.last_steps, t=time.perf_counter() - t0,
                        sig_dtype=str(sig.dtype), sig_type=type(sig).__name__)
        print(f"{probe:9s} kappa={kappa:.4g} {ch} [{dev}]  QFI={out[dev]['F']:.6f}  tr={out[dev]['tr']:.6f}  "
              f"|rho|={out[dev]['rho_norm']:.4f}  |sigma|={out[dev]['sig_norm']:.4f}  lam=[{out[dev]['lam_min']:.1e}, "
              f"{out[dev]['lam_max']:.6f}]  steps={out[dev]['steps']}  {out[dev]['t']:.1f}s  {out[dev]['sig_type']}/{out[dev]['sig_dtype']}",
              flush=True)
    rel = out["gpu"]["F"] / out["cpu"]["F"] - 1 if out["cpu"]["F"] else float("nan")
    print(f"   -> gpu/cpu QFI rel. diff {rel:+.2e}", flush=True)
# the one-call API too
psi = make_probe("coherent", 160, 5)
print("rp_qfi_labframe coherent 1 kHz: cpu", rp_qfi_labframe(psi, kappa_ba=k1000), " gpu", rp_qfi_labframe(psi, kappa_ba=k1000, device="gpu"), flush=True)
