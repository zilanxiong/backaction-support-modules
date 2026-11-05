#!/usr/bin/env python
"""QFI vs frequency sweep through the package channel, integrated in the lab frame
(backaction_qfi.labframe) on GPU when CuPy is available, else CPU.

One row per (probe, frequency) is appended to results/<tag>/qfi_vs_frequency.csv as
soon as it finishes, so a job that stops early resumes where it left off.  Each point
climbs a Fock-cutoff ladder until two consecutive cutoffs agree to --rtol; points that
never do are kept but flagged converged=False.

Usage (channel loss 0.95, all five probes, 25 frequencies 10 Hz – 1 kHz):
    python scripts/sweep_labframe.py --tag channel_loss_eta0p95 --eta-ch 0.95
Physical stack with 200 mrad phase noise:
    python scripts/sweep_labframe.py --tag physical_pn0p2 --eta-in 0.95 --pn-in 0.2 --eta-ch 0.99 --eta-out 0.90
One job per probe (parallel GPUs; each probe writes its own CSV, merge with pandas):
    for f in "Squeezed vac." Coherent "Even cat" "Squeezed cat" Fock; do
        sbatch scripts/sweep_gpu.slurm --tag backaction_only --families "$f"; done
Same sweep through the package's own channel and QFI (GPU via qutip-cuquantum), one job per probe:
    sbatch scripts/sweep_gpu.slurm --tag channel_loss_eta0p95 --eta-ch 0.95 --engine package --families Fock
Timing check only (three frequencies, squeezed vacuum):
    python scripts/sweep_labframe.py --tag timing --eta-ch 0.95 --freqs 100 31.6 10 --families "Squeezed vac."
"""
import argparse, csv, os, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

import quantum_sensing as qs
from backaction_qfi import make_probe
from backaction_qfi.labframe import rp_qfi_labframe

FAMILIES = {"Squeezed vac.": "squeezed", "Coherent": "coherent", "Even cat": "cat",
            "Squeezed cat": "squeezed_cat", "Fock": "fock"}
FIELDS = ["family", "freq_hz", "kappa_ba", "qfi", "qfi_extrap", "cutoff", "converged", "seconds", "ladder"]


def pick_device(requested):
    if requested == "cpu":
        return "cpu"
    try:
        import cupy
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"device: gpu ({name})", flush=True)
        return "gpu"
    except Exception as err:
        if requested == "gpu":
            raise SystemExit(f"--device gpu requested but CuPy/GPU unavailable: {err}")
        print(f"device: cpu (no CuPy/GPU: {err})", flush=True)
        return "cpu"


GROWTH = 1.4


def cutoff_ladder(kappa, n_rungs, N0_min=160, per_kappa=150, growth=GROWTH):
    """Cutoffs to try. The sheared state's extent grows with kappa; at kappa ~ 9.5 (10 Hz)
    the lab frame needs several thousand levels, so the ladder starts higher there."""
    N0 = int(max(N0_min, per_kappa * kappa))
    return [int(N0 * growth ** i) for i in range(n_rungs)]


def extrapolate(history):
    """Geometric (Richardson-type) extrapolation of the last three rungs: with a fixed
    cutoff ratio, consecutive differences shrink by a constant factor r once the
    truncation error is a power law in N; the remaining tail is d_last * r / (1 - r)."""
    if len(history) < 3:
        return history[-1][1], float("nan")
    f1, f2, f3 = (h[1] for h in history[-3:])
    d1, d2 = f2 - f1, f3 - f2
    if d1 == 0.0 or abs(d2) >= abs(d1):
        return f3, float("nan")
    r = d2 / d1
    tail = d2 * r / (1.0 - r)
    return f3 + tail, abs(tail / (f3 + tail))


def qfi_point(psi, kappa, N, device, channel, solver_rtol, solver_atol, engine, padded):
    if engine == "port":
        return rp_qfi_labframe(psi, kappa_ba=kappa, N_basis=N, device=device,
                               rtol=solver_rtol, atol=solver_atol, **channel)
    # the package wants the probe at the channel's cutoff: use the pre-padded ket
    return qs.calculate_qfi(dynamics=qs.get_state_single_mode_rp, param_type="epsilon_a",
                            rho=padded[N], kappa_ba=kappa, N_basis=N, **channel)


def converged_point(psi, kappa, ladder, device, rtol, channel, solver_rtol, solver_atol, engine="port", padded=None):
    history = []
    for N in ladder:
        t0 = time.perf_counter()
        F = qfi_point(psi, kappa, N, device, channel, solver_rtol, solver_atol, engine, padded)
        history.append((N, F, time.perf_counter() - t0))
        if not np.isfinite(F) or F <= 0.0:
            raise RuntimeError(f"suspicious QFI {F!r} at N={N}, kappa={kappa}, {channel}: ladder so far {history}")
        if len(history) >= 2 and abs(history[-1][1] - history[-2][1]) <= rtol * abs(history[-1][1]):
            return F, N, True, history
        if len(history) >= 3:
            _, remaining = extrapolate(history)
            if remaining == remaining and remaining < rtol:        # extrapolated tail below rtol
                return F, N, True, history
    return F, N, False, history


def solver_atol_pkg(args):
    # finite differences with step 1e-5 need tighter tolerances than the sensitivity equation
    return min(args.solver_atol, 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="results/<tag>/ holds the CSV")
    ap.add_argument("--eta-in", type=float, default=1.0)
    ap.add_argument("--pn-in", type=float, default=0.0)
    ap.add_argument("--eta-ch", type=float, default=1.0)
    ap.add_argument("--eta-out", type=float, default=1.0)
    ap.add_argument("--nbar", type=float, default=5.0)
    ap.add_argument("--freqs", type=float, nargs="*", default=None, help="Hz; default 25 log-spaced 10–1000")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES), help=f"subset of {list(FAMILIES)}")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--engine", choices=["port", "package"], default="port",
                    help="port: backaction_qfi.labframe (NumPy/CuPy, sensitivity equation); "
                         "package: quantum_sensing's own channel + calculate_qfi, on GPU via qutip-cuquantum")
    ap.add_argument("--rtol", type=float, default=1e-3, help="ladder convergence criterion")
    ap.add_argument("--rungs", type=int, default=6, help="ladder length (top rung = N0 * 1.4^(rungs-1))")
    ap.add_argument("--solver-rtol", type=float, default=1e-8, help="DOP853 relative tolerance")
    ap.add_argument("--solver-atol", type=float, default=1e-10, help="DOP853 absolute tolerance")
    ap.add_argument("--n-probe", type=int, default=400, help="cutoff the probe is built at (then zero-padded)")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    device = pick_device(args.device)
    channel = dict(eta_in=args.eta_in, pn_in=args.pn_in, eta_ch=args.eta_ch, eta_out=args.eta_out)
    freqs = np.logspace(1, 3, 25) if args.freqs is None else np.array(args.freqs, float)
    out_dir = Path(args.results) / args.tag; out_dir.mkdir(parents=True, exist_ok=True)
    # one job per probe may run concurrently: a family subset gets its own CSV
    slug = "" if sorted(args.families) == sorted(FAMILIES) else "_" + "_".join(
        re.sub(r"[^a-z0-9]+", "", f.lower()) for f in args.families)
    csv_path = out_dir / f"qfi_vs_frequency{slug}.csv"

    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        done = set(zip(prev.family, prev.freq_hz.round(6)))
    todo = [(label, f) for f in sorted(freqs, reverse=True) for label in args.families
            if (label, round(float(f), 6)) not in done]
    print(f"channel {channel}  nbar {args.nbar}  {len(todo)} points to do, {len(done)} already in {csv_path}", flush=True)

    probes = {label: make_probe(FAMILIES[label], args.n_probe, args.nbar) for label in args.families}
    padded = {}
    if args.engine == "package":
        import types, qutip as qt, quantum_sensing.dynamics as dyn
        # every cutoff the ladders can reach, as zero-padded kets, built BEFORE the GPU
        # backend becomes the default dtype (it rejects rectangular kets)
        cutoffs = sorted({N for f in freqs for N in cutoff_ladder(float(qs.rp_kappa(2 * np.pi * f)), args.rungs)})
        for label, psi in probes.items():
            v = psi.full()
            padded[label] = {N: qt.Qobj(np.pad(v, ((0, max(N - v.shape[0], 0)), (0, 0)))[:N]) for N in cutoffs}
        # the package passes no solver settings: inject tolerances / step budget into its mesolve calls
        opts = dict(atol=solver_atol_pkg(args), rtol=args.solver_rtol, nsteps=1_000_000)

        class _QT(types.ModuleType):
            def __getattr__(self, name):
                return getattr(qt, name)

            def mesolve(self, *a, **kw):
                kw.setdefault("options", {}).update({k: v for k, v in opts.items() if k not in kw["options"]})
                return qt.mesolve(*a, **kw)

        dyn.qt = _QT("qutip")
        if device == "gpu":
            import qutip_cuquantum
            from cuquantum.densitymat import WorkStream
            qutip_cuquantum.set_as_default(WorkStream())
            print("engine: package channel on GPU via qutip-cuquantum (CuVern7)", flush=True)
        else:
            print("engine: package channel on CPU", flush=True)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        for label, f in todo:
            kappa = float(qs.rp_kappa(2 * np.pi * f))
            ladder = cutoff_ladder(kappa, args.rungs)
            F, N, ok, hist = converged_point(probes[label], kappa, ladder, device, args.rtol, channel,
                                             args.solver_rtol if args.engine == "port" else min(args.solver_rtol, 1e-10),
                                             args.solver_atol, args.engine, padded.get(label))
            F_ex, _ = extrapolate(hist)
            row = dict(family=label, freq_hz=float(f), kappa_ba=kappa, qfi=F, qfi_extrap=F_ex, cutoff=N, converged=ok,
                       seconds=round(sum(h[2] for h in hist), 1),
                       ladder=" ".join(f"{n}:{v:.6g}" for n, v, _ in hist))
            writer.writerow(row); fh.flush()
            print(f"{label:14s} {f:8.2f} Hz  kappa {kappa:7.4f}  QFI {F:10.5f} (extrap {F_ex:.5f})  N {N:5d}  "
                  f"{'ok' if ok else 'NOT CONVERGED'}  {row['seconds']:7.1f} s   ladder {row['ladder']}", flush=True)
    print("done:", csv_path, flush=True)


if __name__ == "__main__":
    sys.exit(main())
