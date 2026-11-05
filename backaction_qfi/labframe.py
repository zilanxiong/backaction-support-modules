"""Lab-frame port of the package channel onto NumPy or CuPy arrays.

The Python counterpart of the group's Julia CUDA extension: the three stages of
``quantum_sensing.get_state_single_mode_rp`` (input loss / dephasing, sensing with
``H = g x² + ε_a (a + a†)`` and concurrent channel loss, output loss) are integrated
in the lab frame on a dense ``N × N`` density matrix, with the operators applied as
row / column shifts so that no ``N² × N²`` generator is ever formed.  The same code
runs on NumPy (``device="cpu"``) and CuPy (``device="gpu"``).

Two differences from the package's own integration, both improvements:

* the derivative ``∂ρ/∂ε_a`` is integrated alongside ``ρ`` (sensitivity equation)
  instead of being taken by central difference, so one solve replaces three and
  the solver tolerance can be ~1e-8 instead of ~1e-10;
* the integrator is Dormand–Prince 8(5,3) (scipy's DOP853 coefficients and step
  control, re-implemented on the array namespace).

Parameter conversions (``ba_to_g``, ``loss_to_kappa``, ``phirms_to_chi``) are the
package's.  The QFI is the package's eigen-sum with the same small-eigenvalue
cutoff.  Conventions: ħ = 1, ``x = (a + a†)/√2``, unit stage duration, signal on
``epsilon_a``.
"""
import numpy as np
from scipy.integrate._ivp import dop853_coefficients as _dc

from quantum_sensing.conversions import ba_to_g, loss_to_kappa, phirms_to_chi

__all__ = ["array_module", "LabFrameChannel", "channel_labframe", "rp_qfi_labframe",
           "qfi_from_rho_drho"]

# DOP853 tableau (scipy), as plain numpy for the stage loop
_A, _B, _C = _dc.A[:_dc.N_STAGES, :_dc.N_STAGES], _dc.B, _dc.C[:_dc.N_STAGES]
_E3, _E5, _NS = _dc.E3, _dc.E5, _dc.N_STAGES
_SAFETY, _MIN_FACTOR, _MAX_FACTOR, _ERR_EXP = 0.9, 0.2, 10.0, -1.0 / 8.0


def array_module(device="cpu"):
    """NumPy for ``"cpu"``, CuPy for ``"gpu"`` (imported lazily)."""
    if device == "cpu":
        return np
    if device == "gpu":
        import cupy
        return cupy
    raise ValueError("device must be 'cpu' or 'gpu'")


def _to_dense_dm(rho, N):
    """qutip Qobj (ket or operator) or ndarray -> dense numpy density matrix at cutoff N."""
    if hasattr(rho, "full"):
        arr = rho.full()
        if arr.shape[1] == 1:                   # ket
            arr = arr @ arr.conj().T
    else:
        arr = np.asarray(rho, dtype=complex)
        if arr.ndim == 1 or arr.shape[1] == 1:
            arr = arr.reshape(-1, 1); arr = arr @ arr.conj().T
    n = arr.shape[0]
    if N is None or N == n:
        return np.ascontiguousarray(arr, dtype=complex)
    if N < n:
        return np.ascontiguousarray(arr[:N, :N], dtype=complex)
    out = np.zeros((N, N), dtype=complex); out[:n, :n] = arr
    return out


class LabFrameChannel:
    """The package channel at cutoff ``N`` on the chosen device."""

    def __init__(self, N, *, device="cpu", rtol=1e-8, atol=1e-10, max_steps=10_000_000):
        self.N, self.device, self.rtol, self.atol, self.max_steps = N, device, rtol, atol, max_steps
        xp = self.xp = array_module(device)
        n = np.arange(N, dtype=float)
        self.sq = xp.asarray(np.sqrt(n))                      # sqrt(m)
        self.sq_col = self.sq[1:, None]                       # for left shifts
        self.sq_row = self.sq[None, 1:]                       # for right shifts
        self.n_col = xp.asarray(n)[:, None]
        self.n_row = xp.asarray(n)[None, :]
        self.deph = xp.asarray(-0.5 * (n[:, None] - n[None, :]) ** 2)   # D[n] is elementwise

    # ---- a, a† acting on a matrix (shifts) -------------------------------------
    def a_left(self, r):            # (a r)[m, :] = sqrt(m+1) r[m+1, :]
        out = self.xp.zeros_like(r); out[:-1] = self.sq_col * r[1:]; return out

    def adag_left(self, r):         # (a† r)[m, :] = sqrt(m) r[m-1, :]
        out = self.xp.zeros_like(r); out[1:] = self.sq_col * r[:-1]; return out

    def a_right(self, r):           # (r a)[:, k] = sqrt(k) r[:, k-1]
        out = self.xp.zeros_like(r); out[:, 1:] = self.sq_row * r[:, :-1]; return out

    def adag_right(self, r):        # (r a†)[:, k] = sqrt(k+1) r[:, k+1]
        out = self.xp.zeros_like(r); out[:, :-1] = self.sq_row * r[:, 1:]; return out

    def q_left(self, r):            # (a + a†) r
        return self.a_left(r) + self.adag_left(r)

    def q_right(self, r):           # r (a + a†)
        return self.a_right(r) + self.adag_right(r)

    # ---- Lindbladian of one stage ---------------------------------------------
    def make_rhs(self, *, g=0.0, eps_a=0.0, gamma=0.0, chi=0.0, with_sensitivity=False, source=False):
        """Return f(t, y) for dy/dt. y = rho (N×N) or stacked (rho, sigma) with
        dsigma/dt = L(sigma) [- i[(a + a†), rho] if source]  (sensitivity to eps_a;
        the source term exists only while the signal acts, i.e. in the sensing stage)."""
        def lind(r):
            out = self.xp.zeros_like(r)
            if g != 0.0 or eps_a != 0.0:
                # H r - r H with H = g (a+a†)²/2 + eps_a (a+a†)
                ql, qr = self.q_left(r), self.q_right(r)
                hr = eps_a * ql; rh = eps_a * qr
                if g != 0.0:
                    hr = hr + (0.5 * g) * self.q_left(ql)
                    rh = rh + (0.5 * g) * self.q_right(qr)
                out -= 1j * (hr - rh)
            if gamma != 0.0:
                out += gamma * (self.adag_right(self.a_left(r)) - 0.5 * (self.n_col * r + r * self.n_row))
            if chi != 0.0:
                out += chi * (self.deph * r)
            return out

        if not with_sensitivity:
            return lambda t, y: lind(y)

        def f(t, y):
            rho, sig = y[0], y[1]
            out = self.xp.empty_like(y)
            out[0] = lind(rho)
            out[1] = lind(sig)
            if source:
                out[1] -= 1j * (self.q_left(rho) - self.q_right(rho))
            return out
        return f

    # ---- DOP853 on arrays of any shape -----------------------------------------
    def integrate(self, f, y, t_end):
        xp, rtol, atol = self.xp, self.rtol, self.atol
        n_el = y.size
        t, fy = 0.0, f(0.0, y)

        def rms(z):
            return float(xp.sqrt(xp.sum(xp.abs(z) ** 2).real / n_el))

        # initial step (scipy's select_initial_step)
        scale = atol + rtol * xp.abs(y)
        d0, d1 = rms(y / scale), rms(fy / scale)
        h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        f1 = f(h0, y + h0 * fy)
        d2 = rms((f1 - fy) / scale) / h0
        h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** (1.0 / 9.0)
        h = min(100 * h0, h1, t_end)

        K = [None] * (_NS + 1)
        steps = 0
        while t < t_end:
            if steps > self.max_steps:
                raise RuntimeError("DOP853: too many steps")
            h = min(h, t_end - t)
            K[0] = fy
            for s in range(1, _NS):
                dy = _A[s, 0] * K[0]
                for j in range(1, s):
                    if _A[s, j] != 0.0:
                        dy = dy + _A[s, j] * K[j]
                K[s] = f(t + _C[s] * h, y + h * dy)
            inc = _B[0] * K[0]
            for j in range(1, _NS):
                if _B[j] != 0.0:
                    inc = inc + _B[j] * K[j]
            y_new = y + h * inc
            f_new = f(t + h, y_new)
            K[_NS] = f_new
            scale = atol + rtol * xp.maximum(xp.abs(y), xp.abs(y_new))
            err5 = _E5[0] * K[0]; err3 = _E3[0] * K[0]
            for j in range(1, _NS + 1):
                if _E5[j] != 0.0:
                    err5 = err5 + _E5[j] * K[j]
                if _E3[j] != 0.0:
                    err3 = err3 + _E3[j] * K[j]
            e5 = float(xp.sum(xp.abs(err5 / scale) ** 2).real)
            e3 = float(xp.sum(xp.abs(err3 / scale) ** 2).real)
            if e5 == 0.0 and e3 == 0.0:
                err_norm = 0.0
            else:
                err_norm = abs(h) * e5 / np.sqrt((e5 + 0.01 * e3) * n_el)
            steps += 1
            if err_norm < 1.0:
                t, y, fy = t + h, y_new, f_new
                factor = _MAX_FACTOR if err_norm == 0.0 else min(_MAX_FACTOR, _SAFETY * err_norm ** _ERR_EXP)
                h *= factor
            else:
                h *= max(_MIN_FACTOR, _SAFETY * err_norm ** _ERR_EXP)
        self.last_steps = steps
        return y

    # ---- the three stages -----------------------------------------------------
    def run(self, rho0, *, kappa_ba=0.0, eta_in=1.0, pn_in=0.0, eta_ch=1.0, eta_out=1.0,
            t_final=1.0, epsilon_a=0.0):
        """Return (rho_out, drho_out/d eps_a) on the device, as the package orders the stages."""
        xp = self.xp
        rho = xp.asarray(_to_dense_dm(rho0, self.N))
        # stage 1: input noise (H = 0), unit duration
        g_in = loss_to_kappa(1 - eta_in) if eta_in < 1.0 else 0.0
        chi_in = phirms_to_chi(pn_in) if pn_in > 0.0 else 0.0
        if g_in or chi_in:
            rho = self.integrate(self.make_rhs(gamma=g_in, chi=chi_in), rho, 1.0)
        # stage 2: sensing with back-action and concurrent channel loss, duration t_final
        g = ba_to_g(kappa_ba, t_final) if kappa_ba != 0.0 else 0.0
        g_ch = loss_to_kappa(1 - eta_ch) if eta_ch < 1.0 else 0.0
        y = xp.stack([rho, xp.zeros_like(rho)])
        y = self.integrate(self.make_rhs(g=g, eps_a=epsilon_a, gamma=g_ch, with_sensitivity=True, source=True),
                           y, t_final)
        # stage 3: output noise (H = 0), unit duration; the sensitivity rides along
        g_out = loss_to_kappa(1 - eta_out) if eta_out < 1.0 else 0.0
        if g_out:
            y = self.integrate(self.make_rhs(gamma=g_out, with_sensitivity=True), y, 1.0)
        rho, sig = y[0], y[1]
        tr = float(xp.trace(rho).real)
        finite = bool(xp.isfinite(rho).all()) and bool(xp.isfinite(sig).all())
        if not finite or not np.isfinite(tr) or abs(tr - 1.0) > 1e-6:
            raise RuntimeError(f"lab-frame channel: trace(rho_out) = {tr!r}, finite = {finite} on {self.device} "
                               f"at N={self.N} (kappa_ba={kappa_ba}, steps={getattr(self, 'last_steps', None)})")
        return 0.5 * (rho + rho.conj().T), 0.5 * (sig + sig.conj().T)


def qfi_from_rho_drho(rho, drho, prec_eig=1e-8, xp=np):
    """Package formula: sum_{ij} 2 |<i|drho|j>|² / (p_i + p_j) over p_i + p_j > prec_eig."""
    vals, vecs = xp.linalg.eigh(rho)
    d = vecs.conj().T @ drho @ vecs
    denom = vals[:, None] + vals[None, :]
    mask = xp.abs(denom) > prec_eig
    return float(xp.sum(xp.where(mask, 2.0 * xp.abs(d) ** 2 / xp.where(mask, denom, 1.0), 0.0)).real)


def channel_labframe(rho, *, N_basis=None, device="cpu", rtol=1e-8, atol=1e-10, **channel):
    """(rho_out, drho_out/d eps_a) as numpy arrays."""
    N = N_basis or (rho.shape[0])
    ch = LabFrameChannel(N, device=device, rtol=rtol, atol=atol)
    r, s = ch.run(rho, **channel)
    if device == "gpu":
        r, s = r.get(), s.get()
    return r, s


def rp_qfi_labframe(rho, *, kappa_ba=0.0, eta_in=1.0, pn_in=0.0, eta_ch=1.0, eta_out=1.0,
                    t_final=1.0, N_basis=None, device="cpu", rtol=1e-8, atol=1e-10, prec_eig=1e-8):
    """QFI for ``epsilon_a`` of ``rho`` (qutip ket / density matrix, or ndarray) through the
    package channel, integrated in the lab frame on ``device`` ("cpu" or "gpu")."""
    N = N_basis or rho.shape[0]
    ch = LabFrameChannel(N, device=device, rtol=rtol, atol=atol)
    r, s = ch.run(rho, kappa_ba=kappa_ba, eta_in=eta_in, pn_in=pn_in, eta_ch=eta_ch,
                  eta_out=eta_out, t_final=t_final)
    return qfi_from_rho_drho(r, s, prec_eig=prec_eig, xp=ch.xp)
