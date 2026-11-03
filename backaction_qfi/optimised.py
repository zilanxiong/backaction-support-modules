"""The group's previously optimised probe states, oriented for the GW signal.

The data in ``data/optimised_states/`` was produced for a channel with loss
and phase noise but no radiation pressure, and for the signal ``epsilon_p``
(generator ``p``).  The GW signal in the back-action channel is ``epsilon_a``
(generator ``x``).  Without back-action the two are related by a quarter-turn
rotation of phase space, which is a symmetry of that channel, so the stored
states are still optimal for ``epsilon_a`` once rotated.  With back-action the
rotation is no longer a symmetry, so the orientation has to be right *before*
the state enters the channel: :func:`load_probe` rotates by default.
"""

import os
from pathlib import Path

import numpy as np
import qutip as qt

import quantum_sensing as qs

__all__ = ["DATA_DIR", "LOSS_KWARG", "list_entries", "best_entry", "rotate",
           "quarter_turn", "load_probe", "channel_kwargs"]

def _unpack_if_needed(d):
    """The repository stores ``states.h5.gz`` (2.5 MB); unpack to ``states.h5`` (29 MB) once."""
    gz, h5 = d / "states.h5.gz", d / "states.h5"
    if gz.exists() and not h5.exists():
        import gzip
        import shutil
        with gzip.open(gz, "rb") as fin, open(h5, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return h5.exists()


def _find_data_dir():
    """``data/optimised_states/`` next to this package if present, else the directory
    named by ``QUANTUM_SENSING_DATA_DIR`` (the package's own convention, used when
    this study layer lives inside the upstream repository)."""
    here = Path(__file__).resolve().parent.parent / "data" / "optimised_states"
    if _unpack_if_needed(here):
        return here
    env = os.environ.get("QUANTUM_SENSING_DATA_DIR")
    if env and _unpack_if_needed(Path(env)):
        return Path(env)
    return here        # may not exist; the first load then raises a clear error


DATA_DIR = _find_data_dir()

#: How each stored ``loss_config`` maps onto the package channel's loss argument.
LOSS_KWARG = {"loss_in": "eta_in", "loss_ch": "eta_ch", "loss_out": "eta_out"}


def _use_data_dir():
    if not _unpack_if_needed(DATA_DIR):
        raise FileNotFoundError(
            f"optimised-state data not found at {DATA_DIR}; put states.h5 (or states.h5.gz) and "
            "master_results.csv there or set QUANTUM_SENSING_DATA_DIR to the directory that holds them")
    qs.set_data_dir(DATA_DIR)


def list_entries(state_type, loss_config, eta, pn):
    """All stored optimisation results for one ``(state_type, loss_config, eta, pn)`` slice."""
    _use_data_dir()
    return qs.load_state_params(state_type, loss_config, eta, pn)


def best_entry(state_type, loss_config, eta, pn, N_target):
    """The highest-QFI stored entry at ``N_target``.  Raises if the slice is empty."""
    entries = [e for e in list_entries(state_type, loss_config, eta, pn)
               if np.isclose(e["N_target"], N_target)]
    if not entries:
        raise LookupError(f"no stored {state_type}/{loss_config} state at eta={eta}, pn={pn}, N_target={N_target}")
    return max(entries, key=lambda e: e["qfi"])


def rotate(psi, theta):
    """Phase-space rotation ``exp(-i theta n)``: ``c_n -> e^{-i n theta} c_n``.

    Heisenberg picture: ``a -> a e^{-i theta}``, so ``theta = pi/2`` maps
    ``x -> p`` (the new ``x`` is the old ``p``).
    """
    n = np.arange(psi.shape[0])
    return qt.Qobj(psi.full().ravel() * np.exp(-1j * n * theta))


def quarter_turn(psi):
    """Rotate by 90°: a state elongated along ``p`` becomes elongated along ``x``."""
    return rotate(psi, 0.5 * np.pi)


def load_probe(state_type, loss_config, eta, pn, N_target, N_basis=None, oriented_for="epsilon_a"):
    """Reconstruct the best stored probe as a ket at cutoff ``N_basis``.

    ``oriented_for="epsilon_a"`` (default) applies the quarter turn so the
    state is oriented for the GW signal; ``"epsilon_p"`` returns it as stored.
    ``N_basis`` defaults to the cutoff the optimisation used.
    Returns ``(psi, entry)``.
    """
    entry = best_entry(state_type, loss_config, eta, pn, N_target)
    N = int(N_basis or entry["N_basis"])
    psi = qs.reconstruct_state(entry, N_basis=N).unit()
    if oriented_for == "epsilon_a":
        psi = quarter_turn(psi)
    elif oriented_for != "epsilon_p":
        raise ValueError("oriented_for must be 'epsilon_a' or 'epsilon_p'")
    return psi, entry


def channel_kwargs(entry):
    """The package-channel arguments the stored QFI was computed with
    (loss at the stored placement, phase noise at the input)."""
    return {LOSS_KWARG[entry["loss_config"]]: entry["eta"], "pn_in": entry["pn"]}
