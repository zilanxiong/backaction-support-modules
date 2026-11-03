# Optimised probe states (no back-action)

The group's optimised single-mode probes for the loss + phase-noise channel
*without* radiation pressure: Fock superpositions (`fock_sup`), squeezed-vacuum
superpositions (`sqz_vac`) and squeezed-coherent superpositions (`sqz_coh`),
over loss placement (`loss_in`, `loss_ch`, `loss_out`), transmissivity `eta`,
phase-noise rms `pn` and target photon number `N_target`.

* `states.h5.gz`: state parameters (gzipped, 2.5 MB). The loader unpacks it to `states.h5`
  (29 MB, gitignored) on first use; `quantum_sensing.states.load_state_params` reads that.
* `master_results.csv`: one row per optimisation run (`qfi`, `N_actual`, `N_basis`, ...).

Copied unchanged from the archived repository (`reference/quantum_sensing/data`).

**Orientation.** These states were optimised for the signal `epsilon_p`
(generator `p`, displaces `x`): their stored `qfi` reproduces under
`param_type="epsilon_p"`, not under the GW signal `epsilon_a`. Without
back-action the two differ only by a 90° phase-space rotation, so
`backaction_qfi.optimised.load_probe` rotates each state by a quarter turn
(`c_n -> (-i)^n c_n`) so that it is oriented for `epsilon_a`.

**Known discrepancy (to raise with the package author).** For the two squeezed
families (`sqz_vac`, `sqz_coh`) the Python reconstruction gives a mean photon
number about 1.7-1.8 % *above* the stored `N_actual` (e.g. 5.085 vs 5.000 at
`N_target = 5`), independent of cutoff, while the stored `qfi` still
reproduces to 1e-4. Fock superpositions reproduce `N_actual` exactly. The
Julia optimiser and `quantum_sensing.states.reconstruct_sqz_*` therefore
differ slightly in how the photon-number constraint was evaluated.
`tests/test_optimised_states.py` pins the current numbers.
