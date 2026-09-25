"""Population transfer through Emma's real RF field, as a function of velocity.

This replaces ``centrex_beam.projection.ramsey_probabilities``, which is the analytic
two-factor model

    P1(v) = sin^2(Omega * L_p / v) * cos^2(pi/4 + phi/2 + pi * df * L_R / v)

i.e. an idealized pair of square pulses separated by field-free precession. The short-IR
coils do not look like that: their fields overlap, have spatial sign structure, and sit
inside an E-field plateau whose fringes move nu_je by up to 1.3 kHz. So P(v) is obtained
here by integrating the two-level Schroedinger equation through the actual field.

The cost is made irrelevant by two things:

1. **Vectorization over velocity.** At each z step the rotation generator is the same for
   every molecule; only the time spent in the step, dt = dz / v, differs. So one pass
   over the z grid propagates the whole velocity grid at once.
2. **Tabulate then interpolate.** P(v) is smooth, so it is built once per switch state on
   a fine velocity grid and interpolated afterwards. Downstream code then sees exactly
   the same interface as ``ramsey_probabilities``.

Convention (identical to Emma's ramsey4matrix.py and to the 2-level notebook):

    H = 0.5 * [[-delta, conj(Omega_c)], [Omega_c, delta]]
    Omega_c(z) = 2*pi*M(E(z)) * [B1(z) + B2(z)*exp(i*phase_2)]
    delta(z)   = 2*pi*(nu_RF - nu_je(z) - shifts)

with no extra factor of 1/2 in the Rabi rate.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


C_LIGHT = 299_792_458.0


def propagate_velocities(field, velocities, *, phase_2, detuning_Hz=0.0,
                         b_par_G=0.0, dnu_dB_par=0.0, dnu_transverse_Hz=0.0,
                         motional=None):
    """Return P_e for each velocity, integrating through the real field.

    Parameters
    ----------
    field : ShortIRField
    velocities : (n_v,) array, m/s
    phase_2 : coil-2 relative phase in rad (operating phase + calibration switch)
    detuning_Hz : added to nu_je; carries the detuning switch and the CP shift
    motional : None, or (kappa [Hz/G^2], b_perp [G], e_sign [+-1]). Adds the
        E-DEPENDENT part of the transverse Zeeman shift from the motional field,
            kappa * [(b_perp + B_mot)^2 - b_perp^2],   B_mot = -e_sign v E(z) / c^2,
        on top of dnu_transverse_Hz (which already holds kappa * |B_perp,static|^2).
        B_mot depends on v, so |h| is then velocity dependent at every z step; the
        propagation is still vectorised over velocity, just with (n_v,) arrays per step.

    Returns
    -------
    (n_v,) array of excited-state populations.

    Exact 2x2 propagator per step: H is traceless Hermitian, so with
    H = h.sigma and |h| = 0.5*sqrt(delta^2 + |Omega|^2),

        U = cos(|h| dt) I - i sin(|h| dt)/|h| H

    |h| and H depend only on z, dt only on v -- which is what makes the batching work.
    """
    v = np.asarray(velocities, dtype=float)
    if v.ndim != 1 or v.size == 0 or np.any(v <= 0):
        raise ValueError("velocities must be a nonempty 1-D array of positive floats")

    omega = field.rabi_complex(phase_2)
    delta = field.detuning(b_par_G=b_par_G, dnu_dB_par=dnu_dB_par,
                           dnu_transverse_Hz=dnu_transverse_Hz,
                           extra_Hz=detuning_Hz)

    # Midpoint values on each interval.
    om_m = 0.5 * (omega[:-1] + omega[1:])
    dl_m = 0.5 * (delta[:-1] + delta[1:])
    dz = field.dz

    hx = 0.5 * om_m.real
    hy = 0.5 * om_m.imag
    hz = -0.5 * dl_m
    hn = np.sqrt(hx * hx + hy * hy + hz * hz)
    hn = np.where(hn < 1e-30, 1e-30, hn)

    # psi[:, 0] = amplitude in j, psi[:, 1] = amplitude in e
    a = np.ones(v.size, dtype=complex)
    b = np.zeros(v.size, dtype=complex)
    inv_v = 1.0 / v

    if motional is not None:
        return _propagate_with_motional(field, v, hx, hy, dl_m, dz, motional, a, b)

    for k in range(dz.size):
        theta = hn[k] * dz[k] * inv_v          # (n_v,)
        c = np.cos(theta)
        s = np.sin(theta) / hn[k]
        h00, h11 = hz[k], -hz[k]
        h01 = hx[k] - 1j * hy[k]
        h10 = hx[k] + 1j * hy[k]
        a_new = c * a - 1j * s * (h00 * a + h01 * b)
        b_new = c * b - 1j * s * (h10 * a + h11 * b)
        a, b = a_new, b_new

    return np.abs(b) ** 2


def _propagate_with_motional(field, v, hx, hy, dl_m, dz, motional, a, b):
    """Same exact 2x2 stepping, with a velocity-dependent detuning at every z step."""
    kappa, b_perp, e_sign = motional
    e_vm = 0.5 * (field.E_kVcm[:-1] + field.E_kVcm[1:]) * 1e5          # V/m, midpoints
    g_per_ms = -float(e_sign) * e_vm / C_LIGHT**2 * 1e4                  # B_mot/v in G/(m/s)
    inv_v = 1.0 / v
    for k in range(dz.size):
        bm = g_per_ms[k] * v                                             # (n_v,) G
        dnu = kappa * (2.0 * b_perp * bm + bm * bm)                      # Hz, E-dependent part
        hz = -0.5 * (dl_m[k] - 2 * np.pi * dnu)                          # delta = 2pi(nu_RF - nu)
        hn = np.sqrt(hx[k] ** 2 + hy[k] ** 2 + hz ** 2)
        hn = np.where(hn < 1e-30, 1e-30, hn)
        theta = hn * dz[k] * inv_v
        c = np.cos(theta)
        s = np.sin(theta) / hn
        h01 = hx[k] - 1j * hy[k]
        h10 = hx[k] + 1j * hy[k]
        a, b = c * a - 1j * s * (hz * a + h01 * b), c * b - 1j * s * (h10 * a - hz * b)
    return np.abs(b) ** 2


def motional_args(cfg, e_sign):
    """``motional=`` argument for propagate_velocities, or None if switched off."""
    if not getattr(cfg, "include_motional_field", False):
        return None
    return (cfg.transverse_zeeman_Hz_per_G2, cfg.b_perp_motional_G, int(e_sign))


def find_operating_phase(field, cfg, *, bracket=(-np.pi, np.pi), n_scan=25):
    """Coil-2 phase giving P_e = 0.5 at the nominal velocity (the fringe zero crossing).

    Solve once and freeze. The readout is the channel asymmetry A = 1 - 2 P_e, so the
    zero crossing is where the phase slope is largest.
    """
    v0 = np.array([cfg.mean_velocity])

    def f(phase):
        return propagate_velocities(
            field, v0, phase_2=float(phase),
            b_par_G=cfg.b_par_G, dnu_dB_par=cfg.dnu_dB_par,
            dnu_transverse_Hz=cfg.dnu_transverse_Hz)[0] - 0.5

    grid = np.linspace(*bracket, n_scan)
    values = np.array([f(p) for p in grid])
    crossings = np.nonzero(np.diff(np.sign(values)))[0]
    if crossings.size == 0:
        raise RuntimeError("no P_e = 0.5 crossing found; check the field amplitude")
    best = crossings[np.argmax(np.abs(np.diff(values)[crossings]))]
    return float(brentq(f, grid[best], grid[best + 1], xtol=1e-12))


class RamseyTable:
    """P(v) for one switch state, tabulated once and interpolated afterwards.

    Drop-in for ``centrex_beam.projection.ramsey_probabilities``: call it with an array
    of velocities and it returns channel-1 probabilities.

    **Sign convention -- note the mismatch.** Here channel 1 is the state the molecules
    START in (j), so ``P_1(v) = 1 - P_e(v)``. ``centrex_beam`` uses the OPPOSITE: its
    ``ramsey_probabilities`` returns sin^2(Omega L_p / v) * cos^2(...), which vanishes
    when the drive vanishes, so its channel 1 is the TRANSFERRED state (e), not the
    initial one. Take v -> infinity to see it: no time in the field, no transfer, and
    ``ramsey_probabilities`` -> 0 while a ground-state population would -> 1.

    The mismatch is a global sign on the asymmetry A = (N_1 - N_2)/n, and it does not
    affect anything measured. Every estimator in ``centrex_beam.switching`` is a ratio of
    parities, and the parities are linear in A:

        f_CP = df_sw * A_cp / A_detuning        T = scale * A_detuning / A_phase
        phi_CP = phi_cal * A_cp / A_phase       contrast = |A_phase / phi_cal|

    so A -> -A cancels throughout (verified numerically). Only the sign of the offset
    A_0 and the vertical orientation of the asymmetry plots depend on the choice.
    """

    def __init__(self, field, cfg, *, phase_2, detuning_Hz=0.0, n_v=None,
                 v_range=None, e_sign=None):
        self.field = field
        self.cfg = cfg
        self.phase_2 = float(phase_2)
        self.detuning_Hz = float(detuning_Hz)
        n_v = cfg.n_v_table if n_v is None else int(n_v)
        lo, hi = cfg.velocity_table_range() if v_range is None else v_range
        self.v_grid = np.linspace(lo, hi, n_v)
        self.p_excited = propagate_velocities(
            field, self.v_grid, phase_2=self.phase_2,
            detuning_Hz=self.detuning_Hz, b_par_G=cfg.b_par_G,
            dnu_dB_par=cfg.dnu_dB_par, dnu_transverse_Hz=cfg.dnu_transverse_Hz,
            motional=None if e_sign is None else motional_args(cfg, e_sign))
        self.p_channel1 = 1.0 - self.p_excited
        self._interp = interp1d(self.v_grid, self.p_channel1, kind="cubic",
                                bounds_error=False,
                                fill_value=(self.p_channel1[0], self.p_channel1[-1]))

    def __call__(self, velocities):
        return np.clip(self._interp(np.asarray(velocities, dtype=float)), 0.0, 1.0)

    def bin_average(self, edges, weights_pdf):
        """Weighted mean of P_1 over each velocity bin.

        Using the bin-averaged probability rather than the bin-centre value makes the
        binned sampler in ``fast_shot`` correct to second order in the bin width, so a
        few hundred bins suffice.

        ``weights_pdf`` is called on velocities and must return the (unnormalized)
        velocity density.
        """
        edges = np.asarray(edges, dtype=float)
        out = np.empty(edges.size - 1)
        for i in range(edges.size - 1):
            vv = np.linspace(edges[i], edges[i + 1], 9)
            w = weights_pdf(vv)
            total = np.trapezoid(w, vv)
            out[i] = (np.trapezoid(self(vv) * w, vv) / total if total > 0
                      else self(0.5 * (edges[i] + edges[i + 1])))
        return out


def build_switch_tables(field, cfg, schedule, *, operating_phase_rad,
                        cp_frequency_Hz, verbose=True):
    """One RamseyTable per distinct (phase_rad, detuning_hz, cp_state) in the schedule.

    The schedule comes from ``centrex_beam.switching.make_three_switch_schedule``, whose
    columns are reused verbatim. The CP reversal enters as a frequency shift added to
    nu_je, so the propagator produces its velocity-dependent phase automatically rather
    than assuming the analytic 2*pi*f_CP*L/v form. The CP switch is also the E reversal,
    so it sets the sign of the motional field (``cfg.include_motional_field``).
    """
    keys = schedule[["phase_rad", "detuning_hz", "cp_state"]].drop_duplicates()
    tables = {}
    for n, (_, row) in enumerate(keys.iterrows(), start=1):
        key = (float(row.phase_rad), float(row.detuning_hz), int(row.cp_state))
        extra = key[1] + key[2] * cp_frequency_Hz
        tables[key] = RamseyTable(field, cfg,
                                  phase_2=operating_phase_rad + key[0],
                                  detuning_Hz=extra,
                                  e_sign=key[2])      # CP switch = E reversal
        if verbose:
            print(f"  [{n}/{len(keys)}] phase={key[0]:+.6f} rad  "
                  f"detuning={key[1]:+.3f} Hz  cp={key[2]:+d}")
    return tables


def table_key(row):
    """Schedule row -> the key used by ``build_switch_tables``."""
    return (float(row["phase_rad"]), float(row["detuning_hz"]), int(row["cp_state"]))
