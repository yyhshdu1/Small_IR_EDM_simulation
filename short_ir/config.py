"""Short interaction region parameters.

Everything the short-IR Monte Carlo needs, in one place. Values marked TODO are not
yet measured; the rest are either Emma's COMSOL settings (which are deliberately NOT
changed here) or the beam/detection model from ``centrex_asymmetry``.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

EMMA_REPO = Path(
    r"C:/Users/yuanhang.yang/Documents/GitHub/Ramsey-Simulation-for-Short-Interaction-Region"
)
CENTREX_ASYMMETRY = Path(r"C:/Users/yuanhang.yang/Documents/GitHub/centrex_asymmetry")


@dataclass
class ShortIRConfig:
    # ---- Emma's COMSOL field files. Do not change FIELD_SCALE or B2_PHASE. ----
    repo: Path = EMMA_REPO
    b1_file: str = "B_Ramsey_May6_FINAL_proc.csv"           # position[m], B1[G]
    b2_file: str = "B_Ramsey_May6_FINAL_proc_mirrored.csv"  # mirrored -> coil 2
    e_file: str = "E_Ramsey_May6_FINAL_proc.csv"            # position[m], E[V/m]
    coupling_file: str = "Coupling.csv"                     # E[kV/cm], |M|[Hz/G]
    sibsplit_file: str = "sib_split_vs_E.csv"               # E[kV/cm], nu_je[kHz]
    field_scale: float = 0.365      # Emma's value, fixed
    b2_phase: float = np.pi / 2     # Emma's value, fixed

    # ---- operating phase -------------------------------------------------
    # Coil-2 phase that puts the channel asymmetry at its zero crossing. Solved once
    # and then held fixed; re-solving per configuration would absorb the shifts being
    # measured. None means "solve it at construction".
    operating_phase_rad: float | None = None

    # ---- beam (centrex_asymmetry model) ----------------------------------
    mean_velocity: float = 184.0      # m/s. NOTE Emma's script hardcodes 180; 184 is
                                      # the design-paper / centrex_asymmetry value.
    velocity_spread: float = 16.0     # m/s, within one shot
    shot_velocity_spread: float = 4.0 # m/s, shot-to-shot scatter of the shot mean
    start_time_spread: float = 1e-3   # s
    mean_start_time: float = 0.0      # s

    # ---- geometry --------------------------------------------------------
    flight_length: float = 5.2        # m, source to detector (short IR).
                                      # Does not enter the Ramsey physics at all -- P(v)
                                      # comes from integrating Emma's field, which has its
                                      # own geometry. It sets arrival TIME, and therefore
                                      # how well an arrival-time bin sorts molecules by
                                      # velocity, which is what makes the bin-first
                                      # analysis work. centrex_asymmetry uses 7.9 m.

    # ---- magnetic field (measured, no drift) -----------------------------
    # Plot Bx is PARALLEL to E -> Hamiltonian z. By and Bz are transverse.
    b_par_G: float = 0.45e-3
    dnu_dB_par: float = -2486.4860    # Hz/G, from centrex_tlf
    dnu_transverse_Hz: float = 0.011496   # Hz, the measured transverse contribution,
                                          # from direct diagonalization at the full
                                          # measured field vector

    # ---- magnetic field DRIFT (single component, one direction) -----------
    # Only B_par drifts: it is the component that shifts nu_je at FIRST order
    # (dnu_dB_par = -2486 Hz/G). The transverse drift in the same record (Bz ~0.30 mG
    # over 18 h) enters at second order, ~0.8 mHz, and is neglected.
    # Default rate from the 18 h record: ~0.05 mG of B_par drift in 18 h, taken as a
    # linear, one-directional drift -> 0.0667 mG/day -> -0.166 Hz/day in nu_je.
    # Set to 0 to switch the drift off; flip the sign for the other direction.
    b_par_drift_G_per_day: float = 0.0   # OFF. Measured record: 0.05e-3 * 24 / 18
                                          # = 6.67e-5 G/day; set that to re-enable.
    # Time between consecutive runs. The field is held constant WITHIN a run (a run is
    # a short data-taking session; no shot repetition rate is defined in this model),
    # and steps between runs.
    run_spacing_days: float = 1.0

    # ---- motional magnetic field ------------------------------------------
    # A molecule moving at v through E sees B_mot = v x E / c^2 -- 60.8 uG at 184 m/s and
    # the 29.7 kV/cm plateau -- transverse to both v and E, and REVERSING with E. A static
    # field component b_perp along the same axis turns the second-order transverse Zeeman
    # shift kappa (b_perp + B_mot)^2 into something with an E-odd cross term,
    #     2 kappa b_perp B_mot    -> a fake CP-odd frequency, 25.3 nHz per uG of b_perp,
    # i.e. 50.5 nHz/uG between +E and -E (the CeNTREX proposal, arXiv:2010.01451 sec 3.3.2,
    # quotes ~50 nHz/uG). It is also proportional to v, so it varies across arrival bins.
    # The CP switch of this model is taken to be the E reversal: c = +1 is +E.
    include_motional_field: bool = True
    # Static field along v x E. The measured record's position is quoted as z = 44-88 cm,
    # i.e. its z is the BEAM axis; so the plot's By is the component perpendicular to both
    # v and E, and the plot's Bz is along the beam (no motional coupling). This axis
    # assignment is inferred, not stated -- swap the two values if it is the other way.
    b_perp_motional_G: float = -0.80e-3
    b_along_beam_G: float = -7.40e-3

    # kappa: second-order Zeeman coefficient for a field transverse to E. A FIXED physical
    # constant, obtained once from the measured record: dnu_transverse_Hz (the directly
    # diagonalised shift, 0.011496 Hz) divided by the measured |B_perp|^2 = 0.80^2 + 7.40^2
    # mG^2, assuming the response is isotropic about E (m_J = +-1 states about the E axis
    # have no preferred transverse direction). Do NOT re-derive it from b_perp_motional_G:
    # changing the field must not change the coefficient.
    transverse_zeeman_Hz_per_G2: float = 207.509

    # Values from run_full_16block_fluorescence.py / apply_saved_fluorescence.py, the
    # authoritative main-IR production path. The two readout channels are NOT identical:
    # their total rotational+vibrational pump-out probabilities differ.
    pumpout_probabilities: tuple = (0.4822585, 0.4756322)
    detection_efficiencies: tuple = (0.052, 0.052)
    cycles_per_window: int = 50       # 50 saturated emission chances per 5 us on-window
    laser_repeat_s: float = 10e-6     # same channel is addressed every 10 us
    detection_length: float = 0.025   # m, detector transit length (NOT 0.04; that was
                                      # an unused placeholder). Sets the residence budget
                                      # M = 1 + floor((L_det/v) / laser_repeat_s).

    @property
    def pumpout_probability(self) -> float:
        """Channel-1 value, for the analytic single-channel excess-noise factor."""
        return float(self.pumpout_probabilities[0])

    @property
    def detection_efficiency(self) -> float:
        return float(self.detection_efficiencies[0])

    # ---- numerics --------------------------------------------------------
    n_z: int = 4000                   # z steps through the field
    n_v_table: int = 1500             # velocity grid for the P(v) table
    v_table_sigma: float = 6.0        # table spans mean +- this many velocity_spread
    n_v_bins: int = 200               # velocity bins for the fast sampler
    bin_width: float = 10e-6          # s, ACQUISITION arrival-bin width.
        # 10 us, as centrex_asymmetry records. Fine bins make a clean time-of-flight
        # trace; they are not what the extraction should be run at.

    analysis_bin_width: float = 1000e-6   # s, ANALYSIS arrival-bin width.
        # Matched to the TOF velocity resolution, which the start-time spread caps at
        # sigma_ts * v^2 / L = 6.5 m/s -- about 1 ms of arrival time. Finer bins add no
        # velocity information, only noise per bin.

    def b_par_at_run(self, run_index: int) -> float:
        """B_par (G) during run ``run_index``, counted from this config's b_par_G."""
        return (self.b_par_G
                + self.b_par_drift_G_per_day * self.run_spacing_days * run_index)

    def drifted(self, run_index: int) -> "ShortIRConfig":
        """Copy of this config with B_par advanced to run ``run_index``.

        Always call it on the BASE config (run 0); calling it on an already-drifted copy
        would add the drift twice.
        """
        return replace(self, b_par_G=self.b_par_at_run(run_index))

    def path(self, name: str) -> Path:
        return self.repo / name

    def velocity_table_range(self) -> tuple[float, float]:
        half = self.v_table_sigma * self.velocity_spread + 4.0 * self.shot_velocity_spread
        return max(1.0, self.mean_velocity - half), self.mean_velocity + half


# Switch amplitudes, matching centrex_asymmetry's reference settings.
DETUNING_SWITCH_HZ = 0.5
CALIBRATION_PHASE_RAD = np.pi / 32
