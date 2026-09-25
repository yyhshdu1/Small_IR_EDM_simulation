"""LIF readout for the short IR, using centrex_beam's explicit cycling model.

Ported from the authoritative main-IR path (``run_full_16block_fluorescence.py`` plus
``apply_saved_fluorescence.py``). Nothing about the cycling physics is reimplemented --
``centrex_beam.photon_cycling.simulate_alternating_cycling`` is called directly, so the
short-IR readout is identical to the main-IR one apart from geometry.

What the readout adds to the projection-only model
--------------------------------------------------
Until now the analysis used the exact molecule counts N_1, N_2 with pure binomial
projection variance, and the photon budget entered only as the analytic factor
F = sqrt(1 + var/mean^2) ~ 3.21, never applied to the sampled data. That is optimistic in
a specific way: it is a scalar multiplying a quoted sigma, not extra noise in the numbers
the estimator actually sees.

Here the chain is the real one:

1. Each 10 us arrival bin is split into the two alternating 5 us laser windows -- even
   windows address channel 1, odd windows channel 2.
2. Within a window a molecule gets ``cycles_per_window`` saturated emission chances. After
   every emitted photon it is pumped dark with probability q_j (different for the two
   channels: 0.4822585 and 0.4756322). The pump-out photon itself is counted.
3. Bright survivors carry a residence budget of
   ``M = 1 + floor((L_det / v_ref) / laser_repeat_s)`` same-channel windows and rejoin
   later arrivals. At 50 cycles the first-window survival probability is (1-q)^50 ~ 1e-14,
   so in practice no one survives -- but the path is exercised, not assumed away.
4. Every emitted photon is detected independently with probability eta.
5. The analysis then works on *estimated* populations, not true counts:

       N_hat_j = D_j / (eta * n_bar_j),    n_bar_j = 1 / q_j

Uncertainties follow the main-IR treatment exactly: independent readout variance per
channel added to the saved projection variance, with the projection covariance between
channels kept NEGATIVE (N_1 + N_2 is fixed, so their projection noises anticorrelate).

    delta_j = q_j (1/eta - 1) + (1 - q_j)   ->  9.3097 and 9.1955 at eta = 0.052
    sigma inflation  = sqrt(1 + mean(delta)) = 3.2020

**What it changes.** Measured variance budget of the additive floor (f_CP = 0, 4 blocks,
40 seeds, paired so the common parts cancel): LIF readout 63%, shot-to-shot velocity
scatter 30%, spin projection 7%. The readout inflates the spin-projection term by 3.12
against the predicted 3.20. Photon noise, not velocity scatter, is the dominant term once
it is actually in the sampler.
"""
import numpy as np
import pandas as pd

from centrex_beam.photon_cycling import (detected_photon_moments,
                                         simulate_alternating_cycling)

CHANNELS = (1, 2)


def residence_windows(cfg, reference_velocity=None):
    """Same-channel laser windows a molecule is exposed to while crossing the detector.

    Uses a reference velocity rather than the bin's own velocity, matching the main-IR
    replay: the saved bins do not carry velocity moments. Immaterial while first-window
    survival is ~1e-14, but it would matter for slower cycling.
    """
    v = cfg.mean_velocity if reference_velocity is None else float(reference_velocity)
    return int(np.floor((cfg.detection_length / v) / cfg.laser_repeat_s)) + 1


def apply_fluorescence_to_shot(groups, cfg, *, rng, windows=None):
    """Cycle one shot's arrival bins and append the estimated-population columns.

    ``groups`` is the per-shot table from ``fast_shot`` (or ``run_short_ir_blocks``),
    carrying ``N_1``, ``N_2`` and ``variance_N_1`` per 10 us arrival bin. Returns a copy
    with the columns ``linear_switch_block_bin_analysis_from_estimates`` requires.
    """
    if not np.isclose(cfg.bin_width, cfg.laser_repeat_s):
        raise ValueError(
            f"Fluorescence cycling assumes one arrival bin per laser repeat period, but "
            f"bin_width = {cfg.bin_width:g} s and laser_repeat_s = {cfg.laser_repeat_s:g} s. "
            f"Cycle at the acquisition bin width and rebin afterwards for analysis.")
    windows = residence_windows(cfg) if windows is None else int(windows)
    frame = groups.sort_values("time_start_s").copy()

    # Interleave the two channels into consecutive 5 us laser windows.
    arrivals = np.empty(2 * len(frame), dtype=np.int64)
    arrivals[0::2] = frame.N_1.to_numpy(dtype=np.int64)
    arrivals[1::2] = frame.N_2.to_numpy(dtype=np.int64)

    result = simulate_alternating_cycling(
        arrivals,
        pumpout_probabilities=tuple(cfg.pumpout_probabilities),
        detection_efficiencies=tuple(cfg.detection_efficiencies),
        cycles_per_window=cfg.cycles_per_window,
        residence_windows=windows, rng=rng, return_cycle_trace=False)
    w = result["windows"]
    per_channel = (w.iloc[0::2].reset_index(drop=True), w.iloc[1::2].reset_index(drop=True))

    moments = [detected_photon_moments(q, eta, cfg.cycles_per_window * windows)
               for q, eta in zip(cfg.pumpout_probabilities, cfg.detection_efficiencies)]

    projection_variance = frame.variance_N_1.to_numpy(dtype=float)
    for channel, table, moment in zip(CHANNELS, per_channel, moments):
        frame[f"emitted_photons_{channel}"] = table.photons_emitted.to_numpy(np.int64)
        frame[f"detected_photons_{channel}"] = table.photons_detected.to_numpy(np.int64)
        frame[f"carried_bright_{channel}"] = table.carried_to_next.to_numpy(np.int64)
        frame[f"expected_detected_per_molecule_{channel}"] = moment["mean_detected"]
        frame[f"estimated_N_{channel}"] = (
            frame[f"detected_photons_{channel}"] / moment["mean_detected"])
        readout_factor = moment["variance_detected"] / moment["mean_detected"] ** 2
        frame[f"variance_estimated_N_{channel}"] = (
            projection_variance + frame[f"estimated_N_{channel}"] * readout_factor)

    # N_1 + N_2 is fixed per bin, so the projection parts anticorrelate.
    frame["covariance_estimated_N_1_N_2"] = -projection_variance

    total = frame.estimated_N_1 + frame.estimated_N_2
    frame["fluorescence_asymmetry"] = np.where(
        total > 0, (frame.estimated_N_1 - frame.estimated_N_2) / total, np.nan)
    return frame


def apply_fluorescence_to_run(out, cfg, *, seed=6062, verbose=True):
    """Cycle every shot of a ``run_short_ir_blocks`` result.

    Returns a new dict with ``shot_bins`` replaced by the fluorescence-augmented table.
    The original projection columns are preserved alongside, so the two readout models can
    be compared on identical spin realisations -- the only thing that changes is whether
    the estimator sees true counts or photon-derived estimates.
    """
    windows = residence_windows(cfg)
    children = np.random.SeedSequence(seed).spawn(len(out["bins"]))
    tables, carried = [], 0
    for (_, shot), frame, child in zip(out["summary"].iterrows(), out["bins"], children):
        lit = apply_fluorescence_to_shot(frame, cfg,
                                         rng=np.random.default_rng(child),
                                         windows=windows)
        lit["shot"] = int(shot["acquisition_order"])
        carried = max(carried, int(lit[["carried_bright_1", "carried_bright_2"]].max().max()))
        tables.append(lit)
    shot_bins = pd.concat(tables, ignore_index=True)

    truth = out["shot_bins"][["N_1", "N_2"]].sum()
    est = shot_bins[["estimated_N_1", "estimated_N_2"]].sum()
    closure = (est.to_numpy() / truth.to_numpy() - 1.0) * 100.0
    if verbose:
        print(f"fluorescence: {len(tables)} shots, residence budget {windows} windows, "
              f"max carryover {carried}")
        print(f"  calibrated totals vs truth: {closure[0]:+.4f}% / {closure[1]:+.4f}%")
        print(f"  sigma inflation sqrt(1 + mean delta) = "
              f"{readout_inflation(cfg):.4f}")
    merged = dict(out)
    merged["shot_bins"] = shot_bins
    merged["shot_bins_projection"] = out["shot_bins"]
    merged["fluorescence"] = dict(residence_windows=windows, max_carryover=carried,
                                  closure_percent=closure.tolist(),
                                  inflation=readout_inflation(cfg))
    return merged


def readout_inflation(cfg):
    """sqrt(1 + mean(delta_j)) with delta_j = q_j (1/eta - 1) + (1 - q_j).

    The closed form from the main-IR manuscript; 3.2020 at the reference numbers. The
    truncated-geometric moments used elsewhere give 3.2109, the difference being the
    50-cycle truncation.
    """
    q = np.asarray(cfg.pumpout_probabilities, dtype=float)
    eta = np.asarray(cfg.detection_efficiencies, dtype=float)
    delta = q * (1.0 / eta - 1.0) + (1.0 - q)
    return float(np.sqrt(1.0 + delta.mean()))
