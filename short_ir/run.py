"""Drive a three-switch schedule and hand the result to centrex_beam's analysis.

The schedule and every downstream estimator come from ``centrex_beam.switching``
unchanged. Only the probability model is replaced: instead of the analytic
``ramsey_probabilities``, each switch state gets a ``RamseyTable`` built by integrating
through Emma's measured RF field.
"""
import time

import numpy as np
import pandas as pd

from centrex_beam.switching import (average_time_bins_by_switch,
                                    fit_block_frequency_from_bin_parities,
                                    linear_switch_analysis,
                                    linear_switch_block_bin_analysis,
                                    linear_switch_block_bin_analysis_from_estimates,
                                    make_three_switch_schedule)

from .fast_shot import simulate_shot_binned, simulate_shot_streaming
from .ramsey import build_switch_tables, table_key

SWITCH_COLUMNS = ("detuning_state", "phase_state", "cp_state")


def run_short_ir_blocks(field, cfg, *, n_blocks, n_molecules, shots_per_state=1,
                        detuning_hz, calibration_phase_rad, cp_frequency_hz,
                        operating_phase_rad, seed=3030, schedule_seed=3031,
                        shot_mean_seed=None, method="binned", randomize=True,
                        verbose=True):
    """Run balanced eight-state blocks and pool them by switch state.

    Returns a dict with ``schedule``, ``summary``, ``bins``, ``state_bins`` and
    ``tables``, matching what ``centrex_beam.switching.run_three_switch_shots``
    produces so the same analysis functions apply. ``summary`` records each shot's
    mean velocity in ``shot_mean``.

    ``shot_mean_seed``: if given, every shot's mean velocity is drawn up front from its
    own random stream. Two runs with the same ``shot_mean_seed`` and schedule then see
    the SAME beam, shot for shot, whatever else differs (RF amplitude, readout) -- a
    paired comparison. If None, the shot mean is drawn inside the sampler as before.
    """
    if method not in ("binned", "streaming"):
        raise ValueError("method must be 'binned' or 'streaming'")

    schedule = make_three_switch_schedule(
        n_blocks, shots_per_state, detuning_hz=detuning_hz,
        calibration_phase_rad=calibration_phase_rad,
        cp_frequency_hz=cp_frequency_hz, randomize=randomize, seed=schedule_seed)

    if verbose:
        print(f"building P(v) tables for {len(schedule)} shots "
              f"({len(schedule.groupby(list(SWITCH_COLUMNS)))} distinct states)")
    t0 = time.perf_counter()
    tables = build_switch_tables(field, cfg, schedule,
                                 operating_phase_rad=operating_phase_rad,
                                 cp_frequency_Hz=cp_frequency_hz, verbose=verbose)
    t_tables = time.perf_counter() - t0

    simulate = simulate_shot_binned if method == "binned" else simulate_shot_streaming
    rng = np.random.default_rng(seed)

    shot_means = (None if shot_mean_seed is None else
                  np.random.default_rng(shot_mean_seed).normal(
                      cfg.mean_velocity, cfg.shot_velocity_spread, len(schedule)))

    rows, bins = [], []
    t0 = time.perf_counter()
    for i, (_, shot) in enumerate(schedule.iterrows()):
        result = simulate(cfg, tables[table_key(shot)], n_molecules, rng=rng,
                          shot_mean=None if shot_means is None else float(shot_means[i]))
        whole = result["whole_shot"]
        row = {c: int(shot[c]) for c in SWITCH_COLUMNS}
        row.update(
            acquisition_order=int(shot["acquisition_order"]),
            block=int(shot["block"]), switch_label=shot["switch_label"],
            detuning_hz=float(shot["detuning_hz"]),
            phase_rad=float(shot["phase_rad"]),
            cp_frequency_hz=float(shot["cp_frequency_hz"]),
            n=whole["n"], N_1=whole["N_1"], N_2=whole["N_2"],
            asymmetry=whole["asymmetry"], sigma_asymmetry=whole["sigma_asymmetry"],
            expected_N_1=result["expected_N_1"],
            shot_mean=result["shot_mean"])
        rows.append(row)
        frame = result["groups"].copy()
        frame["block"] = int(shot["block"])
        for c in SWITCH_COLUMNS:
            frame[c] = int(shot[c])
        frame["switch_label"] = shot["switch_label"]
        bins.append(frame)
    t_shots = time.perf_counter() - t0

    summary = pd.DataFrame(rows)
    state_bins = average_time_bins_by_switch(summary, bins, bin_width=cfg.bin_width)
    shot_bins = pd.concat(bins, ignore_index=True)

    if verbose:
        print(f"\ntables {t_tables:.2f} s, {len(schedule)} shots {t_shots:.2f} s "
              f"({t_shots/len(schedule)*1e3:.1f} ms/shot, method={method})")
    return dict(schedule=schedule, summary=summary, bins=bins,
                state_bins=state_bins, shot_bins=shot_bins, tables=tables,
                timing=dict(tables_s=t_tables, shots_s=t_shots, method=method))


def extract(state_bins, *, calibration_phase_rad, detuning_hz, tail_fraction=0.01):
    """Pooled three-switch estimate: centrex_beam's linear estimator, unchanged.

    Returns ``aggregate`` (one estimate over the whole retained window) and
    ``time_bins`` (one estimate per arrival bin). The aggregate mixes velocities,
    which is exactly what makes it vulnerable to shot-to-shot velocity scatter --
    prefer ``extract_bin_first`` below.
    """
    return linear_switch_analysis(state_bins,
                                  calibration_phase_rad=calibration_phase_rad,
                                  detuning_hz=detuning_hz,
                                  tail_fraction=tail_fraction)


def extract_bin_first(out, *, calibration_phase_rad, detuning_hz, bin_width,
                      tail_fraction=0.01, min_count_per_state=100):
    """Bin-first extraction from true molecule counts (projection-only readout).

    Molecules in the same arrival-time bin share (to the extent the start-time spread
    allows) the same velocity, hence the same interaction time and pulse area. So the
    three-switch linearization holds bin by bin, instead of being applied to a mixture
    of velocities. This is the route ``centrex_asymmetry_study.ipynb`` uses.

    The per-bin CP frequency is the ratio A_cp / A_detuning, which blows up wherever
    A_detuning passes through zero. ``fit_block_frequency_from_bin_parities`` avoids
    that by fitting a single slope r = f_CP / df_sw to A_cp = r * A_detuning across all
    bins of a block. Run it at the analysis bin width (~1 ms, the TOF velocity
    resolution), not at the 10 us acquisition width.
    """
    window = _window(out, calibration_phase_rad, detuning_hz, tail_fraction)
    per_bin = linear_switch_block_bin_analysis(
        out["shot_bins"], calibration_phase_rad=calibration_phase_rad,
        detuning_hz=detuning_hz, time_start_s=float(window["time_start_s"]),
        time_end_s=float(window["time_end_s"]),
        min_count_per_state=min_count_per_state, bin_width=bin_width)
    return _fit_and_combine(window, per_bin, detuning_hz)


def extract_bin_first_fluorescence(out, *, calibration_phase_rad, detuning_hz,
                                   bin_width, tail_fraction=0.01,
                                   min_count_per_state=100):
    """Bin-first extraction from LIF-estimated populations instead of true counts.

    Same structure as ``extract_bin_first``, but the per-bin asymmetry is formed from
    ``estimated_N_1/2`` with the readout variance folded in and the negative projection
    covariance retained -- via ``linear_switch_block_bin_analysis_from_estimates``, the
    function the main-IR production path uses. ``out`` must come from
    ``fluorescence.apply_fluorescence_to_run``.

    Returns the same keys as ``extract_bin_first``, so the two are directly comparable on
    the same spin realisations: the ONLY difference is whether the estimator sees exact
    molecule counts or photon-derived estimates.
    """
    required = {"estimated_N_1", "variance_estimated_N_1", "covariance_estimated_N_1_N_2"}
    missing = required.difference(out["shot_bins"].columns)
    if missing:
        raise ValueError(
            f"shot_bins lacks the fluorescence columns {sorted(missing)}; run "
            f"short_ir.fluorescence.apply_fluorescence_to_run first.")
    window = _window(out, calibration_phase_rad, detuning_hz, tail_fraction)
    per_bin = linear_switch_block_bin_analysis_from_estimates(
        out["shot_bins"], calibration_phase_rad=calibration_phase_rad,
        detuning_hz=detuning_hz, time_start_s=float(window["time_start_s"]),
        time_end_s=float(window["time_end_s"]),
        min_count_per_state=min_count_per_state, bin_width=bin_width)
    return _fit_and_combine(window, per_bin, detuning_hz)


def _window(out, calibration_phase_rad, detuning_hz, tail_fraction):
    """Common retained arrival window: 1% of the molecules cut from each tail."""
    return linear_switch_analysis(
        out["state_bins"], calibration_phase_rad=calibration_phase_rad,
        detuning_hz=detuning_hz, tail_fraction=tail_fraction)["aggregate"]


def _fit_and_combine(window, per_bin, detuning_hz):
    """Per-block slope fit, then a PLAIN MEAN over blocks.

    The plain mean matches the main-IR production path (apply_saved_fluorescence.py).
    Inverse-variance weights would use sigma_block, which depends on the realised detuning
    parity of that block, and so correlate weight with the block's own error.

    ``block_sem_hz`` is the empirical uncertainty and includes shot-to-shot velocity
    scatter; ``sigma_conditional_hz`` propagates counting (and, for LIF, readout) noise
    only and understates the real spread. Quote the former.
    """
    blocks = fit_block_frequency_from_bin_parities(per_bin, detuning_hz=detuning_hz)
    f = blocks["frequency_hz"].to_numpy(dtype=float)
    s = blocks["sigma_frequency_hz"].to_numpy(dtype=float)
    combined = dict(
        frequency_hz=float(np.mean(f)),
        sigma_conditional_hz=float(np.sqrt(np.sum(s**2)) / f.size),
        block_sd_hz=float(np.std(f, ddof=1)) if f.size > 1 else np.nan,
        block_sem_hz=(float(np.std(f, ddof=1)/np.sqrt(f.size))
                      if f.size > 1 else np.nan),
        contrast=float(window["contrast"]),
        n_blocks=int(f.size), n_bins=int(len(per_bin)))
    return dict(window=window, per_bin=per_bin, blocks=blocks, combined=combined)
