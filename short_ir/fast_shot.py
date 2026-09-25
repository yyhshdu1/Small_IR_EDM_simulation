"""Two shot samplers producing identical output structures.

``simulate_shot_binned`` is the fast one; ``simulate_shot_streaming`` is the
per-molecule reference used to validate it.

Why the fast one is legitimate
------------------------------
Per shot the statistical model is: draw N molecules, each with velocity v ~ N(vbar, sv),
start time ts ~ N(0, st), spin ~ Bernoulli(p(v)), arriving at t = ts + L/v. Spin and
start time are independent given v, and p depends only on v. So the exact joint
distribution of counts over (velocity bin, arrival bin, spin) factorizes:

    n_j          ~ Multinomial(N, q_j)            q_j = mass of velocity bin j
    N1_j         ~ Binomial(n_j, pbar_j)          pbar_j = bin-averaged p
    N1_j -> bins ~ Multinomial(N1_j, w_jk)        w_jk = arrival-bin mass given v_j
    N2_j -> bins ~ Multinomial(N2_j, w_jk)        drawn independently

Two approximations, both controlled:
  * p is replaced by its bin average (see RamseyTable.bin_average), so the error is
    second order in the velocity bin width, not first;
  * the flight time L/v is evaluated at the bin centre. The spread of L/v across a bin
    is L*dv/v^2, distributed as a top-hat; it is folded into the arrival weights by
    adding its variance to the start-time variance, so nothing is neglected to the
    accuracy of treating that convolution as Gaussian.

Cost is ~2 multinomial draws per velocity bin instead of N Bernoulli draws: roughly
20 ms versus 6 s for a 70M-molecule shot.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.special import erf

from centrex_beam.beam import (arrival_times, generate_start_times,
                               generate_velocities)
from centrex_beam.projection import project_spins


def _gauss_bin_mass(edges, mean, sigma):
    """Mass of N(mean, sigma) in each bin, exactly, via the error function."""
    z = (np.asarray(edges, dtype=float) - mean) / (sigma * np.sqrt(2.0))
    return np.diff(0.5 * (1.0 + erf(z)))


def _assemble(n_total, n1, edges, n_molecules, variance_n1, expected_n1):
    """Build the groups/whole_shot structure that centrex_beam's analysis expects."""
    n2 = n_total - n1
    populated = n_total > 0
    asym = np.full(n_total.size, np.nan)
    sigma = np.full(n_total.size, np.nan)
    model = np.full(n_total.size, np.nan)
    asym[populated] = (n1[populated] - n2[populated]) / n_total[populated]
    sigma[populated] = 2 * np.sqrt(variance_n1[populated]) / n_total[populated]
    model[populated] = 2 * expected_n1[populated] / n_total[populated] - 1
    groups = pd.DataFrame(dict(
        group=np.arange(n_total.size), n=n_total, N_1=n1, N_2=n2,
        time_start_s=edges[:-1], time_end_s=edges[1:],
        time_center_s=(edges[:-1] + edges[1:]) / 2,
        asymmetry=asym, expected_asymmetry=model, sigma_asymmetry=sigma,
        variance_N_1=variance_n1))
    total_1 = int(n1.sum())
    whole = dict(N_1=total_1, N_2=int(n_total.sum()) - total_1, n=int(n_total.sum()),
                 asymmetry=(2 * total_1 - n_total.sum()) / max(n_total.sum(), 1),
                 sigma_asymmetry=2 * np.sqrt(variance_n1.sum()) / max(n_total.sum(), 1))
    return dict(groups=groups, whole_shot=whole,
                expected_N_1=float(expected_n1.sum()))


def _arrival_edges(cfg, shot_mean, n_sigma=6.0):
    """Arrival-bin edges wide enough to hold essentially the whole distribution."""
    v_lo = max(1.0, shot_mean - n_sigma * cfg.velocity_spread)
    v_hi = shot_mean + n_sigma * cfg.velocity_spread
    t_lo = cfg.flight_length / v_hi - n_sigma * cfg.start_time_spread
    t_hi = cfg.flight_length / v_lo + n_sigma * cfg.start_time_spread
    lo = np.floor(t_lo / cfg.bin_width)
    hi = np.ceil(t_hi / cfg.bin_width)
    return np.arange(lo, hi + 1) * cfg.bin_width


def simulate_shot_binned(cfg, prob_fn, n_molecules, *, rng, shot_mean=None,
                         n_v_bins=None, check_bin_width=True):
    """Fast binned sampler. ``prob_fn(v)`` returns the channel-1 probability."""
    n_molecules = int(n_molecules)
    if n_molecules <= 0:
        raise ValueError("n_molecules must be positive")
    if shot_mean is None:
        shot_mean = float(generate_velocities(
            cfg.mean_velocity, cfg.shot_velocity_spread, 1, rng=rng)[0])
    n_v_bins = cfg.n_v_bins if n_v_bins is None else int(n_v_bins)

    # --- velocity bins ---
    half = 6.0 * cfg.velocity_spread
    v_edges = np.linspace(max(1.0, shot_mean - half), shot_mean + half, n_v_bins + 1)
    v_centres = 0.5 * (v_edges[:-1] + v_edges[1:])
    q = _gauss_bin_mass(v_edges, shot_mean, cfg.velocity_spread)
    q = q / q.sum()

    pdf = lambda vv: np.exp(-0.5*((vv - shot_mean)/cfg.velocity_spread)**2)
    pbar = (prob_fn.bin_average(v_edges, pdf) if hasattr(prob_fn, "bin_average")
            else prob_fn(v_centres))

    # Within-bin flight-time spread, folded into the arrival weights as a Gaussian of
    # equal variance (top-hat of width w has variance w^2/12).
    dv = v_edges[1] - v_edges[0]
    flight_spread = cfg.flight_length * dv / shot_mean**2
    sigma_t_eff = np.sqrt(cfg.start_time_spread**2 + flight_spread**2 / 12.0)
    if check_bin_width and flight_spread > cfg.start_time_spread:
        warnings.warn(
            f"velocity bin width {dv:.3f} m/s maps to {flight_spread*1e6:.1f} us of "
            f"flight-time spread, exceeding the {cfg.start_time_spread*1e6:.1f} us "
            f"start-time spread. The Gaussian fold-in is then a poor approximation; "
            f"increase n_v_bins.", RuntimeWarning)

    # --- counts per velocity bin, then spins within each ---
    n_v = rng.multinomial(n_molecules, q)
    n1_v = rng.binomial(n_v, pbar)
    n2_v = n_v - n1_v

    # --- spread each velocity bin's counts over arrival bins ---
    t_edges = _arrival_edges(cfg, shot_mean)
    n_t = t_edges.size - 1
    tot = np.zeros(n_t, dtype=np.int64)
    one = np.zeros(n_t, dtype=np.int64)
    expected = np.zeros(n_t)
    variance = np.zeros(n_t)
    for j in range(n_v_bins):
        if n_v[j] == 0:
            continue
        w = _gauss_bin_mass(t_edges,
                            cfg.mean_start_time + cfg.flight_length/v_centres[j],
                            sigma_t_eff)
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        c1 = rng.multinomial(int(n1_v[j]), w)
        c2 = rng.multinomial(int(n2_v[j]), w)
        one += c1
        tot += c1 + c2
        expected += n_v[j] * pbar[j] * w
        variance += n_v[j] * pbar[j] * (1 - pbar[j]) * w

    result = _assemble(tot, one, t_edges, n_molecules, variance, expected)
    result["shot_mean"] = float(shot_mean)
    return result


def simulate_shot_streaming(cfg, prob_fn, n_molecules, *, rng, shot_mean=None,
                            chunk_size=1_000_000):
    """Per-molecule reference sampler, chunked to bound memory.

    Mirrors ``centrex_beam.streaming.streaming_shot`` but takes ``prob_fn`` instead of
    calling the analytic ``ramsey_probabilities``. Used to validate the binned sampler.
    """
    n_molecules = int(n_molecules)
    if shot_mean is None:
        shot_mean = float(generate_velocities(
            cfg.mean_velocity, cfg.shot_velocity_spread, 1, rng=rng)[0])
    t_edges = _arrival_edges(cfg, shot_mean)
    n_t = t_edges.size - 1
    tot = np.zeros(n_t, dtype=np.int64)
    one = np.zeros(n_t, dtype=np.int64)
    expected = np.zeros(n_t)
    variance = np.zeros(n_t)

    for start in range(0, n_molecules, chunk_size):
        size = min(chunk_size, n_molecules - start)
        v = generate_velocities(shot_mean, cfg.velocity_spread, size, rng=rng)
        p = prob_fn(v)
        s1, _ = project_spins(p, rng=rng)
        ts = generate_start_times(cfg.mean_start_time, cfg.start_time_spread,
                                  size, rng=rng)
        t = ts + arrival_times(0.0, v, cfg.flight_length)
        idx = np.clip(np.searchsorted(t_edges, t, side="right") - 1, 0, n_t - 1)
        tot += np.bincount(idx, minlength=n_t)
        one += np.bincount(idx[s1], minlength=n_t)
        expected += np.bincount(idx, weights=p, minlength=n_t)
        variance += np.bincount(idx, weights=p * (1 - p), minlength=n_t)

    result = _assemble(tot, one, t_edges, n_molecules, variance, expected)
    result["shot_mean"] = float(shot_mean)
    return result


def photon_excess_noise_factor(cfg):
    """sqrt(1 + var/mean^2) per molecule for the fluorescence readout.

    Uses centrex_beam's own moment formulas. With the reference numbers this is ~3.21,
    i.e. any sensitivity quoted at 1/sqrt(N_molecules) is optimistic by that factor.
    """
    from centrex_beam.photon_cycling import detected_photon_moments
    m = detected_photon_moments(cfg.pumpout_probability, cfg.detection_efficiency,
                                cfg.cycles_per_window)
    return float(np.sqrt(1.0 + m["variance_detected"] / m["mean_detected"] ** 2)), m
