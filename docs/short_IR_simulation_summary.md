# CeNTREX short interaction region — EDM (Schiff-moment) signal simulation

Summary of the method, the simulation chain, the noise sources, and what they imply for
the achievable CP-odd frequency sensitivity, data-taking time and magnetic shielding.

Code: `Small_IR_EDM_simulation/short_ir/` (Python package) and
`short_ir_monte_carlo.ipynb`. Beam, switching and photon-cycling models are reused from
`centrex_asymmetry` (`centrex_beam`); its authoritative production path is
`run_full_16block_fluorescence.py` + `apply_saved_fluorescence.py`.

All numbers below were measured with this code unless marked as taken from a paper.

---

## 1. Idea in one paragraph

The short IR (~0.7 m of field, 5.2 m source-to-detector) uses the same E field (29.7 kV/cm),
state preparation and detection as the main IR, but is shorter and less well shielded.
We simulate a Ramsey measurement on one pair of hyperfine levels (j ↔ e, 119.52 kHz) using
**Emma's COMSOL RF and E field maps integrated exactly**, instead of the idealised
two-square-pulse Ramsey formula, and feed the result into `centrex_asymmetry`'s beam
Monte Carlo, three-switch scheme and explicit LIF photon-cycling readout. A CP-odd
frequency $f_{\rm CP}$ is injected, and extracted exactly as an experiment would. The
scatter of the extracted value over many simulated runs is the sensitivity; any offset
is a systematic.

## 2. Physics model

### 2.1 Two-level Ramsey through the real field

$$H(z)=\tfrac12\begin{pmatrix}-\delta(z)&\Omega^*(z)\\ \Omega(z)&\delta(z)\end{pmatrix},\qquad
\Omega(z)=2\pi M(E(z))\,\big[B_1(z)+B_2(z)e^{i\phi_2}\big]$$

$$\delta(z)=2\pi\big[\nu_{\rm RF}-\nu_{je}(E(z))-\tfrac{\partial\nu}{\partial B_\parallel}B_\parallel-\Delta\nu_\perp-\nu_{\rm switch}-\nu_{\rm mot}(z,v)\big]$$

* Coil 2 is coil 1 mirrored about the domain centre; the two coils overlap (peaks at
  0.803 and 1.197 m, 0.394 m apart), so this is not a clean two-pulse sequence.
* $M(E)$ and $\nu_{je}(E)$ are looked up from tables along $z$, so the fringe-field region
  (where $\nu_{je}$ moves by up to 1.3 kHz) is included.
* Exact 2×2 propagator per $z$ step. The generator depends only on $z$ and the time step
  is $dz/v$, so the whole velocity grid is propagated in one pass (1500 velocities ×
  4000 steps ≈ 0.1 s). $P(v)$ is tabulated once per switch state and interpolated.
* Operating point: the coil-2 phase is solved so that $P_e=0.5$ (zero crossing, maximum
  slope) at 184 m/s — 1.6831 rad at Emma's amplitude, not exactly π/2 — then frozen.

### 2.2 Key parameters

| quantity | value | source |
|---|---|---|
| RF amplitude `FIELD_SCALE` | 0.365 (Emma); **0.3828 recommended** (§5) | Emma / this work |
| E plateau | 29.72 kV/cm | COMSOL |
| beam velocity | 184 ± 16 m/s within a shot | centrex_asymmetry |
| shot-to-shot mean-velocity scatter | 4 m/s | centrex_asymmetry |
| source start-time spread | 1 ms | centrex_asymmetry |
| flight length | 5.2 m (mean arrival 28.3 ms) | short IR |
| interaction time $T$ | ≈ 2.4 ms (2.0–2.9 ms across arrival bins) | measured |
| $B_\parallel$ (along E) | 0.45 mG → −1.119 Hz | measured record |
| $B_\perp$ (static) | 0.80 mG (⊥ v and E), 7.40 mG (along beam) → +0.0115 Hz | measured record |
| $\partial\nu/\partial B_\parallel$ | −2486.5 Hz/G (= 2.49 mHz/µG, as CeNTREX) | centrex_tlf |
| $\kappa$ (2nd-order transverse Zeeman) | 207.5 Hz/G² | from the two lines above |

Axis note: the field record's position is quoted as z = 44–88 cm, so its z was taken as
the beam axis and its y as the axis perpendicular to both v and E. This is inferred.

### 2.3 Magnetic effects included

| effect | how it enters | parity under E reversal |
|---|---|---|
| static $B_\parallel$, first-order Zeeman | constant detuning, same in all switch states | even → rejected |
| static transverse field, 2nd order | constant detuning $\kappa\|B_\perp\|^2$ | even → rejected |
| slow one-directional $B_\parallel$ drift | $B_\parallel$ stepped between runs (1 run = 1 day); **default off** | even → rejected |
| **motional field** $B_{\rm mot}=\mathbf v\times\boldsymbol{\mathcal E}/c^2$ | $\kappa[(b_\perp+B_{\rm mot})^2-b_\perp^2]$ per $z$ step and velocity; sign follows the CP switch | **odd → fake EDM** |

$B_{\rm mot}=60.8\ \mu$G at 184 m/s. The E-odd cross term $2\kappa b_\perp B_{\rm mot}$ gives
**−25.1 nHz per µG of $b_\perp$** (half the ±E difference; 50.2 nHz/µG for the difference,
matching the CeNTREX proposal's ≈50 nHz/µG, arXiv:2010.01451 §3.3.2), proportional to $v$.

## 3. Simulation chain

| step | what happens | code |
|---|---|---|
| 1. field | load COMSOL CSVs (explicit `header=`), interpolate on a 4000-point $z$ grid | `fields.py` |
| 2. operating phase | solve $P_e=0.5$ at 184 m/s, freeze | `ramsey.find_operating_phase` |
| 3. switch tables | one $P(v)$ table per switch state $(d,p,c)$: $\phi_2=\phi_{\rm op}+p\,\phi_{\rm cal}$, $\nu_{\rm switch}=d\,\Delta f_{\rm sw}+c\,f_{\rm CP}$, motional sign $=c$ | `ramsey.build_switch_tables` |
| 4. schedule | $N_{\rm blocks}$ blocks; each block = the 8 states once, random order | `centrex_beam.switching` |
| 5. shot sampling | shot mean velocity $\sim\mathcal N(184,4)$; counts per (velocity bin, arrival bin, spin) drawn directly (§3.1) | `fast_shot.simulate_shot_binned` |
| 6. LIF readout | each 10 µs arrival bin → two alternating 5 µs laser windows; explicit photon cycling; populations estimated from detected photons | `fluorescence.py` → `photon_cycling` |
| 7. bin-first extraction | per block, per 1 ms arrival bin: switch parities; per block, through-origin fit $A_{c}=r\,A_d$ across bins, $f_{\rm CP}=r\,\Delta f_{\rm sw}$ | `run.extract_bin_first(_fluorescence)` |
| 8. combination | plain mean over blocks; error = empirical block SEM | `run._fit_and_combine` |

### 3.1 Fast binned sampler

Given $v$, spin and arrival time are conditionally independent, so the joint distribution
of counts factorises exactly:

$$n_j\sim\text{Mult}(N,q_j),\quad N_{1j}\sim\text{Bin}(n_j,\bar p_j),\quad
\{c_{1jk}\},\{c_{2jk}\}\sim\text{Mult}(N_{1j},w_{jk}),\ \text{Mult}(N_{2j},w_{jk})$$

($q_j$: velocity-bin mass by erf; $\bar p_j$: density-weighted bin average of $p(v)$;
$w_{jk}$: arrival-bin mass with the within-bin flight-time spread folded into the
start-time width). Only the within-bin spin–arrival correlation, $O(\Delta v^2)$, is lost.
Cost is independent of $N$: **46 ms vs 4.4 s per 2×10⁷-molecule shot** (per-molecule
reference); $E[A]$ agrees to 5.8×10⁻⁶. Details: `docs/binned_sampler_explained.md`.

### 3.2 Three-switch estimators

$$A_{dpc}\simeq A_0+G_A\big[p\,\phi_{\rm cal}+d\,2\pi\Delta f_{\rm sw}T+c\,\phi_{\rm CP}\big],\qquad
f_{\rm CP}=\Delta f_{\rm sw}\,\frac{A_c}{A_d}$$

$G_A$ and $T$ cancel in the ratio. $\Delta f_{\rm sw}=0.5$ Hz, $\phi_{\rm cal}=\pi/32$. ACME
analogues: $\tilde c\leftrightarrow\tilde N\tilde E$, $\tilde d\leftrightarrow\tilde B$ (known
precession), $\tilde p\leftrightarrow\tilde\theta$ (contrast). Unlike ACME, the switches are not
nested by speed; all three flip on a ~1.7-shot timescale.

**Why bin-first.** Molecules in one arrival bin share a velocity, hence $T$ and pulse
area, so the linearisation holds bin by bin. Pooling the whole run instead carries the
run's single CP-odd velocity imbalance $\delta v_c$ straight into the answer (a fake slope
across arrival time, and a ~4 mHz offset with a 0.04 mHz quoted error in one test run).

## 4. Noise sources

### 4.1 Statistical terms

| source | origin | scales with molecules/shot |
|---|---|---|
| spin projection | Bernoulli outcome of the Ramsey measurement | $1/\sqrt N$ |
| **LIF photon readout** | geometric number of photons per molecule (pump-out $q\approx0.48$, ~2.07 photons) × binomial detection ($\eta=5.2\%$; ~90% of molecules give no detected photon). Per-molecule relative variance $\delta=q(1/\eta-1)+(1-q)\approx9.3$; $\sigma_A$ inflated by $\sqrt{1+\bar\delta}=3.20$ (measured 3.12) | $1/\sqrt N$ |
| **shot-to-shot velocity scatter** | every molecule of a shot shares a mean-velocity offset, changing pulse area and $T$; leaks into the EDM channel through $\delta v_c=\tfrac18\sum c_i\bar v_{{\rm shot},i}$ | **does not scale** |

Variance budget, $f_{\rm CP}=0$, 4-block runs, $10^7$ molecules/shot, Emma's 0.365
(paired, 40 seeds): LIF readout **63%**, velocity scatter **30%**, spin projection **7%**.

Per-block scatter of $f_{\rm CP}$ (8-shot block, LIF readout, SD known to ~7%):

| | 0.365 (Emma) | 0.3828 (local null) | 0.3828, no shot velocity scatter |
|---|---|---|---|
| $10^7$ molecules/shot | 33.1 mHz | 24.2 mHz | — |
| $10^{10}$ molecules/shot | 17.0 mHz | **3.49 mHz** | 0.69 mHz |

At $10^{10}$ the photon and spin terms have dropped by $\sqrt{1000}$ and the **residual
(second-order) velocity scatter is ~96% of the variance**.

### 4.2 Dependence on the injected $f_{\rm CP}$

$\sigma^2(f_{\rm CP})=\sigma_0^2+(\kappa_m f_{\rm CP})^2$: an additive floor, flat below ~0.1 Hz
($f_{\rm CP}=0$ and $3\times10^{-7}$ Hz agree to 0.1%), plus a multiplicative term from the
noise on the ruler $A_d$, dominant above ~0.4–0.6 Hz. A physical signal (~10⁻⁷ Hz) sits
on the floor, so **only $f_{\rm CP}\lesssim0.1$ Hz gives physical error bars**; the earlier
π/128 = 1.61 Hz injection does not. The notebook uses 0.01–0.1 Hz; final sensitivity runs
use 0.

### 4.3 Biases

* **Regression dilution** of the through-origin fit: slope attenuated by
  $\lambda=1-\sum w\,{\rm var}_d/\sum wA_d^2$, ∝ 1/(counts per bin). At 1 ms bins: −0.3%
  (projection) and **−3.2% with LIF** (measured −3.3% in a 20-run campaign). It is
  *multiplicative*, so it vanishes at $f_{\rm CP}=0$ and is irrelevant for a physical
  measurement; it matters only for calibration runs with a large injected signal.
* **Motional field**: +20.1 µHz at the measured $b_\perp=-0.80$ mG (j/e only). Additive,
  does not average down.
* **$B_\parallel$ drift** at the measured 0.0667 mG/day: walks the fringe offset $A_0$ by
  −2.4×10⁻³/day exactly as predicted, but leaves $f_{\rm CP}$ and its error unchanged (common
  mode). At 10× the rate the working point leaves the zero crossing (contrast 0.98 → 0.89,
  error +50%): a sensitivity loss curable by daily re-tuning, not a fake EDM.

## 5. RF amplitude optimisation

At 0.365 the peak of $P_e(v)$ is at 175 m/s, not 184, so the asymmetry responds to the shot
velocity at first order. Two nulls:

| null | criterion | FIELD_SCALE | relevant for |
|---|---|---|---|
| local | $dA/dv=0$ at 184 m/s | **0.3828** (+4.9%) | bin-first (what we use) |
| shot-averaged | $d\langle A\rangle_{\rm shot}/d\bar v_{\rm shot}=0$ | 0.3747 (+2.7%) | pooled estimator |

Paired MC (same beam at every amplitude), coupling of the per-block $f_{\rm CP}$ to $\delta v_c$:
+12.8 mHz/(m/s) (r = 0.59) at 0.365 → −0.4 mHz/(m/s) (r = −0.03) at 0.3828. Error bar
0.73× at $10^7$ (≈54% of the beam time), **0.21× at $10^{10}$ (≈4% of the beam time)**.
Contrast is unchanged. The more molecules per shot, the more the amplitude matters.

## 6. Sensitivity expectations

### 6.1 Data-taking time

Blocks are independent (campaign Birge ratio ≈ 1), so

$$T=\frac{8}{R\,\eta_{\rm duty}}\left(\frac{\sigma_{\rm block}}{\sigma_{\rm target}}\right)^2$$

with $\sigma_{\rm block}$ from §4.1. For **$10^{10}$ molecules/shot, $R=50$ Hz, 90% duty
cycle** (one block = 0.18 s):

| target $\sigma(f_{\rm CP})$ | 0.3828, 4 m/s scatter | 0.365, 4 m/s scatter | 0.3828, no shot velocity scatter |
|---|---|---|---|
| 10 mHz | 0.2 s (1 block) | 0.5 s | 0.2 s |
| 1 mHz | 2.2 s | 51 s | 0.2 s |
| 100 µHz | 3.6 min | 1.4 h | 8 s |
| 10 µHz | **6.0 h** | 6 d | 14 min |
| 1 µHz | **25 d** | 1.6 yr | 23 h |
| 300 nHz | **278 d** | 18 yr | 11 d |

### 6.2 Magnetic shielding (motional field only)

Requirement: fake shift ≤ target, $b_{\perp,\max}=\sigma_{\rm target}/k$, with $k=25.1$ nHz/µG
(j/e only, this model) or 2.8 nHz/µG with CeNTREX's M modulation (e↔j vs h↔k; suppression
by the proposal's ratio 5.6/50, independent of convention). Independent of molecule number.

| target | $b_{\perp,\max}$, j/e only | $b_{\perp,\max}$, with M |
|---|---|---|
| 10 mHz | 400 mG | 3.6 G |
| 1 mHz | 40 mG | 360 mG |
| 100 µHz | 4.0 mG | 36 mG |
| 10 µHz | **0.40 mG** | 3.6 mG |
| 1 µHz | 40 µG | **0.36 mG** |
| 300 nHz | **12 µG** | **107 µG** |

With today's $b_\perp=0.8$ mG at $10^{10}$ molecules/shot the measurement becomes
systematics-limited after **~1.5 h** (j/e only, 20 µHz) or **~5 days** (with M, 2.2 µHz).

### 6.3 Reading the tables

* ≥100 µHz: current shielding suffices; minutes to hours of data.
* 10 µHz: ~6 h; needs M modulation, or shielding to 0.4 mG.
* 1 µHz: ~25 d; shielding to 0.36 mG even with M (today ×2 too high).
* 300 nHz (main-IR scale): ~9 months, limited by **shot-to-shot velocity scatter**
  (removing it: 11 d), and shielding to ~0.1 mG with M (today ×7.5 too high).

### 6.4 Comparison with the main IR

Normalised per shot and molecule, the short IR is **5.60× worse** than the main-IR
production run (block SEM 4.84×10⁻⁴ Hz over 128 shots at 7×10⁷), against an
interaction-time ratio of 5.61×: no penalty beyond the shorter $T$, i.e. **31× the data**
for the same precision. CeNTREX projects ≈90 nHz for the main IR in 300 h at 50 Hz
(arXiv:2010.01451).

## 7. Validation performed

* Two independent codes (this package vs Emma's `ramsey4matrix.py`, after fixing its file
  names and a pandas header bug) agree to 3.7×10⁻¹⁵.
* Binned vs per-molecule sampler: $E[A]$ to 5.8×10⁻⁶, realisations within shot noise.
* LIF: calibrated totals within 0.07% of truth; residence 14 windows, inflation 3.2020 and
  zero carry-over, as in the main-IR path; paired readout inflation 3.12 vs 3.20.
* Variance budget closes to −7.6% (within the 11% SD uncertainty).
* Motional: coupling linear from 1 µG to 100 mG; propagated/analytic 0.994; ±E difference
  50.2 vs the proposal's ≈50 nHz/µG. Full-chain IPV (±0.4 G): −16.3 ± 4.4 vs −25.1 nHz/µG
  — right sign, 2σ low, unresolved (re-run at $10^{10}$ molecules for a 7× sharper check).
* Common-mode B drift: $A_0$ slope matches prediction to <1%.

## 8. Not modelled / open items

* **Other E-odd magnetic channels**, so the shielding numbers are necessary, not
  sufficient: leakage-current field (needs ≲1 nA, independent of shielding),
  $B^{\rm nr}\times\mathcal E^{\rm nr}$ (needs an E-dependent Zeeman coefficient and an
  asymmetric E reversal), field gradients × trajectory (needs transverse motion).
* **M modulation** (second transition pair h↔k): only its suppression factor is used.
* **8-level manifold**: out of scope; the pulse pull (+0.16 Hz, ∝ $B_{\rm RF}^2$) is common
  to all switch states.
* **Background light, laser-power and detector-efficiency drifts**: absent; background adds
  to the photon noise and does not shrink with molecule number.
* **Molecule number per shot, duty cycle, rep rate** are assumptions; the times in §6.1
  scale as $1/(N\,\eta)$ only while photon/spin noise dominate.
* **Shot-to-shot velocity scatter** (4 m/s) is taken from `centrex_asymmetry`; at high
  molecule number it is the dominant term, so its measured value matters most.
* Transverse-field drift, non-monotonic (e.g. diurnal) drift, drift within a run.
* Axis assignment of the measured field ($b_\perp$ = By) is inferred.

## 9. Code map

| file | contents |
|---|---|
| `short_ir/config.py` | all parameters (`ShortIRConfig`), drift (`drifted`), motional-field settings |
| `short_ir/fields.py` | COMSOL loading, mirroring, $M(E)$, $\nu_{je}(E)$, detuning |
| `short_ir/ramsey.py` | velocity-vectorised propagator (with motional term), `RamseyTable`, switch tables |
| `short_ir/fast_shot.py` | binned and per-molecule shot samplers |
| `short_ir/fluorescence.py` | LIF readout via `centrex_beam.photon_cycling` |
| `short_ir/run.py` | block runner (`shot_mean_seed` for paired beams), pooled and bin-first extraction |

Notebook `short_ir_monte_carlo.ipynb`: setup and operating phase → $P(v)$ vs the analytic
model → sampler validation → three-switch run → per-shot asymmetry and switching-block
diagram → pooled extraction → bin-first extraction (with its own four-panel view) → RF
amplitude optimisation → motional field (deterministic; IPV check behind
`RUN_MOTIONAL_IPV`) → open items → multi-day campaign (LIF, local-null amplitude, optional

#### drift diagnostics).
