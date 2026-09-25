"""Short-IR Monte Carlo: centrex_asymmetry's method with Emma's measured RF field.

The Ramsey step is the only thing replaced. Beam sampling, the three-switch schedule,
time-bin pooling and the linear parity estimators all come from ``centrex_beam``.
"""
from .config import (CALIBRATION_PHASE_RAD, CENTREX_ASYMMETRY, DETUNING_SWITCH_HZ,
                     EMMA_REPO, ShortIRConfig)
from .fields import ShortIRField, load_interp
from .fluorescence import (apply_fluorescence_to_run, apply_fluorescence_to_shot,
                           readout_inflation, residence_windows)
from .ramsey import (RamseyTable, build_switch_tables, find_operating_phase,
                     propagate_velocities, table_key)
from .fast_shot import (photon_excess_noise_factor, simulate_shot_binned,
                        simulate_shot_streaming)
from .run import (extract, extract_bin_first, extract_bin_first_fluorescence,
                  run_short_ir_blocks)

__all__ = [
    "CALIBRATION_PHASE_RAD", "CENTREX_ASYMMETRY", "DETUNING_SWITCH_HZ", "EMMA_REPO",
    "ShortIRConfig", "ShortIRField", "load_interp", "RamseyTable",
    "build_switch_tables", "find_operating_phase", "propagate_velocities",
    "table_key", "photon_excess_noise_factor", "simulate_shot_binned",
    "extract_bin_first",
    "extract_bin_first_fluorescence", "apply_fluorescence_to_run",
    "apply_fluorescence_to_shot", "readout_inflation", "residence_windows",
    "simulate_shot_streaming", "extract", "run_short_ir_blocks",
]
