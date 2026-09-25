"""Emma's COMSOL fields for the short IR, on a z grid.

Mirroring convention: COMSOL simulated one coil over the full 2 m domain, and coil 2 is
that field mirrored about the domain centre. The two are summed with a 90 degree
relative phase, so the drive is a single complex amplitude at every z:

    B_total(z) = B1(z) + B2(z) * exp(1j * phase_2)

This is NOT two separated pulses; the coil fields overlap in space (peaks at x = 0.803
and 1.197 m, i.e. 0.394 m apart, both inside the E-field plateau).
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

E_CONVERSION = 1e5      # V/m per kV/cm


def load_interp(path, header=None, kind="cubic"):
    """Cubic interpolation with extrapolation.

    ``header`` must be given explicitly: the field profiles and coupling_constant.csv
    are headerless while sib_split_vs_E.csv and abs_M_xE_vs_E.csv carry text headers.
    pandas' default of header=0 silently eats the first DATA row of the headerless
    ones, which shifts the domain start and changes the total flight time by ~1%.
    """
    frame = pd.read_csv(path, header=header)
    x = frame.iloc[:, 0].to_numpy(dtype=float)
    y = frame.iloc[:, 1].to_numpy(dtype=float)
    return x, interp1d(x, y, kind=kind, fill_value="extrapolate")


class ShortIRField:
    """Fields sampled on a uniform z grid, plus the E-dependent lookups.

    Attributes
    ----------
    z        : (n_z,) position along the beam [m]
    dz       : (n_z-1,) step sizes [m]
    B1, B2   : (n_z,) coil fields [G], already multiplied by field_scale
    E_kVcm   : (n_z,) local electric field [kV/cm]
    M        : (n_z,) local coupling |<j|mu_x|e>| [Hz/G]
    nu_je    : (n_z,) local transition frequency [Hz]
    nu_rf    : RF reference frequency [Hz], the plateau value of nu_je
    """

    def __init__(self, cfg):
        self.cfg = cfg
        pos_b1, f_b1 = load_interp(cfg.path(cfg.b1_file))
        pos_b2, f_b2 = load_interp(cfg.path(cfg.b2_file))
        pos_e, f_e = load_interp(cfg.path(cfg.e_file))
        e_c, f_m = load_interp(cfg.path(cfg.coupling_file))
        e_s, f_nu = load_interp(cfg.path(cfg.sibsplit_file), header=0)

        self.z_start = max(pos_b1.min(), pos_b2.min(), pos_e.min())
        self.z_end = min(pos_b1.max(), pos_b2.max(), pos_e.max())
        self.z = np.linspace(self.z_start, self.z_end, cfg.n_z)
        self.dz = np.diff(self.z)

        self.B1 = f_b1(self.z) * cfg.field_scale
        self.B2 = f_b2(self.z) * cfg.field_scale
        raw_e = f_e(self.z) / E_CONVERSION
        self.E_kVcm = np.clip(raw_e, e_c.min(), e_c.max())
        self.clipped = bool(np.any(raw_e < e_c.min()) or np.any(raw_e > e_c.max()))
        self.M = f_m(self.E_kVcm)
        self.nu_je = np.abs(f_nu(np.clip(self.E_kVcm, e_s.min(), e_s.max()))) * 1e3

        plateau = self.E_kVcm > 0.9 * self.E_kVcm.max()
        self.nu_rf = float(np.median(self.nu_je[plateau]))
        self.E_plateau = float(np.median(self.E_kVcm[plateau]))

        self.z_peak1 = float(self.z[np.argmax(np.abs(self.B1))])
        self.z_peak2 = float(self.z[np.argmax(np.abs(self.B2))])
        self.separation = abs(self.z_peak1 - self.z_peak2)
        self.length = self.z_end - self.z_start

    def rabi_complex(self, phase_2):
        """Complex Rabi rate along z in rad/s, for a given coil-2 relative phase.

        Independent of velocity: only the time spent at each z depends on v.
        """
        return 2 * np.pi * self.M * (self.B1 + self.B2 * np.exp(1j * phase_2))

    def detuning(self, *, b_par_G=0.0, dnu_dB_par=0.0, dnu_transverse_Hz=0.0,
                 extra_Hz=0.0):
        """delta(z) = 2*pi*(nu_RF - nu_je(z) - Zeeman - transverse - extra), rad/s.

        ``extra_Hz`` carries the detuning switch and the injected CP shift.
        """
        nu = (self.nu_je
              + dnu_dB_par * b_par_G      # Zeeman, first order, B parallel to E
              + dnu_transverse_Hz         # transverse, precomputed scalar
              + extra_Hz)
        return 2 * np.pi * (self.nu_rf - nu)

    def summary(self):
        c = self.cfg
        return "\n".join([
            f"domain        z = {self.z_start:.3f} .. {self.z_end:.3f} m "
            f"({self.length:.3f} m, {c.n_z} steps)",
            f"coil peaks    {self.z_peak1:.3f} and {self.z_peak2:.3f} m "
            f"-> separation {self.separation:.3f} m",
            f"B peak        {np.abs(self.B1).max():.4f} G "
            f"(field_scale = {c.field_scale})",
            f"E             {self.E_kVcm.min():.4f} .. {self.E_kVcm.max():.4f} kV/cm, "
            f"plateau {self.E_plateau:.4f}",
            f"|M|           {self.M.min():.2f} .. {self.M.max():.2f} Hz/G",
            f"nu_je         {self.nu_je.min():.1f} .. {self.nu_je.max():.1f} Hz",
            f"nu_rf         {self.nu_rf:.4f} Hz",
            ("WARNING: E field outside the coupling table range, clipped"
             if self.clipped else "E field inside the lookup range"),
        ])
