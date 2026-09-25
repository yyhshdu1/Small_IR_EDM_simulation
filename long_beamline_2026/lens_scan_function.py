import copy
import pickle
import inspect
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from rich import progress
from scipy import constants
import cupy as cp
# from centrex_tlf import hamiltonian, states
from centrex_trajectories import (
    Coordinates,
    Gravity,
    PropagationOptions,
    PropagationType,
    Velocities,
    propagate_trajectories,
)
from centrex_trajectories.beamline_objects import (
    Bore,
    CircularAperture,
    ElectrostaticQuadrupoleLens,
    RectangularAperture,
    Section,
)
from centrex_trajectories.particles import TlF
with open("fit_coeffs.pkl", "rb") as f:
    fit_coeffs = pickle.load(f)
fit_coeff_mF0 = fit_coeffs["mF0"]
fit_coeff_mF2 = fit_coeffs["mF2"]
fit_coeff_mF3 = fit_coeffs["mF3"]
options = PropagationOptions(verbose=False, n_cores=24)
particle = TlF()
gravity = Gravity(0, -9.81 * particle.mass, 0)

# EQL parameters
L = 0.6
R = 1.75 * 25.4e-3 / 2
V = 28_000


# [NEW] don't limit DET aperture size, do post selection later
wy = 1

# conversion factors
in_to_m = 25.4e-3

n_nipples = 3

# beamline lengths
distance_lens_bbexit = 36 * in_to_m
lens_chamber_length = (24 + 5 / 8) * in_to_m
lens_reducer_flange = (7 / 8 + 3 + 1 / 8) * in_to_m
lens_electrode_length = L  # m
nipple_length = 39 * in_to_m
distance_det_center = 5.25 * in_to_m
lens_reducer_flange = (3 + 1 / 8 + 7 / 8) * in_to_m
bs_flange = 3 / 4 * in_to_m
rc_chamber_length_no_flanges = 10.5 * in_to_m
rc_chamber_center_from_bs_front = (16 + 3 / 8) * in_to_m
rc_aperture_from_center = 3.56 * in_to_m
rc_aperture_radius = 0.011
fourK = Section(
    name="4K shield",
    objects=[CircularAperture(0, 0, 1.75 * in_to_m, in_to_m / 2)],
    start=0,
    stop=(1.75 + 0.25) * 0.0254,
    save_collisions=False,
    propagation_type=PropagationType.ballistic,
)
fourtyK = Section(
    name="40K shield",
    objects=[CircularAperture(0, 0, fourK.stop + 1.25 * in_to_m, in_to_m / 2)],
    start=fourK.stop,
    stop=fourK.stop + (1.25 + 0.25) * 0.0254,
    save_collisions=False,
    propagation_type=PropagationType.ballistic,
)
bbexit = Section(
    name="Beamsource Exit",
    objects=[CircularAperture(0, 0, fourtyK.stop + 2.5 * in_to_m, 2 * in_to_m)],
    start=fourtyK.stop,
    stop=fourtyK.stop + (2.5 + 0.75) * in_to_m,
    save_collisions=False,
    propagation_type=PropagationType.ballistic,
)

rc = Section(
    name="Rotational cooling",
    objects=[
        CircularAperture(
            x=0,
            y=0,
            z=bbexit.stop + rc_chamber_center_from_bs_front + rc_aperture_from_center,
            r=rc_aperture_radius,
        )
    ],
    start=bbexit.stop
    + bs_flange
    + rc_chamber_center_from_bs_front
    - rc_chamber_length_no_flanges / 2,
    stop=bbexit.stop
    + rc_chamber_center_from_bs_front
    + rc_chamber_length_no_flanges / 2,
    save_collisions=False,
    propagation_type=PropagationType.ballistic,
)

spa = Section(
    name="State Prep A",
    objects=[],
    start=rc.stop,
    stop=bbexit.stop + (19.6 + 0.375 + 9.625 + 0.375) * in_to_m,
    save_collisions=False,
    propagation_type=PropagationType.ballistic,
)

det = Section(
        name="Detection",
        objects=[
            RectangularAperture(
                0,
                0,
                5.2-1.125*in_to_m,
                wx=2e-2,
                wy=wy,
            )
        ],
        start=5,
        stop=5.2 ,
        save_collisions=False,
        propagation_type=PropagationType.ballistic,
        force=None,
    )
det_test = Section(
        name="Detection",
        objects=[
            RectangularAperture(
                0,
                0,
                1.92,
                wx=2.52e-2,
                wy=wy,
            )
        ],
        start=1.915,
        stop=1.93 ,
        save_collisions=False,
        propagation_type=PropagationType.ballistic,
        force=None,
    )
# det = det_test
det_short = Section(
        name="Detection",
        objects=[
            RectangularAperture(
                0,
                0,
                1.91-1.125*in_to_m,
                wx=1,
                wy=1,
            )
        ],
        start=1.8,
        stop=1.91 ,
        save_collisions=False,
        propagation_type=PropagationType.ballistic,
        force=None,
    )

lens_start = (
    bbexit.stop
    + distance_lens_bbexit
    + (lens_chamber_length - lens_electrode_length) / 2
)

lens_stop = (
    bbexit.stop
    + distance_lens_bbexit
    + lens_chamber_length
    - (lens_chamber_length - lens_electrode_length) / 2
)


import gc

def lens_scan_function_0kV(
    origin: Coordinates,
    velocities: Velocities,
    length: float = 0.6,
    radius: float = 0.022225,
    include_det_short: bool = False,
):
    lens_start = bbexit.stop + distance_lens_bbexit + (lens_chamber_length - length) / 2
    lens_stop = lens_start + length
    eql = Section(
        name="Electrostatic Lens",
        objects=[
            Bore(
                x=0,
                y=0,
                z=lens_start,
                length=length,
                radius=radius,
            )
        ],
        start=lens_start,
        stop=lens_stop,
        save_collisions=False,
    )

    if include_det_short:
        sections = [fourK, fourtyK, bbexit, rc, spa, eql, det_short, det]
    else:
        sections = [fourK, fourtyK, bbexit, rc, spa, eql, det]
    section_data, trajectories = propagate_trajectories(
        sections,
        origin,
        velocities,
        particle,
        force=gravity,
        options=options,
    )
    return section_data, trajectories, sections

def generate_dist_before_lens(    origin: Coordinates,
    velocities: Velocities,):

    sections = [fourK, fourtyK, bbexit, rc]
    section_data, trajectories = propagate_trajectories(
        sections,
        origin,
        velocities,
        particle,
        force=gravity,
        options=options,
    )
    del section_data
    n = len(trajectories)                     # number of stored Trajectory objects

    # allocate the whole block once
    pos_arr = np.empty((n, 3), dtype=float)   # columns: x, y, z
    vel_arr = np.empty((n, 3), dtype=float)   # columns: vx, vy, vz

    for i, tr in enumerate(trajectories.values()):
        # fetch the last entry of each vector only once
        pos_arr[i, 0] = tr.x [-1]
        pos_arr[i, 1] = tr.y [-1]
        pos_arr[i, 2] = tr.z [-1]

        vel_arr[i, 0] = tr.vx[-1]
        vel_arr[i, 1] = tr.vy[-1]
        vel_arr[i, 2] = tr.vz[-1]

    last_positions  = Coordinates(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2])
    last_velocities = Velocities(vel_arr[:, 0], vel_arr[:, 1], vel_arr[:, 2])
    del trajectories
    gc.collect(1)   
    return last_positions, last_velocities

def lens_scan_function_with_input_before_lens(
    origin: Coordinates,
    velocities: Velocities,
    voltage: float,
    length: float = 0.6,
    radius: float = 0.022225,
    stark_potential: npt.NDArray[np.float64] = fit_coeff_mF0,
    tilt: float = 0,
    y_offset: float = 0,
    include_det_short: bool = False,
):
    lens_start = bbexit.stop + distance_lens_bbexit + (lens_chamber_length - length) / 2
    lens_stop = lens_start + length

    if voltage == 0:
        eql = Section(
            name="Electrostatic Lens",
            objects=[
                Bore(
                    x=0,
                    y=0,
                    z=lens_start,
                    length=length,
                    radius=radius,
                )
            ],
            start=lens_start,
            stop=lens_stop,
            save_collisions=False,
        )
    else:
        # print(f"Creating EQL with tilt {tilt} rad and y offset {y_offset} m")
        eql = ElectrostaticQuadrupoleLens(
            name="Electrostatic Lens",
            objects=[Bore(x=0, y=0, z=lens_start, length=length, radius=radius)],
            start=lens_start,
            stop=lens_stop,
            V=voltage,
            R=radius,
            tilt=tilt,
            y=y_offset,
            save_collisions=False,
            stark_potential=stark_potential,
        )
    sections = [eql, det]    
    section_data, trajectories = propagate_trajectories(
        sections,
        origin,
        velocities,
        particle,
        force=gravity,
        options=options,
    )
    if include_det_short:
        sections_det_short = [det_short]
        _, trajectories_det_short = propagate_trajectories(
            sections_det_short,
            origin,
            velocities,
            particle,
            force=gravity,
            options=options,
        )
        # Iterate through the particle IDs in the smaller collection
        for particle_id in trajectories:
            # Access the specific Trajectory objects
            traj1 = trajectories[particle_id]
            traj2 = trajectories_det_short[particle_id]
            idx_to_update = -2  # index of the last entry to update
            # 1. Update the last timestamp
            traj1.t[idx_to_update] = traj2.t[-1]
            
            # 2. Update the last coordinate values
            # Accessing the .x, .y, .z arrays via the Trajectory properties
            traj1.coordinates.x[idx_to_update] = traj2.coordinates.x[-1]
            traj1.coordinates.y[idx_to_update] = traj2.coordinates.y[-1]
            traj1.coordinates.z[idx_to_update] = traj2.coordinates.z[-1]
            
            # 3. Update the last velocity values
            traj1.velocities.vx[idx_to_update] = traj2.velocities.vx[-1]
            traj1.velocities.vy[idx_to_update] = traj2.velocities.vy[-1]
            traj1.velocities.vz[idx_to_update] = traj2.velocities.vz[-1]

    
    return section_data, trajectories, sections
def generate_vz_distribution(n_trajectories: int, vz_mean: float = 184, vz_sigma: float = 16):
    vz = cp.random.randn(n_trajectories) * vz_sigma + vz_mean
    vz = vz[(vz > 100) & (vz < 300)]
    while len(vz) < n_trajectories:
        additional_vz = cp.random.randn(n_trajectories - len(vz)) * vz_sigma + vz_mean
        additional_vz = additional_vz[(additional_vz > 100) & (additional_vz < 300)]
        vz = cp.concatenate((vz, additional_vz))
    vz = vz[:n_trajectories]
    return vz