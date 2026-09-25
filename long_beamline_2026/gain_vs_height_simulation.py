import os
import pickle
import numpy as np
from rich.progress import track
from joblib import Parallel, delayed, parallel_backend
from pathlib import Path
from uncertainties import ufloat

import logging

# Configure logging to write to a file
# logging.basicConfig(
#     filename='parallel_debug.log', 
#     level=logging.INFO,
#     format='%(processName)s: %(message)s'
# )

# --- Worker for a single height (no changes) ---
def calculate_survived_for_height(height, trajectory_endpoints, interpolator):
    _nr_survived = 0
    debug_flag = True
    for entry in trajectory_endpoints:
        if len(entry) == 5:
            end_x, end_y, end_vz, end_vx, weight = entry
            # logging.info(f"Debug: height={height:.3e}, end_y={end_y*1000:.2f}mm, weight={weight:.3f}") if debug_flag else None

        else:
            end_x, end_y, end_vz, end_vx = entry
            # if debug_flag:
            #     logging.info(f"Warning: Entry with unexpected length {len(entry)}. Expected 5, got {len(entry)}. Defaulting weight to 1.0") if debug_flag else None
            #     debug_flag = False  # Only print for the first entry to avoid clutter
            # logging.info(f"Debug: height={height:.3e}, end_y={end_y*1000:.2f}mm, weight={weight:.3f}") if debug_flag else None
            weight = 1.0  
        weight = 1
        # debug_flag = False  # Only print for the first entry to avoid clutter
        nr_photons = interpolator(180, abs(end_y - height), abs(end_vx)) * weight
        # nr_photons = 2 * weight if abs(end_y - height) <= 1e-3 else 0
        if np.isnan(nr_photons):
            continue
        _nr_survived += nr_photons 
    return _nr_survived

# --- NEW: Worker function to process one entire file ---
# This function will be called in parallel by the outer loop.
def process_single_file(file_path, interpolator_path, heights):
    """
    Loads, pre-processes, and calculates results for a single file.
    This function contains the INNER parallel loop for heights.
    """
    try:
        interpolator = pickle.load(open(interpolator_path, "rb"))

        path_parts = file_path.parts
        # k1, k2, k3, k4 = path_parts[1], path_parts[2], path_parts[3], file_path.stem
        k1, k2, k3, tilt, y_offset, k4 = path_parts[1], path_parts[2], path_parts[3], path_parts[-3].split("_")[1], path_parts[-2].split("_")[2], file_path.stem

        # Load and pre-process data
        print(f"Processing {file_path}")
        with open(file_path, "rb") as f:
            trajectories = pickle.load(f)
        print(f"Loaded trajectories from {file_path}")
        trajectory_endpoints = [
            (trajectories[t].coordinates[-1].x, trajectories[t].coordinates[-1].y,
             trajectories[t].velocities[-1].vz, trajectories[t].velocities[-1].vx)
            for t in trajectories
            if abs(trajectories[t].coordinates[-1].x) <= 1e-2 and abs(trajectories[t].coordinates[-1].y) <= 2.5e-2 # and abs(trajectories[t].velocities[-1].vx) < 1
        ]
        del trajectories
        print(f"Filtered to {len(trajectory_endpoints)} endpoints within x-range.")
        # --- INNER Parallel Loop (for heights) ---
        # This will use 11 cores for each file being processed.
        with parallel_backend('loky', inner_max_num_threads=1):
            survived_counts = Parallel(n_jobs=len(heights))(
                delayed(calculate_survived_for_height)(h, trajectory_endpoints, interpolator) for h in heights
            )
        
        # Assemble results for this file
        file_results = {height: count for height, count in zip(heights, survived_counts)}
        
        # Return the keys and results so they can be merged later
        return (k1, k2, k3, tilt, y_offset, k4, file_results)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_single_trajectories(trajectories, interpolator_path, heights):
    """
    Loads, pre-processes, and calculates results for a single file.
    This function contains the INNER parallel loop for heights.
    """
    try:
        interpolator = pickle.load(open(interpolator_path, "rb"))

        trajectory_endpoints = [
            (trajectories[t].coordinates[-1].x, trajectories[t].coordinates[-1].y,
             trajectories[t].velocities[-1].vz, trajectories[t].velocities[-1].vx)
            for t in trajectories
            if abs(trajectories[t].coordinates[-1].x) <= 1e-2 and abs(trajectories[t].coordinates[-1].y) <= 2.5e-2 # and abs(trajectories[t].velocities[-1].vx) < 1
        ]
        del trajectories
        print(f"Filtered to {len(trajectory_endpoints)} endpoints within x-range.")
        # --- INNER Parallel Loop (for heights) ---
        # This will use 11 cores for each file being processed.
        with parallel_backend('loky', inner_max_num_threads=1):
            survived_counts = Parallel(n_jobs=len(heights))(
                delayed(calculate_survived_for_height)(h, trajectory_endpoints, interpolator) for h in heights
            )
        
        # Assemble results for this file
        file_results = {height: count for height, count in zip(heights, survived_counts)}
        
        # Return the keys and results so they can be merged later
        return file_results
    except Exception as e:
        print(f"Error processing : {e}")
        return None

def process_single_trajectories_serial(trajectories, interpolator_path, heights, nr_trajectories: int, aperture: bool = True):
    """
    Loads, pre-processes, and calculates results for a single file.
    This function contains the INNER parallel loop for heights.
    """
    try:
        interpolator = pickle.load(open(interpolator_path, "rb"))
        if aperture:
            trajectory_endpoints = [
                (trajectories[t].coordinates[-1].x, trajectories[t].coordinates[-1].y,
                trajectories[t].velocities[-1].vz, trajectories[t].velocities[-1].vx)
                for t in trajectories
                if abs(trajectories[t].coordinates[-1].x) <= 1e-2 and abs(trajectories[t].coordinates[-1].y) <= 2.5e-2 # and abs(trajectories[t].velocities[-1].vx) < 1
            ]
        else:
            trajectory_endpoints = [
                (trajectories[t].coordinates[-1].x, trajectories[t].coordinates[-1].y,
                trajectories[t].velocities[-1].vz, trajectories[t].velocities[-1].vx)
                for t in trajectories
                if abs(trajectories[t].coordinates[-1].x) <= 8e-2 and abs(trajectories[t].coordinates[-1].y) <= 2.5e-2
            ]
        del trajectories
        # print(f"Filtered to {len(trajectory_endpoints)} endpoints within x-range.")
        # --- INNER Parallel Loop (for heights) ---

        survived_counts = []
        for h in heights:
            count = calculate_survived_for_height(h, trajectory_endpoints, interpolator)
            survived_counts.append(count)

        # Assemble results for this file
        file_results = {height: ufloat(count/nr_trajectories, np.sqrt(count)/nr_trajectories) for height, count in zip(heights, survived_counts)}
        
        # Return the keys and results so they can be merged later
        return file_results
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing : {e}")
        return None

def process_single_trajectories_parallel(trajectories, interpolator, heights, nr_trajectories: int, aperture: bool = True, det_short_interpolator=None):
    """
    Loads, pre-processes, and calculates results for a single file.
    Uses NumPy for data storage and joblib for parallelizing the height loop.
    """
    try:
        endpoints_data = []
        
        # 1. Efficient Filtering and Extraction
        for t in trajectories:
            coords = trajectories[t].coordinates[-1]
            vels = trajectories[t].velocities[-1]
            if det_short_interpolator is not None:
                coords_short = trajectories[t].coordinates[-2]
                # logging.info(f"Trajectory {t}: end_z={coords_short.z:.2f}m") if False else None
            if (aperture and abs(coords.x) <= 1e-2 and abs(coords.y) <= 2.5e-2) or \
               (not aperture and abs(coords.x) <= 8e-2 and abs(coords.y) <= 2.5e-2):
                if det_short_interpolator is not None:
                    weight = max(det_short_interpolator(coords_short.y * 1000), 0) # Convert to mm for interpolation
                else:
                    weight = 1
                # Data structure: (x, y, vz, vx)
                endpoints_data.append((coords.x, coords.y, vels.vz, vels.vx, weight))

        # Convert to a NumPy array for fast subsequent calculations
        trajectory_endpoints = np.array(endpoints_data, dtype=np.float64)
        
        # Free up memory early
        del trajectories 

        # 2. Parallelizing the Inner Loop with joblib
        survived_counts = Parallel(n_jobs=-1)(
            delayed(calculate_survived_for_height)(h, trajectory_endpoints, interpolator)
            for h in heights
        )

        # 3. Assemble results (assuming ufloat is correctly imported)
        file_results = {
            height: ufloat(count/nr_trajectories, np.sqrt(count)/nr_trajectories) if count > 0 else ufloat(0, 0)
            for height, count in zip(heights, survived_counts)
        }
        
        return file_results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing : {e}")
        return None
    
def main():
    # --- Setup ---
    # HEIGHT_SCANS = np.array([-13., -11.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1., 0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   9.,  11.])*1e-3
    HEIGHT_SCANS = np.arange(-25,25,2)*1e-3
    
    nr_survived = {}

    root = Path("saved_trajectories_lens_tilt_shift")
    all_dirs = list(root.glob("**/*.pkl"))
    # Filter files before processing
    filtered_dirs = []
    for fp in all_dirs:
        parts = fp.parts
        if len(parts) < 6:
            continue
        k2 = parts[2]
        k3 = parts[3]
        tilt = float(parts[-3].split("_")[1])
        y_offset = float(parts[-2].split("_")[2])
        k4 = fp.stem
        if k2 == "gaussian" and k3 in ("0.005", "0.004", "0.007"):
            try:
                k4_float = float(k4)
                if k4_float in (0, 22.5e3, 26.25e3, 30e3, 20e3):
                    filtered_dirs.append(fp)
            except ValueError:
                continue
    all_dirs = filtered_dirs
    # --- Determine Concurrency Level ---
    cores_per_file = len(HEIGHT_SCANS) # 11 cores needed per file
    total_cores = os.cpu_count()
    # Calculate how many files can be processed at once
    num_concurrent_files = max(1, total_cores // cores_per_file)
    
    print(f"💻 System has {total_cores} cores.")
    print(f"📂 Each file will use {cores_per_file} cores for height calculations.")
    print(f"🚀 Processing {num_concurrent_files} files concurrently.")
    
    # --- OUTER Parallel Loop (for files) ---
    # This loop assigns files to different sets of cores.
    tasks = (delayed(process_single_file)(fp, "3d_interpolator.pkl", HEIGHT_SCANS) for fp in all_dirs)

    # `track` gives a nice progress bar for the file processing
    results_list = track(
        Parallel(n_jobs=num_concurrent_files, verbose=10)(tasks),
        description="Processing files...",
        total=len(all_dirs)
    )

    # --- Merge results from all parallel tasks ---
    print("\nMerging results...")
    output_path = Path(root / "lens_signal_vs_height_with_aperture.pkl")
    
    if output_path.exists():
        with open(output_path, "rb") as f:
            nr_survived = pickle.load(f)
    else:
        nr_survived = {}
    for result in results_list:
        if result: # In case a file was skipped or failed
            k1, k2, k3, tilt, y_offset, k4, file_results = result
            # setdefault is a clean way to build nested dictionaries
            nr_survived.setdefault(k1, {}).setdefault(k2, {}).setdefault(k3, {}).setdefault(tilt, {}).setdefault(y_offset, {})[k4] = file_results

    # --- Save the final aggregated results ---
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(nr_survived, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Processing complete. Results saved to {output_path}")

# This guard is essential for multiprocessing
if __name__ == "__main__":
    main()