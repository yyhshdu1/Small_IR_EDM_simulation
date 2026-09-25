import pickle
import numpy as np
from uncertainties import ufloat
from rich import progress
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def process_single_combination(task_keys, shared_data):
    """
    This is the worker function that runs on a separate core.
    It processes one combination of keys.
    """
    # Unpack the keys that define this specific task
    k1, k2, k3, k4, aperture_size_x, aperture_size_y, aperture_y_offset = task_keys
    
    # Unpack the data that is shared across all workers
    counts_vs_vx = shared_data['counts_vs_vx']
    signal_vs_freq = shared_data['signal_vs_freq']
    
    # --- This is the core logic from your original loop ---
    h = counts_vs_vx[k1][k2][k3][k4][aperture_size_x][aperture_size_y][aperture_y_offset]
    # counts_0kV = counts_vs_vx[k1][k2][k3]['0.0'][aperture_size_x][aperture_size_y][aperture_y_offset]
    
    # counts_0kV_with_err = np.array([ufloat(c, np.sqrt(c)) if c!=0 else ufloat(0,1) for c in counts_0kV])
    # signal_0kV = np.convolve(counts_0kV_with_err, signal_vs_freq, "same")
    
    counts_with_err = np.array([ufloat(_h, np.sqrt(_h)) if _h!=0 else ufloat(0,1) for _h in h])
    signal = np.convolve(counts_with_err, signal_vs_freq, "same")
    
    # gain = np.array([s / s0 if s0 != 0 else ufloat(0,1) for s, s0 in zip(signal, signal_0kV)])
    
    # Return the keys and the calculated gain so the main process can assemble the final dict
    return (task_keys, signal)

if __name__ == "__main__":
    # --- 1. Initial Data Loading (Done once in the main process) ---
    with open("counts_vs_vx.pkl", "rb") as f:
        counts_vs_vx = pickle.load(f)
    with open("results_vertical_expanded_beam.pkl", "rb") as f:
        results_scan_expanded_beam = pickle.load(f)

    # Assuming bin_centers and v_to_Γ are defined somewhere
    # Example placeholder values:
    bin_edges = np.linspace(-2, 2, 41)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    v_to_Γ = 0.9173012617949181

    res = results_scan_expanded_beam[0.5]
    freqs = np.arange(res[0].min(), res[0].max(), np.diff(bin_centers*v_to_Γ)[0])
    signal_vs_freq_unbroadened = np.interp(freqs, res[0], res[1])
    # apply broadening assuming 1MHz FWHM
    def gaussian_broadening(freq, fwhm):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-0.5 * (freq / sigma) ** 2)
    def lorentzian_broadening(freq, fwhm):
        return 1 / (freq**2 + (fwhm/2)**2)
    with open("2025_5_30.pickle", "rb") as f:
        cs_laser_broadening_fcn = pickle.load(f)
    laser_broadening = np.convolve(cs_laser_broadening_fcn(freqs), lorentzian_broadening(freqs, 0.15), mode='same')
    # signal_vs_freq = np.convolve(signal_vs_freq_unbroadened, gaussian_broadening(freqs, 1), mode='same')
    #     # signal_vs_freq = signal_vs_freq_unbroadened
    signal_vs_freq = np.convolve(signal_vs_freq_unbroadened, laser_broadening, mode='same')
    # --- 2. Flatten the nested loops into a list of tasks ---
    print("Preparing tasks for parallel execution...")
    tasks = []
    for k1, data1 in counts_vs_vx.items():
        for k2, data2 in data1.items():
            for k3, data3 in data2.items():
                for k4, data4 in data3.items():
                    # # We only want to process voltages that are not 0.0
                    # if k4 == '0.0':
                    #     continue
                    for aperture_size_x, data5 in data4.items():
                        for aperture_size_y, data6 in data5.items():
                            for aperture_y_offset in data6:
                                tasks.append((k1, k2, k3, k4, aperture_size_x, aperture_size_y, aperture_y_offset))
    
    # --- 3. Run tasks in parallel ---
    gain_vs_detuning = {}
    shared_data = {
        'counts_vs_vx': counts_vs_vx,
        'signal_vs_freq': signal_vs_freq
    }

    # Use 'partial' to create a new function with 'shared_data' already included.
    # This is a clean way to pass constant data to the worker function.
    worker_func = partial(process_single_combination, shared_data=shared_data)

    print(f"Starting processing of {len(tasks)} tasks on multiple cores...")
    with progress.Progress(
        progress.TextColumn("[progress.description]{task.description}"),
        progress.BarColumn(bar_width=None),
        progress.MofNCompleteColumn(),
        progress.TimeRemainingColumn(),
    ) as pb:
        task_id = pb.add_task("Calculating signal", total=len(tasks))
        
        # ProcessPoolExecutor manages the pool of cores
        with ProcessPoolExecutor(max_workers=61) as executor:
            # executor.map applies the worker function to each task and returns results as they complete
            for result in executor.map(worker_func, tasks):
                # --- 4. Collect results and build the final dictionary ---
                task_keys, gain = result
                k1, k2, k3, k4, aperture_size_x, aperture_size_y, aperture_y_offset = task_keys
                
                # This is the same logic as your original script to build the nested dictionary
                gain_vs_detuning.setdefault(k1, {}) \
                                .setdefault(aperture_size_x, {}) \
                                .setdefault(aperture_size_y, {}) \
                                .setdefault(k4, {}) \
                                .setdefault(k2, {}) \
                                .setdefault(k3, {}) \
                                .setdefault(aperture_y_offset, gain)
                
                pb.advance(task_id, 1)

    # --- 5. Save the final result ---
    print("\nProcessing complete. Saving results...")
    with open("signal_vs_detuning_broadened.pkl", "wb") as f:
        pickle.dump(gain_vs_detuning, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Done.")