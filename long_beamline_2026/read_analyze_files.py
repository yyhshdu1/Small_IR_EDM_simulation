import os
import pickle
import time
import threading
from pathlib import Path
from functools import partial
import numpy as np
from rich import progress
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import gc
import argparse
import queue
# --- Assume these are defined elsewhere in your script ---
aperture_size_xs = np.array([25.4]) * 1e-3
aperture_size_ys = np.array([10]) * 1e-3
aperture_ys_offset = np.array([0]) * 1e-3
# ---------------------------------------------------------

class ProgressUpdater:
    """Manages Rich progress updates from a separate thread."""
    def __init__(self, pb, progress_queue):
        self.pb = pb
        self.progress_queue = progress_queue
        self.active_tasks = {}
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run)

    def run(self):
        """Listen for progress messages and update the Rich progress bar."""
        while not self._stop_event.is_set():
            try:
                # Wait for a message with a timeout to allow checking the stop event
                message = self.progress_queue.get(timeout=0.1)
                msg_type, task_id, *data = message

                if msg_type == 'add':
                    file_size = data[0]
                    self.active_tasks[task_id] = self.pb.add_task(f"[yellow]Loading {task_id}", total=file_size)
                elif msg_type == 'update':
                    advance_by = data[0]
                    if task_id in self.active_tasks:
                        self.pb.update(self.active_tasks[task_id], advance=advance_by)
                elif msg_type == 'remove':
                    if task_id in self.active_tasks:
                        self.pb.remove_task(self.active_tasks.pop(task_id))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in progress thread: {e}")
                break

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self.thread.join()

class QueueFileWrapper:
    """A file-like wrapper that sends progress updates to a queue."""
    def __init__(self, f, progress_queue, task_id):
        self.f = f
        self.progress_queue = progress_queue
        self.task_id = task_id

    def read(self, size=-1):
        chunk = self.f.read(size)
        if chunk:
            self.progress_queue.put(('update', self.task_id, len(chunk)))
        return chunk

    def __getattr__(self, attr):
        # Delegate other file methods (like 'readline') to the original file object
        return getattr(self.f, attr)

def deep_merge_dicts(d1, d2):
    """Recursively merges dictionary d2 into d1."""
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            deep_merge_dicts(d1[k], v)
        else:
            d1[k] = v

def process_file(file_path, aperture_xs, aperture_ys, aperture_ys_offset, progress_queue):
    """Processes a single pickle file, sending progress updates to the queue."""
    vs_distribution_partial, nr_survived_partial = {}, {}
    path_parts = file_path.parts
    k1, k2, k3, k4 = path_parts[-4], path_parts[-3], path_parts[-2], file_path.stem

    # --- Progress reporting for file loading ---
    task_id = file_path.name
    file_size = os.path.getsize(file_path)
    progress_queue.put(('add', task_id, file_size))

    with file_path.open("rb") as f:
        wrapped_f = QueueFileWrapper(f, progress_queue, task_id)
        trajectories = pickle.load(wrapped_f)

    progress_queue.put(('remove', task_id))
    # --- End of progress reporting section ---

    for aperture_size_x in aperture_xs:
        for aperture_size_y in aperture_ys:
            for aperture_offset_y in aperture_ys_offset:
                _nr_survived = 0
                vxs_initialized = np.zeros(1_000_000, dtype=np.float64)
                for t in trajectories:
                    end = trajectories[t].coordinates[-1]
                    end_velocity = trajectories[t].velocities[-1]
                    if (abs(end.x) < aperture_size_x / 2 and
                        abs(end.y-aperture_offset_y) < aperture_size_y / 2):
                        vxs_initialized[_nr_survived] = float(end_velocity.vx)
                        _nr_survived += 1

                vxs = vxs_initialized[:_nr_survived]
                vs_distribution_partial.setdefault(k1, {}).setdefault(k2, {}).setdefault(k3, {}).setdefault(k4, {}).setdefault(aperture_size_x, {}).setdefault(aperture_size_y, {})[aperture_offset_y] = vxs
                nr_survived_partial.setdefault(k1, {}).setdefault(k2, {}).setdefault(k3, {}).setdefault(k4, {}).setdefault(aperture_size_x, {}).setdefault(aperture_size_y, {})[aperture_offset_y] = _nr_survived
    del trajectories
    del vxs_initialized
    gc.collect(1)  # Optional: force garbage collection to free memory
    return vs_distribution_partial, nr_survived_partial

# --- Main execution block ---
if __name__ == "__main__":
    default_root = "saved_trajectories_vz_160ms_mean"
    
    parser = argparse.ArgumentParser(description="Process pickle trajectory files.")
    parser.add_argument("--root", "-r", default=default_root, help="Root path for pickle files")
    parser.add_argument("--aperture_xs", nargs='+', type=float, 
                        default=aperture_size_xs, help="List of aperture sizes in x")
    parser.add_argument("--aperture_ys", nargs='+', type=float, 
                        default=aperture_size_ys, help="List of aperture sizes in y") 
    parser.add_argument("--aperture_ys_offset", nargs='+', type=float, 
                        default=aperture_ys_offset, help="List of aperture offsets in y")   
    args = parser.parse_args()

    # Simply initialize Path with the argument received
    # If the user didn't provide one, it uses default_root automatically
    root = Path(args.root)
    # 3. Convert the lists to numpy arrays
    xs = np.array(args.aperture_xs)
    ys = np.array(args.aperture_ys)
    ys_offset = np.array(args.aperture_ys_offset)

    print(f"XS Array: {xs} | Type: {type(xs)}")
    print(f"YS Array: {ys} | Type: {type(ys)}")
    print(f"YS Offset Array: {ys_offset} | Type: {type(ys_offset)}")
    print(f"Current root is: {root}")
    all_dirs = list(root.glob("**/*.pkl"))
    final_vs_distribution, final_nr_survived = {}, {}

    # Use a Manager to create a queue that can be shared between processes
    with Manager() as manager:
        progress_queue = manager.Queue()
        # NEW: uses the 'xs' and 'ys' variables you parsed from argparse
        worker_func = partial(process_file, aperture_xs=xs, aperture_ys=ys, aperture_ys_offset=ys_offset, progress_queue=progress_queue)

        print(f"Starting parallel processing of {len(all_dirs)} files on 32 cores...")

        with progress.Progress(
            progress.TextColumn("[bold blue]{task.description}", justify="right"),
            progress.BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%", "•",
            progress.MofNCompleteColumn(), "•",
            progress.TimeElapsedColumn(), "|", progress.TimeRemainingColumn(),
        ) as pb:
            # Start the progress listener thread
            progress_updater = ProgressUpdater(pb, progress_queue)
            progress_updater.start()

            main_task = pb.add_task("Overall Progress", total=len(all_dirs))

            with ProcessPoolExecutor(max_workers=8) as executor:
                results = executor.map(worker_func, all_dirs)
                for result in results:
                    vs_partial, nr_partial = result
                    deep_merge_dicts(final_vs_distribution, vs_partial)
                    deep_merge_dicts(final_nr_survived, nr_partial)
                    pb.update(main_task, advance=1)

            # Stop the progress listener thread
            progress_updater.stop()
    pickle.dump(final_vs_distribution, open("final_vs_distribution.pkl", "wb"))
    pickle.dump(final_nr_survived, open("final_nr_survived.pkl", "wb"))
    print("\nProcessing complete.")