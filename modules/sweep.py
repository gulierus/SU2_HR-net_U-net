import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as mc
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import itertools
import time
import os
from datetime import timedelta
from tqdm.auto import tqdm
from modules.tracking import (
    DetectionParams, BTrackParams, LapTrackParams,
    run_tracking_on_validation, CCPDetector, hota, track_with_laptrack
)
from modules.utils import open_tiff_file
from modules.config import DEVICE

def save_tracking_gif(data, image_stack, output_path="tracking_result.gif",
                     y_min=512, y_max=768, x_min=256, x_max=512,
                     tail_length=10, color='yellow', fps=10):
    """
    Save CCP trajectories animation as a GIF.
    """
    if isinstance(data, str):
        trajectories_df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        trajectories_df = data.copy()
    else:
        raise TypeError("`data` must be a CSV file path or a pandas DataFrame.")

    # Filter tracks in ROI
    tracks_in_roi = trajectories_df.groupby('track_id').filter(
        lambda t: (y_min < t.y.mean() < y_max) and (x_min < t.x.mean() < x_max)
    )
    
    print(f"Generating GIF for {len(tracks_in_roi['track_id'].unique())} trajectories...")

    cropped_stack = image_stack[:, y_min:y_max, x_min:x_max]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cropped_stack[0], cmap='magma')
    ax.set_title(f"Tracking Result (HOTA Best)")
    ax.axis('off')

    particles = tracks_in_roi['track_id'].unique()
    
    # Initialize graphics objects
    line_collections = {pid: mc.LineCollection([], linewidths=1, colors=color) for pid in particles}
    for lc in line_collections.values():
        ax.add_collection(lc)
    
    dot = ax.scatter([], [], s=5, c=color)

    def animate(i):
        im.set_array(cropped_stack[i])
        
        # Get data for current frame window
        window = tracks_in_roi[
            (tracks_in_roi['frame'] >= i - tail_length) &
            (tracks_in_roi['frame'] <= i)
        ]
        
        # Update current positions (dots)
        now = window[window['frame'] == i]
        if len(now) > 0:
            coords = np.column_stack((now.x.values - x_min, now.y.values - y_min))
            dot.set_offsets(coords)
        else:
            dot.set_offsets(np.empty((0, 2)))
            
        # Update tails (lines)
        for pid in particles:
            traj = window[window['track_id'] == pid].sort_values('frame')
            if len(traj) >= 2:
                segs = [
                    [(x0 - x_min, y0 - y_min), (x1 - x_min, y1 - y_min)]
                    for (x0, y0, x1, y1) in zip(
                        traj.x.values[:-1], traj.y.values[:-1],
                        traj.x.values[1:], traj.y.values[1:]
                    )
                ]
                line_collections[pid].set_segments(segs)
            else:
                line_collections[pid].set_segments([])
                
        return [im, dot] + list(line_collections.values())

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=cropped_stack.shape[0], interval=1000/fps, blit=True)
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print("Done!")

# ============================================================================
# BTRACK SWEEP 
# ============================================================================
def sweep_and_save_gif(
    model,
    det_param_grid,
    btrack_param_grid,
    y_min=512, y_max=768,
    x_min=256, x_max=512,
    gif_output="best_tracking.gif"
):
    """
    Sweep parameters for BTrack, find best HOTA, and save the best result as a GIF.
    """
    det_items = list(det_param_grid.items())
    bt_items  = list(btrack_param_grid.items())

    det_keys, det_vals = zip(*det_items) if det_items else ([], [])
    bt_keys,  bt_vals  = zip(*bt_items)  if bt_items  else ([], [])

    det_combos = list(itertools.product(*det_vals)) if det_vals else [()]
    bt_combos  = list(itertools.product(*bt_vals))  if bt_vals  else [()]

    total_runs = len(det_combos) * len(bt_combos)
    print(f"\nStarting BTrack Sweep: {total_runs} combinations...")

    best_HOTA = -1.0
    best_DetA = 0.0  
    best_AssA = 0.0 
    best_det_params = None
    best_bt_params = None
    best_tracks_df = None
    
    # Load validation data (assuming path from notebook)
    val_path = "/content/val_data/val.tif"
    if os.path.exists(val_path):
        val_input = open_tiff_file(val_path).astype(np.float64)
    else:
        print("Warning: Validation TIFF not found. GIF generation might fail.")
        val_input = None

    run_idx = 0
    for d_vals in det_combos:
        det_kwargs = dict(zip(det_keys, d_vals))
        det_params = DetectionParams(**det_kwargs)

        for b_vals in bt_combos:
            run_idx += 1
            bt_kwargs = dict(zip(bt_keys, b_vals))
            bt_params = BTrackParams(**bt_kwargs)

            print(f"Run {run_idx}/{total_runs}: Det={det_kwargs}, BTrack={bt_kwargs}")
            
            # Call the existing pipeline function
            tracks_df, results = run_tracking_on_validation(
                model,
                use_validation_data=True,
                tracking_method="btrack",
                detection_params=det_params,
                btrack_params=bt_params,
                y_min=y_min, y_max=y_max,
                x_min=x_min, x_max=x_max,
                show_visualization=False
            )

            if results:
                hota_score = results.get('HOTA', 0)
                deta_score = results.get('DetA', 0)
                assa_score = results.get('AssA', 0)
                print(f"  -> HOTA: {hota_score:.4f} | DetA: {deta_score:.4f} | AssA: {assa_score:.4f}")

                if hota_score > best_HOTA:
                    best_HOTA = hota_score
                    best_DetA = results.get('DetA', 0)  
                    best_AssA = results.get('AssA', 0)
                    best_det_params = det_params
                    best_bt_params = bt_params
                    best_tracks_df = tracks_df
                    print(f"  *** New Best HOTA: {best_HOTA:.4f} ***")


    # Print Best Results
    print("\n" + "="*80)
    print("BEST BTRACK RESULTS")
    print("="*80)
    print(f"HOTA: {best_HOTA:.4f} | DetA: {best_DetA:.4f} | AssA: {best_AssA:.4f}")
    print(f"\nBest Det Params: {best_det_params}")
    print(f"Best BTrack Params: {best_bt_params}")
    print("="*80)

    # Save Best HOTA to txt
    hota_output_path = os.path.join(os.path.dirname(gif_output), "best_hota_btrack.txt")
    with open(hota_output_path, "w") as f:
        f.write(f"Best HOTA: {best_HOTA:.4f}\n")
        f.write(f"Best Det Params: {best_det_params}\n")
        f.write(f"Best BTrack Params: {best_bt_params}\n")
    print(f"Best HOTA score saved to {hota_output_path}")

    # Save GIF of best result
    if best_tracks_df is not None and val_input is not None:
        print(f"\nGenerating GIF for best result...")
        save_tracking_gif(
            best_tracks_df, 
            val_input, 
            output_path=gif_output,
            y_min=y_min, y_max=y_max, 
            x_min=x_min, x_max=x_max
        )
        
    return best_det_params, best_bt_params, best_tracks_df


# ============================================================================
# LAPTRACK SWEEP
# ============================================================================
def sweep_laptrack_and_save_gif(
    model,
    det_param_grid,
    laptrack_param_grid,
    y_min=512, y_max=768,
    x_min=256, x_max=512,
    gif_output="best_tracking_laptrack.gif",
    val_data_path="/content/val_data"
):
    """
    Sweep parameters for LapTrack, find best HOTA, and save the best result as a GIF.
    
    This function uses pre-computed detections for efficiency - detections are computed
    once for each detection parameter set, then reused for all tracking parameter combinations.
    
    Args:
        model: Trained detection model
        det_param_grid: Dict of detection parameters to sweep
        laptrack_param_grid: Dict of LapTrack parameters to sweep
        y_min, y_max, x_min, x_max: ROI coordinates
        gif_output: Path to save best tracking GIF
        val_data_path: Path to validation data directory
    
    Returns:
        best_det_params: Best detection parameters
        best_lap_params: Best LapTrack parameters
        best_tracks_df: Best tracking result DataFrame
        all_results: List of all sweep results
    """
    
    # Load validation data
    val_tif_path = os.path.join(val_data_path, "val.tif")
    val_csv_path = os.path.join(val_data_path, "val.csv")
    
    if not os.path.exists(val_tif_path):
        raise FileNotFoundError(f"Validation TIFF not found: {val_tif_path}")
    
    val_input = open_tiff_file(val_tif_path).astype(np.float64)
    val_gt = pd.read_csv(val_csv_path)
    
    # Filter GT to ROI
    val_gt = val_gt.groupby('track_id').filter(
        lambda t: (y_min < t.y.mean() < y_max) and (x_min < t.x.mean() < x_max)
    )
    
    val_roi = val_input[:, y_min:y_max, x_min:x_max]
    
    # Generate all parameter combinations
    det_keys = list(det_param_grid.keys())
    det_values = list(det_param_grid.values())
    det_combinations = list(itertools.product(*det_values))
    
    lap_keys = list(laptrack_param_grid.keys())
    lap_values = list(laptrack_param_grid.values())
    lap_combinations = list(itertools.product(*lap_values))
    
    total_runs = len(det_combinations) * len(lap_combinations)
    print(f"\n" + "="*80)
    print(f"LAPTRACK PARAMETER SWEEP")
    print("="*80)
    print(f"Total combinations: {total_runs}")
    print(f"  Detection configs: {len(det_combinations)}")
    print(f"  LapTrack configs: {len(lap_combinations)}")
    print("="*80)
    
    best_hota = 0.0
    best_det_params = None
    best_lap_params = None
    best_tracks = None
    all_results = []
    
    # =========================================================================
    # STEP 1: Pre-compute detections for each detection parameter set
    # =========================================================================
    print("\n[STEP 1/2] Pre-computing detections for efficiency...")
    det_cache = {}
    
    for det_combo in tqdm(det_combinations, desc="Detection configs"):
        det_params = DetectionParams(**dict(zip(det_keys, det_combo)))
        detector = CCPDetector(model, device=DEVICE, params=det_params)
        
        detections_per_frame = []
        for frame_idx in range(len(val_roi)):
            frame = val_roi[frame_idx]
            frame_norm = (frame - frame.mean()) / (frame.std() + 1e-8)
            _, detections = detector.detect(frame_norm)
            detections_per_frame.append(detections)
        
        det_cache[det_combo] = detections_per_frame
    
    print(f"✓ Detections pre-computed for {len(det_combinations)} configs")
    
    # =========================================================================
    # STEP 2: Run tracking sweep with pre-computed detections
    # =========================================================================
    print("\n[STEP 2/2] Running LapTrack parameter sweep...")
    pbar = tqdm(total=total_runs, desc="Sweep progress")
    
    for det_combo in det_combinations:
        det_params = DetectionParams(**dict(zip(det_keys, det_combo)))
        detections_per_frame = det_cache[det_combo]
        
        for lap_combo in lap_combinations:
            lap_params = LapTrackParams(**dict(zip(lap_keys, lap_combo)))
            
            try:
                # Run LapTrack
                tracks_df = track_with_laptrack(detections_per_frame, lap_params)
                
                # Adjust coordinates to full image
                if not tracks_df.empty:
                    tracks_df['x'] += x_min
                    tracks_df['y'] += y_min
                
                # Evaluate
                if not tracks_df.empty:
                    results = hota(val_gt, tracks_df)
                else:
                    results = {'HOTA': 0.0, 'DetA': 0.0, 'AssA': 0.0}
                
                all_results.append({
                    'det_params': det_params,
                    'lap_params': lap_params,
                    **results
                })
                
                if results['HOTA'] > best_hota:
                    best_hota = results['HOTA']
                    best_det_params = det_params
                    best_lap_params = lap_params
                    best_tracks = tracks_df.copy()
                    
                    tqdm.write(f"\n  New Best HOTA: {best_hota:.4f} | DetA: {results['DetA']:.4f} | AssA: {results['AssA']:.4f}")
                    tqdm.write(f"     Det: {det_params}")
                    tqdm.write(f"     Lap: {lap_params}")
            
            except Exception as e:
                tqdm.write(f"\n  Error: {e}")
            
            pbar.update(1)
    
    pbar.close()
    
    # =========================================================================
    # Print and save best results
    # =========================================================================
    print("\n" + "="*80)
    print("BEST LAPTRACK RESULTS")
    print("="*80)
    
    if best_det_params and best_lap_params:
        final_results = [r for r in all_results 
                        if r['det_params'] == best_det_params 
                        and r['lap_params'] == best_lap_params][0]
        
        print(f"HOTA: {final_results['HOTA']:.4f} | DetA: {final_results['DetA']:.4f} | AssA: {final_results['AssA']:.4f}")
        print(f"\nBest Detection Params: {best_det_params}")
        print(f"Best LapTrack Params: {best_lap_params}")
        
        # Save to text file
        hota_output_path = os.path.join(os.path.dirname(gif_output), "best_hota_laptrack.txt")
        with open(hota_output_path, "w") as f:
            f.write(f"Best HOTA: {final_results['HOTA']:.4f}\n")
            f.write(f"DetA: {final_results['DetA']:.4f}\n")
            f.write(f"AssA: {final_results['AssA']:.4f}\n")
            f.write(f"\nBest Detection Params:\n{best_det_params}\n")
            f.write(f"\nBest LapTrack Params:\n{best_lap_params}\n")
        print(f"\n✓ Results saved to: {hota_output_path}")
        
        # Save GIF of best result
        if best_tracks is not None and val_input is not None:
            print(f"\nGenerating GIF for best result...")
            save_tracking_gif(
                best_tracks,
                val_input,
                output_path=gif_output,
                y_min=y_min, y_max=y_max,
                x_min=x_min, x_max=x_max
            )
    else:
        print("No valid results found!")
    
    print("="*80)
    
    return best_det_params, best_lap_params, best_tracks, all_results