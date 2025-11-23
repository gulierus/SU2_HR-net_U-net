
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
from modules.tracking import DetectionParams, BTrackParams, run_tracking_on_validation
from modules.utils import open_tiff_file

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

def sweep_and_save_gif(
    model,
    det_param_grid,
    btrack_param_grid,
    y_min=512, y_max=768,
    x_min=256, x_max=512,
    gif_output="best_tracking.gif"
):
    """
    Sweep parameters, find best HOTA, and save the best result as a GIF.
    """
    det_items = list(det_param_grid.items())
    bt_items  = list(btrack_param_grid.items())

    det_keys, det_vals = zip(*det_items) if det_items else ([], [])
    bt_keys,  bt_vals  = zip(*bt_items)  if bt_items  else ([], [])

    det_combos = list(itertools.product(*det_vals)) if det_vals else [()]
    bt_combos  = list(itertools.product(*bt_vals))  if bt_vals  else [()]

    total_runs = len(det_combos) * len(bt_combos)
    print(f"\nStarting Sweep: {total_runs} combinations...")

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
    print("BEST TRACKING RESULTS")
    print("="*80)
    print(f"HOTA: {best_HOTA:.4f} | DetA: {best_DetA:.4f} | AssA: {best_AssA:.4f}")
    print(f"\nBest Det Params: {best_det_params}")
    print(f"Best BTrack Params: {best_bt_params}")
    print("="*80)

    # Save Best HOTA to txt
    hota_output_path = os.path.join(os.path.dirname(gif_output), "best_hota.txt")
    with open(hota_output_path, "w") as f:
        f.write(f"Best HOTA: {best_HOTA:.4f}\n")
        f.write(f"Best Det Params: {best_det_params}\n")
        f.write(f"Best BTrack Params: {best_bt_params}\n")
    print(f"Best HOTA score saved to {hota_output_path}")

    # 2. Save GIF of best result
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
