import torch
import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Optional, Dict, Any
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import measure
from scipy import ndimage as ndi
from scipy import spatial, optimize
from dataclasses import dataclass
import json
import os
from tqdm.auto import tqdm

from .config import DEVICE
from .utils import open_tiff_file

# ============================================================================
# DETECTOR
# ============================================================================
@dataclass
class DetectionParams:
    threshold: float = 0.5
    min_area: int = 5
    nms_min_dist: Optional[float] = None

class CCPDetector:
    """
    CCP detector using trained U-Net++ model.
    Includes Test Time Augmentation (TTA) and Watershed segmentation.
    """
    def __init__(self, model, device=DEVICE, params: Optional[DetectionParams] = None):
        self.model = model.to(device)
        self.device = device
        self.params = params if params is not None else DetectionParams()
        self.model.eval()

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        if image.ndim == 3:
            img_base = image[0]
        else:
            img_base = image

        rotations = [
            img_base,
            np.rot90(img_base, k=1),
            np.rot90(img_base, k=2),
            np.rot90(img_base, k=3)
        ]

        batch_tensors = [torch.from_numpy(r.copy()).float().unsqueeze(0) for r in rotations]
        batch = torch.stack(batch_tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            outputs = torch.sigmoid(outputs)
            preds = outputs.cpu().numpy()

        p0 = preds[0, 0]
        p90 = np.rot90(preds[1, 0], k=-1)
        p180 = np.rot90(preds[2, 0], k=-2)
        p270 = np.rot90(preds[3, 0], k=-3)

        avg_map = (p0 + p90 + p180 + p270) / 4.0
        mask = (avg_map > self.params.threshold).astype(float)
        detections = self.extract_detections(mask)

        return mask, detections

    def extract_detections(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return []

        distance = ndi.distance_transform_edt(mask_bool)
        min_dist = self.params.nms_min_dist if self.params.nms_min_dist is not None else 3
        local_maxi = peak_local_max(distance, min_distance=min_dist, labels=mask_bool, exclude_border=False)
        markers = np.zeros(distance.shape, dtype=int)
        markers[local_maxi[:, 0], local_maxi[:, 1]] = np.arange(len(local_maxi)) + 1
        labels = watershed(-distance, markers, mask=mask_bool)

        detections = []
        for region in measure.regionprops(labels):
            if region.area >= self.params.min_area:
                y, x = region.centroid
                detections.append((int(x), int(y)))
        return detections

# ============================================================================
# TRACKING HELPERS
# ============================================================================
def link_detections(detections_per_frame: List[List[Tuple[int, int]]], max_dist: float = 7.0) -> pd.DataFrame:
    next_track_id = 0
    active_tracks = {}
    records = []

    for frame_idx, detections in enumerate(detections_per_frame):
        assigned = [False] * len(detections)
        detection_track_id = [None] * len(detections)
        updated_tracks = {}

        for track_id, (tx, ty, last_frame) in list(active_tracks.items()):
            best_dist = max_dist
            best_idx = None
            for i, (x, y) in enumerate(detections):
                if assigned[i]: continue
                dist = math.hypot(x - tx, y - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None:
                assigned[best_idx] = True
                detection_track_id[best_idx] = track_id
                updated_tracks[track_id] = (detections[best_idx][0], detections[best_idx][1], frame_idx)

        for i, (x, y) in enumerate(detections):
            if not assigned[i]:
                track_id = next_track_id
                next_track_id += 1
                detection_track_id[i] = track_id
                updated_tracks[track_id] = (x, y, frame_idx)

        active_tracks = updated_tracks
        for i, (x, y) in enumerate(detections):
            tid = detection_track_id[i]
            records.append({'frame': frame_idx, 'x': x, 'y': y, 'track_id': tid})

    return pd.DataFrame(records)

@dataclass
class BTrackParams:
    max_search_radius: float = 20.0
    do_optimize: bool = True
    dist_thresh: float = 10.0
    time_thresh: int = 3
    segmentation_miss_rate: float = 0.1
    apoptosis_rate: float = 0.0
    allow_divisions: bool = False
    min_track_len: int = 5
    gap_closing_max_frame_count: int = 3
    out_config_path: str = "cell_config_custom.json"

def build_btrack_config(params: BTrackParams) -> str:
    from btrack import datasets as btrack_datasets
    base_cfg_path = btrack_datasets.cell_config()
    with open(base_cfg_path, "r") as f:
        cfg = json.load(f)

    hyp = cfg.get("hypothesis_model", {})
    hyp["dist_thresh"] = params.dist_thresh
    hyp["time_thresh"] = params.time_thresh
    hyp["segmentation_miss_rate"] = params.segmentation_miss_rate
    hyp["apoptosis_rate"] = params.apoptosis_rate

    if not params.allow_divisions and "hypotheses" in hyp:
        hyp["hypotheses"] = [h for h in hyp["hypotheses"] if "DIVIDE" not in h.upper()]

    cfg["hypothesis_model"] = hyp
    
    out_path = params.out_config_path
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return out_path

def track_with_btrack(detections_per_frame, val_roi, btrack_params: Optional[BTrackParams] = None):
    import btrack
    from btrack import BayesianTracker

    if btrack_params is None:
        btrack_params = BTrackParams()

    objects = []
    for t, dets in enumerate(detections_per_frame):
        for x, y in dets:
            obj = btrack.btypes.PyTrackObject()
            obj.x = float(x)
            obj.y = float(y)
            obj.z = 0.0
            obj.t = t
            objects.append(obj)

    config_file = build_btrack_config(btrack_params)

    with BayesianTracker() as tracker:
        tracker.configure(config_file)
        tracker.max_search_radius = float(btrack_params.max_search_radius)
        tracker.volume = ((0, val_roi.shape[2]), (0, val_roi.shape[1]), (-1e5, 1e5))
        tracker.append(objects)
        tracker.track()
        if btrack_params.do_optimize:
            tracker.optimize()
        data, properties, graph = tracker.to_napari(ndim=2)

    tracks_df = pd.DataFrame(data, columns=["track_id", "frame", "y", "x"])
    
    # Filter by min_track_len
    if not tracks_df.empty:
        track_counts = tracks_df['track_id'].value_counts()
        valid_tracks = track_counts[track_counts >= btrack_params.min_track_len].index
        tracks_df = tracks_df[tracks_df['track_id'].isin(valid_tracks)]
        
    return tracks_df[["frame", "x", "y", "track_id"]]


# ============================================================================
# LAPTRACK TRACKING
# ============================================================================
@dataclass
class LapTrackParams:
    """
    Parameters for LapTrack (Linear Assignment Problem) tracking.
    
    LapTrack uses graph-based optimization for linking detections across frames.
    It's particularly good at handling:
    - Dense scenes with many particles
    - Temporary occlusions (gap closing)
    - Variable particle velocities
    """
    max_distance: float = 15.0              # Max linking distance between frames
    track_cost_cutoff: float = 15.0         # Cost cutoff for linking
    gap_closing_max_frame_count: int = 2    # Max frames to close gaps
    gap_closing_cost_cutoff: float = 15.0   # Cost cutoff for gap closing
    min_track_len: int = 5                  # Minimum track length to keep

def track_with_laptrack(
    detections_per_frame: List[List[Tuple[int, int]]],
    params: Optional[LapTrackParams] = None
) -> pd.DataFrame:
    """
    Track detections using LapTrack (Linear Assignment Problem tracking).
    
    LapTrack uses graph-based optimization to link detections across frames,
    providing robust tracking even in dense scenes with occlusions.
    
    Args:
        detections_per_frame: List of detections per frame, each as [(x, y), ...]
        params: LapTrack parameters (uses defaults if None)
    
    Returns:
        DataFrame with columns ['frame', 'x', 'y', 'track_id']
    """
    try:
        from laptrack import LapTrack
        import networkx as nx
    except ImportError:
        raise ImportError(
            "LapTrack not installed. Install with: pip install laptrack"
        )
    
    if params is None:
        params = LapTrackParams()
    
    if len(detections_per_frame) == 0:
        return pd.DataFrame(columns=['frame', 'x', 'y', 'track_id'])
    
    # Prepare coordinate array per frame
    coords_per_frame = []
    for frame_idx, dets in enumerate(detections_per_frame):
        if len(dets) > 0:
            coords = np.array([[x, y] for x, y in dets])
        else:
            coords = np.empty((0, 2))
        coords_per_frame.append(coords)
    
    # Initialize LapTrack
    tracker = LapTrack(
        track_cost_cutoff=params.track_cost_cutoff,
        gap_closing_cost_cutoff=params.gap_closing_cost_cutoff,
        gap_closing_max_frame_count=params.gap_closing_max_frame_count,
    )
    
    # Run tracking - returns networkx DiGraph
    graph = tracker.predict(coords_per_frame)
    
    # Extract tracks from graph
    # Nodes are tuples (frame_idx, det_idx)
    # Connected components = tracks
    records = []
    
    # Get weakly connected components (each component is one track)
    for track_id, component in enumerate(nx.weakly_connected_components(graph)):
        for node in component:
            frame_idx, det_idx = node
            if frame_idx < len(coords_per_frame) and det_idx < len(coords_per_frame[frame_idx]):
                x, y = coords_per_frame[frame_idx][det_idx]
                records.append({
                    'frame': frame_idx,
                    'x': x,
                    'y': y,
                    'track_id': track_id
                })
    
    track_df = pd.DataFrame(records)
    
    # Sort by frame
    if not track_df.empty:
        track_df = track_df.sort_values(['track_id', 'frame']).reset_index(drop=True)
        
        # Filter by min track length
        if params.min_track_len > 1:
            track_counts = track_df['track_id'].value_counts()
            valid_tracks = track_counts[track_counts >= params.min_track_len].index
            track_df = track_df[track_df['track_id'].isin(valid_tracks)]
    
    return track_df[["frame", "x", "y", "track_id"]]


# ============================================================================
# METRICS
# ============================================================================
def hota(gt: pd.DataFrame, tr: pd.DataFrame, threshold: float = 5) -> dict[str, float]:
    """
    Calculate HOTA metrics for tracking evaluation.
    """
    # Ensure particle ids are sorted from 0 to max(n)
    gt = gt.copy()
    tr = tr.copy()

    if gt.empty:
        return {'HOTA': 0.0, 'AssA': 0.0, 'DetA': 0.0, 'LocA': 0.0}
    if tr.empty:
        return {'HOTA': 0.0, 'AssA': 0.0, 'DetA': 0.0, 'LocA': 0.0}

    gt.track_id = gt.track_id.map({old: new for old, new in
                                   zip(gt.track_id.unique(), range(gt.track_id.nunique()))})
    tr.track_id = tr.track_id.map({old: new for old, new in
                                   zip(tr.track_id.unique(), range(tr.track_id.nunique()))})

    # Initialization
    num_gt_ids = gt.track_id.nunique()
    num_tr_ids = tr.track_id.nunique()

    frames = sorted(set(gt.frame.unique()) | set(tr.frame.unique()))

    potential_matches_count = np.zeros((num_gt_ids, num_tr_ids))
    gt_id_count = np.zeros((num_gt_ids, 1))
    tracker_id_count = np.zeros((1, num_tr_ids))

    HOTA_TP, HOTA_FN, HOTA_FP = 0, 0, 0
    LocA = 0.0

    # Compute similarities (inverted normalized distance)
    similarities = [1 - np.clip(spatial.distance.cdist(gt[gt.frame == t][['x', 'y']],
                                                       tr[tr.frame == t][['x', 'y']]) / threshold, 0, 1)
                    for t in frames]

    # Accumulate global track information
    for t_idx, t in enumerate(frames):
        gt_ids_t = gt[gt.frame == t].track_id.to_numpy()
        tr_ids_t = tr[tr.frame == t].track_id.to_numpy()
        
        if len(gt_ids_t) == 0 or len(tr_ids_t) == 0:
            continue

        similarity = similarities[t_idx]
        sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
        sim_iou = np.zeros_like(similarity)
        sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
        potential_matches_count[gt_ids_t[:, None], tr_ids_t[None, :]] += sim_iou

        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tr_ids_t] += 1

    global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
    matches_count = np.zeros_like(potential_matches_count)

    # Calculate scores for each timestep
    for t_idx, t in enumerate(frames):
        gt_ids_t = gt[gt.frame == t].track_id.to_numpy()
        tr_ids_t = tr[tr.frame == t].track_id.to_numpy()

        if len(gt_ids_t) == 0:
            HOTA_FP += len(tr_ids_t)
            continue

        if len(tr_ids_t) == 0:
            HOTA_FN += len(gt_ids_t)
            continue

        similarity = similarities[t_idx]
        score_mat = global_alignment_score[gt_ids_t[:, None], tr_ids_t[None, :]] * similarity

        match_rows, match_cols = optimize.linear_sum_assignment(-score_mat)

        actually_matched_mask = similarity[match_rows, match_cols] > 0
        alpha_match_rows = match_rows[actually_matched_mask]
        alpha_match_cols = match_cols[actually_matched_mask]

        num_matches = len(alpha_match_rows)

        HOTA_TP += num_matches
        HOTA_FN += len(gt_ids_t) - num_matches
        HOTA_FP += len(tr_ids_t) - num_matches

        if num_matches > 0:
            LocA += sum(similarity[alpha_match_rows, alpha_match_cols])
            matches_count[gt_ids_t[alpha_match_rows], tr_ids_t[alpha_match_cols]] += 1

    ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
    AssA = np.sum(matches_count * ass_a) / np.maximum(1, HOTA_TP)
    DetA = HOTA_TP / np.maximum(1, HOTA_TP + HOTA_FN + HOTA_FP)
    HOTA_score = np.sqrt(DetA * AssA)

    return {
        'HOTA': HOTA_score,
        'AssA': AssA,
        'DetA': DetA,
        'LocA': LocA,
        'HOTA TP': HOTA_TP,
        'HOTA FN': HOTA_FN,
        'HOTA FP': HOTA_FP
    }

def run_tracking_on_validation(
    model, 
    val_input=None, 
    tracking_method="btrack", 
    detection_params: Optional[DetectionParams] = None,
    btrack_params: Optional[BTrackParams] = None,
    laptrack_params: Optional[LapTrackParams] = None,
    y_min=512, y_max=768, x_min=256, x_max=512,
    use_validation_data=False,
    show_visualization=False
):
    # Load validation data if requested
    val_gt = None
    if use_validation_data:
        val_tif_path = "/content/val_data/val.tif"
        val_csv_path = "/content/val_data/val.csv"
        
        if os.path.exists(val_tif_path):
            val_input = open_tiff_file(val_tif_path).astype(np.float64)
        
        if os.path.exists(val_csv_path):
            val_gt = pd.read_csv(val_csv_path)
            # Filter GT to ROI
            val_gt = val_gt.groupby('track_id').filter(
                lambda t: (y_min < t.y.mean() < y_max) and (x_min < t.x.mean() < x_max)
            )

    if val_input is None:
        raise ValueError("val_input must be provided or use_validation_data=True with valid paths.")

    val_roi = val_input[:, y_min:y_max, x_min:x_max]
    
    # Initialize detector with params
    detector = CCPDetector(model, device=DEVICE, params=detection_params)
    
    # print("Detecting CCPs...")
    detections_per_frame = []
    # Use tqdm only if visualization is on or for single run, to avoid spamming logs during sweep
    iterator = range(len(val_roi))
    if show_visualization:
        iterator = tqdm(iterator, desc="Detecting")
        
    for frame_idx in iterator:
        frame = val_roi[frame_idx]
        frame_norm = (frame - frame.mean()) / (frame.std() + 1e-8)
        mask, detections = detector.detect(frame_norm)
        detections_per_frame.append(detections)

    # print(f"Linking with {tracking_method}...")
    if tracking_method == "btrack":
        tracks_df = track_with_btrack(detections_per_frame, val_roi, btrack_params)
    elif tracking_method == "laptrack":
        tracks_df = track_with_laptrack(detections_per_frame, laptrack_params)
    else:
        tracks_df = link_detections(detections_per_frame)

    # Adjust coordinates back to full image
    tracks_df['x'] += x_min
    tracks_df['y'] += y_min
    
    results = {}
    if val_gt is not None and not tracks_df.empty:
        results = hota(val_gt, tracks_df)
    
    return tracks_df, results