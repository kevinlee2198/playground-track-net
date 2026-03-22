from __future__ import annotations

import math

import torch


def heatmap_to_position(
    heatmap: torch.Tensor, threshold: float = 0.5
) -> tuple[float, float, bool]:
    """Extract ball position from a single-frame heatmap.

    Args:
        heatmap: (H, W) tensor with values in [0, 1].
        threshold: Minimum value to consider as ball detection.

    Returns:
        (x, y, detected): x/y coordinates of the centroid and whether
        a ball was detected.
    """
    binary = (heatmap > threshold).float()
    if binary.sum() == 0:
        return 0.0, 0.0, False

    # Find centroid of all above-threshold pixels
    ys, xs = torch.where(binary > 0)
    x = xs.float().mean().item()
    y = ys.float().mean().item()
    return x, y, True


def compute_detection_metrics(
    pred: tuple[float, float, bool],
    gt: tuple[float, float, bool],
    distance_threshold: float = 4.0,
) -> tuple[int, int, int]:
    """Compute TP/FP/FN for a single frame.

    Args:
        pred: (x, y, detected) predicted position.
        gt: (x, y, detected) ground truth position.
        distance_threshold: Maximum Euclidean distance for a true positive.

    Returns:
        (tp, fp, fn) counts.
    """
    pred_x, pred_y, pred_detected = pred
    gt_x, gt_y, gt_detected = gt

    if not pred_detected and not gt_detected:
        return 0, 0, 0
    if pred_detected and not gt_detected:
        return 0, 1, 0
    if not pred_detected and gt_detected:
        return 0, 0, 1

    # Both detected -- check distance
    dist = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
    if dist <= distance_threshold:
        return 1, 0, 0  # True positive
    else:
        return 0, 1, 1  # Both FP and FN


def aggregate_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision, recall, and F1 from TP/FP/FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    detection_threshold: float = 0.5,
    distance_threshold: float = 4.0,
) -> dict[str, float]:
    """Run full-epoch detection and return precision/recall/F1.

    Args:
        model: Trained model returning (B, 3, H, W) sigmoid heatmaps.
        dataloader: Val/test DataLoader yielding (frames, heatmaps).
        device: Device to run inference on.
        detection_threshold: Heatmap threshold for ball detection.
        distance_threshold: Euclidean pixel distance for TP matching.

    Returns:
        Dict with keys: precision, recall, f1, tp, fp, fn.
    """
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0

    for frames, gt_heatmaps in dataloader:
        frames = frames.to(device)
        gt_heatmaps = gt_heatmaps.to(device)

        preds = model(frames)

        batch_size = preds.shape[0]
        num_frames = preds.shape[1]

        for b in range(batch_size):
            for f in range(num_frames):
                pred_pos = heatmap_to_position(
                    preds[b, f], threshold=detection_threshold
                )
                gt_pos = heatmap_to_position(
                    gt_heatmaps[b, f], threshold=detection_threshold
                )
                tp, fp, fn = compute_detection_metrics(
                    pred_pos, gt_pos, distance_threshold=distance_threshold
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn

    metrics = aggregate_metrics(total_tp, total_fp, total_fn)
    metrics["tp"] = total_tp
    metrics["fp"] = total_fp
    metrics["fn"] = total_fn
    return metrics
