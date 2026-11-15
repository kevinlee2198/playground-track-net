from __future__ import annotations

import math

import torch

from inference.postprocess import heatmap_to_coordinates

OUTCOMES = ("tp", "tn", "fp1", "fp2", "fn")


def heatmap_to_position(
    heatmap: torch.Tensor, threshold: float = 0.5
) -> tuple[float, float, bool]:
    """Extract ball position from a single-frame heatmap.

    The position is the centroid of the largest connected component above
    ``threshold``, the same rule inference uses, so a small noise blob does
    not pull the position away from the ball.

    Args:
        heatmap: (H, W) tensor with values in [0, 1].
        threshold: Minimum value to consider as ball detection.

    Returns:
        (x, y, detected): x/y in heatmap pixels and whether a ball was
        detected. (0.0, 0.0, False) when nothing exceeds the threshold.
    """
    height, width = heatmap.shape
    coords = heatmap_to_coordinates(
        heatmap.detach().float().cpu().numpy(),
        orig_width=width,
        orig_height=height,
        threshold=threshold,
    )
    if coords is None:
        return 0.0, 0.0, False
    x, y, _confidence = coords
    return float(x), float(y), True


def classify_frame(
    pred: tuple[float, float, bool],
    gt: tuple[float, float, bool],
    distance_threshold: float = 4.0,
) -> str:
    """Classify one frame as "tp", "tn", "fp1", "fp2" or "fn".

    fp1 is a detection too far from a present ball (distance above
    ``distance_threshold``); fp2 is a detection when there is no ball.

    Args:
        pred: (x, y, detected) predicted position.
        gt: (x, y, detected) ground truth position.
        distance_threshold: Maximum Euclidean distance for a true positive.
    """
    pred_x, pred_y, pred_detected = pred
    gt_x, gt_y, gt_detected = gt

    if not gt_detected:
        return "fp2" if pred_detected else "tn"
    if not pred_detected:
        return "fn"
    distance = math.hypot(pred_x - gt_x, pred_y - gt_y)
    return "tp" if distance <= distance_threshold else "fp1"


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def aggregate_metrics(counts: dict[str, int]) -> dict[str, float]:
    """Compute detection metrics from per-outcome frame counts.

    Two recall conventions are reported: ``recall`` leaves fp1 frames out of
    its denominator (they count only against precision), while
    ``recall_strict`` also counts them as misses.

    Args:
        counts: Frame counts keyed by every name in ``OUTCOMES``.

    Returns:
        Dict with precision, recall, f1, recall_strict, f1_strict and
        accuracy. Any ratio with a zero denominator is 0.0.
    """
    tp, tn, fp1, fp2, fn = (counts[outcome] for outcome in OUTCOMES)
    precision = _ratio(tp, tp + fp1 + fp2)
    recall = _ratio(tp, tp + fn)
    recall_strict = _ratio(tp, tp + fn + fp1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "recall_strict": recall_strict,
        "f1_strict": _ratio(2 * precision * recall_strict, precision + recall_strict),
        "accuracy": _ratio(tp + tn, tp + tn + fp1 + fp2 + fn),
    }


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    detection_threshold: float = 0.5,
    distance_threshold: float = 4.0,
) -> dict[str, float]:
    """Run full-epoch detection and return the detection metrics.

    Args:
        model: Trained model returning (B, 3, H, W) sigmoid heatmaps.
        dataloader: Val/test DataLoader yielding (frames, heatmaps).
        device: Device to run inference on.
        detection_threshold: Heatmap threshold for ball detection.
        distance_threshold: Euclidean distance for TP matching, in heatmap
            pixels.

    Returns:
        The keys of ``aggregate_metrics`` plus the counts tp, tn, fp1, fp2, fn.
    """
    model.eval()
    counts = dict.fromkeys(OUTCOMES, 0)

    for frames, gt_heatmaps in dataloader:
        preds = model(frames.to(device)).cpu()
        for pred_heatmap, gt_heatmap in zip(
            preds.flatten(0, 1), gt_heatmaps.flatten(0, 1), strict=True
        ):
            pred_pos = heatmap_to_position(pred_heatmap, threshold=detection_threshold)
            gt_pos = heatmap_to_position(gt_heatmap, threshold=detection_threshold)
            counts[classify_frame(pred_pos, gt_pos, distance_threshold)] += 1

    return {**aggregate_metrics(counts), **counts}
