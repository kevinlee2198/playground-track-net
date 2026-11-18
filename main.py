import argparse
import csv
import json
import logging
from pathlib import Path

from inference.postprocess import heatmap_to_coordinates, trajectory_rectification
from inference.tracker import KalmanBallTracker
from inference.video_preprocess import create_sliding_windows, extract_frames

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracknet",
        description="TrackNet V2 ball tracking system",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer steps in total (overrides the config)",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Trainer checkpoint (.pt) to resume from",
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a model on the val or test split"
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    eval_parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights or trainer checkpoint (.pt file)",
    )
    eval_parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Split to evaluate (default: test)",
    )

    # Check-data subcommand
    check_parser = subparsers.add_parser(
        "check-data",
        help="Draw labels on sample frames of every split and log label counts",
    )
    check_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    check_parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Directory to write the overlay PNGs to",
    )
    check_parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of overlays, spread across train, val and test (default: 12)",
    )

    # Extract-frames subcommand
    extract_parser = subparsers.add_parser(
        "extract-frames",
        help="Decode the badminton dataset's rally videos into labelled frame images",
    )
    extract_parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Badminton dataset root, holding <part>/<match>/video and csv",
    )
    extract_parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality of the written frames, 0-100 (default: 95)",
    )

    # Infer subcommand
    infer_parser = subparsers.add_parser("infer", help="Run inference on a video")
    infer_parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file or image directory",
    )
    infer_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights (.pt file)",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file",
    )
    infer_parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Path to output annotated video (optional)",
    )
    infer_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Heatmap detection threshold (default: 0.5)",
    )

    return parser


def run_inference(args: argparse.Namespace) -> None:
    """Run the full inference pipeline."""
    # 1. Extract and preprocess frames
    print(f"Extracting frames from {args.video}...")
    frames, metadata = extract_frames(args.video)
    orig_w = metadata["original_width"]
    orig_h = metadata["original_height"]
    print(f"Extracted {len(frames)} frames ({orig_w}x{orig_h})")

    # 2. Load the model as it was trained
    import torch

    print(f"Loading model from {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models import load_model

    model, stored = load_model(args.model, device)

    # 3. Create sliding windows, each with the video's median background frame
    # after its three frames if the model was trained with one
    background = None
    if frames and stored is not None and stored.get("background_frame", False):
        from data import background_indices, median_frame

        background = median_frame(frames[i] for i in background_indices(len(frames)))
    windows = create_sliding_windows(
        frames, window_size=3, stride=3, background=background
    )
    print(f"Created {len(windows)} sliding windows")

    # 4. Run inference on each window
    all_detections: list[tuple[float, float, float] | None] = []
    model.eval()
    with torch.no_grad():
        for window in windows:
            input_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            heatmaps = model(input_tensor)  # (1, 3, H, W)
            heatmaps = heatmaps.squeeze(0).cpu().numpy()  # (3, H, W)

            for hm in heatmaps:
                result = heatmap_to_coordinates(
                    hm,
                    orig_width=orig_w,
                    orig_height=orig_h,
                    threshold=args.threshold,
                )
                all_detections.append(result)

    # Trim to actual frame count (sliding windows may overshoot)
    all_detections = all_detections[: len(frames)]

    # 5. Trajectory rectification
    positions = [(d[0], d[1]) if d is not None else None for d in all_detections]
    rectified = trajectory_rectification(positions, window=8)

    # 6. Kalman smoothing
    tracker = KalmanBallTracker()
    consecutive_missing = 0
    max_gap_before_reset = 10
    final_results = []
    for i, det in enumerate(all_detections):
        confidence = det[2] if det is not None else 0.0
        pos = rectified[i]
        if pos is not None:
            consecutive_missing = 0
            sx, sy = tracker.update(pos[0], pos[1])
            visibility = 1
        else:
            consecutive_missing += 1
            if consecutive_missing >= max_gap_before_reset:
                tracker.reset()
            sx, sy = 0.0, 0.0
            visibility = 0
        final_results.append(
            {
                "frame": i,
                "x": sx,
                "y": sy,
                "confidence": confidence,
                "visibility": visibility,
            }
        )

    # 7. Write CSV output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Frame", "X", "Y", "Confidence", "Visibility"]
        )
        writer.writeheader()
        for r in final_results:
            writer.writerow(
                {
                    "Frame": r["frame"],
                    "X": f"{r['x']:.2f}",
                    "Y": f"{r['y']:.2f}",
                    "Confidence": f"{r['confidence']:.4f}",
                    "Visibility": r["visibility"],
                }
            )
    print(f"Results written to {args.output}")

    # 8. Optional annotated video output
    if args.output_video:
        import cv2

        from utils.visualization import draw_ball_on_frame

        cap = cv2.VideoCapture(args.video)
        fps = metadata.get("fps", 30.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output_video, fourcc, fps, (orig_w, orig_h))

        for r in final_results:
            ret, frame = cap.read()
            if not ret:
                break
            if r["visibility"] == 1:
                frame = draw_ball_on_frame(
                    frame,
                    r["x"],
                    r["y"],
                    r["confidence"],
                )
            out.write(frame)

        cap.release()
        out.release()
        print(f"Annotated video written to {args.output_video}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command != "infer":  # infer reports progress with print
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    if args.command == "infer":
        run_inference(args)
    elif args.command == "train":
        from training.run import load_config, run_training

        config = load_config(args.config)
        if args.max_steps is not None:
            config["max_steps"] = args.max_steps
        run_training(config, resume=args.resume)
    elif args.command == "evaluate":
        from training.run import load_config, run_evaluation

        result = run_evaluation(load_config(args.config), args.weights, args.split)
        print(json.dumps(result, indent=2))
    elif args.command == "check-data":
        from training.run import load_config, write_label_overlays

        paths = write_label_overlays(load_config(args.config), args.out, args.count)
        print(f"Wrote {len(paths)} overlays to {args.out}")
    elif args.command == "extract-frames":
        from data import extract_badminton_frames

        extract_badminton_frames(args.root, jpeg_quality=args.jpeg_quality)


if __name__ == "__main__":
    main()
