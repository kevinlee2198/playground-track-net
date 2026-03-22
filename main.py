import argparse
import csv
import sys
from pathlib import Path

import torch

from inference.video_preprocess import create_sliding_windows, extract_frames
from inference.postprocess import heatmap_to_coordinates, trajectory_rectification
from inference.tracker import KalmanBallTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracknet",
        description="TrackNet V2 ball tracking system",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand (placeholder)
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml")

    # Evaluate subcommand (placeholder)
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument("--config", type=str, default="configs/default.yaml")
    evaluate_parser.add_argument("--weights", type=str, required=True)

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

    # 2. Create sliding windows
    windows = create_sliding_windows(frames, window_size=3, stride=3)
    print(f"Created {len(windows)} sliding windows")

    # 3. Load model
    print(f"Loading model from {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model loading assumes models/tracknet.py exists and provides TrackNet class
    # This will be implemented by the model subsystem
    from models.tracknet import TrackNet

    model = TrackNet()
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True)
    )
    model.to(device)

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
                if result is not None:
                    all_detections.append(result)
                else:
                    all_detections.append(None)

    # Trim to actual frame count (sliding windows may overshoot)
    all_detections = all_detections[: len(frames)]

    # 5. Trajectory rectification
    positions = [(d[0], d[1]) if d is not None else None for d in all_detections]
    rectified = trajectory_rectification(positions, window=8)

    # 6. Kalman smoothing
    tracker = KalmanBallTracker()
    final_results = []
    for i, det in enumerate(all_detections):
        confidence = det[2] if det is not None else 0.0
        pos = rectified[i]
        if pos is not None:
            sx, sy = tracker.update(pos[0], pos[1])
            visibility = 1
        else:
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output_video, fourcc, 30.0, (orig_w, orig_h))

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

    if args.command == "infer":
        run_inference(args)
    elif args.command == "train":
        print("Training not yet implemented. See training/ subsystem.")
        sys.exit(1)
    elif args.command == "evaluate":
        print("Evaluation not yet implemented. See training/ subsystem.")
        sys.exit(1)


if __name__ == "__main__":
    main()
