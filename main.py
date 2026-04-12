import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from inference.postprocess import heatmap_to_coordinates, trajectory_rectification
from inference.tracker import KalmanBallTracker
from inference.video_preprocess import create_sliding_windows, extract_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracknet",
        description="TrackNet V2 ball tracking system",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand (placeholder)
    _train_parser = subparsers.add_parser("train", help="Train the model")
    _train_parser.add_argument("--config", type=str, default="configs/default.yaml")

    # Evaluate subcommand (placeholder)
    _eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    _eval_parser.add_argument("--config", type=str, default="configs/default.yaml")
    _eval_parser.add_argument("--weights", type=str, required=True)

    # Score subcommand (Week 1: stub implementation)
    score_parser = subparsers.add_parser("score", help="Score a match (MVP)")
    score_parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to match video file",
    )
    score_parser.add_argument(
        "--player1",
        type=str,
        default="Player 1",
        help="Name of player 1",
    )
    score_parser.add_argument(
        "--player2",
        type=str,
        default="Player 2",
        help="Name of player 2",
    )
    score_parser.add_argument(
        "--sport",
        type=str,
        default="tennis",
        choices=["tennis", "badminton", "pickleball"],
        help="Sport type",
    )
    score_parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    score_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (show stats every 5 points)",
    )
    score_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to TrackNet model weights (.pt file). If not provided, runs in stub mode with fake data.",
    )
    score_parser.add_argument(
        "--yolo",
        type=str,
        default=None,
        help="Path to YOLO model for player tracking (e.g., yolov8x-pose.pt). If not provided, uses stub player tracking.",
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


def score_match(args: argparse.Namespace) -> None:
    """
    Score a match with real or fake ball tracking.

    Week 1: Stub mode with fake data
    Week 2: Real ball tracking with TrackNet
    Week 3+: Real player tracking, event detection, scoring
    """
    from datetime import datetime

    from utils.console_logger import ConsoleMatchLogger
    from utils.json_exporter import MatchJSONExporter

    print(f"Processing video: {args.video}")

    # Determine mode based on which models are provided
    ball_mode = "real" if args.model is not None else "stub"
    player_mode = "real" if args.yolo is not None else "stub"
    # Enable event detection if we have both ball and player tracking
    event_mode = "real" if (ball_mode == "real" and player_mode == "real") else "stub"

    print(f"Ball tracking: {ball_mode.upper()} mode" + (f" ({args.model})" if ball_mode == "real" else ""))
    print(f"Player tracking: {player_mode.upper()} mode" + (f" ({args.yolo})" if player_mode == "real" else ""))
    print(f"Event detection: {event_mode.upper()} mode")
    print("Scoring: STUB mode (Week 5+)")
    print()

    # Generate match ID
    match_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Initialize console logger
    logger = ConsoleMatchLogger([args.player1, args.player2], args.sport)

    if ball_mode == "stub" and player_mode == "stub":
        # === FULL STUB MODE: Fake data ===
        print("Reading video... (STUB)")
        num_frames = 300  # Fake 10 seconds at 30fps
        fps = 30.0

        print("Tracking ball... (STUB)")
        print("Tracking players... (STUB)")
        print("Detecting events... (STUB)")
        print()

        ball_positions = []  # Will use fake events instead
        player_tracks = []  # Will use fake events instead

    else:
        # === REAL TRACKING MODE (ball and/or players) ===
        import cv2

        # Read video
        print("Reading video...")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {args.video}")
            sys.exit(1)

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {orig_width}x{orig_height} @ {fps:.1f}fps, {num_frames} frames")

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"Loaded {len(frames)} frames\n")

        # Initialize ball tracker if needed
        ball_positions = []
        if ball_mode == "real":
            from models.ball_tracker_wrapper import BallTrackerWrapper

            print(f"Initializing ball tracker...")
            ball_tracker = BallTrackerWrapper(model_path=args.model)
            print("Ball tracker ready\n")

            # Track ball frame by frame
            print("Tracking ball...")
            for i in range(len(frames)):
                # Get 3-frame window
                frame_window = []
                for offset in [-1, 0, 1]:
                    idx = max(0, min(len(frames) - 1, i + offset))
                    frame_window.append(frames[idx])

                # Track ball
                ball_pos = ball_tracker.track_ball(
                    frame_window, orig_width=orig_width, orig_height=orig_height
                )
                ball_positions.append(ball_pos)

                # Progress indicator
                if (i + 1) % 100 == 0:
                    detected_count = sum(1 for p in ball_positions if p is not None)
                    detection_rate = detected_count / len(ball_positions) * 100
                    print(
                        f"  Frame {i+1}/{len(frames)} - Detection rate: {detection_rate:.1f}%"
                    )

            detected_count = sum(1 for p in ball_positions if p is not None)
            detection_rate = detected_count / len(ball_positions) * 100
            print(f"Ball tracking complete - Detection rate: {detection_rate:.1f}%\n")
        else:
            print("Ball tracking: STUB mode\n")

        # Initialize player tracker if needed
        player_tracks = []
        if player_mode == "real":
            from models.player_detector import PlayerDetector
            from models.trackers.simple_tracker import SimpleTracker

            print(f"Initializing player detector...")
            player_detector = PlayerDetector(stub=False, model_name=args.yolo)
            player_tracker = SimpleTracker(iou_threshold=0.3, max_age=30)
            print("Player detector ready\n")

            # Track players frame by frame
            print("Tracking players...")
            for i, frame in enumerate(frames):
                # Detect players
                detections = player_detector.detect(frame)

                # Track players
                tracks = player_tracker.update(detections)
                player_tracks.append(tracks)

                # Progress indicator
                if (i + 1) % 100 == 0:
                    avg_detections = np.mean([len(t) for t in player_tracks])
                    print(f"  Frame {i+1}/{len(frames)} - Avg players tracked: {avg_detections:.1f}")

            avg_detections = np.mean([len(t) for t in player_tracks])
            print(f"Player tracking complete - Avg players tracked: {avg_detections:.1f}\n")
        else:
            print("Player tracking: STUB mode\n")

        # Event detection (requires both ball and player tracking)
        detected_events = []
        if event_mode == "real":
            from inference.event_detector import EventDetector

            print("Detecting events...")
            event_detector = EventDetector(sport=args.sport)

            for i in range(len(frames)):
                ball_pos = ball_positions[i] if i < len(ball_positions) else None
                # Extract (x, y) from ball position tuple if present
                if ball_pos is not None and len(ball_pos) >= 2:
                    ball_pos = (ball_pos[0], ball_pos[1])

                player_track = player_tracks[i] if i < len(player_tracks) else np.array([]).reshape(0, 5)

                events = event_detector.process_frame(i, ball_pos, player_track)
                detected_events.extend(events)

                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"  Frame {i+1}/{len(frames)} - {len(detected_events)} events detected")

            print(f"Event detection complete - {len(detected_events)} events detected\n")

            # Print event summary
            event_types = {}
            for event in detected_events:
                event_type = event["type"]
                event_types[event_type] = event_types.get(event_type, 0) + 1

            print("Event summary:")
            for event_type, count in sorted(event_types.items()):
                print(f"  {event_type}: {count}")
            print()
        else:
            print("Event detection: STUB mode\n")

    # Use real events if available, otherwise fake events
    if event_mode == "real" and len(detected_events) > 0:
        # Log real detected events to console
        print("Logging detected events to console...\n")
        for event in detected_events:
            frame_id = event["frame"]
            timestamp = frame_id / fps
            logger.log_event(event, frame_id, timestamp)
    else:
        # Fake events for a short rally (stub mode)
        fake_events = [
            # Serve
            {
                "type": "serve_start",
                "frame": 10,
                "player_id": 0,
                "data": {"is_first_serve": True},
            },
            {
                "type": "bounce",
                "frame": 25,
                "data": {"zone": "left_service_box", "valid": True},
            },
            {"type": "hit", "frame": 30, "player_id": 1, "data": {"shot_type": "forehand"}},
            {"type": "hit", "frame": 50, "player_id": 0, "data": {"shot_type": "backhand"}},
            {
                "type": "bounce",
                "frame": 70,
                "data": {"zone": "out", "valid": False},
            },
            {"type": "rally_end", "frame": 70, "winner": 0, "reason": "out"},
            # Second point - ace
            {
                "type": "serve_start",
                "frame": 100,
                "player_id": 0,
                "data": {"is_first_serve": True},
            },
            {
                "type": "bounce",
                "frame": 115,
                "data": {"zone": "left_service_box", "valid": True},
            },
            {"type": "rally_end", "frame": 115, "winner": 0, "reason": "ace"},
        ]

        # Log fake events to console
        for event in fake_events:
            frame_id = event["frame"]
            timestamp = frame_id / 30.0  # Assume 30fps
            logger.log_event(event, frame_id, timestamp)

    # Initialize scoring engine
    from inference.scoring import TennisScorer

    scorer = TennisScorer(args.player1, args.player2, best_of=3)

    # For now, award some fake points to demonstrate scoring
    # In a real implementation, this would be based on rally end detection
    if event_mode == "stub":
        # Fake scoring for demonstration
        scorer.award_point(0, reason="ace")
        logger.log_score(scorer.get_score_string())

        scorer.award_point(0, reason="winner")
        logger.log_score(scorer.get_score_string())
    else:
        # With real events, we would detect rally ends and award points
        # For now, just show the initial score
        logger.log_score(scorer.get_score_string())
        print("Note: Automatic scoring from events not yet implemented (Week 6)")
        print("      Events detected but not converted to points yet\n")

    # Get stats from scorer
    p0_stats = scorer.get_player_stats(0)
    p1_stats = scorer.get_player_stats(1)

    # Get final score and winner
    final_score_str = scorer.get_final_score() if scorer.match_over else "Match incomplete"
    winner_id = scorer.get_winner() if scorer.match_over else 0

    # Log final summary
    logger.log_final_summary(
        final_score=final_score_str,
        winner_id=winner_id,
        player_stats=[p0_stats, p1_stats],
        duration_seconds=600,  # Placeholder - would calculate from timestamps
    )

    # Export JSON (REAL)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minimal_path = output_dir / f"{match_id}_minimal.json"
    exporter = MatchJSONExporter()
    exporter.export_minimal(
        output_path=str(minimal_path),
        match_id=match_id,
        date=datetime.now().isoformat(),
        sport=args.sport,
        player_names=[args.player1, args.player2],
        winner_id=0,
        final_score="6-0 (FAKE)",
        player_stats=[
            {
                "player_id": 0,
                "points": 2,
                "aces": 1,
                "winners": 0,
                "errors": 0,
                "first_serve_pct": 100.0,
            },
            {
                "player_id": 1,
                "points": 0,
                "aces": 0,
                "winners": 0,
                "errors": 1,
                "first_serve_pct": 0.0,
            },
        ],
    )

    print(f"\n{'═' * 70}")
    if ball_mode == "stub" and player_mode == "stub":
        print("Week 1 STUB processing complete!")
        print(f"Output: {minimal_path}")
        print("Next: Use --model and --yolo for real tracking")
    elif ball_mode == "real" and player_mode == "stub":
        print("Week 2 processing complete (real ball tracking)!")
        print(f"Output: {minimal_path}")
        print("Next: Use --yolo for player tracking")
    elif ball_mode == "real" and player_mode == "real" and event_mode == "stub":
        print("Week 3 processing complete (real ball + player tracking)!")
        print(f"Output: {minimal_path}")
        print("Next: Event detection automatic (requires both ball + player tracking)")
    elif ball_mode == "real" and player_mode == "real" and event_mode == "real":
        print("Week 4 processing complete (ball + player + events)!")
        print(f"Output: {minimal_path}")
        if len(detected_events) > 0:
            print(f"Detected {len(detected_events)} events")
        print("Next: Add scoring engine (Week 5+)")
    else:
        print(f"Processing complete (ball: {ball_mode}, players: {player_mode}, events: {event_mode})!")
        print(f"Output: {minimal_path}")
    print("═" * 70 + "\n")


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
    import torch

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

    if args.command == "score":
        score_match(args)
    elif args.command == "infer":
        run_inference(args)
    elif args.command == "train":
        print("Training not yet implemented. See training/ subsystem.")
        sys.exit(1)
    elif args.command == "evaluate":
        print("Evaluation not yet implemented. See training/ subsystem.")
        sys.exit(1)


if __name__ == "__main__":
    main()
