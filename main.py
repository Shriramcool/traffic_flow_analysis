import os
import sys
import time
import math
import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


DEFAULT_YT_URL = "https://www.youtube.com/watch?v=MNn9qKG2UFI"

# COCO vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def ensure_dirs(video_dir: Path, output_dir: Path):
    video_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def have_exe(name: str) -> bool:
    """Check if an executable is on PATH (Windows-safe)."""
    from shutil import which
    return which(name) is not None


def download_with_ytdlp(url: str, out_path: Path):
    """
    Reliable YouTube download using yt-dlp.
    Saves MP4 up to 720p into out_path.
    """
    if not have_exe("yt-dlp"):
        raise RuntimeError(
            "yt-dlp not found. Install it by running: pip install yt-dlp"
        )
    print("[info] Downloading video from YouTube using yt-dlp...")
    
    cmd = [
        "yt-dlp",
        "-f", "mp4[height<=720]/bv*[height<=720]+ba/best[height<=720]",
        "-o", str(out_path),
        url,
    ]
    subprocess.run(cmd, check=True)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("yt-dlp finished but output file was not created.")


def get_device(force_cpu: bool = False) -> str:
    """
    Pick device for YOLO. Prefers CUDA if available and not forced to CPU.
    """
    if force_cpu:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_lanes(frame_height: int, mode: str = "equal_thirds"):
    """
    Define 3 lane regions. For simplicity & robustness on this video,
    we treat them as 3 horizontal bands. You can switch mode later if needed.
    """
    margin = max(8, frame_height // 120)
    band = frame_height // 3
    lanes = [
        (margin, band - margin),
        (band + margin, 2 * band - margin),
        (2 * band + margin, frame_height - margin),
    ]
    return lanes  # list of (ymin, ymax)


def draw_translucent_lanes(img, lanes):
    overlay = img.copy()
    H, W = img.shape[:2]
    colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255)]  # BGR
    for i, (ymin, ymax) in enumerate(lanes):
        cv2.rectangle(overlay, (0, ymin), (W, ymax), colors[i % len(colors)], thickness=-1)
        cv2.putText(
            overlay, f"Lane {i+1}", (10, max(25, ymin + 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3
        )
        cv2.putText(
            overlay, f"Lane {i+1}", (10, max(25, ymin + 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
    return cv2.addWeighted(overlay, 0.15, img, 0.85, 0)


def format_time(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def process_video(
    source_path: Path,
    csv_path: Path,
    annotated_path: Path,
    weights: str = "yolov8n.pt",
    conf: float = 0.35,
    iou: float = 0.5,
    save_annotated: bool = True,
    resize_width: int | None = 1280,
    force_cpu: bool = False,
    max_seconds: int | None = None,
):
    """
    Core pipeline: detect+track vehicles, lane assign, counting, CSV + annotated video.
    """
    device = get_device(force_cpu)
    print(f"[info] Using device: {device}")
    print("[info] Loading YOLO model...")
    model = YOLO(weights)
    if device == "cuda":
        model.to("cuda")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resize_width and resize_width < src_w:
        scale = resize_width / src_w
        width = resize_width
        height = int(src_h * scale)
    else:
        scale = 1.0
        width = src_w
        height = src_h

    lanes = build_lanes(height)
    lane_counts = [0, 0, 0]
    lane_id_sets = [set(), set(), set()]  # to avoid duplicate counts per lane

    # CSV rows: Vehicle ID, Lane number, Frame index, Timestamp seconds
    rows = []

    writer = None
    if save_annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, src_fps, (width, height))

    frame_idx = 0
    start_time = time.time()

    print("[info] Processing... (focus the video window and press 'q' to quit)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

       
        if scale != 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        # Inference + tracking (ByteTrack under the hood)
        results = model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        vis = draw_translucent_lanes(frame, lanes)

        if results and len(results) > 0:
            res = results[0]
            if res.boxes is not None:
                boxes = res.boxes
                track_ids = boxes.id  # may be tensor 

                for j in range(len(boxes)):
                    cls_id = int(boxes.cls[j].item())
                    if cls_id not in VEHICLE_CLASS_IDS:
                        continue

                    x1, y1, x2, y2 = map(int, boxes.xyxy[j].tolist())
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    track_id = None
                    if track_ids is not None and len(track_ids) > j and track_ids[j] is not None:
                        try:
                            track_id = int(track_ids[j].item())
                        except Exception:
                            try:
                                track_id = int(track_ids[j])
                            except Exception:
                                track_id = None

                    # Draw detection
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 200, 40), 2)
                    tag = f"ID {track_id}" if track_id is not None else "ID ?"
                    cv2.putText(vis, tag, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(vis, tag, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

                    # Lane assignment by centroid Y
                    for lane_idx, (ymin, ymax) in enumerate(lanes):
                        if ymin <= cy <= ymax:
                            if track_id is not None and track_id not in lane_id_sets[lane_idx]:
                                lane_id_sets[lane_idx].add(track_id)
                                lane_counts[lane_idx] += 1
                                timestamp = frame_idx / src_fps
                                rows.append([track_id, lane_idx + 1, frame_idx, f"{timestamp:.2f}"])
                            break

        # HUD: lane counters + FPS
        for i, c in enumerate(lane_counts):
            cv2.putText(vis, f"Lane {i+1}: {c}", (20, 40 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            cv2.putText(vis, f"Lane {i+1}: {c}", (20, 40 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        elapsed = max(1e-6, time.time() - start_time)
        fps_live = frame_idx / elapsed
        cv2.putText(vis, f"FPS: {fps_live:.1f}", (width - 200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(vis, f"FPS: {fps_live:.1f}", (width - 200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer:
            writer.write(vis)

        cv2.imshow("Traffic Flow Analysis", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if max_seconds is not None and (frame_idx / src_fps) >= max_seconds:
            print(f"[info] Reached max_seconds={max_seconds}, stopping early.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Save CSV
    pd.DataFrame(rows, columns=["Vehicle_ID", "Lane", "Frame", "Timestamp"]).to_csv(csv_path, index=False)

    # Summary / Output
    print("\n=== SUMMARY ===")
    for i, c in enumerate(lane_counts):
        print(f"Lane {i+1}: {c} vehicles")
    print(f"CSV saved to: {csv_path}")
    if writer:
        print(f"Annotated video saved to: {annotated_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis: detect, track, and count vehicles per lane."
    )
    parser.add_argument("--yt", type=str, default=DEFAULT_YT_URL,
                        help="YouTube URL to download if local file not present.")
    parser.add_argument("--video_dir", type=str, default="data", help="Folder to keep/download the video.")
    parser.add_argument("--output_dir", type=str, default="output", help="Folder for outputs (CSV, annotated video).")
    parser.add_argument("--video_name", type=str, default="traffic.mp4", help="Video filename to use.")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights (e.g., yolov8n.pt, yolov8s.pt).")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold.")
    parser.add_argument("--no-save-video", action="store_true", help="Do not save annotated video.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--resize-width", type=int, default=1280, help="Resize width for speed (<= original width).")
    parser.add_argument("--max-seconds", type=int, default=None, help="Process only the first N seconds (for demo).")

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    video_path = video_dir / args.video_name
    annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
    csv_path = output_dir / "traffic_count.csv"

    ensure_dirs(video_dir, output_dir)

    if video_path.exists() and video_path.stat().st_size > 0:
        print(f"[info] Using local video file: {video_path}")
    else:
        print(f"[info] Local file not found: {video_path}")
        print("[info] Attempting to download from YouTube...")
        try:
            download_with_ytdlp(args.yt, video_path)
            print(f"[info] Download complete: {video_path}")
        except Exception as e:
            print(f"[error] Could not obtain video automatically: {e}")
            print(f"[hint] Manually download the video and place it at: {video_path}")
            sys.exit(1)

    process_video(
        source_path=video_path,
        csv_path=csv_path,
        annotated_path=annotated_path,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        save_annotated=not args.no-save_video if hasattr(args, "no-save_video") else True,  # safety
        resize_width=args.resize_width if args.resize_width > 0 else None,
        force_cpu=args.force_cpu,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    main()
