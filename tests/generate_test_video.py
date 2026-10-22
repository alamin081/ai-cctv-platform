"""
Utility script to generate a realistic sample CCTV test video with moving persons.
Used for Phase 2 (Video Reader) and Phase 3 (YOLO Person Detection) testing.
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics.utils import ASSETS


def generate_sample_cctv_video(
    output_path: Path,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    total_frames: int = 90,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter for {output_path}")

    # Load real person cutout from Ultralytics assets
    zidane_path = ASSETS / "zidane.jpg"
    person_patch = None
    if zidane_path.exists():
        src_img = cv2.imread(str(zidane_path))
        if src_img is not None:
            # Crop the person figure (approximate bbox in zidane image)
            # zidane image dimensions: approx 720 x 1280
            h_src, w_src = src_img.shape[:2]
            y1_crop, y2_crop = int(h_src * 0.15), int(h_src * 0.95)
            x1_crop, x2_crop = int(w_src * 0.15), int(w_src * 0.65)
            person_crop = src_img[y1_crop:y2_crop, x1_crop:x2_crop]
            # Resize to realistic surveillance scale
            target_h = 240
            aspect = person_crop.shape[1] / person_crop.shape[0]
            target_w = int(target_h * aspect)
            person_patch = cv2.resize(person_crop, (target_w, target_h))

    for i in range(total_frames):
        # CCTV dark surveillance corridor ambiance
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (40, 44, 48)

        # Draw corridor perspective lines
        cv2.line(frame, (0, 360), (width, 360), (60, 65, 70), 2)
        cv2.line(frame, (100, 360), (220, height), (55, 60, 65), 1)
        cv2.line(frame, (540, 360), (420, height), (55, 60, 65), 1)

        # Overlay moving person
        if person_patch is not None:
            p_h, p_w = person_patch.shape[:2]
            # Smooth horizontal walking motion
            start_x = int(-50 + (i * (width + 100)) / total_frames)
            pos_y = 150

            # Compute overlapping regions
            fx1 = max(0, start_x)
            fy1 = pos_y
            fx2 = min(width, start_x + p_w)
            fy2 = min(height, pos_y + p_h)

            px1 = max(0, -start_x)
            py1 = 0
            px2 = px1 + (fx2 - fx1)
            py2 = fy2 - fy1

            if fx2 > fx1 and fy2 > fy1 and px2 > px1 and py2 > py1:
                frame[fy1:fy2, fx1:fx2] = person_patch[py1:py2, px1:px2]

        # Surveillance camera watermark overlay
        cv2.putText(
            frame,
            "CCTV CAM-01 [BANGLALINK GATE-01]",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 128),
            2,
        )
        timestamp_str = f"FRAME: {i + 1:03d}/{total_frames:03d} | TIME: {i / fps:05.2f}s"
        cv2.putText(
            frame,
            timestamp_str,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )

        writer.write(frame)

    writer.release()
    print(f"Sample CCTV video generated successfully: {output_path.resolve()} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "data" / "videos" / "test.mp4"
    generate_sample_cctv_video(target)
