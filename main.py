"""
AI CCTV Platform - Main Entry Point
Phase: Phase 9 - Footfall Analytics Engine
Pipeline: Video Reader -> YOLO Person Detector -> ByteTrack Tracker -> Track State Manager -> Zone Engine -> Zone Counting Engine -> Entry/Exit Engine -> Footfall Analytics Engine
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import yaml

from src.analytics import (
    EntryExitEngine,
    EntryExitError,
    EntryExitEvent,
    EntryExitResult,
    FootfallAnalyticsEngine,
    FootfallBucket,
    FootfallError,
    FootfallResult,
    FootfallSummary,
    MovementType,
)
from src.detection import (
    DetectionError,
    InvalidFrameError,
    ModelLoadError,
    YOLOPersonDetector,
)
from src.tracking import (
    ByteTrackPersonTracker,
    InvalidStateManagerInputError,
    InvalidTrackerInputError,
    TrackerConfigError,
    TrackingError,
    TrackStateManager,
)
from src.video_input import (
    VideoFileReader,
    VideoInputError,
    VideoNotFoundError,
    VideoOpenError,
)
from src.zones import (
    DuplicateZoneIdError,
    InvalidPolygonError,
    InvalidZoneConfigError,
    TrackZoneAssignment,
    Zone,
    ZoneCountingEngine,
    ZoneCountingResult,
    ZoneEngine,
    ZoneError,
    ZoneEvent,
    ZoneOccupancy,
    ZoneResult,
)

# Configure standard Python logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file, with safe fallbacks."""
    default_config = {
        "video_input": {"source": "data/videos/test.mp4"},
        "detection": {
            "model_name": "data/models/yolov8n.pt",
            "confidence": 0.40,
            "iou": 0.45,
            "device": "auto",
        },
        "tracking": {
            "tracker": "bytetrack",
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "fuse_score": True,
            "save_visualization": False,
        },
        "track_state": {
            "enabled": True,
            "max_missed_frames": 30,
        },
        "zones": {
            "enabled": True,
            "assignment_mode": "center",
            "boundary_policy": "inside",
            "definitions": [
                {
                    "id": "entrance",
                    "name": "Entrance",
                    "polygon": [[0, 0], [320, 0], [320, 480], [0, 480]],
                },
                {
                    "id": "waiting_area",
                    "name": "Waiting Area",
                    "polygon": [[320, 0], [640, 0], [640, 480], [320, 480]],
                },
            ],
        },
        "zone_counting": {
            "enabled": True,
        },
        "entry_exit": {
            "enabled": True,
            "allow_initial_entry": True,  # Allows tracking entry of persons present in initial frame
            "definitions": [
                {"id": "main_entrance", "from": None, "to": "entrance", "type": "ENTRY"},
                {"id": "main_exit", "from": "entrance", "to": None, "type": "EXIT"},
            ],
        },
        "footfall": {
            "enabled": True,
            "bucket_minutes": 15,
            "deduplicate_events": True,
        },
    }

    if not config_path.exists():
        logger.warning("Config file not found at %s. Using default configuration.", config_path)
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            return loaded if loaded else default_config
    except Exception as e:
        logger.error("Failed to parse config.yaml: %s. Using defaults.", e)
        return default_config


def run_pipeline(project_root: Path, video_source: Optional[Any] = None) -> None:
    """Run full Phase 8 pipeline: Video Reader -> YOLO -> ByteTrack -> State Manager -> Zone Engine -> Zone Counting Engine -> EntryExitEngine."""
    config_file = project_root / "config.yaml"
    config = load_config(config_file)

    video_cfg = config.get("video_input", {})
    detection_cfg = config.get("detection", {})
    tracking_cfg = config.get("tracking", {})
    state_cfg = config.get("track_state", {})
    zones_cfg = config.get("zones", {})
    counting_cfg = config.get("zone_counting", {})
    entry_exit_cfg = config.get("entry_exit", {})
    footfall_cfg = config.get("footfall", {})

    if video_source is not None:
        video_path = Path(video_source) if Path(video_source).is_absolute() else project_root / video_source
    else:
        video_rel_path = video_cfg.get("source", "data/videos/test.mp4")
        video_path = project_root / video_rel_path

    model_name = detection_cfg.get("model_name", "data/models/yolov8n.pt")
    model_path = project_root / model_name if not Path(model_name).is_absolute() else Path(model_name)
    if not model_path.exists():
        model_path = Path("yolov8n.pt")

    confidence_threshold = float(detection_cfg.get("confidence", 0.40))
    iou_threshold = float(detection_cfg.get("iou", 0.45))
    device_setting = str(detection_cfg.get("device", "auto"))

    # Tracking parameters
    track_high_thresh = float(tracking_cfg.get("track_high_thresh", 0.5))
    track_low_thresh = float(tracking_cfg.get("track_low_thresh", 0.1))
    new_track_thresh = float(tracking_cfg.get("new_track_thresh", 0.6))
    track_buffer = int(tracking_cfg.get("track_buffer", 30))
    match_thresh = float(tracking_cfg.get("match_thresh", 0.8))
    fuse_score = bool(tracking_cfg.get("fuse_score", True))
    save_visualization = bool(tracking_cfg.get("save_visualization", False))

    # Track State Manager parameters
    max_missed_frames = int(state_cfg.get("max_missed_frames", 30))

    # Zone Engine & Counting parameters
    zones_enabled = bool(zones_cfg.get("enabled", True))
    counting_enabled = bool(counting_cfg.get("enabled", True))
    entry_exit_enabled = bool(entry_exit_cfg.get("enabled", True))
    footfall_enabled = bool(footfall_cfg.get("enabled", True))

    print("=" * 65)
    print("      AI CCTV Video Analytics Platform - Phase 9 Pipeline")
    print("=" * 65)

    # 1. Initialize Video Reader (Phase 2)
    reader = VideoFileReader(video_path)
    try:
        reader.open()
    except VideoNotFoundError as e:
        logger.error("Video file not found: %s", e)
        print(f"[!] Error: {e}")
        return
    except VideoOpenError as e:
        logger.error("Failed to open video file: %s", e)
        print(f"[!] Error: {e}")
        return
    except VideoInputError as e:
        logger.error("Video input error: %s", e)
        print(f"[!] Error: {e}")
        return

    # 2. Initialize YOLO Person Detector (Phase 3)
    try:
        detector = YOLOPersonDetector(
            model_name_or_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device_setting,
        )
    except ModelLoadError as e:
        logger.error("Failed to initialize YOLO detector: %s", e)
        print(f"[!] Error: {e}")
        reader.release()
        return

    # 3. Initialize ByteTrack Person Tracker (Phase 4)
    try:
        tracker = ByteTrackPersonTracker(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
    except TrackerConfigError as e:
        logger.error("Failed to initialize ByteTrack tracker: %s", e)
        print(f"[!] Error: {e}")
        reader.release()
        return

    # 4. Initialize Track State Manager (Phase 5)
    state_manager = TrackStateManager(max_missed_frames=max_missed_frames)

    # 5. Initialize Zone Engine (Phase 6)
    zone_engine = None
    if zones_enabled:
        try:
            zone_engine = ZoneEngine.from_config(zones_cfg)
        except ZoneError as e:
            logger.error("Failed to initialize Zone Engine: %s", e)
            print(f"[!] Error: {e}")
            reader.release()
            return

    # 6. Initialize Zone Counting Engine (Phase 7)
    counting_engine = None
    if counting_enabled and zone_engine is not None:
        counting_engine = ZoneCountingEngine.from_zone_engine(zone_engine)

    # 7. Initialize Entry / Exit Engine (Phase 8)
    entry_exit_engine = None
    if entry_exit_enabled:
        try:
            entry_exit_engine = EntryExitEngine.from_config(entry_exit_cfg)
        except EntryExitError as e:
            logger.error("Failed to initialize EntryExitEngine: %s", e)
            print(f"[!] Error: {e}")
            reader.release()
            return

    # 8. Initialize Footfall Analytics Engine (Phase 9)
    footfall_engine = None
    if footfall_enabled:
        try:
            footfall_engine = FootfallAnalyticsEngine.from_config(footfall_cfg)
        except FootfallError as e:
            logger.error("Failed to initialize FootfallAnalyticsEngine: %s", e)
            print(f"[!] Error: {e}")
            reader.release()
            return

    # Optional video writer for visual inspection
    video_writer = None
    output_video_path = project_root / "data" / "outputs" / "entry_exit_output.mp4"
    if save_visualization:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            reader.fps,
            (reader.width, reader.height),
        )

    # Startup Reporting
    print(f"Model                   : {model_path.name}")
    print(f"Detector Device         : {detector.device_display_name}")
    print(f"Tracker                 : ByteTrack (CPU)")
    print(f"Track State Manager     : Enabled (max_missed_frames={max_missed_frames})")
    if zone_engine:
        zone_summary = ", ".join(z.zone_id for z in zone_engine.zones)
        print(f"Zone Engine             : Enabled ({len(zone_engine.zones)} zones: {zone_summary})")
        print(f"Zone Assignment Mode    : {zone_engine.assignment_mode}")
        print(f"Zone Boundary Policy    : {zone_engine.boundary_policy}")
    else:
        print(f"Zone Engine             : Disabled")
    print(f"Zone Counting Engine    : {'Enabled' if counting_engine else 'Disabled'}")
    print(f"Entry / Exit Engine     : {'Enabled' if entry_exit_engine else 'Disabled'}")
    print(f"Footfall Engine         : {'Enabled' if footfall_engine else 'Disabled'}")
    if footfall_engine:
        print(f"Footfall Bucket Interval: {footfall_engine.bucket_seconds / 60:.1f} min ({footfall_engine.bucket_seconds:.0f}s)")
        print(f"Footfall Deduplication  : {footfall_engine.deduplicate_events}")
    print(f"Target Video            : {video_path.name}")
    print(f"Input FPS               : {reader.fps:.2f}")
    print(f"Resolution              : {reader.width}x{reader.height}")
    print(f"Total Video Frames      : {reader.total_frames}")
    print("-" * 65)

    frames_processed = 0
    start_total_time = time.perf_counter()
    sample_json = None

    try:
        while True:
            frame = reader.read()
            if frame is None:
                # Video stream reached end
                break

            frames_processed += 1

            # Step 1: Detect Persons (YOLO)
            det_result = detector.detect(frame)

            # Step 2: Track Association (ByteTrack)
            track_result = tracker.update(
                detections=det_result,
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                frame_bgr=frame.data,
            )

            # Step 3: Manage Track Lifecycle State (Track State Manager)
            state_result = state_manager.update(
                tracks=track_result,
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
            )

            # Step 4: Spatial Zone Membership & Transition Detection (Zone Engine)
            zone_result = None
            if zone_engine is not None:
                zone_result = zone_engine.update(
                    tracks=state_result,
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                    removed_track_ids=[t.track_id for t in state_manager.removed_tracks],
                )

            # Step 5: Zone Occupancy & Event Generation (Zone Counting Engine)
            counting_result = None
            if counting_engine is not None and zone_result is not None:
                counting_result = counting_engine.update(
                    zone_result=zone_result,
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                    removed_track_ids=[t.track_id for t in state_manager.removed_tracks],
                )

            # Step 6: Entry / Exit Movement Analytics (Entry / Exit Engine)
            entry_exit_result = None
            if entry_exit_engine is not None and zone_result is not None:
                entry_exit_result = entry_exit_engine.update(
                    zone_result=zone_result,
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                    removed_track_ids=[t.track_id for t in state_manager.removed_tracks],
                )

            # Step 7: Footfall Aggregation & Analytics (Footfall Analytics Engine)
            footfall_result = None
            if footfall_engine is not None and entry_exit_result is not None:
                footfall_result = footfall_engine.update(
                    events=entry_exit_result,
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                )

            # Capture sample JSON at frame 30 or first active frame
            if frames_processed == 30 and sample_json is None and zone_result and zone_result.count > 0:
                sample_json = {
                    "footfall_output": footfall_result.to_dict() if footfall_result else None,
                    "entry_exit_output": entry_exit_result.to_dict() if entry_exit_result else None,
                    "occupancy_output": counting_result.to_dict() if counting_result else None,
                    "zone_output": zone_result.to_dict(),
                    "state_output": state_result.to_dict(),
                }

            # Optional visual annotation
            if video_writer is not None:
                vis_frame = TrackStateManager.draw_state_tracks(frame.data, state_result.tracks)
                if zone_engine is not None and zone_result is not None:
                    vis_frame = zone_engine.draw_zones(vis_frame, zone_result)
                if counting_engine is not None and counting_result is not None:
                    vis_frame = counting_engine.draw_occupancy(vis_frame, counting_result)
                if entry_exit_engine is not None and entry_exit_result is not None:
                    vis_frame = entry_exit_engine.draw_entry_exit(vis_frame, entry_exit_result)
                if footfall_engine is not None and footfall_result is not None:
                    vis_frame = footfall_engine.draw_footfall(vis_frame, footfall_result)
                video_writer.write(vis_frame)

            # Log movement events immediately when they occur
            if entry_exit_result:
                for ev in entry_exit_result.events:
                    if ev.movement_type == "ENTRY":
                        logger.info("MOVEMENT: ENTRY - Track ID %d entered '%s' at frame %d", ev.track_id, ev.zone_id, frame.frame_number)
                    elif ev.movement_type == "EXIT":
                        logger.info("MOVEMENT: EXIT - Track ID %d left '%s' at frame %d", ev.track_id, ev.zone_id, frame.frame_number)
                    elif ev.movement_type == "ZONE_TRANSITION":
                        logger.info(
                            "MOVEMENT: ZONE_TRANSITION - Track ID %d: '%s' -> '%s' at frame %d",
                            ev.track_id,
                            ev.previous_zone_id,
                            ev.current_zone_id,
                            frame.frame_number,
                        )

            # Periodic log update
            if frames_processed == 1 or frames_processed % 30 == 0 or frames_processed == reader.total_frames:
                active_ids_states = [f"{t.track_id}:{t.state}" for t in state_result.tracks]
                occ_summary = ""
                if counting_result:
                    occ_parts = [f"{zid}={occ.count}{occ.track_ids if occ.count > 0 else ''}" for zid, occ in counting_result.occupancy.items()]
                    occ_summary = " | Occupancy: " + ", ".join(occ_parts)

                movement_summary = ""
                if entry_exit_result:
                    movement_summary = f" | Total Movements: in={entry_exit_result.total_entries}, out={entry_exit_result.total_exits}, trans={entry_exit_result.total_zone_transitions}"

                footfall_log = ""
                if footfall_result:
                    footfall_log = f" | Footfall: in={footfall_result.summary.total_entries}, out={footfall_result.summary.total_exits}, net={footfall_result.summary.net_flow:+d}"

                logger.info(
                    "Frame %03d/%03d | Detections: %d | Tracks: %s%s%s%s",
                    frame.frame_number,
                    reader.total_frames,
                    det_result.count,
                    active_ids_states if active_ids_states else "None",
                    occ_summary,
                    movement_summary,
                    footfall_log,
                )

    finally:
        reader.release()
        if video_writer is not None:
            video_writer.release()
            logger.info("Visualization video saved to %s", output_video_path)

    total_pipeline_time = time.perf_counter() - start_total_time
    avg_pipeline_fps = (
        (frames_processed / total_pipeline_time) if total_pipeline_time > 0 else 0.0
    )

    print("-" * 65)
    print("                     Execution Summary")
    print("-" * 65)
    print(f"Model                   : {model_path.name}")
    print(f"Detector Device         : {detector.device_display_name}")
    print(f"Tracker                 : ByteTrack")
    print(f"Track State Manager     : Active")
    print(f"Zone Engine             : {'Active' if zone_engine else 'Disabled'}")
    print(f"Zone Counting Engine    : {'Active' if counting_engine else 'Disabled'}")
    print(f"Entry / Exit Engine     : {'Active' if entry_exit_engine else 'Disabled'}")
    print(f"Frames Processed        : {frames_processed}")
    print(f"Frames With Tracks      : {tracker.frames_with_tracks}")
    print(f"Unique Track IDs Seen   : {len(tracker.unique_track_ids)} (IDs: {sorted(list(tracker.unique_track_ids))})")
    print("-" * 65)
    print("                  Track State Manager Metrics")
    print("-" * 65)
    print(f"Total Tracks Created    : {state_manager.total_tracks_created}")
    print(f"Currently Active Tracks : {state_manager.currently_active_tracks}")
    print(f"Currently Lost Tracks   : {state_manager.currently_lost_tracks}")
    print(f"Currently Removed Tracks: {state_manager.currently_removed_tracks}")
    print(f"Maximum Concurrent Tracks: {state_manager.max_concurrent_tracks}")
    print(f"Average Track Age       : {state_manager.average_track_age:.2f} frames")
    print(f"Average Active Frames   : {state_manager.average_active_frames:.2f} frames")
    print(f"Total Track Lifetime    : {state_manager.total_track_lifetime} frame-intervals")
    if counting_engine and counting_result:
        print("-" * 65)
        print("               Zone Counting / Occupancy Metrics")
        print("-" * 65)
        final_occ = {zid: f"{occ.count} (IDs: {occ.track_ids})" for zid, occ in counting_result.occupancy.items()}
        print(f"Final Frame Occupancy   : {final_occ}")
    if entry_exit_engine:
        print("-" * 65)
        print("               Entry / Exit Analytics Metrics")
        print("-" * 65)
        print(f"Frames Evaluated        : {entry_exit_engine.frames_evaluated}")
        print(f"Total Entries           : {entry_exit_engine.total_entries}")
        print(f"Total Exits             : {entry_exit_engine.total_exits}")
        print(f"Total Zone Transitions  : {entry_exit_engine.total_zone_transitions}")
    if footfall_engine:
        ff_summary = footfall_engine.get_summary()
        print("-" * 65)
        print("               Footfall Analytics Metrics")
        print("-" * 65)
        print(f"Total Entries           : {ff_summary.total_entries}")
        print(f"Total Exits             : {ff_summary.total_exits}")
        print(f"Total Movement Events   : {ff_summary.total_movement_events}")
        print(f"Net Flow                : {ff_summary.net_flow:+d}")
        print(f"Duplicate Events Rejected: {footfall_engine.duplicate_events_rejected}")
        print(f"Time Buckets Generated  : {len(ff_summary.buckets)}")
        for idx, b in enumerate(ff_summary.buckets):
            print(f"  Bucket {idx+1} [{b.bucket_start:.1f}s - {b.bucket_end:.1f}s]: in={b.entries}, out={b.exits}, net={b.net_flow:+d}, rate_in={b.entry_rate:.2f}/min, rate_out={b.exit_rate:.2f}/min")
        if ff_summary.entrance_breakdown:
            print(f"Entrance Breakdown      : {ff_summary.entrance_breakdown}")
    print("-" * 65)
    print(f"Average Pipeline FPS    : {avg_pipeline_fps:.2f} FPS")
    print(f"Total Runtime           : {total_pipeline_time:.2f} seconds")
    print(f"Status                  : Completed Gracefully")
    print("=" * 65)

    if sample_json:
        print("\nSample Structured Output (Frame 30):")
        print(json.dumps(sample_json, indent=2))


def main() -> None:
    """Main application entry point without automatic test video execution."""
    project_root = Path(__file__).resolve().parent
    if len(sys.argv) > 1 and sys.argv[1] in ("--run", "-r"):
        run_pipeline(project_root)
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        run_pipeline(project_root, video_source=sys.argv[1])
    else:
        logger.info(
            "AI CCTV Platform initialized. Automatic test video execution disabled. "
            "Call run_pipeline(project_root) or pass a video path to process."
        )


if __name__ == "__main__":
    main()
