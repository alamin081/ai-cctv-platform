"""
AI CCTV Platform - FastAPI Web Application
Provides RESTful APIs for video ingestion, AI pipeline processing, and live analytics streaming.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.pipeline_service import PipelineService
from src.api.playback_service import PlaybackService
from src.zones import ZoneConfigStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api.app")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
pipeline_service = PipelineService(PROJECT_ROOT)
playback_service = PlaybackService(PROJECT_ROOT)
zone_store = ZoneConfigStore(PROJECT_ROOT)

app = FastAPI(
    title="AI CCTV Video Analytics Platform",
    description="Professional CCTV Monitoring & AI Video Analytics Platform",
    version="0.9.0",
)

# Enable CORS for local development and integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure web directories exist
WEB_DIR = PROJECT_ROOT / "src" / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
(WEB_DIR / "css").mkdir(parents=True, exist_ok=True)
(WEB_DIR / "js").mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# Pydantic schemas
class JobCreateRequest(BaseModel):
    video_filename: str


class PlaybackSessionRequest(BaseModel):
    video_filename: str


class PlaybackPlayRequest(BaseModel):
    timestamp: float = 0.0
    speed: float = 1.0


class PlaybackPauseRequest(BaseModel):
    timestamp: float = 0.0


class PlaybackSeekRequest(BaseModel):
    timestamp: float = 0.0
    frame_number: Optional[int] = None


class PlaybackSyncRequest(BaseModel):
    current_time: float = 0.0
    is_playing: bool = False
    speed: float = 1.0


class PlaybackProcessFrameRequest(BaseModel):
    timestamp: float
    frame_number: Optional[int] = None


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """Return system and AI engine health status."""
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    return {
        "status": "healthy",
        "version": "0.9.0",
        "platform": "ai-cctv-platform",
        "device": device_name,
        "cuda_available": cuda_avail,
        "engines": {
            "yolo_detector": "active (YOLOv8n)",
            "tracker": "active (ByteTrack)",
            "state_manager": "active (TrackStateManager)",
            "zone_engine": "active (ZoneEngine)",
            "zone_counting": "active (ZoneCountingEngine)",
            "entry_exit_engine": "active (EntryExitEngine)",
            "footfall_engine": "active (FootfallAnalyticsEngine)",
        },
    }


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    """Return active platform configuration with zones from persistent store."""
    cfg = pipeline_service.get_config()
    cfg["zones"] = zone_store.load()
    return cfg


@app.get("/api/videos")
def list_videos():
    """List all available raw input videos."""
    return {"videos": pipeline_service.list_available_videos()}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a new CCTV video file with robust sanitization and metadata extraction."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    raw_name = Path(file.filename).name
    stem = Path(raw_name).stem
    ext = Path(raw_name).suffix.lower()

    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{ext}'. Supported: {', '.join(valid_exts)}",
        )

    # Sanitize filename: remove special characters (e.g. #, @, %, &, +, spaces) to ensure URL and filesystem safety
    clean_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)
    clean_stem = re.sub(r"_+", "_", clean_stem).strip("_")
    if not clean_stem:
        clean_stem = f"cctv_upload_{int(time.time())}"

    safe_filename = f"{clean_stem}{ext}"
    target_path = pipeline_service.videos_dir / safe_filename

    # If file with same name exists, make uniquely identifiable
    if target_path.exists():
        safe_filename = f"{clean_stem}_{int(time.time())}{ext}"
        target_path = pipeline_service.videos_dir / safe_filename

    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("Failed to save uploaded video: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Optimize MP4 container with moov atom at beginning (faststart) for instant browser playback
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        faststart_tmp = target_path.with_name(f"{target_path.stem}_faststart{target_path.suffix}")
        cmd = [
            ffmpeg_exe, "-y", "-i", str(target_path),
            "-c:v", "copy", "-movflags", "+faststart",
            str(faststart_tmp)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        if res.returncode == 0 and faststart_tmp.exists() and faststart_tmp.stat().st_size > 0:
            faststart_tmp.replace(target_path)
    except Exception as faststart_err:
        logger.debug("Faststart stream copy skipped: %s", faststart_err)

    # Probe metadata using OpenCV
    w, h, fps, total_frames, dur = 0, 0, 0.0, 0, 0.0
    try:
        cap = cv2.VideoCapture(str(target_path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur = round(total_frames / fps, 2) if fps > 0 else 0.0
            cap.release()
    except Exception:
        pass

    video_url = f"/media/videos/{urllib.parse.quote(safe_filename)}"

    return {
        "status": "success",
        "filename": safe_filename,
        "video_url": video_url,
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": dur,
        "size_mb": round(target_path.stat().st_size / (1024 * 1024), 2),
    }


# ==============================================================================
# Real-Time Playback Analysis Endpoints
# ==============================================================================

@app.post("/api/playback/session")
def create_playback_session(req: PlaybackSessionRequest):
    """Initialize or retrieve a playback analysis session for a video."""
    try:
        session = playback_service.create_or_get_session(
            req.video_filename, zone_store=zone_store,
        )

        # Send normalized zone data from persistent store
        zones_data = zone_store.load().get("definitions", [])

        init_frame = session.frames_cache.get(session.current_frame_number)
        return {
            "status": "success",
            "session_id": session.session_id,
            "video_filename": session.video_filename,
            "video_url": f"/media/videos/{urllib.parse.quote(session.video_filename)}",
            "width": session.width,
            "height": session.height,
            "fps": round(session.fps, 2),
            "total_frames": session.total_frames,
            "duration_seconds": round(session.duration_seconds, 2),
            "zones": zones_data,
            "device": session.detector_manager.device_display_name,
            "initial_frame": init_frame,
            "metrics": session.get_current_metrics(),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to create playback session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/playback/{session_id}/play")
def play_playback_session(session_id: str, req: PlaybackPlayRequest):
    """Start or resume playback analysis at given timestamp and playback speed."""
    session = playback_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Playback session '{session_id}' not found")
    session.set_speed(req.speed)
    session.set_playing(True, req.timestamp)
    return {
        "status": "playing",
        "session_id": session_id,
        "playback_speed": session.playback_speed,
        "stride": session.stride,
    }


@app.post("/api/playback/{session_id}/pause")
def pause_playback_session(session_id: str, req: PlaybackPauseRequest):
    """Pause real-time frame analysis and freeze analytics state."""
    session = playback_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Playback session '{session_id}' not found")
    session.set_playing(False, req.timestamp)
    return {
        "status": "paused",
        "session_id": session_id,
        "current_timestamp": session.current_timestamp,
    }


@app.post("/api/playback/{session_id}/seek")
def seek_playback_session(session_id: str, req: PlaybackSeekRequest):
    """
    Seek video reader position, reset tracking/zone state to prevent carry-over,
    and immediately analyze and return the seeked frame.
    """
    session = playback_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Playback session '{session_id}' not found")
    frame_data = session.seek(req.timestamp, req.frame_number)
    return {
        "status": "seeked",
        "session_id": session_id,
        "frame_number": session.current_frame_number,
        "timestamp": session.current_timestamp,
        "frame_data": frame_data,
        "metrics": frame_data.get("metrics", session.get_current_metrics()),
    }


@app.post("/api/playback/{session_id}/process_frame")
def process_playback_frame(session_id: str, req: PlaybackProcessFrameRequest):
    """
    Directly analyze a presented video frame during playback.
    Maintains ByteTrack tracking continuity and associates source frame and timestamp.
    """
    session = playback_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Playback session '{session_id}' not found")
    data = session.process_frame_at(req.timestamp, req.frame_number)
    if data is None:
        raise HTTPException(status_code=404, detail="Frame could not be read or is beyond stream end")
    return {
        "status": "success",
        "session_id": session_id,
        "frame_number": data["frame_number"],
        "timestamp": data["timestamp"],
        "frame_data": data,
        "metrics": data.get("metrics", session.get_current_metrics()),
    }


@app.get("/api/playback/{session_id}/stream")
async def stream_playback_frames(session_id: str):
    """SSE stream of real-time analyzed frames synchronized with video playback."""
    session = playback_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Playback session '{session_id}' not found")
    return StreamingResponse(
        playback_service.sse_frame_generator(session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/playback/{session_id}/sync")
def sync_playback_session(session_id: str, req: PlaybackSyncRequest):
    """Client sync ping keeping server aligned with client playback time and speed."""
    session = playback_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Playback session '{session_id}' not found")
    session.set_speed(req.speed)
    session.set_playing(req.is_playing, req.current_time)
    frame_idx = max(1, int(round(req.current_time * session.fps)) + 1)
    cached_frame = session.frames_cache.get(frame_idx)
    return {
        "status": "synced",
        "session_id": session_id,
        "current_frame": session.current_frame_number,
        "frame_data": cached_frame,
        "metrics": session.get_current_metrics(),
    }


@app.delete("/api/playback/{session_id}")
def close_playback_session(session_id: str):
    """Close playback session and release resources."""
    closed = playback_service.close_session(session_id)
    if not closed:
        raise HTTPException(status_code=404, detail=f"Playback session '{session_id}' not found")
    return {"status": "closed", "session_id": session_id}


@app.post("/api/jobs")
def create_job(request: JobCreateRequest):
    """Start an AI video processing job."""
    try:
        job = pipeline_service.create_job(request.video_filename)
        return {"status": "created", "job": job.to_dict()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to create job: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Zone Configuration CRUD API
# ==============================================================================


@app.get("/api/zones")
def get_zones() -> Dict[str, Any]:
    """Retrieve current zone configuration with normalized coordinates and version."""
    return zone_store.load()


@app.put("/api/zones")
def update_zones(zones_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace zone configuration. Accepts a full zone config object with definitions.
    Validates all zones, persists atomically to data/configs/zones.json,
    and hot-reloads all active playback sessions.
    """
    try:
        saved = zone_store.save(zones_data)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={"message": "Zone validation failed", "errors": str(e).split("\n")},
        )
    except Exception as e:
        logger.exception("Failed to save zones: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Hot-reload all active playback sessions with new zones
    reload_count = playback_service.reload_all_sessions_zones(zone_store)
    return {
        "status": "success",
        "message": f"Saved {len(saved.get('definitions', []))} zone(s), reloaded {reload_count} session(s)",
        "zones": saved,
    }


@app.post("/api/zones/reset")
def reset_zones() -> Dict[str, Any]:
    """Reset zone configuration to config.yaml defaults."""
    try:
        defaults = zone_store.reset()
    except Exception as e:
        logger.exception("Failed to reset zones: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    reload_count = playback_service.reload_all_sessions_zones(zone_store)
    return {
        "status": "success",
        "message": f"Reset to defaults ({len(defaults.get('definitions', []))} zones), reloaded {reload_count} session(s)",
        "zones": defaults,
    }


@app.get("/api/jobs")
def list_jobs():
    """List all recent jobs."""
    return {"jobs": pipeline_service.get_all_jobs()}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Retrieve status, live metrics, and results for a job."""
    job = pipeline_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/frames")
def get_job_frames(job_id: str):
    """Retrieve frame-level tracking and detection data for player synchronization."""
    job = pipeline_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {
        "job_id": job.job_id,
        "total_frames": job.total_frames,
        "frames": job.frames_data,
    }


@app.get("/media/videos/{filename:path}")
def stream_input_video(filename: str):
    """Stream raw input video file with range-request and URL encoding support."""
    decoded_name = urllib.parse.unquote(filename).strip()
    safe_name = Path(decoded_name).name  # Prevent directory traversal
    path = pipeline_service.videos_dir / safe_name

    # If exact name doesn't exist, search for matching file
    if not path.exists():
        matched = None
        for f in pipeline_service.videos_dir.iterdir():
            if f.is_file() and (f.name.lower() == safe_name.lower() or f.stem.lower() == Path(safe_name).stem.lower()):
                matched = f
                break
        if matched:
            path = matched
        else:
            raise HTTPException(status_code=404, detail=f"Video file '{safe_name}' not found")

    ext = path.suffix.lower()
    media_type = "video/mp4"
    if ext in (".webm", ".mkv"):
        media_type = "video/webm"
    elif ext == ".ogg":
        media_type = "video/ogg"

    return FileResponse(
        path=path,
        media_type=media_type,
        filename=path.name,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/media/outputs/{filename:path}")
def stream_output_video(filename: str):
    """Stream generated annotated output video file."""
    decoded_name = urllib.parse.unquote(filename).strip()
    safe_name = Path(decoded_name).name
    path = pipeline_service.outputs_dir / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found")
    return FileResponse(
        path=path,
        media_type="video/mp4",
        filename=path.name,
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serve the side-by-side CCTV Monitoring & Analytics Dashboard."""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>AI CCTV Dashboard Loading...</h1>")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


def start_server(host: str = "127.0.0.1", port: int = 8000):
    """Start uvicorn server."""
    import uvicorn
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start_server()
