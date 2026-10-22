"""
YOLO Person Detector implementation using Ultralytics.
Phase: Phase 3 - YOLO Person Detection
"""

import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO

from .models import (
    Detection,
    DetectionResult,
    InferenceError,
    InvalidFrameError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)

# COCO dataset class index for 'person'
COCO_PERSON_CLASS_ID = 0


class YOLOPersonDetector:
    """
    Lightweight person detector based on Ultralytics YOLO.
    Configured strictly for person detection (COCO class 0).
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        device: str = "auto",
        target_classes: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the YOLO Person Detector.

        Args:
            model_name_or_path: Model checkpoint name (e.g. 'yolov8n.pt') or path to .pt file.
            confidence_threshold: Minimum detection confidence score (0.0 to 1.0).
            iou_threshold: IoU threshold for Non-Maximum Suppression (NMS).
            device: 'auto', 'cuda', or 'cpu'. If 'auto', uses CUDA if available.
            target_classes: Class IDs to detect (defaults to [0] for person).
        """
        self.model_name_or_path = str(model_name_or_path)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.requested_device = device.lower().strip()
        self.target_classes = target_classes if target_classes is not None else [COCO_PERSON_CLASS_ID]

        # Determine and report device
        self.device = self._resolve_device(self.requested_device)
        logger.info("Initializing YOLOPersonDetector with device: %s", self.device_display_name)

        # Load YOLO model
        self.model = self._load_model()

    @property
    def device_display_name(self) -> str:
        """User-friendly device display name (e.g. 'CUDA (NVIDIA GeForce RTX 4050)' or 'CPU')."""
        if "cuda" in self.device:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"
            return f"CUDA ({gpu_name})"
        return "CPU"

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve requested device string to a concrete PyTorch device."""
        cuda_available = torch.cuda.is_available()

        if requested_device == "auto":
            if cuda_available:
                logger.info("CUDA detected. Selecting GPU for YOLO inference.")
                return "cuda:0"
            else:
                logger.info("CUDA not available. Falling back to CPU for YOLO inference.")
                return "cpu"
        elif "cuda" in requested_device:
            if cuda_available:
                return requested_device
            else:
                logger.warning(
                    "CUDA requested ('%s') but not available on this system. Falling back to CPU.",
                    requested_device,
                )
                return "cpu"
        else:
            return "cpu"

    def _load_model(self) -> YOLO:
        """Load pretrained YOLO weights."""
        try:
            logger.info("Loading YOLO model from: %s", self.model_name_or_path)
            model = YOLO(self.model_name_or_path)
            try:
                model.to(self.device)
            except Exception as dev_err:
                if "cuda" in str(dev_err).lower() and self.device != "cpu":
                    logger.warning("CUDA model transfer failed (%s). Falling back to CPU.", dev_err)
                    self.device = "cpu"
                    model.to("cpu")
                else:
                    raise
            logger.info("Successfully loaded model: %s on %s", self.model_name_or_path, self.device)
            return model
        except Exception as e:
            error_msg = f"Failed to load YOLO model '{self.model_name_or_path}': {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def detect(self, frame: Any) -> DetectionResult:
        """
        Perform person detection on a single frame.

        Args:
            frame: Either a numpy ndarray (BGR image matrix) or a Frame object from video_input.

        Returns:
            DetectionResult: Structured detection outputs with bounding boxes, confidences, and timings.

        Raises:
            InvalidFrameError: If frame data is None, empty, or not a valid image array.
            InferenceError: If model prediction encounters an unrecoverable failure.
        """
        # Extract image data and optional metadata
        if isinstance(frame, np.ndarray):
            image = frame
            frame_number = None
            timestamp = None
        elif hasattr(frame, "data"):
            image = frame.data
            frame_number = getattr(frame, "frame_number", None)
            timestamp = getattr(frame, "timestamp", None)
        else:
            image = frame
            frame_number = getattr(frame, "frame_number", None)
            timestamp = getattr(frame, "timestamp", None)

        # Validate frame array
        if image is None:
            raise InvalidFrameError("Supplied frame is None.")

        if not isinstance(image, np.ndarray):
            raise InvalidFrameError(f"Expected numpy.ndarray for image, got {type(image).__name__}")

        if image.size == 0 or len(image.shape) < 2:
            raise InvalidFrameError(f"Supplied frame has invalid shape: {image.shape}")

        try:
            start_time = time.perf_counter()

            # Execute YOLO inference (strictly person class, verbose=False to keep logs clean)
            try:
                results = self.model.predict(
                    source=image,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    classes=self.target_classes,
                    device=self.device,
                    verbose=False,
                )
            except Exception as cuda_err:
                if "cuda" in str(cuda_err).lower() or "cublas" in str(cuda_err).lower():
                    logger.warning("YOLO CUDA execution error encountered (%s). Attempting memory cleanup and CPU fallback.", cuda_err)
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    try:
                        # Try CPU fallback by reloading cleanly on CPU
                        self.device = "cpu"
                        try:
                            self.model.to("cpu")
                        except Exception:
                            self.model = YOLO(self.model_name_or_path)
                            self.model.to("cpu")
                        results = self.model.predict(
                            source=image,
                            conf=self.confidence_threshold,
                            iou=self.iou_threshold,
                            classes=self.target_classes,
                            device="cpu",
                            verbose=False,
                        )
                    except Exception as cpu_err:
                        raise InferenceError(f"YOLO inference failed on CPU fallback: {cpu_err}") from cpu_err
                else:
                    raise

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        except Exception as e:
            error_msg = f"YOLO inference failed: {e}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e

        # Extract structured detections
        detections: List[Detection] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())

                # Filter strictly to target classes (person: 0)
                if cls_id not in self.target_classes:
                    continue

                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                class_name = self.model.names.get(cls_id, "person")

                detection = Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=[round(float(c), 2) for c in xyxy],
                )
                detections.append(detection)

        return DetectionResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            inference_time_ms=elapsed_ms,
        )
