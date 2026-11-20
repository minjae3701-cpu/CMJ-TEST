"""JUMP.LAB MMPose backend for CMJ analysis.

POST /api/analyze-pose accepts WebM/MP4 videos and optional duration_sec.
The model must output 133 COCO-WholeBody keypoints (including feet).
This server estimates pose, not actual force-plate ground reaction forces.
"""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="JUMP.LAB MMPose Analysis API", version="1.0.0")
origins = [
    x.strip() for x in os.getenv(
        "JUMPLAB_ALLOWED_ORIGINS", "https://ai-jump-lab-cmj-sj.vercel.app"
    ).split(",") if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "35"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "1500"))
MAX_MODEL_WIDTH = int(os.getenv("MAX_MODEL_WIDTH", "640"))


@lru_cache(maxsize=1)
def get_model():
    config = os.getenv("MMPOSE_CONFIG", "").strip()
    checkpoint = os.getenv("MMPOSE_CHECKPOINT", "").strip()
    if not config or not checkpoint:
        raise RuntimeError("Set MMPOSE_CONFIG and MMPOSE_CHECKPOINT for a COCO-WholeBody MMPose model")
    if not Path(config).is_file() or not Path(checkpoint).is_file():
        raise RuntimeError("MMPose config/checkpoint file is missing")
    try:
        from mmpose.apis import init_model
    except ImportError as exc:
        raise RuntimeError("Missing MMPose dependency; install PyTorch, MMCV, MMEngine and MMPose") from exc
    return init_model(config, checkpoint, device=os.getenv("MMPOSE_DEVICE", "cpu"))


def decode_metadata(path: str):
    """First pass: collect frame timestamps without retaining all decoded images."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("OpenCV could not open the recorded video. Verify FFmpeg WebM/MP4 support.")
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    raw_positions_ms: list[float] = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            if len(raw_positions_ms) >= MAX_FRAMES:
                raise ValueError(f"Recording exceeds the maximum of {MAX_FRAMES} frames")
            raw_positions_ms.append(float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0))
    finally:
        cap.release()
    if not raw_positions_ms:
        raise ValueError("No decodable frames in uploaded video")
    if width <= 0 or height <= 0:
        raise ValueError("Cannot determine video width and height")
    return reported_fps, raw_positions_ms, width, height


def resolve_timing(
    frame_count: int,
    reported_fps: float,
    raw_positions_ms: list[float],
    duration_sec: float | None,
):
    """Correct MediaRecorder FPS=0/1000 and unreliable frame timestamps."""
    if frame_count <= 0:
        raise ValueError("No frames supplied")
    if duration_sec is not None and not (0 < duration_sec <= MAX_DURATION_SEC + 2):
        raise ValueError("duration_sec must be positive and within allowed recording length")
    reported_valid = np.isfinite(reported_fps) and 0 < reported_fps <= 120
    measured = frame_count / duration_sec if duration_sec else None
    if not reported_valid:
        if measured is None:
            raise ValueError("Invalid OpenCV FPS; duration_sec is needed to calculate timing")
        fps = float(measured)
        method = "frame_count/duration_sec"
    elif measured is not None and abs(reported_fps - measured) / max(1.0, measured) > .3:
        fps = float(measured)
        method = "frame_count/duration_sec (metadata mismatch)"
    else:
        fps = float(reported_fps)
        method = "opencv"
    if not (1 <= fps <= 240):
        raise ValueError(f"Recalculated FPS={fps:.2f} is outside plausible range")
    pos = [x / 1000 for x in raw_positions_ms]
    good_pts = len(pos) == frame_count and all(np.isfinite(x) for x in pos)
    if good_pts and frame_count >= 3:
        dt = (pos[-1] - pos[0]) / (frame_count - 1)
        good_pts = (
            all(pos[i] > pos[i - 1] for i in range(1, frame_count))
            and .5 / fps <= dt <= 2 / fps
        )
    if good_pts:
        offset = pos[0]
        stamps = [round(x - offset, 6) for x in pos]
        method += "+decoded_pts"
    else:
        stamps = [round(i / fps, 6) for i in range(frame_count)]
        method += "+recomputed_timestamps"
    return fps, stamps, method


@app.get("/api/health")
def health():
    config = os.getenv("MMPOSE_CONFIG", "").strip()
    checkpoint = os.getenv("MMPOSE_CHECKPOINT", "").strip()
    return {
        "ok": True,
        "model_configured": bool(config and checkpoint),
        "model_files_present": bool(config and checkpoint and Path(config).is_file() and Path(checkpoint).is_file()),
    }


@app.post("/api/analyze-pose")
async def analyze_pose(video: UploadFile = File(...), duration_sec: float | None = Form(default=None)):
    mime = (video.content_type or "").split(";", 1)[0].strip().lower()
    if mime not in {"video/webm", "video/mp4", "video/quicktime", "application/octet-stream"}:
        raise HTTPException(415, "Unsupported MIME type. Use WebM or MP4.")
    ext = ".mp4" if mime == "video/mp4" else ".mov" if mime == "video/quicktime" else ".webm"
    payload = await video.read(MAX_UPLOAD_BYTES + 1)
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Video is empty or exceeds server upload limit")
    filename = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as out:
            out.write(payload)
            filename = out.name
        reported_fps, raw_ms, width, height = decode_metadata(filename)
        fps, timestamps, source = resolve_timing(len(raw_ms), reported_fps, raw_ms, duration_sec)
        try:
            model = get_model()
            from mmpose.apis import inference_topdown
        except Exception as exc:
            raise HTTPException(503, f"MMPose model unavailable: {exc}") from exc
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            raise HTTPException(422, "Unable to reopen uploaded video")
        frames = []
        try:
            for timestamp in timestamps:
                ok, image = cap.read()
                if not ok:
                    raise HTTPException(422, "Video failed during inference decoding")
                # Limit decode/inference RAM while mapping predictions to original pixels.
                scale = min(1.0, MAX_MODEL_WIDTH / image.shape[1])
                scaled = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale))) if scale < 1 else image
                preds = inference_topdown(model, scaled)
                if not preds or not hasattr(preds[0], "pred_instances"):
                    frames.append({"timestamp": timestamp, "keypoints": [], "scores": []})
                    continue
                inst = preds[0].pred_instances
                xy = np.asarray(inst.keypoints[0], dtype=float)
                scores = np.asarray(inst.keypoint_scores[0], dtype=float)
                if xy.ndim != 2 or xy.shape[1] < 2 or not np.all(np.isfinite(xy)):
                    raise HTTPException(422, "Invalid joint coordinates")
                xy = xy[:, :2] / scale
                frames.append({
                    "timestamp": timestamp,
                    "keypoints": xy.round(3).tolist(),
                    "scores": scores.round(4).tolist(),
                })
        finally:
            cap.release()
        if max((len(frame["keypoints"]) for frame in frames), default=0) < 133:
            raise HTTPException(422, "COCO-WholeBody 133-keypoint model is required for toe/heel detection")
        return {
            "ok": True, "engine": "MMPose", "format": "coco_wholebody",
            "video_width": width, "video_height": height,
            "fps": round(fps, 5), "fps_source": source,
            "reported_fps": reported_fps, "frame_count": len(frames),
            "duration_sec": float(duration_sec) if duration_sec else len(frames) / fps,
            "frames": frames,
        }
    except HTTPException:
        raise
    except (ValueError, cv2.error) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if filename:
            Path(filename).unlink(missing_ok=True)
