"""端到端调试脚本 — 逐帧分析 + 中间图片输出 + 性能计时 + 独立输出目录

用法:
    python debug_pipeline.py

输出目录: debug_output/<run_id>/
    run_metadata.json
    effective_config.yaml
    events.jsonl
    keyframe_stats.csv
    association_events.csv
    object_lifecycle.json
    reports/report.json
    reports/report.csv
    log.txt
    frames/          — 每帧的检测可视化
    keyframes/       — 关键帧的特写
    matches/         — 匹配可视化

按 Ctrl+C 可以提前终止（已处理的帧仍有输出）。
"""
import sys
import json
import hashlib
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── 确保项目根在 sys.path 中 ──
_proj_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_proj_root))

from core.types import (
    Frame, KeyframeDecision, KeyframeTriggerContext,
    ConfirmationStatus, ReviewFlag, RawDetection,
)
from core.video_reader import VideoReader
from core.quality_evaluator import InitialKeyframeFallback, QualityEvaluator
from core.frame_buffer import FrameBuffer
from core.feature_matcher import FeatureMatcher
from core.homography_graph import HomographyGraph
from core.keyframe_selector import KeyframeSelector
from core.detector import Detector
from core.detection_fusion import DetectionFusion
from core.simple_tracker import SimpleDetectionTracker
from core.global_projector import GlobalProjector
from core.object_associator import ObjectAssociator
from core.coverage_map import CoverageMap
from core.report_generator import ReportGenerator, get_reportable_objects
from core.recovery_manager import RecoveryManager, RecoveryState
from core.config_loader import ConfigLoader
from core.debug_events import DebugEvent, DebugStats, DebugEventWriter, PerfTimer, TimedMatcher

# ── 配置 ──
# VIDEO_PATH = r"D:\杭州供电段\头戴设备作业工具识别\01公司拍摄数据20260717\测试用\test_cut.mp4"
VIDEO_PATH = r"D:\杭州供电段\头戴设备作业工具识别\260814拍摄测试\test.mp4"
CONFIG_DIR = _proj_root / "configs"
MAX_FRAMES = None  # None=全部, 设为数字可限制帧数

# ── 颜色映射（每类一种颜色）──
CLASS_COLORS = {}
COLOR_PALETTE = [
    (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (0, 128, 255), (255, 128, 0), (128, 255, 0), (255, 0, 128),
    (0, 200, 100), (200, 0, 200), (200, 200, 0), (0, 100, 255),
    (128, 0, 255), (255, 0, 0), (0, 0, 255),
]


def get_class_color(class_name: str) -> tuple:
    if class_name not in CLASS_COLORS:
        CLASS_COLORS[class_name] = COLOR_PALETTE[len(CLASS_COLORS) % len(COLOR_PALETTE)]
    return CLASS_COLORS[class_name]


# ── 获取 git SHA ──
def get_git_sha():
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(_proj_root)
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def file_hash(path: str) -> str:
    if not Path(path).exists():
        return "not_found"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── 初始化输出目录 ──
GIT_SHA = get_git_sha()
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + GIT_SHA
OUTPUT_DIR = _proj_root / "debug_output" / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "frames").mkdir(exist_ok=True)
(OUTPUT_DIR / "keyframes").mkdir(exist_ok=True)
(OUTPUT_DIR / "matches").mkdir(exist_ok=True)
(OUTPUT_DIR / "reports").mkdir(exist_ok=True)

LOG_FILE = OUTPUT_DIR / "log.txt"
LOG_FILE.write_text("", encoding="utf-8")


PERF_COLLECTION_ACTIVE = False


def log(msg: str):
    """同时输出到控制台和日志文件。"""
    t0 = time.perf_counter()
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    stats = globals().get("dbg")
    if PERF_COLLECTION_ACTIVE and stats is not None:
        stats.add_timing("event_log_io_ms", (time.perf_counter() - t0) * 1000.0)


# ── 事件写入器 ──
event_writer = DebugEventWriter(OUTPUT_DIR / "events.jsonl")


def emit_event(event_type: str, frame_id: int | None, **payload):
    t0 = time.perf_counter()
    event_writer.emit(DebugEvent(event_type=event_type, frame_id=frame_id, payload=payload))
    stats = globals().get("dbg")
    if PERF_COLLECTION_ACTIVE and stats is not None:
        stats.add_timing("event_log_io_ms", (time.perf_counter() - t0) * 1000.0)


# ── 运行元数据 ──
metadata = {
    "run_id": RUN_ID,
    "timestamp": datetime.now().isoformat(),
    "git_sha": GIT_SHA,
    "git_full_sha": get_git_sha(),
    "video_path": VIDEO_PATH,
    "video_hash": file_hash(VIDEO_PATH),
    "model_hash": file_hash(str(_proj_root / "models" / "best.pt")),
}
with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)


def draw_detections(image: np.ndarray, detections: list, level: str,
                    color: tuple = (0, 255, 0)) -> np.ndarray:
    """在图像上绘制检测框。"""
    img = image.copy()
    for det in detections:
        if hasattr(det, 'candidate'):
            det = det.candidate
        x1, y1, x2, y2 = map(int, det.bbox)
        cls_color = get_class_color(det.class_name)
        cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls_color, 1)
    cv2.putText(img, level, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


def draw_match_info(src_img: np.ndarray, dst_img: np.ndarray,
                    match_result, frame_id: int) -> np.ndarray:
    """并排显示两张图 + 匹配信息。"""
    h1, w1 = src_img.shape[:2]
    h2, w2 = dst_img.shape[:2]
    h = max(h1, h2)
    w = w1 + w2

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = src_img
    if dst_img.ndim == 2:
        dst_display = cv2.cvtColor(dst_img, cv2.COLOR_GRAY2BGR)
    else:
        dst_display = dst_img
    canvas[:h2, w1:w1 + w2] = dst_display

    cv2.line(canvas, (w1, 0), (w1, h), (0, 0, 255), 2)

    info_lines = [
        f"Frame {frame_id}",
        f"Valid: {match_result.valid}",
        f"Good matches: {match_result.num_good_matches}",
        f"Inliers: {match_result.num_inliers}",
        f"Inlier ratio: {match_result.inlier_ratio:.3f}",
        f"Reproj err: {match_result.reprojection_error:.2f} px",
        f"Quads src: {match_result.occupied_quadrants_src}",
        f"Quads dst: {match_result.occupied_quadrants_dst}",
    ]
    if match_result.failure_reason:
        info_lines.append(f"FAIL: {match_result.failure_reason}")

    y0 = 60
    for line in info_lines:
        color_code = (0, 0, 255) if line.startswith("FAIL") else (255, 255, 255)
        cv2.putText(canvas, line, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_code, 1)
        y0 += 22

    cv2.putText(canvas, "CURRENT", (w1 // 2 - 50, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(canvas, "PREV KEYFRAME", (w1 + w2 // 2 - 80, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    return canvas


# ══════════════════════════════════════════════════════════════════
# 加载配置（使用统一 ConfigLoader）
# ══════════════════════════════════════════════════════════════════

cfg_loader = ConfigLoader(CONFIG_DIR)
cfg = cfg_loader.pipeline
mcfg = cfg_loader.matcher
acfg = cfg_loader.associator
tcfg = cfg_loader.tracker
detector_cfg = cfg_loader.detector
l3cfg = detector_cfg.get("l3", {})

# 保存 effective config
effective_config = {
    "pipeline": cfg,
    "matcher": mcfg,
    "association": acfg,
    "tracker": tcfg,
    "git_sha": GIT_SHA,
    "model_hash": metadata["model_hash"],
    "video_hash": metadata["video_hash"],
}
with open(OUTPUT_DIR / "effective_config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(effective_config, f, allow_unicode=True)

log("=" * 80)
log("HEAD TOOL COUNTER — DEBUG PIPELINE V2")
log("=" * 80)
log(f"Video: {VIDEO_PATH}")
log(f"Output: {OUTPUT_DIR}")
log(f"Git SHA: {GIT_SHA}")
log(f"Max frames: {MAX_FRAMES or 'ALL'}")
log("")

log("Config loaded:")
log(f"  matcher: min_good={mcfg['min_good_matches']}, min_inliers={mcfg['min_inliers']}, "
    f"min_ratio={mcfg['min_inlier_ratio']}, max_reproj={mcfg['max_reprojection_error_px']}")
log(f"  associator: max_pos={acfg['max_position_distance_px']}, "
    f"min_obs_confirm={acfg['min_observations_confirmed']}")
log(f"  tracker: max_missed={tcfg['max_missed_detection_frames']}, "
    f"min_iou={tcfg['min_iou']}, inactive_min_iou={tcfg.get('inactive_min_iou', 0.30)}")
log(f"  pipeline: l2_interval={cfg.get('l2_interval_frames', 3)}, "
    f"min_kf_interval={cfg.get('min_keyframe_interval_frames', 5)}")
log("")

l2_interval = cfg.get("l2_interval_frames", 3)
max_interval = cfg.get("max_keyframe_interval_frames", 30)
end_window = cfg.get("end_window_frames", 30)
min_kf_interval = cfg.get("min_keyframe_interval_frames", 5)

# ── 初始化模块 ──
log("Initializing modules...")
t0 = time.perf_counter()

reader = VideoReader(VIDEO_PATH, max_fps=cfg.get("max_input_fps", 30))
quality = QualityEvaluator(
    sharpness_threshold=cfg.get("sharpness_threshold", 20.0),
    detection_sharpness_threshold=cfg.get(
        "detection_sharpness_threshold", 63.0
    ),
    quality_evaluation_scale=cfg.get("quality_evaluation_scale", 0.5),
    dark_pixel_threshold=cfg.get("dark_pixel_threshold", 10),
    bright_pixel_threshold=cfg.get("bright_pixel_threshold", 245),
)
frame_buffer = FrameBuffer(max_size=end_window)
dbg = DebugStats()
matcher = TimedMatcher(FeatureMatcher(
    min_good_matches=mcfg.get("min_good_matches", 20),
    min_inliers=mcfg.get("min_inliers", 30),
    min_inlier_ratio=mcfg.get("min_inlier_ratio", 0.30),
    max_reprojection_error=mcfg.get("max_reprojection_error_px", 3.0),
    min_occupied_quadrants=mcfg.get("min_occupied_quadrants", 2),
    min_inlier_bbox_area_ratio=mcfg.get("min_inlier_bbox_area_ratio", 0.10),
    roi_center_ratio=mcfg.get("roi_center_ratio", 0.70),
    max_projected_area_ratio=mcfg.get("max_projected_area_ratio", 10.0),
    min_projected_area_ratio=mcfg.get("min_projected_area_ratio", 0.10),
    max_condition_number=mcfg.get("max_condition_number", 500000),
), dbg)
graph = HomographyGraph()
selector = KeyframeSelector(
    max_interval=max_interval,
    end_window_frames=end_window,
    matcher=matcher,
    min_keyframe_interval_frames=min_kf_interval,
    emergency_keyframe_interval_frames=cfg.get("emergency_keyframe_interval_frames", 2),
)
detector = Detector(model_path=str(_proj_root / "models" / "best.pt"))
fcfg = detector_cfg.get("fusion", {})
fusion = DetectionFusion(
    iou_threshold=fcfg.get("iou_threshold", 0.65),
    center_merge_distance_px=fcfg.get("center_merge_distance_px", 40.0),
    center_merge_min_ios=fcfg.get("center_merge_min_ios", 0.30),
    per_class_center_merge_distances=fcfg.get(
        "per_class_center_merge_distances", {}
    ),
)
tracker = SimpleDetectionTracker(
    max_missed_detection_frames=tcfg.get("max_missed_detection_frames", 5),
    lost_reactivation_frames=tcfg.get("lost_reactivation_frames", 10),
    min_iou=tcfg.get("min_iou", 0.20),
    max_center_distance_ratio=tcfg.get("max_center_distance_ratio", 0.20),
    iou_weight=tcfg.get("iou_weight", 0.60),
    center_weight=tcfg.get("center_weight", 0.40),
    class_compatibility=acfg.get("class_compatibility", {}),
    inactive_min_iou=tcfg.get("inactive_min_iou", 0.30),
    inactive_max_center_distance_ratio=tcfg.get("inactive_max_center_distance_ratio", 0.12),
    quality_drop_trigger_ratio=tcfg.get("quality_drop_trigger_ratio", 0.70),
    quality_drop_rearm_ratio=tcfg.get("quality_drop_rearm_ratio", 0.85),
    quality_drop_min_history=tcfg.get("quality_drop_min_history", 5),
    new_detection_confirmation_runs=tcfg.get(
        "new_detection_confirmation_runs", 3
    ),
)
projector = GlobalProjector()
associator = ObjectAssociator(
    max_position_distance_px=acfg.get("max_position_distance_px", 120.0),
    position_weight=acfg.get("position_weight", 0.55),
    overlap_weight=acfg.get("overlap_weight", 0.20),
    size_weight=acfg.get("size_weight", 0.10),
    class_weight=acfg.get("class_weight", 0.15),
    max_cost=acfg.get("max_cost", 0.75),
    min_observations_confirmed=acfg.get("min_observations_confirmed", 3),
    min_keyframes_confirmed=acfg.get("min_keyframes_confirmed", 2),
    min_top_class_ratio=acfg.get("min_top_class_ratio", 0.60),
    max_votes_per_track=acfg.get("max_votes_per_track", 3),
    class_compatibility=acfg.get("class_compatibility", {}),
    online_gate_ratio=acfg.get("online_gate_ratio", 0.5),
    per_class_gate_ratios=acfg.get("per_class_gate_ratios", {}),
    per_class_position_gates=acfg.get("per_class_position_gates", {}),
    track_reactivate_max_gap_frames=acfg.get("track_reactivate_max_gap_frames", 15),
    centroid_distance_threshold=acfg.get("centroid_distance_threshold", 30.0),
    debug_mode=True,
)
coverage = CoverageMap(grid_resolution=100)
recovery_mgr = RecoveryManager(matcher=matcher)

log(f"  All modules initialized in {time.perf_counter() - t0:.2f}s")
log("")

# ── 状态变量 ──
last_keyframe = None
last_keyframe_frame_id = -1
last_keyframe_node_id: int | None = None
end_window_deque = deque(maxlen=end_window)
processed_keyframe_frame_ids: set[int] = set()
keyframe_images: dict[int, np.ndarray] = {}
fc = 0
# 质量拒绝连续区间跟踪
quality_reject_start: int | None = None
initial_fallback = InitialKeyframeFallback(
    min_sharpness=cfg.get("detection_sharpness_threshold", 63.0),
    max_interval_frames=max_interval,
)

# ── 性能统计 ──
def build_raw_detections(tracked_dets, kf_id: int, sharpness: float, mapping_quality: float):
    raw = []
    for td in tracked_dets:
        c = td.candidate
        x1, y1, x2, y2 = c.bbox
        raw.append(RawDetection(
            frame_id=c.frame_id,
            keyframe_id=kf_id,
            track_id=td.track_id,
            bbox=c.bbox,
            center=((x1 + x2) / 2, (y1 + y2) / 2),
            corners=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
            class_id=c.class_id,
            class_name=c.class_name,
            confidence=c.confidence,
            sharpness=sharpness,
            mapping_quality=mapping_quality,
            source=c.source,
        ))
    return raw


def build_untracked_raw_detections(candidates, kf_id: int, sharpness: float,
                                   mapping_quality: float):
    raw = []
    for c in candidates:
        x1, y1, x2, y2 = c.bbox
        raw.append(RawDetection(
            frame_id=c.frame_id, keyframe_id=kf_id, track_id=None,
            bbox=c.bbox, center=((x1 + x2) / 2, (y1 + y2) / 2),
            corners=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
            class_id=c.class_id, class_name=c.class_name,
            confidence=c.confidence, sharpness=sharpness,
            mapping_quality=mapping_quality, source=c.source,
        ))
    return raw


def process_accepted_keyframe(frame, keyframe_id, H_kf_to_global, l3_regions, l3_required,
                               parent_node_id=None, update_tracker=True):
    """关键帧的完整处理（L1检测→L3检测→融合→追踪→投影→关联）。"""
    with PerfTimer(dbg, "l1_inference_ms"):
        l1_candidates = detector.detect(image=frame.image, level="L1", frame_id=frame.frame_id)
    l3_candidates = []
    if l3_required and l3_regions:
        with PerfTimer(dbg, "l3_inference_ms"):
            l3_candidates = detector.detect(
                image=frame.image, level="L3", frame_id=frame.frame_id, regions=l3_regions,
            )

    with PerfTimer(dbg, "fusion_ms"):
        fused_candidates = fusion.fuse(l1=l1_candidates, l3=l3_candidates)

    tracked = []
    if update_tracker:
        with PerfTimer(dbg, "tracker_update_ms"):
            tracked = tracker.update(fused_candidates, frame_id=frame.frame_id)

    with PerfTimer(dbg, "projection_ms"):
        if update_tracker:
            raw_detections = build_raw_detections(
                tracked, keyframe_id, frame.sharpness_score, frame.mapping_quality,
            )
        else:
            raw_detections = build_untracked_raw_detections(
                fused_candidates, keyframe_id, frame.sharpness_score,
                frame.mapping_quality,
            )
        global_detections = [
            projector.project(detection=rd, H_keyframe_to_global=H_kf_to_global,
                              transform_version=graph.transform_version)
            for rd in raw_detections
        ]

    with PerfTimer(dbg, "association_ms"):
        affected = associator.ingest_frame(frame_id=frame.frame_id, global_detections=global_detections)

    with PerfTimer(dbg, "coverage_update_ms"):
        projected_fov = projector.project_frame_corners(
            image_shape=frame.image.shape, H_keyframe_to_global=H_kf_to_global,
        )
        coverage.update(frame.frame_id, projected_fov)

    tracking_label = str(len(tracked)) if update_tracker else "historical-untracked"
    log(f"  [KEYFRAME #{keyframe_id}] L1={len(l1_candidates)} L3={len(l3_candidates)} "
        f"fused={len(fused_candidates)} tracked={tracking_label} objects_affected={len(affected)}")

    emit_event("KEYFRAME_ACCEPT", frame.frame_id,
               keyframe_id=keyframe_id, l1_count=len(l1_candidates),
               l3_count=len(l3_candidates), fused_count=len(fused_candidates),
               objects_affected=len(affected))

    # 绘制关键帧
    with PerfTimer(dbg, "debug_image_write_ms"):
        viz = draw_detections(frame.image, fused_candidates,
                              f"KEYFRAME #{keyframe_id} | L1:{len(l1_candidates)} L3:{len(l3_candidates)} fused:{len(fused_candidates)}")
        cv2.putText(viz, f"map_q={frame.mapping_quality:.2f} n_objects={len(associator.map.get_all())}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imwrite(str(OUTPUT_DIR / "keyframes" / f"kf_{keyframe_id:04d}_frame_{frame.frame_id:04d}.jpg"), viz)

    return True


log("")
log("=" * 80)
log("STARTING FRAME-BY-FRAME PROCESSING")
log("=" * 80)
log("")

pipeline_start = time.perf_counter()
PERF_COLLECTION_ACTIVE = True
frame_iterator = iter(reader.read())

while True:
    try:
        with PerfTimer(dbg, "video_decode_ms"):
            frame = next(frame_iterator)
    except StopIteration:
        break
    fc += 1
    dbg.total_frames += 1
    frame_start = time.perf_counter()

    # ── 1. 质量评估 ──
    with PerfTimer(dbg, "quality_ms"):
        frame = quality.evaluate(frame)

    emit_event("FRAME_QUALITY", frame.frame_id,
               sharpness=round(frame.sharpness_score, 1),
               exposure=round(frame.exposure_score, 3))

    mapping_eligible = quality.is_acceptable(frame)
    historical_keyframe = False
    if last_keyframe is None:
        initial_fallback.consider(frame)
        fallback_frame = initial_fallback.select(frame.frame_id)
        if not mapping_eligible and fallback_frame is not None:
            historical_keyframe = fallback_frame.frame_id < tracker.current_frame_id
            frame = fallback_frame
            mapping_eligible = True
        if mapping_eligible:
            initial_fallback.clear()

    if not quality.is_acceptable_for_detection(frame):
        # 质量差帧：推进 tracker 时间，但不建图不检测
        tracker.advance_frame(frame.frame_id)
        recovery_mgr.cache_frame(frame)

        if quality_reject_start is None:
            quality_reject_start = frame.frame_id
        log(f"F{frame.frame_id:04d} | QUALITY REJECT | sharpness={frame.sharpness_score:.1f}")
        continue
    else:
        if quality_reject_start is not None:
            log(f"  (quality reject interval: {quality_reject_start}..{frame.frame_id - 1}, "
                f"length={frame.frame_id - quality_reject_start})")
            quality_reject_start = None

    frame_buffer.push(frame)
    end_window_deque.append(frame)

    # ── 2. L2 检测 ──
    l2_was_run = (fc % l2_interval == 0)
    l2_candidates = []
    if l2_was_run:
        with PerfTimer(dbg, "l2_inference_ms"):
            l2_candidates = detector.detect(image=frame.image, level="L2", frame_id=frame.frame_id)
        dbg.record_l2_result(len(l2_candidates))
        emit_event("L2_RESULT", frame.frame_id, count=len(l2_candidates), was_run=True)
    else:
        emit_event("L2_NOT_RUN", frame.frame_id)

    # ── 3. Tracker 预览 ──
    with PerfTimer(dbg, "tracker_preview_ms"):
        preview = tracker.preview(l2_candidates, l2_was_run=l2_was_run)
    emit_event("TRACK_PREVIEW", frame.frame_id,
               l2_was_run=l2_was_run,
               unmatched_count=len(preview.unmatched_detection_indices),
               quality_drop=preview.track_quality_drop)

    if not mapping_eligible:
        if l2_was_run:
            with PerfTimer(dbg, "tracker_update_ms"):
                tracker.update(l2_candidates, frame_id=frame.frame_id)
        recovery_mgr.cache_frame(frame)
        log(
            f"F{frame.frame_id:04d} | QUALITY DETECTION_ONLY | "
            f"sharpness={frame.sharpness_score:.1f} L2={len(l2_candidates)}"
        )
        continue

    # ── 4. 触发上下文 ──
    frames_since_kf = frame.frame_id - last_keyframe_frame_id if last_keyframe_frame_id >= 0 else 999

    l3_regions = Detector.select_l3_regions(l2_candidates, l3cfg)
    trigger_context = KeyframeTriggerContext(
        max_interval_reached=(frames_since_kf >= max_interval),
        l2_new_unmatched_detection=preview.l2_new_unmatched_detection,
        track_quality_drop=preview.track_quality_drop,
        l3_required=bool(l3_regions),
        l3_regions=l3_regions,
    )

    # ── 5. 关键帧评估 ──
    match_ms_before = dbg.feature_match_ms
    decision_t0 = time.perf_counter()
    keyframe_result = selector.evaluate(
        frame=frame, previous_keyframe=last_keyframe, trigger_context=trigger_context,
    )
    decision_ms = (time.perf_counter() - decision_t0) * 1000.0
    match_ms_delta = dbg.feature_match_ms - match_ms_before
    dbg.add_timing("keyframe_decision_ms", max(0.0, decision_ms - match_ms_delta))

    if trigger_context.track_quality_drop:
        dbg.keyframe_trigger_quality_drop += 1
    if trigger_context.l2_new_unmatched_detection:
        dbg.keyframe_trigger_new_det += 1
    if trigger_context.max_interval_reached:
        dbg.keyframe_trigger_max_interval += 1

    # ── 构建状态行 ──
    status_parts = []
    status_parts.append(f"sharp={frame.sharpness_score:.0f}")
    if l2_was_run:
        status_parts.append(f"L2={len(l2_candidates)}")
    if trigger_context.max_interval_reached:
        status_parts.append("MAX_INTERVAL")
    if trigger_context.l2_new_unmatched_detection:
        status_parts.append("NEW_DET")
    if trigger_context.track_quality_drop:
        status_parts.append("QUAL_DROP")

    # ── 6. 决策分支 ──
    if keyframe_result.decision == KeyframeDecision.ACCEPTED:
        dbg.record_consecutive_keyframe(True)
        emit_event("KEYFRAME_TRIGGER", frame.frame_id,
                   reason=keyframe_result.reason, decision="accepted")

        # 添加到图（带显式 parent）
        graph_t0 = time.perf_counter()
        if last_keyframe is None:
            keyframe_id = graph.add_first_keyframe(frame_id=frame.frame_id)
            frame.mapping_quality = 1.0
            parent_node_id = None
        else:
            parent_node_id = last_keyframe_node_id
            if False:
                # 注意：match 已经在 selector.evaluate() 中运行过了
                pass
            keyframe_id = graph.add_keyframe(
                frame_id=frame.frame_id,
                H_current_to_parent=keyframe_result.H_current_to_previous,
                parent_node_id=parent_node_id,
            )
            if keyframe_result.match_result is not None:
                frame.mapping_quality = float(keyframe_result.match_result.inlier_ratio)

        H_kf_to_global = graph.get_transform(keyframe_id)
        keyframe_images[keyframe_id] = frame.image
        dbg.add_timing("graph_update_ms", (time.perf_counter() - graph_t0) * 1000.0)

        emit_event("HOMOGRAPHY_MATCH", frame.frame_id,
                   keyframe_id=keyframe_id, parent_node_id=parent_node_id,
                   valid=keyframe_result.match_result.valid if keyframe_result.match_result else True)

        # 匹配可视化
        if last_keyframe is not None and keyframe_result.match_result is not None:
            with PerfTimer(dbg, "debug_image_write_ms"):
                prev_img = last_keyframe.image if hasattr(last_keyframe, 'image') else last_keyframe.gray
                match_viz = draw_match_info(frame.image, prev_img, keyframe_result.match_result, frame.frame_id)
                cv2.imwrite(str(OUTPUT_DIR / "matches" / f"match_kf{keyframe_id}_f{frame.frame_id:04d}.jpg"), match_viz)

        process_accepted_keyframe(
            frame=frame, keyframe_id=keyframe_id, H_kf_to_global=H_kf_to_global,
            l3_regions=trigger_context.l3_regions if trigger_context.l3_required else [],
            l3_required=trigger_context.l3_required,
            parent_node_id=parent_node_id,
            update_tracker=not historical_keyframe,
        )
        processed_keyframe_frame_ids.add(frame.frame_id)
        recovery_mgr.reset()
        last_keyframe = frame
        last_keyframe_frame_id = frame.frame_id
        last_keyframe_node_id = keyframe_id
        dbg.keyframe_accepted += 1

        if keyframe_result.match_result:
            status_parts.append("MATCH OK")
            status_parts.append(f"KF#{keyframe_id}")
        else:
            status_parts.append("FIRST KF")

        log(f"F{frame.frame_id:04d} | ACCEPTED ({keyframe_result.reason}) | "
            f"{' '.join(status_parts)} | "
            f"{time.perf_counter() - frame_start:.3f}s")

    elif keyframe_result.decision == KeyframeDecision.RECOVERY:
        dbg.record_consecutive_keyframe(False)
        dbg.recovery_attempts += 1
        emit_event("KEYFRAME_RECOVERY", frame.frame_id,
                   reason=keyframe_result.reason)

        status_parts.append(f"RECOVERY: {keyframe_result.reason}")

        recovery_match_before = dbg.feature_match_ms
        recovery_t0 = time.perf_counter()
        recovery_result = recovery_mgr.recover(
            current_frame=frame, previous_keyframe=last_keyframe,
            frame_buffer=frame_buffer, graph=graph,
            keyframe_images=keyframe_images if keyframe_images else None,
        )
        recovery_elapsed_ms = (time.perf_counter() - recovery_t0) * 1000.0
        dbg.add_timing(
            "recovery_ms",
            max(
                0.0,
                recovery_elapsed_ms
                - (dbg.feature_match_ms - recovery_match_before),
            ),
        )

        if recovery_result.state == RecoveryState.RECOVERED:
            dbg.recovery_success += 1
            if recovery_result.H_current_to_anchor is not None:
                if recovery_result.anchor_node_id is not None:
                    # history anchor recovery: 使用 anchor 作为 parent
                    keyframe_id = graph.add_keyframe(
                        frame_id=frame.frame_id,
                        H_current_to_parent=recovery_result.H_current_to_anchor,
                        parent_node_id=recovery_result.anchor_node_id,
                    )
                else:
                    # bridge recovery: 使用 last_keyframe_node_id 作为 parent
                    keyframe_id = graph.add_keyframe(
                        frame_id=frame.frame_id,
                        H_current_to_parent=recovery_result.H_current_to_anchor,
                        parent_node_id=last_keyframe_node_id,
                    )
                frame.mapping_quality = 0.5
                H_kf_to_global = graph.get_transform(keyframe_id)
                keyframe_images[keyframe_id] = frame.image
                process_accepted_keyframe(
                    frame=frame, keyframe_id=keyframe_id, H_kf_to_global=H_kf_to_global,
                    l3_regions=trigger_context.l3_regions if trigger_context.l3_required else [],
                    l3_required=trigger_context.l3_required,
                )
                processed_keyframe_frame_ids.add(frame.frame_id)
                recovery_mgr.reset()
                last_keyframe = frame
                last_keyframe_frame_id = frame.frame_id
                last_keyframe_node_id = keyframe_id
                status_parts.append("RECOVERED OK")
        else:
            dbg.recovery_failed += 1
            recovery_mgr.cache_frame(frame)
            recovery_mgr.cache_detections(frame.frame_id, l2_candidates, frame.sharpness_score)
            status_parts.append("LOST")

        log(f"F{frame.frame_id:04d} | RECOVERY | {' '.join(status_parts)} | "
            f"match_err={keyframe_result.reason}")

    else:
        # SKIP
        dbg.record_consecutive_keyframe(False)
        dbg.keyframe_rejected_by_cooldown += 1
        emit_event("KEYFRAME_SKIP", frame.frame_id)

        # 仅 L2 更新 tracker
        if l2_was_run:
            with PerfTimer(dbg, "tracker_update_ms"):
                tracker.update(l2_candidates, frame_id=frame.frame_id)

        # 普通帧可视化
        if l2_was_run and l2_candidates:
            with PerfTimer(dbg, "debug_image_write_ms"):
                viz = draw_detections(frame.image, l2_candidates, f"L2 F{frame.frame_id} | {' '.join(status_parts)}")
                cv2.imwrite(str(OUTPUT_DIR / "frames" / f"frame_{frame.frame_id:04d}.jpg"), viz)

    # 限制帧数
    if MAX_FRAMES and fc >= MAX_FRAMES:
        log(f"\nReached max frames limit ({MAX_FRAMES}), stopping early.")
        break

main_loop_time = time.perf_counter() - pipeline_start

# ══════════════════════════════════════════════════════════════════
# 后处理
# ══════════════════════════════════════════════════════════════════

log("")
log("=" * 80)
log("POST-PROCESSING")
log("=" * 80)

# 尾帧关键帧——按 frame_id 升序确保正确的 parent 链
end_match_before = dbg.feature_match_ms
end_t0 = time.perf_counter()
end_kfs = selector.select_end_keyframes(list(end_window_deque))
end_elapsed_ms = (time.perf_counter() - end_t0) * 1000.0
dbg.add_timing(
    "end_window_ms",
    max(0.0, end_elapsed_ms - (dbg.feature_match_ms - end_match_before)),
)
log(f"End-window keyframes: {len(end_kfs)} candidates")
for ekf in end_kfs:
    if ekf.frame_id in processed_keyframe_frame_ids:
        continue
    end_eval_match_before = dbg.feature_match_ms
    end_eval_t0 = time.perf_counter()
    result = selector.evaluate(
        ekf,
        last_keyframe,
        KeyframeTriggerContext(force_end_candidate=True),
    )
    dbg.add_timing(
        "end_window_ms",
        max(
            0.0,
            (time.perf_counter() - end_eval_t0) * 1000.0
            - (dbg.feature_match_ms - end_eval_match_before),
        ),
    )
    if result.decision == KeyframeDecision.ACCEPTED:
        parent_id = last_keyframe_node_id
        try:
            end_graph_t0 = time.perf_counter()
            keyframe_id = graph.add_keyframe(
                frame_id=ekf.frame_id,
                H_current_to_parent=result.H_current_to_previous,
                parent_node_id=parent_id,
            )
            H_kf_to_global = graph.get_transform(keyframe_id)
            dbg.add_timing(
                "graph_update_ms",
                (time.perf_counter() - end_graph_t0) * 1000.0,
            )
            log(f"  Adding end KF: frame {ekf.frame_id} → node {keyframe_id} (parent={parent_id})")
            process_accepted_keyframe(frame=ekf, keyframe_id=keyframe_id,
                                      H_kf_to_global=H_kf_to_global,
                                      l3_regions=[], l3_required=False,
                                      update_tracker=False)
            processed_keyframe_frame_ids.add(ekf.frame_id)
            last_keyframe = ekf
            last_keyframe_frame_id = ekf.frame_id
            last_keyframe_node_id = keyframe_id
        except (ValueError, KeyError) as e:
            log(f"  End KF failed: frame {ekf.frame_id} — {e}")

# 最终审查
log("\nRunning final_review()...")
with PerfTimer(dbg, "final_review_ms"):
    associator.final_review()
log(f"  final_review done, merged={associator.stats['objects_merged']} "
    f"blocked_cooccur={associator.stats['merge_blocked_by_cooccurrence']} "
    f"blocked_overlap={associator.stats['merge_blocked_by_frame_overlap']}")

# 验证对象地图
violations = associator.validate_object_map()

# 分配 persistent ID（只给 reportable）
associator.map.assign_persistent_ids()

# ── 报告 ──
with PerfTimer(dbg, "report_ms"):
    # R3: JSON、CSV、控制台、API 使用同一个 reportable_objects
    reportable = associator.get_reportable_objects()
    gen = ReportGenerator(object_map=associator.map)
    gen.find_evidence_frames(reportable)
    json_report = gen.generate_json_report()

    json_path = OUTPUT_DIR / "reports" / "report.json"
    json_path.write_text(json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = OUTPUT_DIR / "reports" / "report.csv"
    csv_path.write_text(gen.generate_csv_report(associator.map.get_all()), encoding="utf-8")

# 关键帧统计
artifact_report_t0 = time.perf_counter()
kf_data = []
for nid, fid, h2g in graph.nodes:
    parent_id = graph.get_parent_node_id(nid)
    kf_data.append({
        "node_id": nid,
        "frame_id": fid,
        "parent_node_id": parent_id,
        "H_to_global_det": float(np.linalg.det(h2g)),
    })
with open(OUTPUT_DIR / "keyframe_stats.csv", "w", newline="", encoding="utf-8") as f:
    import csv as csv_mod
    w = csv_mod.DictWriter(f, fieldnames=["node_id", "frame_id", "parent_node_id", "H_to_global_det"])
    w.writeheader()
    w.writerows(kf_data)

# 关联事件
with open(OUTPUT_DIR / "association_events.csv", "w", newline="", encoding="utf-8") as f:
    w = csv_mod.writer(f)
    w.writerow(["event", "primary_id", "secondary_id", "decision", "reason"])
    for audit in associator.merge_audits:
        w.writerow(["merge_audit", audit.primary_id, audit.secondary_id, audit.decision, audit.reason])

# 对象生命周期
lifecycle = []
for obj in associator.map.get_all():
    lifecycle.append({
        "provisional_id": obj.provisional_id,
        "persistent_id": obj.persistent_id,
        "class_name": obj.class_name,
        "status": obj.confirmation_status.value,
        "observation_count": obj.observation_count,
        "unique_frame_count": len({obs.frame_id for obs in obj.observations}),
        "keyframe_count": len(obj.keyframe_ids),
        "track_count": len(obj.track_ids),
        "rejected_reason": obj.rejected_reason,
        "merged_into_id": obj.merged_into_id,
        "review_flags": [f.value for f in obj.review_flags],
        "centroid_xy": list(obj.centroid_xy),
    })
with open(OUTPUT_DIR / "object_lifecycle.json", "w", encoding="utf-8") as f:
    json.dump(lifecycle, f, indent=2, ensure_ascii=False)
dbg.add_timing("report_ms", (time.perf_counter() - artifact_report_t0) * 1000.0)

# ── 汇总 ──

# 统计 rejected reasons
rejected_reasons = {}
for obj in associator.map.get_all():
    if obj.confirmation_status == ConfirmationStatus.REJECTED:
        reason = obj.rejected_reason or "unexplained"
        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

# 报告一致性检查
reportable_total = json_report["total_objects"]
json_confirmed = json_report["confirmed_count"]
json_tentative = json_report["tentative_count"]
json_uncertain = json_report["uncertain_count"]
r1_check = (reportable_total == json_confirmed + json_tentative + json_uncertain)

# CSV 行数检查
with open(csv_path, newline="", encoding="utf-8") as csv_file:
    csv_data_rows = sum(1 for _ in csv_mod.reader(csv_file)) - 1
all_objects = len(associator.map.get_all())

pipeline_time = time.perf_counter() - pipeline_start
PERF_COLLECTION_ACTIVE = False
wall_ms = pipeline_time * 1000.0

log("")
log("=" * 80)
log("SUMMARY")
log("=" * 80)
log(f"Run ID: {RUN_ID}")
log(f"Pipeline time: {pipeline_time:.1f}s")
log(f"Frames processed: {dbg.total_frames}")
log(f"  L2 run frames: {dbg.l2_run_frames}")
log(f"  L2 empty frames: {dbg.l2_empty_frames}")
log(f"  Keyframe accepted: {dbg.keyframe_accepted}")
log(f"  Keyframe skipped (cooldown): {dbg.keyframe_rejected_by_cooldown}")
log(f"  Max consecutive keyframes: {dbg.consecutive_keyframe_max_run}")
log(f"Graph nodes: {graph.num_keyframes}")
log(f"  Invalid homography nodes: {dbg.invalid_homography_nodes}")
log(f"")
log(f"Recovery:")
log(f"  Attempts: {dbg.recovery_attempts}")
log(f"  Success: {dbg.recovery_success}")
log(f"  Failed: {dbg.recovery_failed}")
log(f"")
log(f"Object statistics:")
log(f"  Provisional objects: {all_objects}")
log(f"  CONFIRMED: {json_report['confirmed_count']}")
log(f"  TENTATIVE: {json_report['tentative_count']}")
log(f"  UNCERTAIN: {json_report['uncertain_count']}")
log(f"  REJECTED: {json_report['rejected_count']}")
log(f"  Reportable (R1={r1_check}): {reportable_total}")
log(f"")
log(f"Rejected reasons:")
for reason, count in sorted(rejected_reasons.items()):
    log(f"  {reason}: {count}")
log(f"  unexplained: {rejected_reasons.get('unexplained', 0)}")
log(f"")
log(f"Merges:")
log(f"  Total merges: {associator.stats['objects_merged']}")
log(f"  Blocked by co-occurrence: {associator.stats['merge_blocked_by_cooccurrence']}")
log(f"  Blocked by frame overlap: {associator.stats['merge_blocked_by_frame_overlap']}")
log(f"")
log(f"Invariants:")
log(f"  track_binding_conflicts: {associator.stats['track_binding_conflicts']}")
log(f"  same_frame_duplicate_observations: {associator.stats['same_frame_duplicate_observations']}")
log(f"  unexplained_rejected_objects: {rejected_reasons.get('unexplained', 0)}")
log(f"  invalid_homography_nodes: {dbg.invalid_homography_nodes}")
log(f"  report_count_inconsistency: {0 if r1_check and csv_data_rows == all_objects else 1}")
log(f"  tracker_time_regressions: {tracker.time_regressions}")
log(f"  l2_run_count_inconsistency: {dbg.l2_run_count_inconsistency}")
log(f"")
log(f"FPS:")
core_fps = dbg.total_frames / (dbg.core_algorithm_ms / 1000.0) if dbg.core_algorithm_ms > 0 else 0
total_fps = dbg.total_frames / pipeline_time if pipeline_time > 0 else 0
log(f"  Core algorithm: {core_fps:.1f} FPS")
log(f"  Total (inc I/O): {total_fps:.1f} FPS")
log(f"  Main loop: {main_loop_time:.3f}s")
log(f"  End-to-end: {pipeline_time:.3f}s")
log("")
log("Performance breakdown:")
log(f"  {'Stage':<24} {'Total(ms)':>10} {'Calls':>8} {'Avg(ms)':>10} {'Wall%':>8}")
performance_labels = {
    "initialization_ms": "Initialization",
    "video_decode_ms": "Video decode",
    "quality_ms": "Quality evaluation",
    "l2_inference_ms": "L2 inference",
    "tracker_preview_ms": "Tracker preview",
    "tracker_update_ms": "Tracker update",
    "keyframe_decision_ms": "Keyframe decision",
    "feature_match_ms": "Feature matching",
    "recovery_ms": "Recovery",
    "l1_inference_ms": "L1 inference",
    "l3_inference_ms": "L3 inference",
    "fusion_ms": "Fusion",
    "graph_update_ms": "Graph update",
    "projection_ms": "Projection",
    "association_ms": "Association",
    "coverage_update_ms": "Coverage update",
    "end_window_ms": "End-window selection",
    "final_review_ms": "Final review",
    "mosaic_ms": "Global mosaic",
    "evidence_ms": "Evidence extraction",
    "session_store_ms": "Session storage",
    "report_ms": "Report generation",
    "event_log_io_ms": "Event/log I/O",
    "debug_image_write_ms": "Debug image I/O",
}
for field, total, calls, average in dbg.performance_rows():
    wall_pct = total / wall_ms * 100.0 if wall_ms > 0 else 0.0
    log(
        f"  {performance_labels[field]:<24} {total:>10.1f} "
        f"{calls:>8d} {average:>10.3f} {wall_pct:>7.1f}%"
    )
unaccounted_ms = dbg.unaccounted_ms(wall_ms)
log(f"  {'Unaccounted overhead':<24} {unaccounted_ms:>10.1f} "
    f"{'-':>8} {'-':>10} {unaccounted_ms / wall_ms * 100.0:>7.1f}%")
log(f"  Accounted coverage: {dbg.coverage_ratio(wall_ms) * 100.0:.1f}%")
log(f"")
log(f"Class distribution (reportable):")
for cls_name, count in sorted(json_report["class_counts"].items()):
    log(f"  {cls_name}: {count}")
log(f"")
log(f"Object details:")
for obj in associator.map.get_all():
    reportable_mark = "✓" if obj.confirmation_status != ConfirmationStatus.REJECTED else "✗(REJECTED)"
    rejected_info = ""
    if obj.rejected_reason:
        rejected_info = f" reason={obj.rejected_reason}"
    if obj.merged_into_id:
        rejected_info += f" merged_into={obj.merged_into_id}"
    log(f"  [{reportable_mark}] {obj.provisional_id} → {obj.persistent_id or 'N/A'} "
        f"class={obj.class_name} status={obj.confirmation_status.value} "
        f"obs={obj.observation_count} kfs={len(obj.keyframe_ids)} tracks={len(obj.track_ids)} "
        f"centroid=({obj.centroid_xy[0]:.0f}, {obj.centroid_xy[1]:.0f}) "
        f"votes={dict(obj.vote_distribution)} flags={[f.value for f in obj.review_flags]}{rejected_info}")
log(f"")
log(f"Violations:")
for vtype, items in violations.items():
    if items:
        log(f"  {vtype}: {len(items)}")
        if vtype == "duplicate_frame_observations":
            for item in items[:5]:
                log(f"    {item}")
log(f"")
log(f"Output files:")
log(f"  {json_path}")
log(f"  {csv_path}")
log(f"  {OUTPUT_DIR / 'events.jsonl'}")
log(f"  {OUTPUT_DIR / 'keyframe_stats.csv'}")
log(f"  {OUTPUT_DIR / 'association_events.csv'}")
log(f"  {OUTPUT_DIR / 'object_lifecycle.json'}")
log(f"  {OUTPUT_DIR / 'effective_config.yaml'}")
log(f"  {LOG_FILE}")
log("")
log("=" * 80)
log("DONE")
log("=" * 80)

# 关闭事件写入器
event_writer.close()
