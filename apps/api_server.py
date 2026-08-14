"""FastAPI 在线处理入口 — 接收分片上传视频，流式处理返回结果"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

from apps.offline_scan import run_pipeline
from core.report_generator import empty_report

app = FastAPI(title="Head Tool Counter API", version="0.1.0")

# ── 作业状态存储 ──
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _run_job(job_id: str, video_path: str, config_dir: str, output_dir: str) -> None:
    try:
        with _lock:
            _jobs[job_id]["status"] = "processing"
        run_pipeline(video_path, config_dir, output_dir)

        # Read report
        report_path = Path(output_dir) / "reports" / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            with _lock:
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["result"] = report
        else:
            with _lock:
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["result"] = empty_report()
    except Exception as e:
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
    finally:
        # Cleanup temp video
        Path(video_path).unlink(missing_ok=True)


@app.post("/scan")
async def scan(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config_dir: str = "configs",
):
    """上传视频文件，启动后台处理。返回 job_id 用于轮询结果。"""
    import uuid

    job_id = str(uuid.uuid4())[:8]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        shutil.copyfileobj(file.file, tmp)
    finally:
        tmp.close()

    output_dir = tempfile.mkdtemp(prefix="htc_")

    with _lock:
        _jobs[job_id] = {"status": "queued", "result": None, "error": None}

    background_tasks.add_task(_run_job, job_id, tmp.name, config_dir, output_dir)

    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/job/{job_id}")
async def job_status(job_id: str):
    """轮询作业状态。status 为 completed 时包含 result。"""
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse(job)


@app.get("/job/{job_id}/report")
async def job_report(job_id: str):
    """获取已完成作业的 JSON 报告。"""
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    if job["status"] != "completed":
        return JSONResponse({"error": "job not yet completed"}, status_code=409)
    return JSONResponse(job["result"])


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
