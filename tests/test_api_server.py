from apps import api_server
from core import report_generator


def test_missing_report_uses_current_empty_report_schema(tmp_path, monkeypatch):
    job_id = "missing-report"
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video")
    output_dir = tmp_path / "output"
    api_server._jobs[job_id] = {"status": "queued"}
    monkeypatch.setattr(api_server, "run_pipeline", lambda *_args: None)

    try:
        api_server._run_job(
            job_id, str(video_path), "configs", str(output_dir)
        )
        job = api_server._jobs[job_id]

        assert job["status"] == "completed"
        assert job["result"] == report_generator.empty_report()
        assert job["result"]["schema_version"] == (
            report_generator.REPORT_SCHEMA_VERSION
        )
        assert job["result"]["objects"] == []
        assert job["result"]["review_candidates"] == []
        assert job["result"]["rejected_objects"] == []
    finally:
        api_server._jobs.pop(job_id, None)
