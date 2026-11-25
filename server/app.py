"""Autosampler server: library, analysis jobs, audio."""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = Path(os.environ.get("AUTOSAMPLER_LIBRARY", ROOT / "Vinyls"))
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".aif", ".aiff", ".opus", ".wma"}

app = FastAPI()
jobs: dict[str, dict] = {}
heavy = threading.Lock()


def library_path(name: str) -> Path:
    p = (LIBRARY / Path(name).name).resolve()
    if p.parent != LIBRARY.resolve() or not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
        raise HTTPException(404, "No such song")
    return p


@app.get("/api/library")
def library():
    LIBRARY.mkdir(exist_ok=True)
    out = []
    for p in sorted(LIBRARY.iterdir(), key=lambda p: p.name.lower()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT and not p.name.startswith("."):
            sid = analysis.song_id(p)
            out.append({"file": p.name, "title": analysis.clean_title(p.name),
                        "ready": (analysis.CACHE / sid / "song.json").exists()})
    return out


@app.post("/api/upload")
async def upload(file: UploadFile):
    name = Path(file.filename or "song").name
    if Path(name).suffix.lower() not in AUDIO_EXT:
        raise HTTPException(400, "Not an audio file")
    LIBRARY.mkdir(exist_ok=True)
    dest = LIBRARY / name
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return {"file": name, "title": analysis.clean_title(name)}


class AnalyzeRequest(BaseModel):
    file: str
    source: str = "nodrums"


def run_job(job: dict, path: Path, source: str):
    def progress(stage, frac):
        job["stage"], job["progress"] = stage, round(float(frac), 3)

    job["stage"] = "Waiting"
    with heavy:
        try:
            job["result"] = analysis.analyze(path, source, progress)
        except Exception as e:  # surfaced to the UI
            job["error"] = str(e) or e.__class__.__name__
    job["done"] = True


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if req.source not in analysis.SOURCES:
        raise HTTPException(400, "Unknown source")
    path = library_path(req.file)
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "stage": "Starting", "progress": 0.0, "done": False}
    jobs[jid] = job
    threading.Thread(target=run_job, args=(job, path, req.source), daemon=True).start()
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "No such job")
    if job["done"]:
        jobs.pop(jid, None)
    return job


@app.get("/api/audio/{sid}/{name}")
def audio(sid: str, name: str):
    if not sid.isalnum():
        raise HTTPException(404)
    stem, _, ext = name.partition(".")
    ok = (stem in analysis.SOURCES and ext == "flac") or (stem in {"kit_kick", "kit_snare", "kit_hat"} and ext == "wav")
    p = analysis.CACHE / sid / name
    if not ok or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
