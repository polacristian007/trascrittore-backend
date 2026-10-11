"""
Backend di trascrizione gratuito (Whisper open source).

- Nessuna API key, nessun costo.
- Divide l'audio in blocchi da N secondi (default 120s = 2 minuti) con ffmpeg
  e li trascrive uno alla volta con faster-whisper (CPU, int8).
- Espone un job asincrono con progresso e testo in streaming (polling).

Avvio:
    pip install -r requirements.txt
    python main.py
Il server ascolta su 0.0.0.0:8000 (raggiungibile dal telefono nella stessa Wi-Fi).
"""

import glob
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="Trascrittore Lezioni - Whisper API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Stato in memoria
# ---------------------------------------------------------------------------
JOBS: Dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
_MODELS: Dict[str, object] = {}
_MODEL_LOCK = threading.Lock()

WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
MAX_CHUNK_SECONDS = int(os.environ.get("MAX_CHUNK_SECONDS", "600"))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def find_ffmpeg() -> Optional[str]:
    """Trova ffmpeg nel PATH oppure usa il binario incluso in imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_model(name: str):
    with _MODEL_LOCK:
        if name not in _MODELS:
            from faster_whisper import WhisperModel

            _MODELS[name] = WhisperModel(
                name, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
            )
        return _MODELS[name]


def split_audio(src_path: str, workdir: str, chunk_seconds: int) -> List[str]:
    """Divide un file audio/video in blocchi WAV mono 16kHz."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg non trovato. Installa ffmpeg oppure 'pip install imageio-ffmpeg'."
        )

    chunk_seconds = max(10, min(int(chunk_seconds), MAX_CHUNK_SECONDS))
    out_pattern = os.path.join(workdir, "chunk_%05d.wav")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        src_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-c:a",
        "pcm_s16le",
        out_pattern,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="ignore").strip()[-600:]
        raise RuntimeError(detail or "ffmpeg ha restituito un errore")

    files = sorted(glob.glob(os.path.join(workdir, "chunk_*.wav")))
    if not files:
        raise RuntimeError("Nessun blocco audio generato dal file caricato")
    return files


def _is_cancelled(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return bool(job and job.get("cancelled"))


def process_job(job_id: str, chunks: List[str], workdir: str, model_name: str, language: str):
    try:
        model = get_model(model_name)
        parts: List[str] = []
        for index, chunk_path in enumerate(chunks):
            if _is_cancelled(job_id):
                return
            segments, _info = model.transcribe(
                chunk_path,
                language=language or "it",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            part = " ".join(seg.text.strip() for seg in segments).strip()
            if part:
                parts.append(part)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    return
                job["text"] = " ".join(parts).strip()
                job["processed_chunks"] = index + 1

        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job["text"] = " ".join(parts).strip()
                job["processed_chunks"] = len(chunks)
                job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job["status"] = "error"
                job["error"] = str(exc)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "Trascrittore Lezioni API", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ffmpeg": bool(find_ffmpeg()),
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE,
        "models_loaded": list(_MODELS.keys()),
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("small"),
    language: str = Form("it"),
    chunk_seconds: int = Form(120),
):
    if model not in ("tiny", "base", "small", "medium", "large-v3"):
        model = "small"

    suffix = os.path.splitext(file.filename or "audio.m4a")[1] or ".m4a"
    fd, src_path = tempfile.mkstemp(prefix="upload_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:  # noqa: BLE001
        try:
            os.remove(src_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=f"Errore durante il caricamento: {exc}")

    workdir = tempfile.mkdtemp(prefix="trascr_")
    try:
        chunks = await run_in_threadpool(split_audio, src_path, workdir, int(chunk_seconds))
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            os.remove(src_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=f"Impossibile leggere il file: {exc}")
    finally:
        try:
            os.remove(src_path)
        except OSError:
            pass

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "processing",
            "processed_chunks": 0,
            "total_chunks": len(chunks),
            "text": "",
            "error": None,
            "cancelled": False,
            "created": time.time(),
        }

    threading.Thread(
        target=process_job,
        args=(job_id, chunks, workdir, model, language),
        daemon=True,
    ).start()

    return {"job_id": job_id, "total_chunks": len(chunks)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job non trovato")
        return JSONResponse(
            {
                "id": job["id"],
                "status": job["status"],
                "processed_chunks": job["processed_chunks"],
                "total_chunks": job["total_chunks"],
                "text": job["text"],
                "error": job["error"],
            }
        )


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job non trovato")
        job["cancelled"] = True
    return {"status": "cancelling"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
