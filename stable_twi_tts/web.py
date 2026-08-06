"""A local web UI: type text, upload a PDF, or point at a URL, and get audio.

Deliberately local-first. It binds to 127.0.0.1 by default and holds one model in memory, because
the realistic use is someone on their own machine turning documents into Twi speech — not a
public service. Two consequences worth being explicit about:

  URL fetching is a server-side request made on the user's behalf, so it refuses private and
  loopback addresses. That guard matters the moment anyone binds this to 0.0.0.0, which the
  --host flag allows and the startup banner warns about.

  Synthesis is serialised behind a lock. onnxruntime sessions are not safely shared across
  threads, and a queue of one is honest for a single-user tool; concurrency belongs in the batch
  endpoint, not in parallel sessions.

Long inputs are split into sentences and synthesised piece by piece, with progress streamed, so a
40-page PDF reports as it goes instead of appearing to hang.
"""
from __future__ import annotations

import io
import ipaddress
import json
import re
import socket
import threading
import time
import urllib.parse
import wave
import zipfile
from pathlib import Path

# Imported at module scope, not inside create_app: `from __future__ import annotations` turns
# every annotation into a string, and pydantic resolves those against module globals. With the
# names local to a function, `UploadFile | None` stays an unresolvable ForwardRef and every
# upload 500s. The flag keeps the friendly "pip install" error for anyone without fastapi.
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, Response, StreamingResponse
    _HAVE_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAVE_FASTAPI = False

MAX_CHARS = 200_000
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class TextExtractionError(RuntimeError):
    pass


# ------------------------------------------------------------------ ingestion

def sentences(text: str, max_chars: int = 600) -> list[str]:
    """Split into utterance-sized pieces.

    Sentence boundaries first, then a hard wrap on anything still oversized — a PDF paragraph
    with no terminal punctuation would otherwise become one enormous utterance, and VITS
    attention degrades badly past what it saw in training.
    """
    out: list[str] = []
    for part in SENT_SPLIT.split(text):
        part = " ".join(part.split())
        if not part:
            continue
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            out.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            out.append(part)
    return out


def text_from_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise TextExtractionError(
            "PDF support needs pypdf: pip install 'stable-twi-tts[web]'") from e
    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise TextExtractionError(
            "no text found — this PDF is probably scanned images, which needs OCR first")
    return text


def _is_public(host: str) -> bool:
    """Reject loopback, private and link-local targets before fetching."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


def text_from_url(url: str, timeout: int = 20) -> str:
    import urllib.request

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise TextExtractionError("only http and https URLs are supported")
    if not parsed.hostname or not _is_public(parsed.hostname):
        raise TextExtractionError(
            "refusing to fetch a private, loopback or link-local address")

    req = urllib.request.Request(url, headers={"User-Agent": "stable-twi-tts/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(8_000_000)
        charset = r.headers.get_content_charset() or "utf-8"
    html = raw.decode(charset, errors="replace")

    try:
        import trafilatura
        text = trafilatura.extract(html) or ""
    except ImportError:
        # Crude fallback: strip script/style, then tags. Good enough for simple pages, and
        # avoids making trafilatura mandatory just to read a URL.
        html = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", html)
        text = re.sub(r"&nbsp;?", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text.strip():
        raise TextExtractionError("no readable text found at that URL")
    return text


# ----------------------------------------------------------------------- app

def create_app(model_dir: str | None = None, repo: str | None = None):
    if not _HAVE_FASTAPI:
        raise ImportError("the web UI needs fastapi and uvicorn:\n"
                          "  pip install 'stable-twi-tts[web]'")

    from .tts import StableTwiTTS

    app = FastAPI(title="Stable Twi TTS")
    static = Path(__file__).parent / "static"

    tts = (StableTwiTTS(model_dir) if model_dir
           else StableTwiTTS.from_pretrained(repo))
    lock = threading.Lock()          # one onnxruntime session, one caller at a time

    def synth(text: str, voice: str, language: str, length_scale: float):
        with lock:
            return tts.synthesize(text, voice=voice, language=language,
                                  length_scale=length_scale)

    def wav_bytes(audio, sample_rate: int) -> bytes:
        import numpy as np
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        return buf.getvalue()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (static / "index.html").read_text(encoding="utf-8")

    @app.get("/api/voices")
    def voices():
        return {
            "voices": [
                {"name": v.name, "language": v.language, "hours": v.hours,
                 "codeswitch_uer": v.codeswitch_uer_avg, "twi_only_uer": v.twi_only_uer,
                 "source": v.source_speaker}
                for v in tts.voices.voices
            ],
            "sample_rate": tts.sample_rate,
        }

    @app.post("/api/speak")
    def speak(text: str = Form(...), voice: str = Form(""), language: str = Form("twi"),
              length_scale: float = Form(1.0)):
        text = text.strip()
        if not text:
            raise HTTPException(400, "no text")
        if len(text) > MAX_CHARS:
            raise HTTPException(413, f"text longer than {MAX_CHARS} characters")
        try:
            s = synth(text, voice or None, language, length_scale)
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        return Response(wav_bytes(s.audio, s.sample_rate), media_type="audio/wav",
                        headers={"X-Duration": f"{s.duration:.2f}",
                                 "X-Phonemes": str(s.n_phonemes),
                                 "X-Voice": s.voice.name})

    @app.post("/api/extract")
    async def extract(file: UploadFile | None = File(None), url: str = Form("")):
        """PDF or URL to text, so the user can review and edit before synthesising."""
        try:
            if file is not None:
                data = await file.read()
                text = text_from_pdf(data)
            elif url.strip():
                text = text_from_url(url.strip())
            else:
                raise HTTPException(400, "provide a PDF or a URL")
        except TextExtractionError as e:
            raise HTTPException(400, str(e))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        chunks = sentences(text)
        return {"text": text[:MAX_CHARS], "n_chars": len(text), "n_chunks": len(chunks)}

    @app.post("/api/batch")
    def batch(text: str = Form(...), voice: str = Form(""), language: str = Form("twi"),
              length_scale: float = Form(1.0), join: bool = Form(True)):
        """Synthesise many lines, streaming progress, then a zip or one joined wav.

        Progress is streamed as newline-delimited JSON because a long document otherwise looks
        like a hang; the final line carries the download id.
        """
        import numpy as np

        chunks = sentences(text)
        if not chunks:
            raise HTTPException(400, "nothing to say")

        def run():
            pieces, results, t0 = [], [], time.time()
            gap = np.zeros(int(0.25 * tts.sample_rate), dtype="float32")
            for i, chunk in enumerate(chunks, 1):
                try:
                    s = synth(chunk, voice or None, language, length_scale)
                    pieces.append(s.audio)
                    results.append({"i": i, "text": chunk, "duration": round(s.duration, 2)})
                except Exception as e:
                    results.append({"i": i, "text": chunk, "error": f"{type(e).__name__}: {e}"})
                yield json.dumps({"done": i, "total": len(chunks),
                                  "elapsed": round(time.time() - t0, 1)}) + "\n"

            ok = [p for p in pieces if p is not None and len(p)]
            if not ok:
                yield json.dumps({"error": "everything failed"}) + "\n"
                return
            buf = io.BytesIO()
            if join:
                joined = ok[0]
                for p in ok[1:]:
                    joined = np.concatenate([joined, gap, p])
                buf.write(wav_bytes(joined, tts.sample_rate))
                name = "speech.wav"
            else:
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for j, p in enumerate(ok, 1):
                        z.writestr(f"{j:04d}.wav", wav_bytes(p, tts.sample_rate))
                    z.writestr("manifest.json", json.dumps(results, ensure_ascii=False, indent=1))
                name = "speech.zip"
            token = str(int(time.time() * 1000))
            DOWNLOADS[token] = (name, buf.getvalue())
            yield json.dumps({"done": len(chunks), "total": len(chunks),
                              "token": token, "filename": name,
                              "failed": sum(1 for r in results if "error" in r)}) + "\n"

        return StreamingResponse(run(), media_type="application/x-ndjson")

    DOWNLOADS: dict[str, tuple[str, bytes]] = {}

    @app.get("/api/download/{token}")
    def download(token: str):
        item = DOWNLOADS.pop(token, None)     # one-shot: the browser has it after this
        if not item:
            raise HTTPException(404, "expired or already downloaded")
        name, data = item
        media = "application/zip" if name.endswith(".zip") else "audio/wav"
        return Response(data, media_type=media,
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="stable-twi-tts-web",
                                 description="Local web UI for Twi speech synthesis")
    ap.add_argument("--model", default=None, help="voice directory; omit to use the Hub")
    ap.add_argument("--repo", default=None, help="Hub repo when --model is omitted")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 exposes this to your network — see the URL-fetch note")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    import uvicorn
    app = create_app(args.model, args.repo)
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"warning: binding {args.host} exposes this UI, and the URL-fetch endpoint makes "
              f"requests from this machine. Private addresses are refused, but do not expose "
              f"this to an untrusted network.")
    print(f"\n  Stable Twi TTS  ->  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
