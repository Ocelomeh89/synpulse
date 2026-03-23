"""Vercel serverless API: transcribe audio via OpenAI Whisper API, analyze via Claude."""

import json
import os
import tempfile

import anthropic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ANALYSIS_PROMPT = """\
You are a senior Sales Engineering analyst with 15+ years of experience closing \
complex B2B deals. Analyze the following transcript from a voice note or call recording.

Your job is to read between the lines: distinguish genuine buying signals \
(budget discussions, timeline commitments, internal champion mentions, technical \
deep-dives) from polite dismissals (vague interest, "let's circle back", no \
concrete next steps, deflecting to committees without timelines).

Return your analysis as a JSON object with exactly this structure:

{
  "key_client": "<Name of the primary person or organization discussed>",
  "action_points": [
    {
      "task": "<Specific action>",
      "owner": "<Person responsible>",
      "deadline": "<Date or timeframe, or 'TBD' if not mentioned>"
    }
  ],
  "summary": "<Exactly 3 sentences. Executive summary of the conversation.>",
  "closing_probability": {
    "percentage": <0-100>,
    "reasoning": "<2-3 sentences explaining the score based on commitment signals, budget mentions, concrete next steps, and any red flags.>"
  }
}

Rules:
- If no client name is identifiable, use "Unknown".
- Extract ALL action points, even implicit ones.
- The closing probability must reflect subtle cues: a call with "we love it" but \
no budget discussion scores lower than one with "let me check our Q3 budget allocation".
- Be skeptical of enthusiasm without specifics.
- Return ONLY the JSON object, no markdown fencing or extra text.

TRANSCRIPT:
{transcript}
"""

ALLOWED_BASE_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a", "audio/x-m4a",
    "audio/wav", "audio/x-wav", "audio/webm", "video/webm",
    "audio/ogg", "audio/aac", "audio/flac",
}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB (OpenAI Whisper API limit)


def _is_allowed_type(content_type: str) -> bool:
    """Check content type, ignoring codec params like ';codecs=opus'."""
    base = content_type.split(";")[0].strip().lower()
    return base in ALLOWED_BASE_TYPES


def transcribe(file_bytes: bytes, filename: str) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(model="whisper-1", file=f)
    finally:
        os.unlink(tmp_path)

    return result.text


def analyze(transcript: str) -> dict:
    """Send transcript to Claude for sales-engineering analysis."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(transcript=transcript)}],
    )
    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]  # remove opening ```json line
        raw = raw.rsplit("```", 1)[0]  # remove closing ```
    return json.loads(raw.strip())


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if file.content_type and not _is_allowed_type(file.content_type):
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, "File exceeds 25 MB limit")

    try:
        transcript = transcribe(file_bytes, file.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(502, f"Transcription failed: {e}")

    try:
        report = analyze(transcript)
    except json.JSONDecodeError as e:
        raise HTTPException(502, f"Claude returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(502, f"Analysis failed: {e}")

    return {"transcript": transcript, "report": report}


@app.get("/")
async def home():
    return HTMLResponse(open("api/static/index.html").read())
