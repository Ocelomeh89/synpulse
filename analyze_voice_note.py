#!/usr/bin/env python3
"""Voice note analyzer: transcribe audio and extract sales intelligence via Claude."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import anthropic
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

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


def convert_to_wav(input_path: str) -> str:
    """Convert mp3/m4a to wav if needed, return path to wav file."""
    ext = Path(input_path).suffix.lower()
    if ext == ".wav":
        return input_path

    fmt_map = {".mp3": "mp3", ".m4a": "m4a"}
    fmt = fmt_map.get(ext)
    if not fmt:
        sys.exit(f"Unsupported format: {ext}. Use mp3, m4a, or wav.")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    audio = AudioSegment.from_file(input_path, format=fmt)
    audio.export(tmp.name, format="wav")
    return tmp.name


def transcribe(audio_path: str, model_name: str = "base") -> str:
    """Transcribe audio file using Whisper."""
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    wav_path = convert_to_wav(audio_path)
    try:
        print("Transcribing audio...")
        result = model.transcribe(wav_path)
    finally:
        if wav_path != audio_path:
            os.unlink(wav_path)

    return result["text"]


def analyze(transcript: str) -> dict:
    """Send transcript to Claude for sales-engineering analysis."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")

    client = anthropic.Anthropic(api_key=api_key)
    print("Analyzing transcript with Claude...")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": ANALYSIS_PROMPT.format(transcript=transcript)}
        ],
    )

    raw = message.content[0].text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(f"Failed to parse Claude response as JSON:\n{raw}")


def format_markdown(report: dict) -> str:
    """Render the analysis report as readable Markdown."""
    lines = [
        "# Voice Note Analysis Report",
        "",
        f"**Key Client:** {report['key_client']}",
        "",
        "## Executive Summary",
        "",
        report["summary"],
        "",
        "## Action Points",
        "",
    ]
    for ap in report["action_points"]:
        lines.append(f"- **{ap['task']}** — Owner: {ap['owner']} | Deadline: {ap['deadline']}")

    cp = report["closing_probability"]
    lines += [
        "",
        "## Closing Probability",
        "",
        f"**{cp['percentage']}%**",
        "",
        f"*{cp['reasoning']}*",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze a voice note for sales intelligence.")
    parser.add_argument("audio_file", help="Path to audio file (mp3, m4a, or wav)")
    parser.add_argument("--model", default=os.getenv("WHISPER_MODEL", "base"),
                        help="Whisper model size (default: base)")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown",
                        help="Output format (default: markdown)")
    parser.add_argument("--output", "-o", help="Write output to file instead of stdout")
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        sys.exit(f"File not found: {args.audio_file}")

    transcript = transcribe(args.audio_file, args.model)
    print(f"Transcription complete ({len(transcript.split())} words).\n")

    report = analyze(transcript)

    if args.format == "json":
        output = json.dumps(report, indent=2)
    else:
        output = format_markdown(report)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
