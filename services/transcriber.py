import os
import tempfile
from openai import OpenAI

# Whisper benefits from a domain hint prompt — improves medical term accuracy
_WHISPER_PROMPT = (
    "Medical dictation. Clinical terminology: tachycardia, hypertension, dyspnea, "
    "myocardial infarction, ecchymosis, auscultation, palpation, edema, erythema, "
    "stenosis, prednisone, metformin, lisinopril, atorvastatin. "
    "Units: mmHg, bpm, mg/dL, SpO2, BMI."
)


def transcribe_audio(audio_bytes: bytes, filename: str, api_key: str) -> str:
    """Send audio to gpt-4o-transcribe and return the raw transcript string."""
    client = OpenAI(api_key=api_key)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp3"

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f,
                response_format="text",
                prompt=_WHISPER_PROMPT,
            )
        return str(response)
    finally:
        os.unlink(tmp_path)
