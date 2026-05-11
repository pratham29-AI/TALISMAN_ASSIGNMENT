  from typing import Generator
from openai import OpenAI

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_CLEAN_TRANSCRIPT_SYSTEM = """\
You are a senior medical transcriptionist. Your task is to take a raw \
speech-to-text output from a clinician's dictation and produce a clean, \
polished, human-readable version.

Rules:
- Preserve EVERY piece of clinical content exactly — no additions, omissions, \
  or alterations of any kind.
- Fix speech-recognition artifacts: run-on sentences, homophones, missing \
  punctuation.
- Apply standard medical abbreviations naturally (BP, HR, RR, SpO₂, BID, PRN, \
  QD, etc.) and normalise units ("millimeters of mercury" → mmHg, \
  "beats per minute" → bpm).
- Organise into clear logical paragraphs that follow the flow of the dictation.
- Output clean prose only — no bullet points, no headers, no SOAP structure.\
"""

_SOAP_SYSTEM = """\
You are a clinical documentation specialist. Convert the provided medical \
dictation transcript into a precise SOAP note.

────────────────────────────────────────
SECTION DEFINITIONS
────────────────────────────────────────
S — SUBJECTIVE
  Patient-reported information only:
  • Chief complaint (in the patient's words or as described on the patient's behalf)
  • History of present illness: onset, duration, character, severity, location,
    radiation, timing, modifying factors, associated symptoms the patient mentions
  • Current medications the patient reports taking
  • Relevant past medical / surgical / family / social history as reported

O — OBJECTIVE
  Clinician-observed or measured data only:
  • Vital signs (BP, HR, RR, Temp, SpO₂, weight / BMI)
  • Physical examination findings (inspection, palpation, auscultation, etc.)
  • Laboratory results and values
  • Imaging and diagnostic findings
  ⚠ CRITICAL: Patient-reported symptoms must NEVER appear in this section. \
    Only measurable, observable, clinician-recorded data belongs here.

A — ASSESSMENT
  • Primary diagnosis or most likely diagnosis
  • Differential diagnoses if stated
  • Brief clinical rationale tying S and O together

P — PLAN
  • Medications prescribed (name, dose, route, frequency, duration)
  • Non-pharmacological treatments (rest, ice, physical therapy, etc.)
  • Referrals and specialist consults
  • Diagnostic tests or imaging ordered
  • Follow-up instructions and timeline
  • Patient education or lifestyle guidance

────────────────────────────────────────
QUALITY CHECK — apply before finalising
────────────────────────────────────────
1. Confirm no patient-reported symptoms appear in the Objective section.
2. Confirm all four sections are populated with content from the transcript.
3. Confirm no clinical detail has been lost or fabricated.

────────────────────────────────────────
OUTPUT FORMAT — use exactly this Markdown structure:
────────────────────────────────────────

## S — Subjective

[content]

## O — Objective

[content]

## A — Assessment

[content]

## P — Plan

[content]\
"""


# ---------------------------------------------------------------------------
# Streaming generators
# ---------------------------------------------------------------------------

def stream_clean_transcript(raw_transcript: str, api_key: str) -> Generator[str, None, None]:
    """Yield GPT-4o token chunks that clean and polish the raw Whisper transcript."""
    client = OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": _CLEAN_TRANSCRIPT_SYSTEM},
            {"role": "user", "content": f"Raw transcript:\n\n{raw_transcript}"},
        ],
        stream=True,
        temperature=0.1,
        max_tokens=2048,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_soap_note(clean_transcript: str, api_key: str) -> Generator[str, None, None]:
    """Yield GPT-4o token chunks that structure the transcript into a SOAP note."""
    client = OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": _SOAP_SYSTEM},
            {"role": "user", "content": f"Medical dictation transcript:\n\n{clean_transcript}"},
        ],
        stream=True,
        temperature=0.1,
        max_tokens=2048,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
