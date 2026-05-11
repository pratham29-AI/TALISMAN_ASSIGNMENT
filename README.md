# MedScribe AI

A production-quality Python pipeline that ingests a medical audio dictation, produces an
accurate transcript, and structures it into a SOAP note — all served through a clean,
dark-themed Streamlit web interface.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Streamlit UI (app.py)                      │
│                                                                     │
│  [Upload audio]  ──►  [Transcribe]  ──►  [Generate SOAP Note]      │
└────────┬─────────────────┬───────────────────────┬─────────────────┘
         │                 │                       │
         ▼                 ▼                       ▼
   User's file     services/transcriber.py    services/llm.py
   (any audio       gpt-4o-transcribe API     gpt-4.1-mini  gpt-4o
    format)         ── raw transcript ──►      cleanup       SOAP gen
                                               (stream)      (stream)
```

### Three-stage pipeline

| Stage | Service | Model | Purpose |
|-------|---------|-------|---------|
| 1. Speech-to-Text | `services/transcriber.py` | `gpt-4o-transcribe` | Verbatim transcription with medical-term accuracy |
| 2. Transcript Cleanup | `services/llm.py` → `stream_clean_transcript` | `gpt-4.1-mini` | Removes STT artefacts, normalises units/abbreviations, produces readable prose |
| 3. SOAP Generation | `services/llm.py` → `stream_soap_note` | `gpt-4o` | Structures the clean transcript into labelled S / O / A / P sections |

Both LLM calls stream tokens back to the UI in real time using
`openai`'s `stream=True` and Streamlit's `st.write_stream`.

---

## Project Structure

```
TRANSCRIBER/
├── app.py                       # Streamlit application — UI logic and rendering
├── services/
│   ├── __init__.py
│   ├── transcriber.py           # gpt-4o-transcribe wrapper — audio → raw transcript
│   └── llm.py                   # LLM wrappers — transcript cleanup + SOAP generation
├── .streamlit/
│   └── config.toml              # Dark theme configuration for Streamlit
├── requirements.txt
├── .env                         # Your API key (not committed to version control)
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/account/api-keys) with access to
  `gpt-4o-transcribe`, `gpt-4.1-mini`, and `gpt-4o`

### Installation

```bash
# 1. Clone / unzip the project
cd TRANSCRIBER

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
echo OPENAI_API_KEY=sk-... > .env
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. **API key** is loaded automatically from `.env` at startup — no UI input required.
2. **Upload an audio file** — drag-and-drop or click to browse. Supported formats:
   MP3, WAV, M4A, OGG, FLAC, MP4, WEBM (max 25 MB, the OpenAI API limit).
3. Click **Transcribe Audio**.
   - `gpt-4o-transcribe` produces a verbatim raw transcript.
   - `gpt-4.1-mini` streams a cleaned, readable version in real time.
4. Click **Generate SOAP Note**.
   - `gpt-4o` streams the structured SOAP note directly to the page.
5. Use the **Download** buttons to save `transcript.txt` and `soap_note.md`.
6. Expand **"View raw gpt-4o-transcribe output"** to compare the raw STT output against
   the cleaned version — useful for evaluating transcription accuracy.

---

## Model Choices and Rationale

### `gpt-4o-transcribe` for STT

OpenAI's latest generation speech-to-text model, replacing the legacy `whisper-1`.

- **Higher accuracy on medical terminology** — `gpt-4o-transcribe` is built on the
  GPT-4o architecture, giving it significantly better contextual understanding of
  clinical vocabulary compared to Whisper.
- **Domain hint prompt** — The API accepts an optional `prompt` parameter that nudges
  the model toward specific spellings. We pre-seed it with common clinical terms and
  units (`tachycardia`, `mmHg`, `lisinopril`, etc.) so the decoder prefers correct
  medical spelling over phonetic approximations.
- **Same interface** — Identical endpoint (`client.audio.transcriptions.create`) and
  parameters as Whisper, making the upgrade a single-line model swap.
- **Format flexibility** — Accepts all common audio codecs without pre-processing.

### `gpt-4.1-mini` for transcript cleanup

- **Cost-efficient for a straightforward task** — cleaning punctuation and normalising
  units does not require a frontier model; `gpt-4.1-mini` delivers the same quality
  at lower latency and cost.
- **Follows strict instructions reliably** — the prompt explicitly forbids adding or
  removing clinical content; the mini model respects these constraints well.
- `temperature=0.1` keeps output deterministic.

### `gpt-4o` for SOAP generation

- **Medical knowledge depth** — `gpt-4o` has the broadest clinical knowledge of the
  three models used, which matters when the dictation is ambiguous about whether a
  finding is patient-reported or clinician-observed.
- **Instruction following** — reliably adheres to the strict Markdown output format
  and the S/O boundary rule enforced by the system prompt.
- `temperature=0.1` minimises hallucination — the model extracts and reorders, it
  does not invent.

---

## Prompt Engineering

### Transcript cleanup prompt (`_CLEAN_TRANSCRIPT_SYSTEM`)

The system prompt casts the model as a *transcriptionist*, not a summariser. Key
constraints:

- **No information loss** — "Preserve EVERY piece of clinical content exactly."
- **Artefact removal** — corrects homophones, run-ons, and missing punctuation that
  the STT model produces.
- **Unit normalisation** — converts spoken units to standard abbreviations.
- **No structure** — explicitly forbids bullet points and headers, keeping output
  as clean prose ready to feed into the SOAP prompt.

### SOAP generation prompt (`_SOAP_SYSTEM`)

The system prompt is a formal clinical documentation specification:

- Each section's **definition is strict and exclusive** — the model is told exactly
  what belongs in each section and, critically, what must *not* appear there
  (patient-reported symptoms must never appear in Objective).
- A **logic check** is embedded: before producing output the model is instructed to
  verify the S/O boundary, confirm all four sections are populated, and confirm no
  content was fabricated.
- The **output format is enforced** with explicit Markdown `##` headers so downstream
  rendering is predictable.

This approach directly implements the "Bonus" requirement from the assignment spec:
*"implement a logic check to ensure no Subjective information ends up in the
Objective section."*

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| API key via `.env` only | Keeps secrets out of the UI entirely; standard 12-factor app practice |
| `gpt-4o-transcribe` over `whisper-1` | OpenAI's latest STT model — higher accuracy on medical vocabulary |
| `gpt-4.1-mini` for cleanup, `gpt-4o` for SOAP | Right-sizes the model to the task — cost and latency savings on the simpler step |
| `.streamlit/config.toml` dark theme | Applies dark base to all native Streamlit widgets; CSS overrides only handle custom elements |
| Separate `services/` package | Clean separation between UI code and AI logic; services are independently testable |
| `stream=True` on both LLM calls | Real-time token streaming reduces perceived latency |
| `st.session_state` for persistence | Transcript and SOAP survive Streamlit reruns without re-calling the API |
| File-change detection | Uploading a new file automatically clears previous results to avoid stale state |
| `just_transcribed` / `just_generated_soap` flags | Prevents double-rendering on the same execution pass where streaming already happened |
| `temperature=0.1` on all LLM calls | Maximises determinism — essential for clinical accuracy |
| Raw transcript expander | Lets reviewers compare raw STT output vs. cleaned version side by side |

---

## Scaling Considerations

- **Chunked transcription** — for audio > 25 MB, the file can be split with
  `pydub` before sending to `gpt-4o-transcribe`. Transcripts are then concatenated.
- **Async processing** — for a multi-user deployment, `asyncio` + FastAPI can replace
  the synchronous Streamlit backend; the `services/` functions are already stateless.
- **Caching** — `@st.cache_data` on `transcribe_audio` with a file-hash key avoids
  re-calling the STT API for the same audio within a session.
- **Model swap** — swapping `gpt-4o-transcribe` for a local Faster-Whisper instance
  requires only a one-line change in `transcriber.py`; the rest of the pipeline is
  unaffected.
