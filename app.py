import os
import streamlit as st
from dotenv import load_dotenv

from services.transcriber import transcribe_audio
from services.llm import stream_clean_transcript, stream_soap_note
from services.docx_export import transcript_to_docx, soap_to_docx

load_dotenv()

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedScribe AI",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS overrides (dark palette set via .streamlit/config.toml) ───────────────
st.markdown(
    """
    <style>
    /* Remove Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }

    /* Centered column */
    .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Sidebar border */
    [data-testid="stSidebar"] {
        border-right: 1px solid #1a1b2c;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.75rem; }

    /* Step label pill */
    .step-pill {
        display: inline-block;
        background: #1a1b35;
        color: #818cf8;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        padding: 3px 11px;
        border-radius: 100px;
        margin-bottom: 8px;
    }

    /* Thin section rule */
    .thin-rule {
        border: none;
        border-top: 1px solid #1a1b2c;
        margin: 1.5rem 0;
    }

    /* Metric cards — darker inset */
    [data-testid="metric-container"] {
        background: #0f1019;
        border: 1px solid #1a1b2c;
        border-radius: 8px;
        padding: 10px 14px;
    }

    /* File uploader drop zone */
    [data-testid="stFileUploaderDropzone"] {
        background: #0f1019 !important;
        border: 1px dashed #252640 !important;
        border-radius: 10px;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6366f1 !important;
    }

    /* Bordered containers (transcript / SOAP boxes) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #0f1019;
        border-color: #1e1f35 !important;
        border-radius: 10px;
        padding: 4px 8px;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #0f1019;
        border: 1px solid #1a1b2c !important;
        border-radius: 8px;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #6366f1;
        border: none;
        letter-spacing: 0.3px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background: #4f52d9;
        border: none;
    }

    /* Secondary / download button */
    .stDownloadButton > button {
        background: transparent;
        border: 1px solid #252640;
        color: #818cf8;
        font-size: 0.82rem;
    }
    .stDownloadButton > button:hover {
        border-color: #6366f1;
        color: #a5b4fc;
    }

    /* Start-over button (secondary) */
    .stButton > button:not([kind="primary"]) {
        background: transparent;
        border: 1px solid #252640;
        color: #565870;
        font-size: 0.82rem;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #3c3e5a;
        color: #818cf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── API key — loaded from .env only ──────────────────────────────────────────
_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "current_file_name": "",
    "raw_transcript": "",
    "clean_transcript": "",
    "soap_note": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### MedScribe AI")
    st.caption("Medical dictation → accurate transcript → SOAP note")
    st.divider()

    st.markdown("**Accepted formats**")
    st.markdown("MP3 · WAV · M4A · OGG · FLAC · MP4 · WEBM")
    st.caption("Maximum file size: 25 MB")

    st.divider()
    st.markdown("**AI pipeline**")
    st.markdown(
        "1. **gpt-4o-transcribe** — speech-to-text  \n"
        "2. **gpt-4.1-mini** — transcript cleanup  \n"
        "3. **gpt-4o** — SOAP note generation"
    )

    st.divider()
    has_file       = bool(st.session_state.current_file_name)
    has_transcript = bool(st.session_state.clean_transcript)
    has_soap       = bool(st.session_state.soap_note)

    def _dot(done: bool) -> str:
        return "●" if done else "○"

    if has_file:
        st.markdown(
            f"{_dot(True)} &nbsp; Audio uploaded  \n"
            f"{_dot(has_transcript)} &nbsp; Transcript ready  \n"
            f"{_dot(has_soap)} &nbsp; SOAP note ready",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Upload a file to begin.")


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# MedScribe AI")
st.markdown(
    "Upload a medical audio dictation, generate a clean transcript, "
    "then produce a structured SOAP note."
)
st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)

# Guard: API key must be present
if not _API_KEY:
    st.error(
        "**OPENAI_API_KEY not found.** "
        "Create a `.env` file in the project root with:  \n"
        "`OPENAI_API_KEY=sk-...`"
    )
    st.stop()


# ── Step 1 — Upload ───────────────────────────────────────────────────────────
st.markdown('<span class="step-pill">Step 1 — Upload audio</span>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop your audio file here or click to browse",
    type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "mpga", "mpeg", "webm"],
    label_visibility="collapsed",
)

if uploaded is not None:
    # Clear state when a different file is chosen
    if uploaded.name != st.session_state.current_file_name:
        st.session_state.current_file_name = uploaded.name
        st.session_state.raw_transcript    = ""
        st.session_state.clean_transcript  = ""
        st.session_state.soap_note         = ""

    # File metadata row
    size_bytes = uploaded.size
    size_str   = (
        f"{size_bytes / 1024:.1f} KB"
        if size_bytes < 1_048_576
        else f"{size_bytes / 1_048_576:.1f} MB"
    )
    ext = uploaded.name.rsplit(".", 1)[-1].upper() if "." in uploaded.name else "AUDIO"

    c1, c2, c3 = st.columns(3)
    label = uploaded.name if len(uploaded.name) <= 24 else uploaded.name[:21] + "..."
    c1.metric("File", label)
    c2.metric("Format", ext)
    c3.metric("Size", size_str)

    # ── Step 2 — Transcription ────────────────────────────────────────────────
    st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)
    st.markdown('<span class="step-pill">Step 2 — Transcription</span>', unsafe_allow_html=True)

    just_transcribed = False

    if not st.session_state.clean_transcript:
        if st.button("Transcribe Audio", type="primary", use_container_width=True):
            audio_bytes = uploaded.read()

            if len(audio_bytes) > 25 * 1024 * 1024:
                st.error(
                    "File exceeds the 25 MB OpenAI limit. "
                    "Please compress or trim the audio before uploading."
                )
                st.stop()

            try:
                with st.spinner("Transcribing with gpt-4o-transcribe..."):
                    raw = transcribe_audio(audio_bytes, uploaded.name, _API_KEY)
                    st.session_state.raw_transcript = raw

                st.caption("Cleaning transcript with gpt-4.1-mini...")
                with st.container(border=True):
                    clean = st.write_stream(
                        stream_clean_transcript(raw, _API_KEY)
                    )
                st.session_state.clean_transcript = str(clean)
                just_transcribed = True

            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
                st.stop()

    # Static display on subsequent renders
    if st.session_state.clean_transcript and not just_transcribed:
        with st.container(border=True):
            st.markdown(st.session_state.clean_transcript)

    if st.session_state.clean_transcript:
        dl1, dl2, _ = st.columns([1, 1, 3])
        dl1.download_button(
            "Download .docx",
            data=transcript_to_docx(st.session_state.clean_transcript),
            file_name="transcript.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        dl2.download_button(
            "Download .md",
            data=st.session_state.clean_transcript,
            file_name="transcript.md",
            mime="text/markdown",
        )

        with st.expander("View raw gpt-4o-transcribe output"):
            st.text(st.session_state.raw_transcript)

        # ── Step 3 — SOAP note ────────────────────────────────────────────────
        st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)
        st.markdown(
            '<span class="step-pill">Step 3 — SOAP note</span>', unsafe_allow_html=True
        )

        just_generated_soap = False

        if not st.session_state.soap_note:
            if st.button("Generate SOAP Note", type="primary", use_container_width=True):
                try:
                    st.caption("Structuring SOAP note with gpt-4o...")
                    with st.container(border=True):
                        soap = st.write_stream(
                            stream_soap_note(st.session_state.clean_transcript, _API_KEY)
                        )
                    st.session_state.soap_note = str(soap)
                    just_generated_soap = True

                except Exception as exc:
                    st.error(f"SOAP generation failed: {exc}")
                    st.stop()

        if st.session_state.soap_note and not just_generated_soap:
            with st.container(border=True):
                st.markdown(st.session_state.soap_note)

        if st.session_state.soap_note:
            dl3, dl4, _ = st.columns([1, 1, 3])
            dl3.download_button(
                "Download .docx",
                data=soap_to_docx(st.session_state.soap_note),
                file_name="soap_note.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            dl4.download_button(
                "Download .md",
                data=st.session_state.soap_note,
                file_name="soap_note.md",
                mime="text/markdown",
            )

        # ── Reset ─────────────────────────────────────────────────────────────
        st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)
        if st.button("Start over"):
            for _k in ["current_file_name", "raw_transcript", "clean_transcript", "soap_note"]:
                st.session_state[_k] = ""
            st.rerun()
