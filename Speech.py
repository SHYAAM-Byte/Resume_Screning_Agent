"""
speech.py — Speech I/O layer for the AI Interviewer.

Wraps:
  - Speech-to-text via OpenAI Whisper (WAV and MP3 input supported).
  - Text-to-speech via Piper TTS or Coqui TTS (returns WAV bytes).

Design notes:
  - All heavy models (Whisper / Piper voice / Coqui TTS) are lazily loaded
    and cached in module-level dicts so repeated calls don't reload them.
  - Every public function returns plain Python types (str / bytes) and
    raises one of the two typed exceptions below on failure — callers
    (e.g. interviewer.py) never need to catch library-specific errors.
  - No Streamlit / Ollama / database concerns live here — this module is
    reusable from a CLI script, a test, or any other front-end.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()
if TYPE_CHECKING:
    from whisper.model import Whisper
# Whisper is imported eagerly (it's the primary, always-used feature here),
# but guarded so importing speech.py never crashes if it isn't installed yet.
try:
    import whisper  # type: ignore
except ImportError:  # pragma: no cover - exercised only when dependency missing
    whisper = None  # noqa: N816


__all__ = [
    "TranscriptionError",
    "SpeechSynthesisError",
    "UnsupportedAudioFormatError",
    "speech_to_text",
    "text_to_speech",
    "convert_to_wav",
    "SUPPORTED_AUDIO_FORMATS",
    "DEFAULT_WHISPER_MODEL",
    "DEFAULT_TTS_ENGINE",
]

SUPPORTED_AUDIO_FORMATS = {"wav", "mp3"}
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_TTS_ENGINE = "piper"


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class TranscriptionError(Exception):
    """Raised when speech-to-text transcription fails for any reason."""


class UnsupportedAudioFormatError(TranscriptionError):
    """Raised when the provided audio isn't a supported WAV/MP3 file."""


class SpeechSynthesisError(Exception):
    """Raised when text-to-speech synthesis fails for any reason."""


# --------------------------------------------------------------------------
# Model caches (populated lazily on first use)
# --------------------------------------------------------------------------
_whisper_models: dict[str, Whisper] = {}
_piper_voices: Dict[str, object] = {}
_coqui_engines: Dict[str, object] = {}


# --------------------------------------------------------------------------
# Speech-to-text
# --------------------------------------------------------------------------
import logging
import os
import tempfile
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Configure your Groq API key
# -------------------------------------------------------
# Recommended:
# set GROQ_API_KEY=your_key (Windows)
#
# or place it in a .env and load it before importing.
# -------------------------------------------------------

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def speech_to_text(audio_bytes: bytes) -> str:
    """
    Convert speech audio into text using Groq Whisper API.

    Args:
        audio_bytes: Raw bytes of the uploaded/recorded audio.

    Returns:
        Transcribed text.

    Raises:
        RuntimeError
    """

    if not audio_bytes:
        raise ValueError("No audio data received.")

    tmp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        with open(tmp_path, "rb") as audio_file:

            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="verbose_json",
                language="en",
                temperature=0.0,
            )

        text = transcription.text.strip()

        logger.info("Speech transcription completed successfully.")

        return text

    except Exception as e:
        logger.exception("Speech transcription failed.")
        raise RuntimeError(f"Speech transcription failed: {e}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _get_whisper_model(model_size: str):
    if whisper is None:
        raise TranscriptionError(
            "The 'openai-whisper' package is not installed. "
            "Install it with `pip install openai-whisper`."
        )
    if model_size not in _whisper_models:
        try:
            logger.info("Loading Whisper model '%s' (first use only)...", model_size)
            _whisper_models[model_size] = whisper.load_model(model_size)
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(
                f"Failed to load Whisper model '{model_size}': {exc}"
            ) from exc
    return _whisper_models[model_size]


# --------------------------------------------------------------------------
# Text-to-speech
# --------------------------------------------------------------------------
def text_to_speech(
    text: str,
    engine: str = DEFAULT_TTS_ENGINE,
    voice: Optional[str] = None,
) -> bytes:
    """
    Synthesize speech audio (WAV bytes) from text using Piper or Coqui TTS.

    Args:
        text: The text to speak (e.g. an interview question).
        engine: Which TTS backend to use — "piper" (default) or "coqui".
        voice: Backend-specific voice identifier:
            - Piper: path to a `.onnx` voice model. Falls back to the
              `PIPER_VOICE_MODEL` environment variable if omitted.
            - Coqui: a model name (e.g. "tts_models/en/ljspeech/tacotron2-DDC").
              Falls back to the `COQUI_TTS_MODEL` environment variable, then
              a bundled default.

    Returns:
        WAV-encoded audio bytes.

    Raises:
        SpeechSynthesisError: If the text is empty, the engine is unknown,
            required dependencies are missing, or synthesis otherwise fails.
    """
    if not text or not text.strip():
        raise SpeechSynthesisError("No text was provided for speech synthesis.")

    engine_name = (engine or "").strip().lower()
    try:
        if engine_name == "piper":
            return _synthesize_with_piper(text, voice_model_path=voice)
        if engine_name == "coqui":
            return _synthesize_with_coqui(text, model_name=voice)
        raise SpeechSynthesisError(
            f"Unknown TTS engine '{engine}'. Supported engines: 'piper', 'coqui'."
        )
    except SpeechSynthesisError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Text-to-speech synthesis failed (engine=%s)", engine)
        raise SpeechSynthesisError(f"Speech synthesis failed: {exc}") from exc


def _synthesize_with_piper(text: str, voice_model_path: Optional[str]) -> bytes:
    try:
        from piper import PiperVoice  # type: ignore
    except ImportError as exc:
        raise SpeechSynthesisError(
            "The 'piper-tts' package is not installed. "
            "Install it with `pip install piper-tts`."
        ) from exc

    model_path = voice_model_path or os.environ.get("PIPER_VOICE_MODEL")
    if not model_path:
        raise SpeechSynthesisError(
            "No Piper voice model was specified. Pass `voice=<path to .onnx model>` "
            "or set the PIPER_VOICE_MODEL environment variable."
        )

    if model_path not in _piper_voices:
        try:
            logger.info("Loading Piper voice '%s' (first use only)...", model_path)
            _piper_voices[model_path] = PiperVoice.load(model_path)
        except Exception as exc:  # noqa: BLE001
            raise SpeechSynthesisError(
                f"Failed to load Piper voice '{model_path}': {exc}"
            ) from exc

    voice = _piper_voices[model_path]

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        voice.synthesize(text, wav_file)
    return buffer.getvalue()


def _synthesize_with_coqui(text: str, model_name: Optional[str]) -> bytes:
    try:
        from TTS.api import TTS  # type: ignore
    except ImportError as exc:
        raise SpeechSynthesisError(
            "The 'TTS' (Coqui) package is not installed. Install it with `pip install TTS`."
        ) from exc

    resolved_model = (
        model_name
        or os.environ.get("COQUI_TTS_MODEL")
        or "tts_models/en/ljspeech/tacotron2-DDC"
    )

    if resolved_model not in _coqui_engines:
        try:
            logger.info("Loading Coqui TTS model '%s' (first use only)...", resolved_model)
            _coqui_engines[resolved_model] = TTS(model_name=resolved_model, progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise SpeechSynthesisError(
                f"Failed to load Coqui TTS model '{resolved_model}': {exc}"
            ) from exc

    tts_engine = _coqui_engines[resolved_model]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        tts_engine.tts_to_file(text=text, file_path=tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        _safe_remove(tmp_path)


# --------------------------------------------------------------------------
# Audio format helpers
# --------------------------------------------------------------------------
def convert_to_wav(audio_bytes: bytes, source_format: Optional[str] = None) -> bytes:
    """
    Convert MP3 (or other ffmpeg-supported) audio bytes into WAV bytes.

    Useful when a downstream component (e.g. an audio player widget or a
    TTS pipeline step) requires WAV specifically.

    Args:
        audio_bytes: Raw source audio bytes.
        source_format: Explicit source format ("wav"/"mp3"). Auto-detected
            from the file header if omitted.

    Returns:
        WAV-encoded audio bytes.

    Raises:
        TranscriptionError: If pydub/ffmpeg isn't available or conversion fails.
    """
    try:
        from pydub import AudioSegment  # type: ignore
    except ImportError as exc:
        raise TranscriptionError(
            "The 'pydub' package is not installed. Install it with `pip install pydub` "
            "(requires ffmpeg on PATH)."
        ) from exc

    fmt = source_format or _detect_audio_format(audio_bytes)
    try:
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        out_buffer = io.BytesIO()
        segment.export(out_buffer, format="wav")
        return out_buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"Failed to convert audio to WAV: {exc}") from exc


def _resolve_audio_format(audio_bytes: bytes, filename: str) -> str:
    """Detect the audio format from file content, falling back to the filename."""
    try:
        return _detect_audio_format(audio_bytes)
    except UnsupportedAudioFormatError:
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext in SUPPORTED_AUDIO_FORMATS:
            return ext
        raise


def _detect_audio_format(audio_bytes: bytes) -> str:
    """Sniff WAV/MP3 from magic bytes. Raises if neither pattern matches."""
    if len(audio_bytes) < 4:
        raise UnsupportedAudioFormatError("Audio data is too short to be a valid file.")

    header = audio_bytes[:4]
    if header == b"RIFF":
        return "wav"
    if header[:3] == b"ID3" or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        return "mp3"

    raise UnsupportedAudioFormatError(
        "Unsupported audio format. Only WAV and MP3 are supported."
    )


def _safe_remove(path: Optional[str]) -> None:
    """Best-effort temp-file cleanup that never raises."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning("Could not remove temporary file: %s", path)