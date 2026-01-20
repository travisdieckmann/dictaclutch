"""
DictaClutch - Voice-to-text input tool using Whisper with real-time streaming.

This package provides:
- Batch mode: Record audio, transcribe, paste (Ctrl+Shift+J)
- Streaming mode: Real-time transcription with corrections (Ctrl+Shift+K)
- Hotkey diagnostic tool for validating key bindings
"""

__version__ = "1.0.0"
__all__ = [
    "DictaClutchApp",
    "Config",
    "DEFAULT_CONFIG",
    "AudioRecorder",
    "StreamingAudioRecorder",
    "Transcriber",
    "StreamingTranscriber",
    "run_diagnostic",
]

# Lazy imports to avoid loading heavy dependencies until needed
def __getattr__(name: str):
    if name == "DictaClutchApp":
        from dictaclutch.app import DictaClutchApp
        return DictaClutchApp
    if name in ("Config", "DEFAULT_CONFIG"):
        from dictaclutch import config
        return getattr(config, name)
    if name == "AudioRecorder":
        from dictaclutch.audio.recorder import AudioRecorder
        return AudioRecorder
    if name == "StreamingAudioRecorder":
        from dictaclutch.audio.recorder import StreamingAudioRecorder
        return StreamingAudioRecorder
    if name == "Transcriber":
        from dictaclutch.transcription.transcriber import Transcriber
        return Transcriber
    if name == "StreamingTranscriber":
        from dictaclutch.transcription.streaming import StreamingTranscriber
        return StreamingTranscriber
    if name == "run_diagnostic":
        from dictaclutch.hotkeys.diagnostic import run_diagnostic
        return run_diagnostic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
