"""
Batch transcription module for DictaClutch.

Provides the Transcriber class for batch (non-streaming) transcription
using faster-whisper.
"""

from typing import Any, Callable

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    """
    Transcribes audio using faster-whisper.

    Used for batch mode where audio is recorded first, then transcribed.
    """

    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        model_loader: Callable[..., WhisperModel] | None = None,
    ):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to use ("cuda" or "cpu")
            compute_type: Computation type (float16, float32, int8)
            model_loader: Optional custom model loader for testing
        """
        print(f"Loading Whisper model '{model_size}' on {device}...")
        if model_loader:
            self.model = model_loader(model_size, device=device, compute_type=compute_type)
        else:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded!")

    def transcribe(self, audio_data: np.ndarray, language: str | None = None) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Audio data as numpy array (float32)
            language: Language code or None for auto-detect

        Returns:
            Transcribed text
        """
        if len(audio_data) == 0:
            return ""

        # faster-whisper can accept numpy array directly
        segments, info = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=5,
            vad_filter=True,  # Filter out silence
        )

        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        return " ".join(text_parts).strip()
