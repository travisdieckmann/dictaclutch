"""
Streaming transcription module for DictaClutch.

Provides real-time streaming transcription with incremental output
and smart diff corrections.
"""

from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from dictaclutch.transcription.buffer import IncrementalBuffer
from dictaclutch.audio.noise import NoiseReducer


class StreamingTranscriber:
    """
    Real-time streaming transcription with incremental output.

    Transcribes at fixed intervals and uses smart diff for minimal corrections.
    """

    def __init__(
        self,
        model: WhisperModel,
        config: dict[str, Any],
        noise_reducer: NoiseReducer | None = None,
    ):
        """
        Initialize the streaming transcriber.

        Args:
            model: WhisperModel instance to use for transcription
            config: Configuration dictionary with streaming settings
            noise_reducer: Optional NoiseReducer instance for noise reduction
        """
        self.model = model
        self.sample_rate = config["sample_rate"]
        self.min_chunk_size = config["streaming_min_chunk"]
        self.max_buffer_size = config["streaming_buffer_max"]
        self.language = config["language"]

        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)

        # Noise reduction (optional)
        if noise_reducer is not None:
            self.noise_reducer = noise_reducer
        elif config.get("noise_reduction_enabled", False):
            self.noise_reducer = NoiseReducer(
                sample_rate=self.sample_rate,
                strength=config.get("noise_reduction_strength", 0.75),
            )
        else:
            self.noise_reducer = None

        # Incremental buffer for smart diff corrections
        self.incremental_buffer = IncrementalBuffer()

        # Tracking
        self.last_transcript = ""

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Add audio chunk to rolling buffer."""
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk.flatten()])

    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self.audio_buffer) / self.sample_rate

    def should_transcribe(self) -> bool:
        """Check if we have enough audio to transcribe (time-based, no VAD wait)."""
        return self.get_buffer_duration() >= self.min_chunk_size

    def transcribe_buffer(self) -> dict:
        """
        Transcribe current buffer and return incremental edit operations.

        Returns:
            Dict with edit operations from IncrementalBuffer.update()
        """
        if len(self.audio_buffer) == 0:
            return {"backspace": 0, "append": ""}

        # Apply noise reduction if enabled
        audio_for_transcription = self.audio_buffer
        if self.noise_reducer is not None:
            audio_for_transcription = self.noise_reducer.reduce(self.audio_buffer)

        # Transcribe
        segments, _ = self.model.transcribe(
            audio_for_transcription,
            language=self.language,
            beam_size=5,
            vad_filter=False,  # Don't filter - we want real-time output
        )

        # Collect transcript
        transcript = " ".join(seg.text for seg in segments).strip()
        self.last_transcript = transcript

        # Get incremental edit (smart diff)
        return self.incremental_buffer.update(transcript)

    def trim_buffer(self) -> None:
        """Trim buffer to max size, keeping recent audio."""
        max_samples = int(self.max_buffer_size * self.sample_rate)
        if len(self.audio_buffer) > max_samples:
            # Keep the most recent audio
            self.audio_buffer = self.audio_buffer[-max_samples:]

    def reset(self) -> None:
        """Clear all buffers."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.incremental_buffer.reset()
        self.last_transcript = ""
