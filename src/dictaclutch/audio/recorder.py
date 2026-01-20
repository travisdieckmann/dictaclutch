"""
Audio recording modules for DictaClutch.

Provides two recorder classes:
- AudioRecorder: Batch recording (accumulates to list)
- StreamingAudioRecorder: Queue-based recording for real-time streaming
"""

import queue
import wave
from typing import Any, Callable

import numpy as np
import sounddevice as sd


class AudioRecorder:
    """
    Records audio from the default microphone for batch processing.

    Audio is accumulated to a list and returned when stopped.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        stream_factory: Callable[..., Any] | None = None,
    ):
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
            stream_factory: Factory for creating audio streams (for testing)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream_factory = stream_factory or sd.InputStream
        self.recording = False
        self.audio_data: list[np.ndarray] = []
        self.stream: Any = None

        # Log the audio input device
        try:
            default_input = sd.query_devices(kind="input")
            print(f"Audio input device: {default_input['name']}")
        except Exception as e:
            print(f"Could not query audio device: {e}")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags | None,
    ) -> None:
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        if self.recording:
            self.audio_data.append(indata.copy())

    def start(self) -> None:
        """Start recording."""
        self.audio_data = []
        self.recording = True
        self.stream = self._stream_factory(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return audio data."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_data:
            return np.array([], dtype=np.float32)

        return np.concatenate(self.audio_data, axis=0).flatten()

    def save_wav(self, audio_data: np.ndarray, filepath: str) -> None:
        """Save audio data to a WAV file."""
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())


class StreamingAudioRecorder:
    """
    Queue-based audio recorder for real-time streaming.

    Audio chunks are placed in a thread-safe queue for consumption
    by the streaming transcriber.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        stream_factory: Callable[..., Any] | None = None,
    ):
        """
        Initialize the streaming audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
            stream_factory: Factory for creating audio streams (for testing)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream_factory = stream_factory or sd.InputStream
        self.chunk_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stream: Any = None
        self.recording = False

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags | None,
    ) -> None:
        """Callback for audio stream - adds chunks to queue."""
        if status:
            print(f"Audio status: {status}")
        if self.recording:
            self.chunk_queue.put(indata.copy())

    def start(self) -> None:
        """Start recording."""
        self.recording = True
        # Clear any old chunks
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break

        self.stream = self._stream_factory(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop(self) -> None:
        """Stop recording."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        """Get next audio chunk (non-blocking with timeout)."""
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None
