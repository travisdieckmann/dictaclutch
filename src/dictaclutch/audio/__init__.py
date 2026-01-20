"""Audio recording and feedback modules."""

from dictaclutch.audio.recorder import AudioRecorder, StreamingAudioRecorder
from dictaclutch.audio.feedback import (
    beep_batch_start,
    beep_batch_stop,
    beep_streaming_start,
    beep_streaming_stop,
    beep_error,
)
from dictaclutch.audio.noise import NoiseReducer

__all__ = [
    "AudioRecorder",
    "StreamingAudioRecorder",
    "NoiseReducer",
    "beep_batch_start",
    "beep_batch_stop",
    "beep_streaming_start",
    "beep_streaming_stop",
    "beep_error",
]
