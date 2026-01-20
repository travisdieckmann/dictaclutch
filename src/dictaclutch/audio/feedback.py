"""
Audio feedback functions for DictaClutch.

Provides beep sounds to indicate recording state changes.
"""

from typing import Callable, Any

import numpy as np
import sounddevice as sd

from dictaclutch.config import DEFAULT_CONFIG


def generate_beep(
    frequency: float,
    duration: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Generate a simple sine wave beep.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Audio sample rate (default: 44100)

    Returns:
        Audio data as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t) * 0.3
    # Apply fade in/out to avoid clicks
    fade_samples = int(sample_rate * 0.01)
    wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
    wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return wave.astype(np.float32)


def play_beep(
    frequency: float = 440,
    duration: float = 0.1,
    audio_player: Any | None = None,
) -> None:
    """
    Play a beep sound.

    Args:
        frequency: Frequency in Hz (default: 440)
        duration: Duration in seconds (default: 0.1)
        audio_player: Optional audio player for testing (default: sounddevice)
    """
    try:
        beep = generate_beep(frequency, duration)
        player = audio_player or sd
        player.play(beep, samplerate=44100)
    except Exception as e:
        print(f"Could not play beep: {e}")


def generate_tone_sequence(
    frequencies: list[float],
    tone_duration: float = 0.10,
    gap_duration: float = 0.03,
) -> np.ndarray:
    """
    Generate a sequence of tones with gaps between them.

    Args:
        frequencies: List of frequencies for each tone
        tone_duration: Duration of each tone in seconds
        gap_duration: Duration of gap between tones in seconds

    Returns:
        Concatenated audio data as numpy array
    """
    sample_rate = 44100
    tones = []
    gap_samples = int(sample_rate * gap_duration)
    gap = np.zeros(gap_samples, dtype=np.float32)

    for i, freq in enumerate(frequencies):
        tone = generate_beep(freq, tone_duration, sample_rate)
        tones.append(tone)
        if i < len(frequencies) - 1:  # Add gap between tones, not after last
            tones.append(gap)

    return np.concatenate(tones)


def play_tone_sequence(
    frequencies: list[float],
    audio_player: Any | None = None,
) -> None:
    """
    Play a sequence of tones.

    Args:
        frequencies: List of frequencies for each tone
        audio_player: Optional audio player for testing (default: sounddevice)
    """
    try:
        sequence = generate_tone_sequence(frequencies)
        player = audio_player or sd
        player.play(sequence, samplerate=44100)
        if hasattr(player, 'wait'):
            player.wait()  # Block until playback finishes
    except Exception as e:
        print(f"Could not play beep sequence: {e}")


def beep_batch_start(audio_player: Any | None = None) -> None:
    """Beep indicating batch recording started (4-tone sequence, lower range)."""
    if DEFAULT_CONFIG["beep_on_start"]:
        play_tone_sequence([523, 392, 440, 330], audio_player=audio_player)


def beep_batch_stop(audio_player: Any | None = None) -> None:
    """Beep indicating batch recording stopped (4-tone sequence, lower range)."""
    if DEFAULT_CONFIG["beep_on_stop"]:
        play_tone_sequence([330, 440, 392, 523], audio_player=audio_player)


def beep_streaming_start(audio_player: Any | None = None) -> None:
    """Beep indicating streaming started (4-tone sequence, higher range)."""
    if DEFAULT_CONFIG["beep_on_start"]:
        play_tone_sequence([784, 587, 659, 523], audio_player=audio_player)


def beep_streaming_stop(audio_player: Any | None = None) -> None:
    """Beep indicating streaming stopped (4-tone sequence, higher range)."""
    if DEFAULT_CONFIG["beep_on_stop"]:
        play_tone_sequence([523, 659, 587, 784], audio_player=audio_player)


def beep_error(audio_player: Any | None = None) -> None:
    """Beep indicating an error."""
    play_beep(220, 0.3, audio_player=audio_player)
