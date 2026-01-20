"""
Noise reduction for DictaClutch streaming mode.

Uses the noisereduce library for stationary noise reduction.
"""

import numpy as np

# Optional noise reduction (install with: pip install noisereduce)
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    nr = None  # type: ignore
    NOISEREDUCE_AVAILABLE = False


class NoiseReducer:
    """
    Real-time noise reduction for audio streams using noisereduce library.

    Reduces stationary background noise (fans, AC, hum) from audio.
    """

    def __init__(self, sample_rate: int = 16000, strength: float = 0.75):
        """
        Initialize noise reducer.

        Args:
            sample_rate: Audio sample rate in Hz
            strength: How aggressively to reduce noise (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.strength = max(0.0, min(1.0, strength))  # Clamp to [0, 1]
        self.enabled = NOISEREDUCE_AVAILABLE

        if not NOISEREDUCE_AVAILABLE:
            print("Noise reduction unavailable (install with: pip install noisereduce)")

    def reduce(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to audio.

        Args:
            audio: Audio data as numpy array (float32, mono)

        Returns:
            Noise-reduced audio
        """
        if not self.enabled or len(audio) == 0 or nr is None:
            return audio

        try:
            # Use stationary noise reduction (faster, good for constant background noise)
            # prop_decrease controls how much noise is reduced (0 = none, 1 = full)
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=self.strength,
                n_fft=512,  # Smaller FFT for lower latency
                hop_length=128,
            )
            return reduced.astype(np.float32)
        except Exception as e:
            # If noise reduction fails, return original audio
            print(f"Noise reduction error: {e}")
            return audio
