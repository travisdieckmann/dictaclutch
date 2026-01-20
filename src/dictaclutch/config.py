"""
DictaClutch configuration management.

This module provides configuration settings for the voice input application,
including hotkeys, model settings, audio settings, and streaming parameters.
"""

from dataclasses import dataclass, field
from typing import Any
from pynput import keyboard


# Default configuration dictionary
DEFAULT_CONFIG: dict[str, Any] = {
    # Batch mode hotkey: Ctrl+Shift+J
    "hotkey_batch": {
        keyboard.Key.alt,
        keyboard.Key.shift,
        keyboard.KeyCode.from_char("j"),
    },
    # Streaming mode hotkey: Ctrl+Shift+K
    "hotkey_streaming": {
        keyboard.Key.alt,
        keyboard.Key.shift,
        keyboard.KeyCode.from_char("k"),
    },
    # Whisper model: tiny, base, small, medium, large-v2, large-v3
    # Recommended for 4GB VRAM: small
    "model_size": "small",
    # Device: "cuda" for GPU, "cpu" for CPU-only
    "device": "cuda",
    # Compute type: float16 for GPU (faster), int8 for CPU, float32 for compatibility
    "compute_type": "float16",
    # Audio settings
    "sample_rate": 16000,
    "channels": 1,
    # Language: None for auto-detect, or "en", "es", "fr", etc.
    "language": "en",
    # Paste method: "clipboard" (Ctrl+V) or "type" (simulate keystrokes - slower but more compatible)
    "paste_method": "type",
    # Audio feedback
    "beep_on_start": True,
    "beep_on_stop": True,
    # Streaming mode settings
    "streaming_min_chunk": 0.5,  # Seconds between transcription attempts
    "streaming_buffer_max": 30.0,  # Maximum buffer size (seconds) before trimming
    # Noise reduction (streaming mode only, requires: pip install noisereduce)
    "noise_reduction_enabled": True,
    "noise_reduction_strength": 0.75,  # How aggressively to reduce noise (0.0 to 1.0)
}


@dataclass
class Config:
    """
    Configuration settings for DictaClutch.

    This dataclass provides typed access to configuration settings.
    """

    # Hotkeys
    hotkey_batch: set = field(default_factory=lambda: DEFAULT_CONFIG["hotkey_batch"])
    hotkey_streaming: set = field(
        default_factory=lambda: DEFAULT_CONFIG["hotkey_streaming"]
    )

    # Model settings
    model_size: str = "small"
    device: str = "cuda"
    compute_type: str = "float16"

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    language: str | None = "en"

    # Input method
    paste_method: str = "type"  # "clipboard" or "type"

    # Audio feedback
    beep_on_start: bool = True
    beep_on_stop: bool = True

    # Streaming settings
    streaming_min_chunk: float = 0.5
    streaming_buffer_max: float = 30.0

    # Noise reduction
    noise_reduction_enabled: bool = True
    noise_reduction_strength: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "hotkey_batch": self.hotkey_batch,
            "hotkey_streaming": self.hotkey_streaming,
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "language": self.language,
            "paste_method": self.paste_method,
            "beep_on_start": self.beep_on_start,
            "beep_on_stop": self.beep_on_stop,
            "streaming_min_chunk": self.streaming_min_chunk,
            "streaming_buffer_max": self.streaming_buffer_max,
            "noise_reduction_enabled": self.noise_reduction_enabled,
            "noise_reduction_strength": self.noise_reduction_strength,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            hotkey_batch=data.get("hotkey_batch", DEFAULT_CONFIG["hotkey_batch"]),
            hotkey_streaming=data.get(
                "hotkey_streaming", DEFAULT_CONFIG["hotkey_streaming"]
            ),
            model_size=data.get("model_size", "small"),
            device=data.get("device", "cuda"),
            compute_type=data.get("compute_type", "float16"),
            sample_rate=data.get("sample_rate", 16000),
            channels=data.get("channels", 1),
            language=data.get("language", "en"),
            paste_method=data.get("paste_method", "type"),
            beep_on_start=data.get("beep_on_start", True),
            beep_on_stop=data.get("beep_on_stop", True),
            streaming_min_chunk=data.get("streaming_min_chunk", 0.5),
            streaming_buffer_max=data.get("streaming_buffer_max", 30.0),
            noise_reduction_enabled=data.get("noise_reduction_enabled", True),
            noise_reduction_strength=data.get("noise_reduction_strength", 0.75),
        )


def validate_config(config: Config) -> list[str]:
    """
    Validate configuration settings.

    Returns a list of error messages (empty if valid).
    """
    errors = []

    if config.model_size not in (
        "tiny",
        "base",
        "small",
        "medium",
        "large-v2",
        "large-v3",
    ):
        errors.append(f"Invalid model_size: {config.model_size}")

    if config.device not in ("cuda", "cpu"):
        errors.append(f"Invalid device: {config.device}")

    if config.compute_type not in ("float16", "float32", "int8"):
        errors.append(f"Invalid compute_type: {config.compute_type}")

    if config.paste_method not in ("clipboard", "type"):
        errors.append(f"Invalid paste_method: {config.paste_method}")

    if not 0.0 <= config.noise_reduction_strength <= 1.0:
        errors.append(
            f"noise_reduction_strength must be 0.0-1.0: {config.noise_reduction_strength}"
        )

    if config.streaming_min_chunk <= 0:
        errors.append(
            f"streaming_min_chunk must be positive: {config.streaming_min_chunk}"
        )

    if config.streaming_buffer_max <= config.streaming_min_chunk:
        errors.append("streaming_buffer_max must be greater than streaming_min_chunk")

    return errors
