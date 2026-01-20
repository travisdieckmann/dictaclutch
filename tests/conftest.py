"""
Shared pytest fixtures for DictaClutch tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Callable, Any


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate test audio data (1 second of 440Hz sine wave)."""
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def short_audio() -> np.ndarray:
    """Generate short test audio (0.1 seconds)."""
    sample_rate = 16000
    samples = int(sample_rate * 0.1)
    t = np.linspace(0, 0.1, samples, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def mock_whisper_model() -> Mock:
    """Mock WhisperModel for transcription tests."""
    model = Mock()
    segment = Mock()
    segment.text = "Hello world"
    model.transcribe.return_value = ([segment], Mock())
    return model


@pytest.fixture
def mock_stream_factory() -> Callable[..., Mock]:
    """Factory for mock audio streams."""
    def _factory(**kwargs: Any) -> Mock:
        stream = Mock()
        stream.start = Mock()
        stream.stop = Mock()
        stream.close = Mock()
        return stream
    return _factory


@pytest.fixture
def mock_keyboard_listener() -> Mock:
    """Mock pynput keyboard listener."""
    listener = Mock()
    listener.start = Mock()
    listener.stop = Mock()
    return listener


@pytest.fixture
def default_config() -> dict:
    """Default configuration for tests."""
    from dictaclutch.config import DEFAULT_CONFIG
    return DEFAULT_CONFIG.copy()
