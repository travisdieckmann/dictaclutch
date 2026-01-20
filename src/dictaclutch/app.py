"""
Main DictaClutch application module.

Coordinates recording, transcription, and text input for both
batch and streaming modes.
"""

import threading
import time
from typing import Any

import numpy as np

from dictaclutch.config import DEFAULT_CONFIG
from dictaclutch.audio.recorder import AudioRecorder, StreamingAudioRecorder
from dictaclutch.audio.feedback import (
    beep_batch_start,
    beep_batch_stop,
    beep_streaming_start,
    beep_streaming_stop,
    beep_error,
)
from dictaclutch.transcription.transcriber import Transcriber
from dictaclutch.transcription.streaming import StreamingTranscriber
from dictaclutch.hotkeys.handler import MultiHotkeyHandler
from dictaclutch.input.text_output import (
    paste_text,
    type_text,
    send_backspace,
    send_ctrl_left,
    send_ctrl_backspace,
    send_end_key,
    get_foreground_window,
)


class DictaClutchApp:
    """
    Main application coordinating recording, transcription, and input.

    Supports two modes:
    - Batch mode: Record audio, transcribe, paste (Ctrl+Shift+J)
    - Streaming mode: Real-time transcription with corrections (Ctrl+Shift+K)
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        recorder: AudioRecorder | None = None,
        streaming_recorder: StreamingAudioRecorder | None = None,
        transcriber: Transcriber | None = None,
        hotkey_handler: MultiHotkeyHandler | None = None,
    ):
        """
        Initialize the application.

        Args:
            config: Configuration dictionary (defaults to DEFAULT_CONFIG)
            recorder: AudioRecorder instance (created if None)
            streaming_recorder: StreamingAudioRecorder instance (created if None)
            transcriber: Transcriber instance (lazy loaded if None)
            hotkey_handler: MultiHotkeyHandler instance (created if None)
        """
        self.config = config or DEFAULT_CONFIG

        # Batch mode recorder
        self.recorder = recorder or AudioRecorder(
            sample_rate=self.config["sample_rate"],
            channels=self.config["channels"],
        )
        self.transcriber = transcriber  # Lazy load (shared model)

        # Streaming mode components
        self.streaming_recorder = streaming_recorder or StreamingAudioRecorder(
            sample_rate=self.config["sample_rate"],
            channels=self.config["channels"],
        )
        self.streaming_transcriber: StreamingTranscriber | None = None  # Lazy load
        self.stop_event = threading.Event()  # For clean shutdown

        # Mode tracking: None, "batch", or "streaming"
        self.mode: str | None = None
        self.target_window: int | None = None  # Window handle for focus tracking

        # Setup multi-hotkey handler
        if hotkey_handler:
            self.hotkey_handler = hotkey_handler
        else:
            hotkey_configs = {
                "batch": self.config["hotkey_batch"],
                "streaming": self.config["hotkey_streaming"],
            }
            self.hotkey_handler = MultiHotkeyHandler(hotkey_configs)

        self.hotkey_handler.register_callback("batch", self.toggle_batch)
        self.hotkey_handler.register_callback("streaming", self.toggle_streaming)
        self.hotkey_handler.register_exit_callback(self._on_exit)

        # Batch mode state
        self.is_recording = False
        self.processing = False
        self.lock = threading.Lock()

        # Exit flag
        self.exit_requested = False

    def _on_exit(self) -> None:
        """Handle ESC key press - clean shutdown."""
        print("\nESC pressed - exiting...")
        self.exit_requested = True
        # Stop any active mode
        if self.mode == "streaming":
            self._stop_streaming()
        elif self.mode == "batch":
            self.recorder.stop()
        self.hotkey_handler.stop()

    def load_model(self) -> None:
        """Load the Whisper model."""
        self.transcriber = Transcriber(
            model_size=self.config["model_size"],
            device=self.config["device"],
            compute_type=self.config["compute_type"],
        )
        # Initialize streaming transcriber with the same model
        self.streaming_transcriber = StreamingTranscriber(
            model=self.transcriber.model,
            config=self.config,
        )

    def toggle_batch(self) -> None:
        """Toggle batch recording mode (Ctrl+Shift+J)."""
        with self.lock:
            # If streaming is active, stop it first
            if self.mode == "streaming":
                self._stop_streaming()

            if self.processing:
                return  # Ignore if still processing previous recording

            if self.mode != "batch":
                self._start_batch()
            else:
                self._stop_batch()

    def toggle_streaming(self) -> None:
        """Toggle streaming mode (Ctrl+Shift+K)."""
        with self.lock:
            # If batch is active, stop it first
            if self.mode == "batch":
                self._stop_batch()

            if self.mode != "streaming":
                self._start_streaming()
            else:
                self._stop_streaming()

    def _start_batch(self) -> None:
        """Start batch recording mode."""
        print("\n[BATCH] Recording... (press Ctrl+Shift+J to stop)")
        self.mode = "batch"
        self.is_recording = True
        self.target_window = get_foreground_window()

        # Play beep BEFORE starting recorder (so beep isn't captured)
        beep_batch_start()
        time.sleep(0.1)  # Let echo die down

        self.recorder.start()

    def _stop_batch(self) -> None:
        """Stop batch recording and process."""
        print("[BATCH] Stopped recording, transcribing...")
        self.is_recording = False
        self.processing = True
        self.mode = None

        # Get audio data BEFORE playing beep (so beep isn't captured)
        audio_data = self.recorder.stop()

        beep_batch_stop()

        # Process in background thread
        thread = threading.Thread(target=self._process_batch_audio, args=(audio_data,))
        thread.start()

    def _start_streaming(self) -> None:
        """Start streaming mode with real-time transcription."""
        # Ensure model is loaded
        if self.streaming_transcriber is None:
            print("Model not loaded yet!")
            return

        print("\n[STREAMING] Mode started (press Ctrl+Shift+K to stop)")
        self.mode = "streaming"
        self.target_window = get_foreground_window()

        # Reset streaming components
        self.stop_event.clear()
        self.streaming_transcriber.reset()

        # Play beep BEFORE starting recorder (so beep isn't captured)
        beep_streaming_start()
        time.sleep(0.1)  # Let echo die down

        # Start audio recording
        self.streaming_recorder.start()

        # Start worker thread
        thread = threading.Thread(target=self._streaming_worker, daemon=True)
        thread.start()

    def _stop_streaming(self) -> None:
        """Stop streaming mode."""
        print("[STREAMING] Mode stopped")

        # Signal worker to stop
        self.stop_event.set()

        # Stop recording
        self.streaming_recorder.stop()

        self.mode = None
        beep_streaming_stop()

    def _streaming_worker(self) -> None:
        """Background worker for real-time streaming transcription with smart diff corrections."""
        last_transcribe_time = time.time()

        while not self.stop_event.is_set():
            # Get audio chunks from recorder
            chunk = self.streaming_recorder.get_chunk(timeout=0.1)

            if chunk is not None and self.streaming_transcriber:
                self.streaming_transcriber.add_chunk(chunk)

            # Check if we should transcribe (time-based, no VAD wait)
            current_time = time.time()
            time_since_last = current_time - last_transcribe_time

            if (
                self.streaming_transcriber
                and self.streaming_transcriber.should_transcribe()
                and time_since_last >= self.config["streaming_min_chunk"]
            ):
                try:
                    # Transcribe and get incremental edit
                    result = self.streaming_transcriber.transcribe_buffer()
                    strategy = result.get("strategy", "backspace")

                    # Check if target window is still focused
                    if (
                        self.target_window
                        and get_foreground_window() != self.target_window
                    ):
                        has_changes = (
                            result.get("backspace", 0)
                            or result.get("append", "")
                            or result.get("word_deletes", 0)
                            or result.get("word_insert", "")
                        )
                        if has_changes:
                            print("    [paused - window not focused]")
                    else:
                        if strategy == "word_jump":
                            # Word-jump strategy: Ctrl+Left to jump words, Ctrl+Backspace to delete
                            word_jumps = result.get("word_jumps", 0)
                            word_deletes = result.get("word_deletes", 0)
                            word_insert = result.get("word_insert", "")

                            if word_jumps > 0 or word_deletes > 0 or word_insert:
                                # Jump backward by words
                                send_ctrl_left(word_jumps)
                                # Delete words
                                send_ctrl_backspace(word_deletes)
                                # Insert replacement
                                if word_insert:
                                    type_text(word_insert)
                                # Return to end
                                send_end_key()
                                print(
                                    f"    [Ctrl+Left x{word_jumps} Ctrl+Back x{word_deletes} +'{word_insert.strip()}' End]"
                                )

                        else:
                            # Backspace strategy: delete from end, retype
                            backspace_count = result.get("backspace", 0)
                            text_to_append = result.get("append", "")

                            if backspace_count > 0:
                                send_backspace(backspace_count)
                                print(f"    [-{backspace_count}]", end="")

                            if text_to_append:
                                type_text(text_to_append)
                                display = (
                                    text_to_append
                                    if len(text_to_append) <= 30
                                    else text_to_append[:27] + "..."
                                )
                                print(f" +'{display}'")
                            elif backspace_count > 0:
                                print()  # Newline after correction-only

                    last_transcribe_time = current_time

                    # Trim buffer if getting too large
                    self.streaming_transcriber.trim_buffer()

                except Exception as e:
                    print(f"    Streaming error: {e}")
                    beep_error()
                    # Stop streaming on error
                    self.stop_event.set()
                    break

    def _process_batch_audio(self, audio_data: np.ndarray) -> None:
        """Process recorded audio for batch mode (runs in background thread)."""
        try:
            if len(audio_data) < self.config["sample_rate"] * 0.3:  # Less than 0.3 seconds
                print("Recording too short, ignoring.")
                return

            if self.transcriber is None:
                print("Model not loaded!")
                return

            # Transcribe
            start_time = time.time()
            text = self.transcriber.transcribe(audio_data, self.config["language"])
            elapsed = time.time() - start_time

            if text:
                # Add space after ending punctuation for better flow
                if text and text[-1] in ".!?":
                    text += " "

                print(f"Transcribed ({elapsed:.2f}s): {text}")

                # Small delay to ensure hotkey is fully released
                time.sleep(0.15)

                # Check if target window is still focused
                if self.target_window and get_foreground_window() != self.target_window:
                    print("Window focus changed, skipping paste")
                else:
                    # Paste text
                    paste_text(text)
            else:
                print("No speech detected.")

        except Exception as e:
            print(f"Error during transcription: {e}")
            beep_error()

        finally:
            with self.lock:
                self.processing = False

    def run(self) -> None:
        """Run the application."""
        print("=" * 60)
        print("DictaClutch - Voice-to-Text Input Tool")
        print("=" * 60)
        print()

        # Load model
        self.load_model()

        # Format hotkeys for display
        def format_hotkey(hotkey_set: set) -> str:
            return " + ".join(
                (
                    k.name
                    if hasattr(k, "name")
                    else (k.char if hasattr(k, "char") else str(k))
                )
                for k in hotkey_set
            )

        batch_hotkey = format_hotkey(self.config["hotkey_batch"])
        streaming_hotkey = format_hotkey(self.config["hotkey_streaming"])

        print(f"\nReady!")
        print(f"   [{batch_hotkey}] - Batch mode (record -> transcribe -> paste)")
        print(f"   [{streaming_hotkey}] - Streaming mode (real-time transcription)")
        print("Press ESC or Ctrl+C to exit.\n")

        # Start hotkey listener
        self.hotkey_handler.start()

        try:
            # Keep main thread alive until exit requested
            while not self.exit_requested:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nCtrl+C - exiting...")
            self._on_exit()
