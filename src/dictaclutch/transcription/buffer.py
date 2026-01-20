"""
Incremental buffer for smart diff corrections in streaming transcription.

This module tracks typed text and computes minimal corrections using
character-level and word-level diff algorithms.
"""


class IncrementalBuffer:
    """
    Tracks typed text and computes minimal corrections using character-level diff.

    Types text as it's spoken and makes minimal corrections when needed.
    Supports two strategies:
    - "backspace": Delete from end, retype (simpler)
    - "word_jump": Use Ctrl+Left to jump words, Ctrl+Backspace to delete (more efficient)
    """

    def __init__(self) -> None:
        """Initialize the incremental buffer."""
        self.typed_text: str = ""  # What's actually been typed to the screen
        self.transcript_history: list[str] = []  # Ring buffer of recent transcripts

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split()) if text.strip() else 0

    def _get_word_at_end(self, text: str, word_count: int) -> str:
        """Get the last N words from text."""
        words = text.split()
        if word_count >= len(words):
            return text
        return " ".join(words[-word_count:])

    def update(self, new_transcript: str) -> dict:
        """
        Compare new transcript with typed text and compute minimal edit.

        Uses word-based Ctrl+Arrow navigation for efficient middle-text edits.

        Strategies:
        - "append": Just add text at end (no changes needed)
        - "backspace": Delete from end, retype (fallback)
        - "word_jump": Use Ctrl+Left to jump words, Ctrl+Backspace to delete words

        Args:
            new_transcript: The new transcription text

        Returns:
            Dict with edit operations:
            {
                "strategy": "append" | "backspace" | "word_jump",
                "backspace": N,              # chars to delete (backspace strategy)
                "append": "text",            # text to add
                "word_jumps": N,             # Ctrl+Left count (word_jump strategy)
                "word_deletes": N,           # Ctrl+Backspace count (word_jump strategy)
                "word_insert": "text",       # text to insert after deletion
            }
        """
        if not new_transcript:
            return {"strategy": "append", "backspace": 0, "append": ""}

        # Normalize: strip and collapse whitespace
        new_transcript = " ".join(new_transcript.split())

        # Track history for debugging
        self.transcript_history.append(new_transcript)
        if len(self.transcript_history) > 10:
            self.transcript_history.pop(0)

        # If nothing typed yet, just type the new transcript
        if not self.typed_text:
            self.typed_text = new_transcript
            return {"strategy": "append", "backspace": 0, "append": new_transcript}

        old = self.typed_text
        new = new_transcript

        # Word-based analysis
        old_words = old.split()
        new_words = new.split()

        # Find common word prefix
        common_prefix_words = 0
        for i in range(min(len(old_words), len(new_words))):
            if old_words[i] == new_words[i]:
                common_prefix_words = i + 1
            else:
                break

        # Find common word suffix (but don't overlap with prefix)
        common_suffix_words = 0
        old_remaining = len(old_words) - common_prefix_words
        new_remaining = len(new_words) - common_prefix_words
        max_suffix = min(old_remaining, new_remaining)

        for i in range(1, max_suffix + 1):
            if old_words[-i] == new_words[-i]:
                common_suffix_words = i
            else:
                break

        # Calculate word-level changes
        old_middle_words = len(old_words) - common_prefix_words - common_suffix_words
        new_middle_words = (
            new_words[common_prefix_words : len(new_words) - common_suffix_words]
            if common_suffix_words > 0
            else new_words[common_prefix_words:]
        )
        new_middle_text = " ".join(new_middle_words)

        # Also do character-level for fallback
        prefix_len = 0
        min_len = min(len(old), len(new))
        for i in range(min_len):
            if old[i] == new[i]:
                prefix_len = i + 1
            else:
                break

        suffix_len = 0
        old_char_remaining = len(old) - prefix_len
        new_char_remaining = len(new) - prefix_len
        max_char_suffix = min(old_char_remaining, new_char_remaining)

        for i in range(1, max_char_suffix + 1):
            if old[-i] == new[-i]:
                suffix_len = i
            else:
                break

        old_middle_len = len(old) - prefix_len - suffix_len
        new_middle_char = (
            new[prefix_len : len(new) - suffix_len]
            if suffix_len > 0
            else new[prefix_len:]
        )
        suffix_text = new[-suffix_len:] if suffix_len > 0 else ""

        # Update what we consider "typed"
        self.typed_text = new_transcript

        # Calculate operation costs
        # Word jump: Ctrl+Left × (suffix_words) + Ctrl+Backspace × (old_middle_words) + type + End
        word_jump_ops = (
            common_suffix_words + old_middle_words + len(new_middle_text) + 1
        )

        # Backspace: backspace × (old_middle_chars + suffix_chars) + type
        backspace_ops = (
            (old_middle_len + suffix_len) + len(new_middle_char) + suffix_len
        )

        # Decide strategy
        # Use word_jump if:
        # 1. There are suffix words to preserve (common_suffix_words >= 1)
        # 2. There are middle words to delete (old_middle_words >= 1)
        # 3. Word approach is more efficient
        use_word_jump = (
            common_suffix_words >= 1
            and old_middle_words >= 1
            and word_jump_ops < backspace_ops
        )

        if use_word_jump:
            # Need to add trailing space if inserting before more words
            insert_text = new_middle_text
            if new_middle_text and common_suffix_words > 0:
                insert_text = new_middle_text + " "

            return {
                "strategy": "word_jump",
                "word_jumps": common_suffix_words,
                "word_deletes": old_middle_words,
                "word_insert": insert_text,
                "backspace": 0,
                "append": "",
            }
        else:
            return {
                "strategy": "backspace",
                "backspace": old_middle_len + suffix_len,
                "append": new_middle_char + suffix_text,
                "word_jumps": 0,
                "word_deletes": 0,
                "word_insert": "",
            }

    def get_stable_prefix(self) -> str:
        """
        Find text that's been stable across recent transcripts.

        Useful for knowing what we're confident about.

        Returns:
            Common prefix across recent transcripts
        """
        if len(self.transcript_history) < 2:
            return ""

        # Find common prefix across last few transcripts
        texts = self.transcript_history[-3:]  # Last 3
        if not texts:
            return ""

        prefix = texts[0]
        for text in texts[1:]:
            # Find common prefix
            common_len = 0
            for i in range(min(len(prefix), len(text))):
                if prefix[i] == text[i]:
                    common_len = i + 1
                else:
                    break
            prefix = prefix[:common_len]

        return prefix

    def reset(self) -> None:
        """Clear all state (on recording start/stop)."""
        self.typed_text = ""
        self.transcript_history.clear()
