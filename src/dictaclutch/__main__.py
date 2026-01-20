#!/usr/bin/env python3
"""
DictaClutch CLI entry point.

Usage:
    dictaclutch              # Run the main voice input application
    dictaclutch --diagnose   # Run hotkey diagnostic tool
    dictaclutch --version    # Show version
"""

import argparse
import sys


def main() -> None:
    """Main entry point for DictaClutch."""
    parser = argparse.ArgumentParser(
        prog="dictaclutch",
        description="Voice-to-text input tool using Whisper with real-time streaming",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run hotkey diagnostic tool to validate key bindings",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output (all key events in diagnostic mode)",
    )

    args = parser.parse_args()

    if args.diagnose:
        _run_diagnostic(verbose=args.verbose)
    else:
        _run_app()


def _get_version() -> str:
    """Get package version."""
    from dictaclutch import __version__
    return __version__


def _run_diagnostic(verbose: bool = False) -> None:
    """Run the hotkey diagnostic tool."""
    from dictaclutch.hotkeys.diagnostic import run_diagnostic
    run_diagnostic(verbose=verbose)


def _run_app() -> None:
    """Run the main DictaClutch application."""
    # Check for CUDA availability
    try:
        import torch
        from dictaclutch.config import DEFAULT_CONFIG

        if DEFAULT_CONFIG["device"] == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            DEFAULT_CONFIG["device"] = "cpu"
            DEFAULT_CONFIG["compute_type"] = "int8"
        elif DEFAULT_CONFIG["device"] == "cuda":
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        from dictaclutch.config import DEFAULT_CONFIG
        if DEFAULT_CONFIG["device"] == "cuda":
            print("WARNING: PyTorch not found, falling back to CPU")
            DEFAULT_CONFIG["device"] = "cpu"
            DEFAULT_CONFIG["compute_type"] = "int8"

    from dictaclutch.app import DictaClutchApp
    app = DictaClutchApp()
    app.run()


if __name__ == "__main__":
    main()
