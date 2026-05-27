"""
Quiet Loading Utilities

Suppresses verbose output during model initialization for a clean user experience.
Third-party libraries (transformers, diffusers, torch, huggingface_hub) often
produce extensive logging during model loading. This module provides utilities
to temporarily suppress that output, including loguru logger messages.

Usage:
    # As a context manager
    with quiet_mode():
        model = AutoModel.from_pretrained(...)

    # Global quiet mode
    enable_quiet_mode()
    # ... load models ...
    disable_quiet_mode()

    # Check if quiet mode is active
    if is_quiet_mode():
        ...
"""

import os
import sys
import logging
from contextlib import contextmanager
from io import StringIO
from typing import Optional, List

# Track whether quiet mode is globally enabled
_quiet_mode_enabled = False

# Store original logging levels to restore later
_original_log_levels: dict = {}

# Store original loguru handlers to restore later
_original_loguru_handlers: list = []

# Libraries that produce verbose output during model loading
_VERBOSE_LIBRARIES = [
    "transformers",
    "diffusers",
    "huggingface_hub",
    "torch",
    "torchaudio",
    "urllib3",
    "filelock",
    "PIL",
    "numpy",
    "tokenizers",
    "accelerate",
    "safetensors",
]


def is_quiet_mode() -> bool:
    """Check if quiet mode is currently enabled."""
    return _quiet_mode_enabled


def _suppress_library_logging(verbose_libraries: Optional[List[str]] = None):
    """Suppress logging output from verbose third-party libraries.

    Sets logging level to ERROR for known verbose libraries to suppress
    INFO and WARNING messages during model loading.

    Args:
        verbose_libraries: List of library names to suppress. If None, uses default list.
    """
    global _original_log_levels

    libraries = verbose_libraries or _VERBOSE_LIBRARIES

    for lib_name in libraries:
        logger = logging.getLogger(lib_name)
        if lib_name not in _original_log_levels:
            _original_log_levels[lib_name] = logger.level
        logger.setLevel(logging.ERROR)


def _restore_library_logging():
    """Restore original logging levels for all suppressed libraries."""
    global _original_log_levels

    for lib_name, level in _original_log_levels.items():
        logger = logging.getLogger(lib_name)
        logger.setLevel(level)

    _original_log_levels.clear()


def _suppress_loguru():
    """Suppress loguru logger output by replacing handlers with a null sink.
    
    Loguru is used extensively in the codebase for model loading messages.
    During quiet mode, we redirect loguru output to a null sink so that
    only ERROR level messages are shown.
    """
    global _original_loguru_handlers
    
    try:
        from loguru import logger
        
        # Save current handlers
        _original_loguru_handlers = list(logger._core.handlers.values())
        
        # Remove all existing handlers
        handler_ids = list(logger._core.handlers.keys())
        for handler_id in handler_ids:
            logger.remove(handler_id)
        
        # Add a handler that only shows ERROR level and above
        logger.add(
            sys.stderr,
            level="ERROR",
            format="{message}",
        )
    except ImportError:
        pass


def _restore_loguru():
    """Restore loguru logger handlers to their original state."""
    global _original_loguru_handlers
    
    try:
        from loguru import logger
        
        # Remove all current handlers (including our error-only one)
        handler_ids = list(logger._core.handlers.keys())
        for handler_id in handler_ids:
            logger.remove(handler_id)
        
        # Restore original handlers
        for handler in _original_loguru_handlers:
            try:
                logger.add(
                    handler._sink,
                    level=handler.levelno,
                    format=handler._format,
                    filter=handler._filter,
                    colorize=handler._colorize,
                    serialize=handler._serialize,
                    backtrace=handler._backtrace,
                    diagnose=handler._diagnose,
                    enqueue=handler._enqueue,
                    catch=handler._catch,
                )
            except Exception:
                pass
        
        _original_loguru_handlers.clear()
    except ImportError:
        pass


def _suppress_tqdm():
    """Disable tqdm progress bars globally."""
    try:
        from tqdm import tqdm
        tqdm.disable = True
    except ImportError:
        pass


def _restore_tqdm():
    """Re-enable tqdm progress bars."""
    try:
        from tqdm import tqdm
        tqdm.disable = False
    except ImportError:
        pass


def enable_quiet_mode(
    suppress_stdout: bool = True,
    suppress_logging: bool = True,
    suppress_tqdm: bool = True,
    suppress_loguru: bool = True,
):
    """Enable quiet mode to suppress verbose model loading output.

    Args:
        suppress_stdout: Redirect stdout to suppress print() statements
        suppress_logging: Set verbose libraries to ERROR level
        suppress_tqdm: Disable tqdm progress bars
        suppress_loguru: Suppress loguru logger messages (set to ERROR level)
    """
    global _quiet_mode_enabled

    if _quiet_mode_enabled:
        return  # Already enabled

    _quiet_mode_enabled = True

    # Suppress third-party library logging
    if suppress_logging:
        _suppress_library_logging()

    # Suppress loguru logger
    if suppress_loguru:
        _suppress_loguru()

    # Suppress tqdm
    if suppress_tqdm:
        _suppress_tqdm()

    # Suppress stdout if requested
    if suppress_stdout:
        sys.stdout = StringIO()

    # Also set TRANSFORMERS_VERBOSITY and HF_HUB_DISABLE_PROGRESS_BARS
    if suppress_logging:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["DIFFUSERS_VERBOSITY"] = "error"


def disable_quiet_mode():
    """Disable quiet mode and restore all output."""
    global _quiet_mode_enabled

    if not _quiet_mode_enabled:
        return  # Not enabled

    _quiet_mode_enabled = False

    # Restore stdout
    if isinstance(sys.stdout, StringIO):
        sys.stdout = sys.__stdout__

    # Restore library logging
    _restore_library_logging()

    # Restore loguru
    _restore_loguru()

    # Restore tqdm
    _restore_tqdm()

    # Restore environment variables
    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    os.environ.pop("DIFFUSERS_VERBOSITY", None)


@contextmanager
def quiet_mode(
    suppress_stdout: bool = True,
    suppress_logging: bool = True,
    suppress_tqdm: bool = True,
    suppress_loguru: bool = True,
):
    """Context manager for quiet model loading.

    Suppresses verbose output from third-party libraries and loguru during
    model initialization. Errors and exceptions are still printed to stderr.

    Args:
        suppress_stdout: Redirect stdout to suppress print() statements
        suppress_logging: Set verbose libraries to ERROR level
        suppress_tqdm: Disable tqdm progress bars
        suppress_loguru: Suppress loguru logger messages (set to ERROR level)

    Example:
        >>> with quiet_mode():
        ...     model = AutoModel.from_pretrained("ACE-Step/Ace-Step1.5")
        >>> # Normal output resumes here
    """
    _prev_stdout = sys.stdout
    _prev_quiet = _quiet_mode_enabled

    try:
        enable_quiet_mode(
            suppress_stdout=suppress_stdout,
            suppress_logging=suppress_logging,
            suppress_tqdm=suppress_tqdm,
            suppress_loguru=suppress_loguru,
        )
        yield
    finally:
        # Restore to previous state
        if not _prev_quiet:
            disable_quiet_mode()
        sys.stdout = _prev_stdout


def quiet_print(message: str, **kwargs):
    """Print a message only when quiet mode is NOT active.

    Use this for informational messages that should be hidden during quiet loading.

    Args:
        message: The message to print
        **kwargs: Additional arguments passed to print()
    """
    if not _quiet_mode_enabled:
        print(message, **kwargs)
