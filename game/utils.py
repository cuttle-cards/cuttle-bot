"""Utility functions for the Cuttle card game.

This module provides utility functions that are used across the game,
including logging and output handling functionality.
"""

import logging
from typing import Any


def log_print(*args: Any, **kwargs: Any) -> None:
    """Print output to console and log it using the game's logger.

    This function combines standard print functionality with logging,
    ensuring that all console output is also captured in the game's log.

    Args:
        *args: Variable number of arguments to print and log.
        **kwargs: Keyword arguments passed to the print function.

    Example:
        >>> log_print("Game started", "with", 2, "players")
        Game started with 2 players
        # Also logs: "Game started with 2 players"
    """
    logger = logging.getLogger("cuttle")
    message = " ".join(str(arg) for arg in args)
    logger.info(message)
