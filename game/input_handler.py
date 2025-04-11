"""Input handling module for the Cuttle card game.

This module provides interactive and non-interactive input handling capabilities for the game.
It includes functions for terminal detection, size management, and user input processing with
filtered options display.

The module supports both interactive terminal environments and non-interactive environments
(such as testing or automated environments), automatically adapting its behavior accordingly.
"""

import sys
import os
from typing import List, Tuple
import errno

def is_interactive_terminal() -> bool:
    """Check if the current environment is an interactive terminal.

    This function determines if the program is running in an interactive terminal
    by checking for TTY presence and terminal attributes. It also handles special
    cases like test environments with pseudo-terminals.

    Returns:
        bool: True if running in an interactive terminal, False otherwise.
    """
    try:
        import termios
        # Only try to get terminal attributes if we have a TTY
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return False
        # Try to get terminal attributes, but don't actually use them
        try:
            fd = sys.stdin.fileno()
            termios.tcgetattr(fd)
            # Also check if we're in a test environment with a pseudo-terminal
            if 'pytest' in sys.modules:
                # If we got here in a test environment, we have a working terminal
                return True
            return os.environ.get('TERM') is not None
        except termios.error:
            return False
    except (ImportError, AttributeError):
        return False

def get_terminal_size() -> Tuple[int, int]:
    """Get the dimensions of the terminal window.

    Returns:
        Tuple[int, int]: A tuple containing (width, height) of the terminal.
        If the size cannot be determined, returns (80, 24) as default values.
    """
    try:
        return os.get_terminal_size()
    except OSError:
        return (80, 24)  # Default size

def clear_lines(num_lines: int) -> None:
    """Clear the specified number of lines above the cursor.

    This function uses ANSI escape sequences to move the cursor up and clear lines.
    It only performs the clearing operation if running in an interactive terminal.

    Args:
        num_lines: The number of lines to clear above the current cursor position.
    """
    if is_interactive_terminal():
        for _ in range(num_lines):
            sys.stdout.write('\033[F')  # Move cursor up one line
            sys.stdout.write('\033[K')  # Clear line

def display_options(prompt: str, current_input: str, pre_filtered_options: List[str], 
                   filtered_options: List[str], selected_idx: int, max_display: int, 
                   terminal_width: int, is_initial_display: bool = False) -> None:
    """Display the prompt and filtered options in an interactive manner.

    This function handles the visual display of the input prompt and available options.
    It supports both interactive and non-interactive modes, adapting the display
    accordingly.

    Args:
        prompt: The prompt text to display to the user.
        current_input: The current user input string.
        pre_filtered_options: List of options before filtering.
        filtered_options: List of options after filtering based on user input.
        selected_idx: Index of the currently selected option.
        max_display: Maximum number of options to display at once.
        terminal_width: Width of the terminal for proper alignment.
        is_initial_display: Whether this is the first display of options.
    """
    if is_interactive_terminal():
        # Clear the entire display area first
        if not is_initial_display:
            if len(pre_filtered_options) == 0:
                clear_lines(min(len(pre_filtered_options), max_display) + 2)
            else:
                clear_lines(min(len(pre_filtered_options), max_display) + 1)
        
        # Print prompt and current input
        sys.stdout.write(f"\r{prompt} {current_input}")
        sys.stdout.flush()
        
        # Print filtered options
        if filtered_options:
            sys.stdout.write("\n")
            for i, option in enumerate(filtered_options[:max_display]):
                prefix = "→ " if i == selected_idx else "  "
                # Clear the entire line first
                sys.stdout.write("\r" + " " * terminal_width + "\r")
                # Write the option aligned to the left
                sys.stdout.write(f"{prefix}{option}\n")
        else:
            sys.stdout.write("\nNo matching options\n")
        sys.stdout.flush()
    else:
        # In non-interactive mode, just print the options once
        print(f"{prompt}")
        for i, option in enumerate(filtered_options[:max_display]):
            print(f"{i}: {option}")

def get_interactive_input(prompt: str, options: List[str]) -> int:
    """Get user input interactively with filtered options display.

    This function provides an interactive input interface where options are filtered
    as the user types. It supports arrow key navigation and handles special keys
    like Enter, Backspace, and Escape sequences.

    Args:
        prompt: The prompt text to display to the user.
        options: List of available options to choose from.

    Returns:
        int: The index of the selected option in the original options list.

    Raises:
        KeyboardInterrupt: If the user presses Ctrl+C.
    """
    # If we're in a test environment or non-interactive terminal, use simple input
    if not is_interactive_terminal():
        return get_non_interactive_input(prompt, options)
        
    try:
        import termios
        import tty
        
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            # Set terminal to raw mode
            tty.setraw(sys.stdin)
            
            # Initialize variables
            current_input = ""
            filtered_options = options
            pre_filtered_options = options
            selected_idx = 0
            max_display = 20  # Maximum number of options to display at once
            
            # Get terminal width for proper alignment
            terminal_width, _ = get_terminal_size()
            
            # Initial display
            display_options(prompt=prompt, current_input=current_input, pre_filtered_options=pre_filtered_options, filtered_options=filtered_options, selected_idx=selected_idx, max_display=max_display, terminal_width=terminal_width, is_initial_display=True)
            
            while True:
                try:
                    # Read a single character
                    char = sys.stdin.read(1)
                    if not char:  # EOF
                        break
                    pre_filtered_options = filtered_options
                    # Handle special keys
                    if ord(char) == 3:  # Ctrl+C
                        raise KeyboardInterrupt
                    elif ord(char) == 13:  # Enter
                        if filtered_options:
                            # Find the original index of the selected option
                            selected_option = filtered_options[selected_idx]
                            original_idx = options.index(selected_option)
                            return original_idx
                    elif ord(char) == 127:  # Backspace
                        if current_input:
                            current_input = current_input[:-1]
                            # Update filtered options
                            filtered_options = [opt for opt in options if current_input.lower() in opt.lower()]
                            selected_idx = 0
                    elif ord(char) == 27:  # Escape sequence
                        next_char = sys.stdin.read(1)
                        if next_char == '[':  # Arrow keys
                            key = sys.stdin.read(1)
                            if key == 'A':  # Up arrow
                                selected_idx = (selected_idx - 1) % len(filtered_options) if filtered_options else 0
                            elif key == 'B':  # Down arrow
                                selected_idx = (selected_idx + 1) % len(filtered_options) if filtered_options else 0
                    elif ord(char) >= 32:  # Printable characters
                        current_input += char
                        # Update filtered options
                        filtered_options = [opt for opt in options if current_input.lower() in opt.lower()]
                        selected_idx = 0
                    
                    # Refresh display after any change
                    display_options(prompt=prompt, current_input=current_input, pre_filtered_options=pre_filtered_options, filtered_options=filtered_options, selected_idx=selected_idx, max_display=max_display, terminal_width=terminal_width, is_initial_display=False)
                except (IOError, OSError) as e:
                    if e.errno == errno.EAGAIN:  # Resource temporarily unavailable
                        continue
                    raise
        
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            # Clear the input area
            if filtered_options:
                clear_lines(min(len(filtered_options), max_display) + 2)
            else:
                clear_lines(2)
            sys.stdout.write("\n")
    except (ImportError, AttributeError, termios.error):
        # Fallback to non-interactive mode if terminal control is not available
        return get_non_interactive_input(prompt, options)

def get_non_interactive_input(prompt: str, options: List[str]) -> int:
    """Get user input in a non-interactive environment.

    This function provides a simple input interface for non-interactive environments
    like testing or automated environments. It displays all options and accepts
    either option text or index numbers as input.

    Args:
        prompt: The prompt text to display to the user.
        options: List of available options to choose from.

    Returns:
        int: The index of the selected option, or -1 for end game command.
    """
    # Display options
    display_options(prompt, "", options, options, 0, len(options), 80)
    
    # Get input (this will use the mocked input in tests)
    response = input().strip()
    
    # Handle 'e' or 'end game' for end game
    if response.lower() in ['e', 'end game']:
        return -1
        
    # Try to match the input against the options
    # First try exact match
    for i, option in enumerate(options):
        if response.lower() in option.lower():
            return i
            
    # Then try to match just the number
    try:
        index = int(response)
        if 0 <= index < len(options):
            return index
    except ValueError:
        pass
        
    # If no match found, return the first option
    return 0
