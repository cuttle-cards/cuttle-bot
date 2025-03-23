import sys
import os
from typing import List, Tuple

def is_interactive_terminal():
    """Check if we're in an interactive terminal."""
    # Check if we're in a test environment
    if 'pytest' in sys.modules or 'unittest' in sys.modules:
        return False
        
    # Check if we have a real terminal
    try:
        import termios
        return sys.stdin.isatty() and os.environ.get('TERM') is not None
    except (ImportError, AttributeError):
        return False

def get_terminal_size() -> Tuple[int, int]:
    """Get the dimensions of the terminal window."""
    try:
        return os.get_terminal_size()
    except OSError:
        return (80, 24)  # Default size

def clear_lines(num_lines: int):
    """Clear the specified number of lines above the cursor."""
    if is_interactive_terminal():
        for _ in range(num_lines):
            sys.stdout.write('\033[F')  # Move cursor up one line
            sys.stdout.write('\033[K')  # Clear line

def display_options(prompt: str, current_input: str, pre_filtered_options: List[str], filtered_options: List[str], selected_idx: int, max_display: int, terminal_width: int):
    """Helper function to display the prompt and options."""
    if is_interactive_terminal():
        # Clear the entire display area first
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
    """
    Display an interactive input prompt with filtered options that update as the user types.
    
    Args:
        prompt: The prompt to display to the user
        options: List of options to choose from
        
    Returns:
        The index of the selected option
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
            max_display = 10  # Maximum number of options to display at once
            
            # Get terminal width for proper alignment
            terminal_width, _ = get_terminal_size()
            
            # Initial display
            display_options(prompt, current_input, pre_filtered_options, filtered_options, selected_idx, max_display, terminal_width)
            
            while True:
                # Read a single character
                char = sys.stdin.read(1)
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
                display_options(prompt, current_input, pre_filtered_options, filtered_options, selected_idx, max_display, terminal_width)
        
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
    """Simple input handler for non-interactive environments."""
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