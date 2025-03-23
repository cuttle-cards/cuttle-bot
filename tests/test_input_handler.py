import unittest
from unittest.mock import patch
import sys
import io
import termios
import tty
import os
import re
from game.input_handler import get_interactive_input

class TestInputHandler(unittest.TestCase):
    def setUp(self):
        # Create a list of test options with index prefixes
        self.test_options = [
            "0: King of Hearts",
            "1: King of Diamonds", 
            "2: Queen of Hearts",
            "3: Queen of Spades",
            "4: Ace of Clubs"
        ]
        
        # Save original stdout and create StringIO for capturing output
        self.original_stdout = sys.stdout
        self.stdout_capture = io.StringIO()
        sys.stdout = self.stdout_capture

        # Create mock terminal settings
        self.mock_termios_settings = termios.tcgetattr(sys.__stdout__.fileno())

        # Number of cleanup characters needed (based on max display lines + prompt)
        self.cleanup_chars = ['\r'] * 12  # Increased from 8 to 12 for more thorough cleanup

    def tearDown(self):
        # Restore original stdout
        sys.stdout = self.original_stdout
        self.stdout_capture.close()

    def get_captured_output(self):
        """Helper to get captured output and reset the buffer"""
        output = self.stdout_capture.getvalue()
        self.stdout_capture.truncate(0)
        self.stdout_capture.seek(0)
        return output

    def clean_ansi(self, text):
        """Remove ANSI escape sequences from text"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def get_last_display(self, output):
        """Get the last displayed state after all updates"""
        displays = output.split("Select a card:")
        return displays[-1] if displays else ""

    def setup_terminal_mocks(self, mock_stdin):
        """Helper to set up terminal-related mocks"""
        mock_stdin.isatty.return_value = True
        mock_stdin.fileno.return_value = 0  # Use 0 as mock file descriptor
        
        # Create mock terminal size tuple
        mock_terminal_size = os.terminal_size((80, 24))
        
        # Patch get_terminal_size in the input_handler module
        patcher = patch('game.input_handler.get_terminal_size', return_value=mock_terminal_size)
        patcher.start()
        self.addCleanup(patcher.stop)
        
        # Patch termios.tcgetattr
        patcher = patch('termios.tcgetattr', return_value=self.mock_termios_settings)
        patcher.start()
        self.addCleanup(patcher.stop)
        
        # Patch termios.tcsetattr
        patcher = patch('termios.tcsetattr')
        patcher.start()
        self.addCleanup(patcher.stop)
        
        # Patch tty.setraw
        patcher = patch('tty.setraw')
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch('game.input_handler.is_interactive_terminal')
    @patch('sys.stdin')
    def test_filtering_as_typing(self, mock_stdin, mock_is_interactive):
        """Test that options are filtered correctly as user types"""
        # Mock interactive terminal
        mock_is_interactive.return_value = True
        self.setup_terminal_mocks(mock_stdin)
        
        # Simulate typing 'king' then Enter, plus cleanup characters
        mock_stdin.read.side_effect = ['k', 'i', 'n', 'g', '\r'] + self.cleanup_chars
        
        # Run the input handler
        selected = get_interactive_input("Select a card:", self.test_options)
        
        # Get captured output and clean ANSI sequences
        output = self.clean_ansi(self.get_captured_output())
        last_display = self.get_last_display(output)
        
        # Verify filtering behavior
        self.assertIn("King of Hearts", last_display)
        self.assertIn("King of Diamonds", last_display)
        self.assertNotIn("Queen of Hearts", last_display)
        self.assertEqual(selected, 0)  # First King should be selected

    @patch('game.input_handler.is_interactive_terminal')
    @patch('sys.stdin')
    def test_arrow_key_navigation(self, mock_stdin, mock_is_interactive):
        """Test arrow key navigation between options"""
        # Mock interactive terminal
        mock_is_interactive.return_value = True
        self.setup_terminal_mocks(mock_stdin)
        
        # Simulate: type 'k' then down arrow then Enter, plus cleanup characters
        mock_stdin.read.side_effect = [
            'k',  # Type 'k'
            '\x1b', '[', 'B',  # Down arrow
            '\r'  # Enter
        ] + self.cleanup_chars
        
        selected = get_interactive_input("Select a card:", self.test_options)
        
        # Get captured output and clean ANSI sequences
        output = self.clean_ansi(self.get_captured_output())
        last_display = self.get_last_display(output)
        
        # Verify second King was selected
        self.assertEqual(selected, 1)  # Should select King of Diamonds
        
        # Verify both Kings were shown in output
        self.assertIn("King of Hearts", last_display)
        self.assertIn("King of Diamonds", last_display)

    @patch('game.input_handler.is_interactive_terminal')
    @patch('sys.stdin')
    def test_backspace_handling(self, mock_stdin, mock_is_interactive):
        """Test handling of backspace key"""
        # Mock interactive terminal
        mock_is_interactive.return_value = True
        self.setup_terminal_mocks(mock_stdin)
        
        # Simulate: type 'queen', backspace twice, type 'g', Enter, plus cleanup
        # Add extra control characters for screen updates
        input_sequence = [
            'q', '\r',  # Type 'q' and redraw
            'u', '\r',  # Type 'u' and redraw
            'e', '\r',  # Type 'e' and redraw
            'e', '\r',  # Type 'e' and redraw
            'n', '\r',  # Type 'n' and redraw
            '\x7f', '\r',  # First backspace and redraw
            '\x7f', '\r',  # Second backspace and redraw
            'g', '\r',  # Type 'g' and redraw
            '\r'  # Final Enter
        ] + self.cleanup_chars + ['\r'] * 4  # Extra cleanup chars
        
        mock_stdin.read.side_effect = input_sequence
        
        selected = get_interactive_input("Select a card:", self.test_options)
        
        # Get captured output and clean ANSI sequences
        output = self.clean_ansi(self.get_captured_output())
        last_display = self.get_last_display(output)
        
        # After backspace, "queg" should match "Queen"
        self.assertIn("Queen", last_display)
        self.assertEqual(selected, 2)  # Should select first Queen

    @patch('game.input_handler.is_interactive_terminal')
    @patch('sys.stdin')
    def test_ctrl_c_handling(self, mock_stdin, mock_is_interactive):
        """Test handling of Ctrl+C (interrupt)"""
        # Mock interactive terminal
        mock_is_interactive.return_value = True
        self.setup_terminal_mocks(mock_stdin)
        
        # Simulate Ctrl+C (ASCII value 3)
        mock_stdin.read.side_effect = ['\x03']
        
        # Verify KeyboardInterrupt is raised
        with self.assertRaises(KeyboardInterrupt):
            get_interactive_input("Select a card:", self.test_options)

    def test_non_interactive_terminal(self):
        """Test fallback behavior for non-interactive terminals"""
        with patch('builtins.input', return_value='0'):
            selected = get_interactive_input("Select a card:", self.test_options)
            self.assertEqual(selected, 0)

        with patch('builtins.input', return_value='king'):
            selected = get_interactive_input("Select a card:", self.test_options)
            self.assertEqual(selected, 0)  # Should match first king

    @patch('game.input_handler.is_interactive_terminal')
    @patch('sys.stdin')
    def test_empty_filter_results(self, mock_stdin, mock_is_interactive):
        """Test behavior when filter matches no options"""
        # Mock interactive terminal
        mock_is_interactive.return_value = True
        self.setup_terminal_mocks(mock_stdin)
        
        # Simulate typing 'xyz' then backspace to all then 'k' then Enter, plus cleanup
        mock_stdin.read.side_effect = [
            'x', 'y', 'z',  # Type "xyz" (no matches)
            '\x7f', '\x7f', '\x7f',  # Backspace all
            'k',  # Type "k"
            '\r'  # Enter
        ] + self.cleanup_chars
        
        selected = get_interactive_input("Select a card:", self.test_options)
        
        # Get captured output and clean ANSI sequences
        output = self.clean_ansi(self.get_captured_output())
        last_display = self.get_last_display(output)
        
        # Should show "No matching options" then recover
        self.assertIn("No matching options", output)
        self.assertIn("King", last_display)  # After backspace and 'k'
        self.assertEqual(selected, 0)  # Should select first King 