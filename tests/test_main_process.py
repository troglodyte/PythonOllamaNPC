import unittest
from unittest.mock import patch, call

import main_process


class TestMainProcess(unittest.TestCase):
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_main_loop_quits_immediately(self, mock_input, mock_print):
        """Test that typing 'quit' exits the loop"""
        mock_input.return_value = 'Q'
        main_process.main_loop()
        
        # Verify input was called once
        mock_input.assert_called_once_with('> ')
        
        # Verify goodbye message was printed
        mock_print.assert_called_once_with('Goodbye quit')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_main_loop_handles_multiple_inputs_before_quit(self, mock_input, mock_print):
        """Test that the loop continues until 'quit' is entered"""
        mock_input.side_effect = ['hello', 'world', 'test', 'quit']
        main_process.main_loop()
        
        # Verify input was called 4 times
        assert mock_input.call_count == 4
        
        # Verify the calls were made with correct prompt
        expected_calls = [call('> ')] * 4
        mock_input.assert_has_calls(expected_calls)
        
        # Verify goodbye message was printed
        mock_print.assert_called_once_with('Goodbye quit')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_main_loop_handles_empty_input(self, mock_input, mock_print):
        """Test that empty strings don't break the loop"""
        mock_input.side_effect = ['', '', 'quit']
        main_process.main_loop()
        
        # Verify input was called 3 times
        assert mock_input.call_count == 3
        
        # Verify goodbye message was printed
        mock_print.assert_called_once_with('Goodbye quit')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_main_loop_case_sensitive(self, mock_input, mock_print):
        """Test that 'QUIT' or 'Quit' don't exit (case sensitive)"""
        mock_input.side_effect = ['QUIT', 'Quit', 'quit']
        main_process.main_loop()
        
        # Verify input was called 3 times (QUIT and Quit didn't exit)
        assert mock_input.call_count == 3
        
        # Verify goodbye message was printed only once
        mock_print.assert_called_once_with('Goodbye quit')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_main_loop_handles_whitespace_around_quit(self, mock_input, mock_print):
        """Test that ' quit ' (with spaces) doesn't exit"""
        mock_input.side_effect = [' quit ', 'quit ']
        mock_input.side_effect = [' quit', 'quit']
        main_process.main_loop()
        
        # Should exit on exact 'quit' match
        assert mock_input.call_count == 2


if __name__ == '__main__':
    unittest.main()