"""
Main test file for testing game functionality through main.py.
Card-specific tests have been moved to separate files:
- King tests: test_main_king.py
- Six tests: test_main_six.py
- Ace tests: test_main_ace.py
- Three tests: test_main_three.py
"""

import unittest

from tests.test_main.test_main_base import MainTestBase


class TestMain(MainTestBase):
    """Base test class for main.py functionality.
    Card-specific tests have been moved to separate test files."""

    pass


if __name__ == "__main__":
    unittest.main()
