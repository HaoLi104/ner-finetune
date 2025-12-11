 #!/usr/bin/env python3
"""
Test file for main module.
"""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import main

def test_main():
    """Test the main function."""
    # This is a simple test to ensure the main function can be called
    try:
        main()
        assert True  # If we get here, the function executed without error
    except Exception as e:
        pytest.fail(f"Main function failed with error: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 