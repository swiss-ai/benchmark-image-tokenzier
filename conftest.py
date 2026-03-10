"""
Pytest configuration for the benchmark-image-tokenizer project.

This file is automatically loaded by pytest and configures:
- Test collection behavior
- Shared fixtures
- Custom markers
"""

import os
import sys

import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Configure test collection to ignore certain directories
collect_ignore_glob = [
    "repos/*",  # Ignore all files in repos directory
    "**/*_backup.py",  # Ignore backup files
    "**/build/*",  # Ignore build directories
    "**/dist/*",  # Ignore distribution directories
]


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to test data directory."""
    return os.path.join(project_root, "tests", "data")


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Create a temporary directory for the test session."""
    return tmp_path_factory.mktemp("test_session")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
