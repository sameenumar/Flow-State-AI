"""Configuration for pytest test discovery and execution."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Fixture examples
@pytest.fixture
def sample_image():
    """Provide a sample image for testing."""
    # TODO: Implement fixture to load/create sample image
    pass


@pytest.fixture
def sample_frame():
    """Provide a sample video frame for testing."""
    # TODO: Implement fixture to load/create sample frame
    pass


@pytest.fixture
def sample_face_region():
    """Provide a sample face region for testing."""
    # TODO: Implement fixture to load/create sample face region
    pass
