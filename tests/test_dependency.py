"""
Tests for the dependency verification functionality.
"""
import os
import pytest
import tempfile
from unittest import mock

# Add the project root to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pallets import paths
from pallets.dependency import verify_cpunks_dependency


class TestDependencyVerification:
    """Test the dependency verification functionality."""
    
    def test_correct_paths(self):
        """Test that the dependency verification passes with correct paths."""
        # This test should pass if the actual paths are correctly set up
        assert verify_cpunks_dependency(raise_error=False) is True
    
    def test_nonexistent_cpunks_path(self):
        """Test that verification fails with a nonexistent path."""
        # Mock paths.CPUNKS_ROOT to a nonexistent path
        with mock.patch('pallets.paths.CPUNKS_ROOT', '/nonexistent/path'):
            # Should return False with raise_error=False
            assert verify_cpunks_dependency(raise_error=False) is False
            
            # Should raise FileNotFoundError with raise_error=True
            with pytest.raises(FileNotFoundError):
                verify_cpunks_dependency(raise_error=True)
    
    def test_missing_images_dir(self):
        """Test that verification fails with missing images directory."""
        # Create a temporary directory to simulate cpunks root
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data directory but no images directory
            os.makedirs(os.path.join(temp_dir, 'data'))
            
            # Mock paths.CPUNKS_ROOT to the temp directory
            with mock.patch('pallets.paths.CPUNKS_ROOT', temp_dir):
                # Should return False with raise_error=False
                assert verify_cpunks_dependency(raise_error=False) is False
                
                # Should raise FileNotFoundError with raise_error=True
                with pytest.raises(FileNotFoundError):
                    verify_cpunks_dependency(raise_error=True)
    
    def test_missing_data_file(self):
        """Test that verification fails with missing data file."""
        # Create a temporary directory to simulate cpunks root
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create images/training directory but not data/punks.json
            os.makedirs(os.path.join(temp_dir, 'images', 'training'))
            
            # Create some dummy image files
            for i in range(200):
                open(os.path.join(temp_dir, 'images', 'training', f'punk{i:04d}.png'), 'w').close()
            
            # Mock paths.CPUNKS_ROOT to the temp directory
            with mock.patch('pallets.paths.CPUNKS_ROOT', temp_dir):
                # Should return False with raise_error=False
                assert verify_cpunks_dependency(raise_error=False) is False
                
                # Should raise FileNotFoundError with raise_error=True
                with pytest.raises(FileNotFoundError):
                    verify_cpunks_dependency(raise_error=True)
                    
    def test_env_variable_override(self):
        """Test that environment variable overrides default path."""
        # Save original value
        original_cpunks_root = paths.CPUNKS_ROOT
        
        try:
            # Set environment variable to a new path
            with mock.patch.dict(os.environ, {'CPUNKS_ROOT_DIR': '/custom/path'}):
                # Reload paths module to pick up environment variable
                import importlib
                importlib.reload(paths)
                
                # Check that CPUNKS_ROOT is updated
                assert paths.CPUNKS_ROOT == '/custom/path'
        finally:
            # Clean up - reset CPUNKS_ROOT to original value
            paths.CPUNKS_ROOT = original_cpunks_root