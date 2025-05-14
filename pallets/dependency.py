"""
Dependency verification and management utilities.
"""
import os
import logging
from . import paths

logger = logging.getLogger("pallets")

def verify_cpunks_dependency(raise_error=True):
    """
    Verify that the cpunks-10k repository is available and properly set up.
    
    Args:
        raise_error (bool): If True, raise FileNotFoundError when dependencies are missing.
                           If False, just return False when dependencies are missing.
    
    Returns:
        bool: True if dependencies are available, False otherwise (when raise_error=False)
        
    Raises:
        FileNotFoundError: If dependencies are missing and raise_error=True
    """
    # Check if CPUNKS_ROOT directory exists
    if not os.path.exists(paths.CPUNKS_ROOT):
        msg = (
            f"CPUNKS_ROOT directory not found at {paths.CPUNKS_ROOT}. "
            "Please clone the cpunks-10k repository in the parent directory "
            "or set the CPUNKS_ROOT_DIR environment variable."
        )
        logger.error(msg)
        if raise_error:
            raise FileNotFoundError(msg)
        return False
    
    # Check if images directory exists and has files
    images_dir = os.path.join(paths.CPUNKS_ROOT, 'images', 'training')
    if not os.path.exists(images_dir):
        msg = (
            f"Training images directory not found at {images_dir}. "
            "Please ensure the cpunks-10k repository is properly set up."
        )
        logger.error(msg)
        if raise_error:
            raise FileNotFoundError(msg)
        return False
    
    # Check if there are at least some images (not necessarily all 10k)
    file_count = len([f for f in os.listdir(images_dir) if f.startswith('punk') and f.endswith('.png')])
    if file_count < 100:  # Arbitrary threshold for a minimum number of files
        msg = (
            f"Not enough training images found in {images_dir} (found {file_count}, expected at least 100). "
            "Please ensure the cpunks-10k repository is properly set up."
        )
        logger.error(msg)
        if raise_error:
            raise FileNotFoundError(msg)
        return False
    
    # Check if data file exists
    data_file = os.path.join(paths.CPUNKS_ROOT, 'data', 'punks.json')
    if not os.path.exists(data_file):
        msg = (
            f"Data file not found at {data_file}. "
            "Please ensure the cpunks-10k repository is properly set up."
        )
        logger.error(msg)
        if raise_error:
            raise FileNotFoundError(msg)
        return False
    
    # Check if artifacts directory exists
    if not os.path.exists(paths.ARTIFACTS_DIR):
        msg = (
            f"Artifacts directory not found at {paths.ARTIFACTS_DIR}. "
            "This directory is required for storing model artifacts."
        )
        logger.error(msg)
        if raise_error:
            raise FileNotFoundError(msg)
        return False
    
    # Check if saved models directory exists
    if not os.path.exists(paths.SAVED_MODELS_DIR):
        msg = (
            f"Saved models directory not found at {paths.SAVED_MODELS_DIR}. "
            "This directory is required for storing trained models."
        )
        logger.error(msg)
        if raise_error:
            raise FileNotFoundError(msg)
        return False
    
    logger.info(f"cpunks-10k dependency verified: {paths.CPUNKS_ROOT}")
    logger.info(f"Found {file_count} training images in {images_dir}")
    return True