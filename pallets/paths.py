import os

PALLETS_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PALLETS_ROOT, '..'))
ARTIFACTS_DIR = os.environ.get('PALLETS_ARTIFACTS_DIR', os.path.join(PROJECT_ROOT, 'artifacts'))
SAVED_MODELS_DIR = os.environ.get('PALLETS_SAVED_MODELS_DIR', os.path.join(PROJECT_ROOT, 'saved'))

# https://github.com/tnn1t1s/cpunks-10k
# Can be overridden with CPUNKS_ROOT_DIR environment variable
DEFAULT_CPUNKS_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'cpunks-10k', 'cpunks'))
CPUNKS_ROOT = os.environ.get('CPUNKS_ROOT_DIR', DEFAULT_CPUNKS_PATH)
