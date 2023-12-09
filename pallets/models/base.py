import os
import json
import torch

from ..logging import logger


RGBA_CHANNELS = 4
ONE_HOT_CHANNELS = 222


def _saved_path():
    """
    Helper function to save and load models from consistent location
    """
    parent_dir = __file__
    for _ in range(3):  # go up three directories
        parent_dir = os.path.dirname(parent_dir)

    models_dir = os.path.join(parent_dir, 'saved')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    return models_dir


def save(modelname, model, train_losses, test_losses):
    """
    Saves model as 'saved/<filename>'
    """
    models_dir = _saved_path()
    modelpath = os.path.join(models_dir, f'{modelname}.pkl')
    metapath = os.path.join(models_dir, f'{modelname}.json')

    torch.save(model, modelpath)
    logger.info(f"model blob saved to {modelpath}")

    losses = {
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    with open(metapath, 'w') as f:
        l_json = json.dumps(losses)
        f.write(l_json)
    logger.info(f"training losses saved to {metapath}")


def load(filename):
    """
    Loads model from `saved/<filename>`
    """
    models_dir = _saved_path()
    filepath = os.path.join(models_dir, filename)
    return torch.load(filepath)


def get_device(require_gpu=True):
    """
    Does its best to find a GPU and falls back to CPU. Set `require_gpu` to
    True to throw exception if GPU isn't found.
    """
    if torch.cuda.is_available():
        logger.info(f"gpu: cuda")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.info(f"gpu: mps")
        return torch.device("mps")
    else:
        if require_gpu:
            raise Exception("No GPU found")
        logger.info(f"gpu: none")
        return torch.device("cpu")
