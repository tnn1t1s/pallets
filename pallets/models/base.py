import os
import json
import torch

from ..logging import logger
from .. import paths


RGBA_CHANNELS = 4
ONE_HOT_CHANNELS = 222


def save(modelname, model, train_losses, test_losses):
    """
    Saves model as 'saved/<filename>'
    """
    modelpath = os.path.join(paths.SAVED_MODELS_DIR, f'{modelname}.pkl')
    metapath = os.path.join(paths.SAVED_MODELS_DIR, f'{modelname}.json')

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


def load(modelname, device):
    """
    Loads model from `saved/<filename>`
    """
    modelpath = os.path.join(paths.SAVED_MODELS_DIR, f'{modelname}.pkl')
    metapath = os.path.join(paths.SAVED_MODELS_DIR, f'{modelname}.json')

    model = torch.load(modelpath)
    model = model.to(device)
    logger.info(f"model blob loaded from {modelpath}")

    with open(metapath, 'r') as f:
        data = json.load(f)
        train_losses = data['train_losses']
        test_losses = data['test_losses']
    logger.info(f"training losses loaded from {modelpath}")

    return model, train_losses, test_losses


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
