import os
import torch


RGBA_CHANNELS = 4
ONE_HOT_CHANNELS = 222


def _saved_path():
    """
    Helper function to save and load models from consistent location
    """
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(parent_dir, 'saved')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    return models_dir


def save(model, filename):
    """
    Saves model as 'saved/<filename>'
    """
    models_dir = _saved_path()
    filepath = os.path.join(models_dir, filename)
    torch.save(model, filepath)


def load(filename):
    """
    Loads model from `saved/<filename>`
    """
    models_dir = _saved_path()
    filepath = os.path.join(models_dir, filename)
    return torch.load(filepath)
