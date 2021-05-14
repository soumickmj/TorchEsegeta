#########################################################################
## Project: Explain-ability and Interpret-ability for segmentation models
## Purpose: Special utility functions
## Author: Arnab Das
#########################################################################

import numpy as np
from lucent.modelzoo.util import get_model_layers
import torch
import os.path as osp
import pickle
from functools import partial
import logging

# Fuction to normalize gradient to be ready for pyplot
def normalize_gradient(grads):

    if np.max(grads) > 1:
        grads = grads - np.min(grads)
        grads = grads / np.max(grads)
    return grads

def get_layers_for_lucent(model):
    print(get_model_layers(model))


def load_checkpoint(fpath):
    r"""Loads checkpoint.
    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.
    Args:
        fpath (str): path to checkpoint.
    Returns:
        dict
    Examples::
        #>>> from torchreid.utils import load_checkpoint
        #>>> fpath = 'log/my_model/model.pth.tar-10'
        #>>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint

def get_logger(logger_name, log_file_path, level=logging.INFO):
    log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    log_handler = logging.FileHandler(log_file_path)
    log_handler.setFormatter(log_formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(log_handler)

    return logger
