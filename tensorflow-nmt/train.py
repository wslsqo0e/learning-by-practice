# -*- coding: utf-8 -*-

#####################################################################
# File Name:  train.py
# Author: shenming
# Created Time: Wed Jan  1 23:10:35 2020
#####################################################################

import os
import sys
from utils import misc_utils as utils

utils.check_tensorflow_version()

def train(hparams, scope=None, target_session=""):
    """Train a translation model."""
    log_device_placement = hparams.log_device_placement


if __name__ == "__main__":
    pass
