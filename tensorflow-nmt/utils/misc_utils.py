# -*- coding: utf-8 -*-

#####################################################################
# File Name:  misc_utils.py
# Author: shenming
# Created Time: Wed Jan  1 23:09:29 2020
#####################################################################

import os
import sys
from distutils import version
import tensorflow as tf


def check_tensorflow_version():
  # LINT.IfChange
  min_tf_version = "1.12.0"
  # LINT.ThenChange(<pwd>/nmt/copy.bara.sky)
  if (version.LooseVersion(tf.__version__) <
      version.LooseVersion(min_tf_version)):
    raise EnvironmentError("Tensorflow version must >= %s" % min_tf_version)

if __name__ == "__main__":
    pass
