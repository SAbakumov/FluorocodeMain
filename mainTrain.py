# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:02:31 2020

@author: Boris
"""
import sys
import numpy as np
import tensorflow as tf
import CNNHelper
from datetime import datetime

tf.debugging.set_log_device_placement(True)

model  =  CNNHelper.VGG19.VGG19(3)

