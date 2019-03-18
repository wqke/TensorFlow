import tensorflow as tf
import sys, os
import numpy as np
from math import cos,sin
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU

import TensorFlowAnalysis as tfa
from tensorflow.python.client import timeline

from ROOT import TFile

if __name__ == "__main__" : 
