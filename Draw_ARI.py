# Copyright 2017 CERN for the benefit of the LHCb collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numpy import *
import tensorflow as tf
import sys, os
import numpy as np
import math
from math import cos,sin,pi

sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU
import root_pandas
import pandas as pd
import TensorFlowAnalysis as tfa
from tensorflow.python.client import timeline
from root_numpy import root2array, rec2array, tree2array
from ROOT import TFile,TChain,TTree
from uncertainties import *


if __name__ == "__main__" :
  # Four body angular phase space is described by 3 angles.
  phsp = tfa.RectangularPhaseSpace( ( (-1., 1.), (-1., 1.), (-math.pi, math.pi) ) )

  # Placeholders for data and normalisation samples (will be used to compile the model)
  data_ph = phsp.data_placeholder
  norm_ph = phsp.norm_placeholder

  binnumber=[1,2,3,4,"total"]
  cut=["q2_true >=3. && q2_true<6.2","q2_true >=6.2 && q2_true<7.6","q2_true >=7.6 && q2_true<8.9","q2_true >=8.9 && q2_true<12","q2_true >=3. && q2_true<12"]
  borders=array([ 3.20305994, 6.2 , 7.6, 8.9, 10.686075  ])
  #Read RapidSim signal sample for either 3pi mode or 3pipi0 mode
  mode = "Bd2DstTauNu"
  #3pi or 3pipi0
  sub_mode = sys.argv[1]
  #Geometry (all or LHCb)
  geom = sys.argv[2]
  #True or reco angles
  type = sys.argv[3]
  #Number of events to run on (in k) - 5, 10, 20, 40, 80
  n = sys.argv[4]
  #The bin number - 1,2,3, 4 or total
  binnum = sys.argv[5]
  if binnum=="total":
    i=4
  else :  
    i=int(binnum)-1
  #Initial guesses in each bin (normalised)
  vals1 = {'I2s': 0.020051490246559067, 'I1s': 0.18071096148133478, 'I2c': -0.07228438459253392, 'I1c': 0.46687790870383206, 'I9': 0.0, 'I8': 0.0, 'I6c': 0.3896425388652342, 'I3': -0.026289731656599664, 'I5': 0.24210317853252794, 'I4': -0.04881671452619071, 'I7': 0.0, 'I6s': -0.15199524705416376}
  vals2 = {'I2s': 0.051449082690773716, 'I1s': 0.3270406806700345, 'I2c': -0.1475671363998936, 'I1c': 0.5350970486572719, 'I9': 0.0, 'I8': 0.0, 'I6c': 0.3715767083222546, 'I3': -0.0751130018612071, 'I5': 0.30975804307365057, 'I4': -0.11433129486838603, 'I7': 0.0, 'I6s': -0.2579101302844987}
  vals3 = {'I2s': 0.09171861086375781, 'I1s': 0.5040071237756012, 'I2c': -0.226179875333927, 'I1c': 0.6179875333926982, 'I9': 0.0, 'I8': 0.0, 'I6c': 0.34995547640249336, 'I3': -0.14870881567230634, 'I5': 0.35262689225289406, 'I4': -0.19412288512911846, 'I7': 0.0, 'I6s': -0.34728406055209265}
  vals4 = {'I2s': 0.1662234042553192, 'I1s': 0.8257978723404258, 'I2c': -0.3617021276595746, 'I1c': 0.7779255319148938, 'I9': 0.0, 'I8': 0.0, 'I6c': 0.29122340425531923, 'I3': -0.3045212765957448, 'I5': 0.3351063829787235, 'I4': -0.33909574468085113, 'I7': 0.0, 'I6s': -0.39095744680851074}
  val_total={'I2s': 0.06457564575645756, 'I1s': 0.3763837638376384, 'I2c': -0.16420664206642066, 'I1c': 0.559040590405904, 'I9': 0.0, 'I8': 0.0, 'I6c': 0.36162361623616235, 'I3': -0.10332103321033212, 'I5': 0.2970479704797048, 'I4': -0.13653136531365315, 'I7': 0.0, 'I6s': -0.25461254612546125}
vals=[vals1,vals2,vals3,vals4,val_total]
