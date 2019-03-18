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
  phsp = tfa.FourBodyAngularPhaseSpace()
  a=tfa.FitParameter("a" , 0., -1.000, 1.000, 0.01) 
  c=tfa.FitParameter("c" , 0., -1.000, 1.000, 0.01) 
  def model(x) : 
    # Get phase space variables
    cosThetast = phsp.CosTheta1(x)     #D* angle costhetast
    cosThetal = phsp.CosTheta2(x)    #Lepton angle costhetal
    chi = phsp.Phi(x)
    # Derived quantities
    sinThetast = tfa.Sqrt( 1.0 - cosThetast * cosThetast )
    sinThetal = tfa.Sqrt( 1.0 - cosThetal * cosThetal )
    sinTheta2st =  (1.0 - cosThetast * cosThetast)
    sinTheta2l =  (1.0 - cosThetal * cosThetal)
    sin2Thetast = (2.0 * sinThetast * cosThetast)
    cos2Thetal = (2.0 * cosThetal * cosThetal - 1.0)
    coschi=tf.cos(chi)
    sinchi=tf.sin(chi)
    cos2chi=2*coschi*coschi-1
    sin2chi=2*sinchi*coschi    
    # Decay density
    pdf  =  c* cosThetast*cosThetast
    pdf +=  a
    return pdf
  
  
  
  
  
