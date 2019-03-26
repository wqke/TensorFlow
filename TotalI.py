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


if __name__ == "__main__" :
  # Four body angular phase space is described by 3 angles.
  phsp = tfa.RectangularPhaseSpace( ( (-1., 1.), (-1., 1.), (-math.pi, math.pi) ) )
  vals = {'I1c':0.78,
            'I1s':0.83,
            'I2c':-0.36,
            'I2s':0.17,
            'I3': -0.305,
            'I4':-0.34,
            'I5': 0.34,
            'I6c':0.29,
            'I6s':-0.39,
            'I7': 0.000,
            'I8': 0.000,
            'I9': 0.000}
  tot_rate = 0.
  for v in vals:
    tot_rate += vals[v]
#  tot_rate=vals["I1c"]
  for v in vals:
    vals[v] = vals[v]/tot_rate
  # Fit parameters of the model
  I8  = tfa.FitParameter("I8",vals["I8"] ,  -1.000, 1.000)
  I7 = tfa.FitParameter("I7",vals["I7"], -1.000, 1.000)
  I6s  = tfa.FitParameter("I6s",vals["I6s"] ,  -1.000, 1.000)
  I6c  = tfa.FitParameter("I6c",vals["I6c"] ,   -1.000, 1.000)
 #  I5 = tfa.FitParameter("I5",vals["I5"],   -1.000, 1.000)
  I4  = tfa.FitParameter("I4",vals["I4"] ,  -1.000, 1.000)
  I3  = tfa.FitParameter("I3",vals["I3"] ,   -1.000, 1.000)
  I2s = tfa.FitParameter("I2s",vals["I2s"],  -1.000, 1.000)
  I2c  = tfa.FitParameter("I2c",vals["I2c"] , -1.000, 1.000)
  I1s  = tfa.FitParameter("I1s",vals["I1s"] , -1.000, 1.000)
  I1c = tfa.FitParameter("I1c",vals["I1c"], -1.000, 1.000)
  I9 = tfa.FitParameter("I9",vals["I9"], -1.000, 1.000)


  #params = [ I1c, I1s, I2c, I2s, I6c, I6s, I3, I9, I4, I8, I5, I7 ]

  ### Start of model description

  def model(x) :
    # Get phase space variables
    cosThetast = phsp.Coordinate(x, 0)     #D* angle costhetast
    cosThetal = phsp.Coordinate(x, 1)  #Lepton angle costhetal
    chi = phsp.Coordinate(x, 2)
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
    pdf  =  I1c* cosThetast*cosThetast
    pdf += I1s * sinTheta2st
    pdf +=  I2c * cosThetast*cosThetast*cos2Thetal
    pdf +=  I2s * sinTheta2st *  cos2Thetal
    pdf +=  I6c *cosThetast*cosThetast *cosThetal
    pdf +=  I6s * sinTheta2st *  cosThetal
    pdf += I3 * cos2chi * sinTheta2l * sinTheta2st
    pdf +=I9 * sin2chi * sinThetal * sinThetal * sinThetast * sinThetast
 #  pdf += (1.0 -I1c -I1s -I2c -I2s -I3 -I4-I5 - I6c -I6s - I7 -I8) * sin2chi * sinThetal * sinThetal * sinThetast * sinThetast
    pdf += I4 * coschi * 2.0 * sinThetal * cosThetal * sin2Thetast
    pdf += I8 * sinchi * 2.0 * sinThetal * cosThetal * sin2Thetast
 #  pdf += I5 * coschi * sinThetal  * sin2Thetast
    pdf +=  (1.0 -I1c -I1s -I2c -I2s -I3 -I4-I9 - I6c -I6s - I7 -I8) * coschi * sinThetal  * sin2Thetast
    pdf +=  I7 * sinchi * sinThetal  * sin2Thetast
    return pdf

  ### End of model description


  # Placeholders for data and normalisation samples (will be used to compile the model)
  data_ph = phsp.data_placeholder
  norm_ph = phsp.norm_placeholder

  # TF initialiser
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)

  # Create normalisation sample (uniform sample in the 3D phase space)
  norm_sample = sess.run( phsp.UniformSample(1000000) )

  tree = TChain("DecayTree")

  tree.Add("/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstTauNu/3pi_all_Total/model_vars.root")
  tree.SetBranchStatus("*",0)
  tree.SetBranchStatus("q2_true",1)
  tree.SetBranchStatus("costheta_D_true",1)
  tree.SetBranchStatus("costheta_L_true",1)
  tree.SetBranchStatus("chi_true",1)
  tree_cut = tree.CopyTree("q2_true >=8.9 && q2_true<12")
  data_sample = tree2array(tree_cut,branches=['costheta_D_true','costheta_L_true','chi_true'])
  data_sample = rec2array(data_sample)
  #array([ 3.20305994, 6.2 , 7.6, 8.9, 10.686075  ])   borders

  data_sample = sess.run(phsp.Filter(data_sample))


  # Estimate the maximum of PDF for toy MC generation using accept-reject method
  majorant = tfa.EstimateMaximum(sess, model(data_ph), data_ph, norm_sample )*1.1
  print "Maximum = ", majorant

  # TF graph for the PDF integral
  norm = tfa.Integral( model(norm_ph) )

  # TF graph for unbinned negalite log likelihood (the quantity to be minimised)
  nll = tfa.UnbinnedNLL( model(data_ph), norm )

  # Options for profiling
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  # Run MINUIT minimisation of the neg. log likelihood
  # Run toy MC corresponding to fitted result


  result = tfa.RunMinuit(sess, nll, { data_ph : data_sample, norm_ph : norm_sample }, options = options, run_metadata = run_metadata, runHesse=True)

  print result
  tfa.WriteFitResults(result, "result.txt")

  fit_result = tfa.RunToyMC( sess, model(data_ph), data_ph, phsp, 1000000, majorant, chunk = 1000)
  f = TFile.Open("result_DstTauNu.root", "RECREATE")
  tfa.FillNTuple("fit_result", fit_result, ["costheta_X", "costheta_L", "chi" ])
  tfa.FillNTuple("data", data_sample, ["costheta_X_true", "costheta_L", "chi" ])
 # chii=f["chi"]
  f.Close()



  # Store timeline profile
  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
  chrome_trace = fetched_timeline.generate_chrome_trace_format()
  with open('timeline.json', 'w') as f:
    f.write(chrome_trace)


                                                 
