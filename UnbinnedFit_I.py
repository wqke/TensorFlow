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
from root_pandas import *

if __name__ == "__main__" :
  # Four body angular phase space is described by 3 angles.
  phsp = tfa.RectangularPhaseSpace( ( (-1., 1.), (-1., 1.), (-math.pi, math.pi) ) )
  binnumber=[1,2,3,4,"total"]
#  cut=["q2_true >=3. && q2_true<6.2","q2_true >=6.2 && q2_true<7.6","q2_true >=7.6 && q2_true<8.9","q2_true >=8.9 && q2_true<12","q2_true >=3. && q2_true<12"]
  cut=["q2_true >=3. and q2_true<6.2","q2_true >=6.2 and q2_true<7.6","q2_true >=7.6 and q2_true<8.9","q2_true >=8.9 and q2_true<12","q2_true >=3. and q2_true<12"]


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
  num_sig = sys.argv[4]
  n=num_sig
  #The bin number - 1,2,3, 4 or total
  binnum = sys.argv[5]
  if binnum=="total":
    i=4
  else :
    i=int(binnum)-1

  # Fit parameters of the model
  
#Total rate  = 3 I1c + 6 I1s - I2c - 2 I2s
  rate = tfa.FitParameter("rate", 1.0)
  
  # Fit parameters
  #I1c = tfa.FitParameter("I1c" , 0.57, -1, 1, 0.01)
  #I1c = (rate - 6 I1s + I2c + 2I2s)/3.
  I1s = tfa.FitParameter("I1s" , 0.43, -1, 1, 0.01)
  I2c = tfa.FitParameter("I2c" , -0.17, -1, 1, 0.01)
  I2s = tfa.FitParameter("I2s" , 0.07, -1, 1, 0.01)
  I6c = tfa.FitParameter("I6c" , 0.35, -1, 1, 0.01)
  I6s = tfa.FitParameter("I6s" , -0.27, -1, 1, 0.01)
  I3  = tfa.FitParameter("I3"  , -0.13, -1, 1, 0.01)
  I4  = tfa.FitParameter("I4"  , -0.15, -1, 1, 0.01)
  I5  = tfa.FitParameter("I5"  , 0.30, -1, 1, 0.01)
  I7  = tfa.FitParameter("I7"  , 0., -1, 1, 0.01)
  I8  = tfa.FitParameter("I8"  , 0., -1, 1, 0.01)
  I9  = tfa.FitParameter("I9"  , 0., -1, 1, 0.01)

  
  """
  I8  = tfa.FitParameter("I8",vals[i]["I8"] ,  -1.000, 1.000)
  I7 = tfa.FitParameter("I7",vals[i]["I7"], -1.000, 1.000)
  I6s  = tfa.FitParameter("I6s",vals[i]["I6s"] ,  -1.000, 1.000)
  I6c  = tfa.FitParameter("I6c",vals[i]["I6c"] ,   -1.000, 1.000)
  I4  = tfa.FitParameter("I4",vals[i]["I4"] ,  -1.000, 1.000)
  I3  = tfa.FitParameter("I3",vals[i]["I3"] ,   -1.000, 1.000)
  I2s = tfa.FitParameter("I2s",vals[i]["I2s"],  -1.000, 1.000)
  I2c  = tfa.FitParameter("I2c",vals[i]["I2c"] , -1.000, 1.000)
  I1s  = tfa.FitParameter("I1s",vals[i]["I1s"] , -1.000, 1.000)
  I1c = tfa.FitParameter("I1c",vals[i]["I1c"], -1.000, 1.000)
  I9 = tfa.FitParameter("I9",vals[i]["I9"], -1.000, 1.000)
 """
 ### Start of model description
  def model(x) :
    # Get phase space variables
    cosThetast = phsp.Coordinate(x, 0)     #D* angle costhetast
    cosThetal = phsp.Coordinate(x, 1)  #Lepton angle costhetal
    chi = phsp.Coordinate(x, 2)
    #Derived quantities
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
    pdf  =  (1.0/3.0)*(rate - 6*I1s + I2c + 2*I2s)* cosThetast*cosThetast
    pdf += I1s * sinTheta2st
    pdf +=  I2c * cosThetast*cosThetast*cos2Thetal
    pdf +=  I2s * sinTheta2st *  cos2Thetal
    pdf +=  I6c *cosThetast*cosThetast *cosThetal
    pdf +=  I6s * sinTheta2st *  cosThetal
    pdf += I3 * cos2chi * sinTheta2l * sinTheta2st
    pdf +=I9 * sin2chi * sinThetal * sinThetal * sinThetast * sinThetast
    pdf += I4 * coschi * 2.0 * sinThetal * cosThetal * sin2Thetast
    pdf += I8 * sinchi * 2.0 * sinThetal * cosThetal * sin2Thetast
    pdf +=  I5 * coschi * sinThetal  * sin2Thetast
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

  print "Loading tree"
  in_file = "/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstTauNu/%s_%s_Total/model_vars_weights.root" % (sub_mode,geom)
  branch_names = ["costheta_D_%s" % type, "costheta_L_%s" % type, "chi_%s" % type]
  data_sample_fit = read_root(in_file,"DecayTree",columns=["costheta_D_%s" % type,"costheta_L_%s" % type ,"chi_%s" % type,'q2_true'])
  data_sample_fit = data_sample_fit.query(cut[i])
  data_sample_fit = data_sample_fit[branch_names]
  data_sample_fit = data_sample_fit.sample(n=int(num_sig)*1000,random_state=42)
  #Convert to ndarray
  data_sample= data_sample_fit.values
  
  
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

  result,covmat = tfa.RunMinuit(sess, nll, { data_ph : data_sample, norm_ph : norm_sample }, options = options, run_metadata = run_metadata, runHesse=True)

  #print fit results (the 12 I coefficients)
  print 'RESULT IN BIN ',binnumber[i],': ',result

  rate1=result['rate'][0]
  i9=result['I9'][0]
  i8=result['I8'][0]
  i7=result['I7'][0]
  i6s=result['I6s'][0]
  i6c=result['I6c'][0]
  i4=result['I4'][0]
  i3=result['I3'][0]
  i2s=result['I2s'][0]
  i2c=result['I2c'][0]
  i1s=result['I1s'][0]
  i5=result['I5'][0]
  #i1c=result['I1c'][0]
  
  (rate1,i1s,i2c,i2s,i6c,i6s,i3,i4,i5,i7,i8,i9)= correlated_values([rate1,i1s,i2c,i2s,i6c,i6s,i3,i4,i5,i7,i8,i9],covmat)
  i1c=(1.0/3.0)*(rate1 - 6.*i1s + i2c + 2.*i2s)
  rab=((1.0/3.0)*(rate1 - 6.*i1s + i2c + 2.*i2s)+2*i1s-3*i2c-6*i2s)/(2*((1.0/3.0)*(rate1 - 6.*i1s + i2c + 2.*i2s)+2*i1s+i2c+2*i2s))
  rlt= ((rate - 6.*i1s + i2c + 2.*i2s)-i2c)/(2*(3*i1s-i2s))
  Gammaq=(3*i1c+6*i1s-i2c-2*i1s)/4.
  afb1=i6c+2*i6s
  afb=(3/8.)*(afb1/Gammaq)
  a3=(1/(np.pi*2))*i3/Gammaq
  a9=(1/(2*np.pi))*i9/Gammaq  
  a6s=(-27/8.)*(i6s/Gammaq)
  a4=(-2/np.pi)*i4/Gammaq
  a8=(2/np.pi)*i8/Gammaq
  a5=(-3/4.)*i5/Gammaq
  a7=(-3/4.)*i7/Gammaq
 # fl=(9*i1c-3*i2c)/(3*i1c+6*i1s-i2c-2*i1s)
  para={'RAB':(rab.n,rab.s),'RLT':(rlt.n,rlt.s),'AFB':(afb.n,afb.s),'A6s':(a6s.n,a6s.s),'A3':(a3.n,a3.s),'A9':(a9.n,a9.s),'A4':(a4.n,a4.s),'A8':(a8.n,a8.s),'A5':(a5.n,a5.s),'A7':(a7.n,a7.s),'I1c':(i1c.n,i1c.s)}

  p = open( "/home/ke/TensorFlowAnalysis/ParamResult/param_%s_%s_%s_%s_bin%s.txt" % (sub_mode,geom,type,n,binnumber[i]), "w")
  slist=['RAB','RLT','AFB','A6s','A3','A9','A4','A8','A5','A7','I5']
  for s in slist:
    a=s+" "
    a += str(para[s][0])
    a += " "
    a += str(para[s][1])
    p.write(a + "\n")
  p.close()
  print para
  tfa.WriteFitResults(result, "/home/ke/TensorFlowAnalysis/UnbinnedResult/result_%s_%s_%s_%s_bin%s.txt" % (sub_mode,geom,type,n,binnumber[i]))

  #Simulation unsing fit results
  fit_result = tfa.RunToyMC( sess, model(data_ph), data_ph, phsp, 1000000, majorant, chunk = 1000)
  f = TFile.Open("/home/ke/TensorFlowAnalysis/UnbinnedResult/result_DstTauNu_bin%s.root" %(binnumber[i]), "RECREATE")
  tfa.FillNTuple("fit_result", fit_result, ["costheta_D", "costheta_L", "chi" ])
  tfa.FillNTuple("data", data_sample, ["costheta_D", "costheta_L", "chi" ])
  f.Close()

  # Store timeline profile
  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
  chrome_trace = fetched_timeline.generate_chrome_trace_format()
  with open('timeline.json', 'w') as f:
    f.write(chrome_trace)



                                                  
