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
  phsp = tfa.RectangularPhaseSpace( ( (-1., 1.), ) )
  vals = {'a': 0.500,
            'b': 0.500,
            'c': 0.500}
  tot_rate = 0.
  for v in vals:
    tot_rate += vals[v]
#  tot_rate=vals["I1c"]
  for v in vals:
    vals[v] = vals[v]/tot_rate
  # Fit parameters of the model
  a  = tfa.FitParameter("a",vals["a"] ,  -1.000, 1.000)
  b = tfa.FitParameter("b",vals["b"], -1.000, 1.000)
  c  = tfa.FitParameter("c",vals["c"] ,  -1.000, 1.000)



  #params = [ I1c, I1s, I2c, I2s, I6c, I6s, I3, I9, I4, I8, I5, I7 ]

  ### Start of model description

  def model(x) :
    # Get phase space variables
    cosThetal = phsp.Coordinate(x, 0)     #D* angle costhetast
    # Derived quantities
    sinThetal = tfa.Sqrt( 1.0 - cosThetal * cosThetal )
    sinTheta2l =  (1.0 - cosThetal * cosThetal)
    cos2Thetal = (2.0 * cosThetal * cosThetal - 1.0)
    # Decay density
    pdf  =  a+b*cosThetal
    pdf += c*cosThetal*cosThetal 
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
  tree.SetBranchStatus("costheta_L_true",1)
  tree_cut = tree.CopyTree("q2_true >=8.9 && q2_true<12")
  data_sample = tree2array(tree_cut,branches=['costheta_L_true'])
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
  tfa.FillNTuple("fit_result", fit_result, ["costheta_L"])
  tfa.FillNTuple("data", data_sample, ["costheta_L"])
 # chii=f["chi"]
  f.Close()



  # Store timeline profile
  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
  chrome_trace = fetched_timeline.generate_chrome_trace_format()
  with open('timeline.json', 'w') as f:
    f.write(chrome_trace)


bin1={'a': (0.8632375916325143, 1.9245036828048743), 'status': 3, 'c': (-0.04886485048968203, 0.06983107749311035), 'b': (0.03480199478341639, 0.04975309695133168), 'iterations': 93, 'loglh': -321.91965093626175}
bin2={'a': (0.8893837394243613, 0.34612563153582776), 'status': 3, 'c': (-0.06008700218952967, 0.026721782068000743), 'b': (-0.1275948168941211, 0.05630222082338587), 'iterations': 98, 'loglh': -2976.0107112816245}
bin3={'a': (0.8586323457963463, 0.2628249854780613), 'status': 3, 'c': (-0.012040285071009738, 0.005077875791454578), 'b': (-0.20183185000563175, 0.06481910490729331), 'iterations': 92, 'loglh': -7131.290809379244}
bin4={'a': (0.8674069243515365, 0.1265442398098331), 'status': 2, 'c': (-0.026521880158885458, 0.005155966769751552), 'b': (-0.17589313254013406, 0.02584112192068555), 'iterations': 91, 'loglh': -5208.034607553979}

binlist=[bin1,bin2,bin3,bin4]

alist=[]
aerrlist=[]
blist=[]
berrlist=[]
clist=[]
cerrlist=[]

for binn in binlist:
  alist.append(binn['a'][0])
  aerrlist.append(binn['a'][1])
  blist.append(binn['b'][0])
  berrlist.append(binn['b'][1])
  clist.append(binn['c'][0])
  cerrlist.append(binn['c'][1])


corr_matrix1=array([    
[ 1.000 , 0.999, -0.999],
[ 0.999 , 1.000 ,-0.998],
[-0.999 ,-0.998 , 1.000]])


corr_matrix2=array([ 
[1.000, -1.000 ,-0.993],
[-1.000 , 1.000  ,0.992],
[-0.993 , 0.992  ,1.000]])

corr_matrix3=array([ 
[1.000 ,-1.000 ,-0.768],
[-1.000 , 1.000  ,0.763],
[-0.768 , 0.763 , 1.000]])

corr_matrix4=array([ 
[1.000 ,-0.998 ,-0.770],
[-0.998 , 1.000 , 0.758],
[-0.770 , 0.758 , 1.000]])

                                                              
a1=bin1['a']
b1=bin1['b']
c1=bin1['c']
a2=bin2['a']
b2=bin2['b']
c2=bin2['c']
a3=bin3['a']
b3=bin3['b']
c3=bin3['c']
a4=bin4['a']
b4=bin4['b']
c4=bin4['c']
(a1,b1,c1)= correlated_values_norm([a1,b1,c1], corr_matrix1)
(a2,b2,c2)= correlated_values_norm([a2,b2,c2], corr_matrix2)
(a3,b3,c3)= correlated_values_norm([a3,b3,c3], corr_matrix3)
(a4,b4,c4)= correlated_values_norm([a4,b4,c4], corr_matrix4)

afb1=b1/(742350/3000000.)
afb2=b2/(767430/3000000.)
afb3=b3/(760410/3000000.)
afb4=b4/(729810/3000000.)

rab1=(a1-c1)/(2*a1+2*c1)
rab2=(a2-c2)/(2*a2+2*c2)
rab3=(a3-c3)/(2*a3+2*c3)
rab4=(a4-c4)/(2*a4+2*c4)

AFB=[afb1.n,afb3.n,afb3.n,afb4.n]
AFBerr=[afb1.s,afb3.s,afb3.s,afb4.s]

RAB=[rab1.n,rab3.n,rab3.n,rab4.n]
RABerr=[rab1.s,rab3.s,rab3.s,rab4.s]
