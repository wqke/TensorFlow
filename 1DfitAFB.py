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
from ROOT import TFile,TChain
if __name__ == "__main__" : 

  # Four body angular phase space is described by 3 angles. 
  phsp = tfa.RectangularPhaseSpace( ( (-1, 1), ) )

  # Fit parameters of the model 
  FL  = tfa.FitParameter("FL" ,  0.600,  0.000, 1.000, 0.01)

  ### Start of model description

  def model(x) : 
    # Get phase space variables
    cosThetast = phsp.Coordinate(x, 0)
    sinThetast=tfa.Sqrt(1-cosThetast*cosThetast)
    # Decay density
    pdf  = 0.25 * (1.0 - FL ) * sinThetast*sinThetast
    pdf += 0.5 * FL * cosThetast*cosThetast
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

  # Estimate the maximum of PDF for toy MC generation using accept-reject method
  majorant = tfa.EstimateMaximum(sess, model(data_ph), data_ph, norm_sample )*1.1
  print "Maximum = ", majorant

  #Total geometry data sample
  tree = TChain("DecayTree")
  tree.Add("/home/ke/calculateI/model_total_new.root")
  data_sample = tree2array(tree,branches=['costheta_X_true'],selection='q2_true >=5.0738137  & q2_true<6.94456747')
  #array([ 3.20305994,  5.0738137 ,  6.94456747,  8.81532123, 10.686075  ])   borders
  #array([4.13843682, 6.00919059, 7.87994435, 9.75069811])   centers 
  data_sample = rec2array(data_sample)

  
  
  # TF graph for the PDF integral
  norm = tfa.Integral( model(norm_ph) )

  # TF graph for unbinned negalite log likelihood (the quantity to be minimised)
  nll = tfa.UnbinnedNLL( model(data_ph), norm )

  # Options for profiling
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()


  # Run MINUIT minimisation of the neg. log likelihood
  result = tfa.RunMinuit(sess, nll, { data_ph : data_sample, norm_ph : norm_sample }, options = options, run_metadata = run_metadata, runHesse=True)
  print result
  tfa.WriteFitResults(result, "result.txt")

  # Run toy MC corresponding to fitted result
  fit_data = tfa.RunToyMC( sess, model(data_ph), data_ph, phsp, 1000000, majorant, chunk = 1000000)
  f = TFile.Open("toyresult.root", "RECREATE")
  tfa.FillNTuple("toy", fit_data, ["cosThetast" ])
  f.Close()

  # Store timeline profile 
  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
  chrome_trace = fetched_timeline.generate_chrome_trace_format()
  with open('timeline.json', 'w') as f:
    f.write(chrome_trace)
    
    
FLlist=[5.33493e-01,4.60898e-01,0.40398019940319396,0.3525863095102372]
FLerr=[7.95262e-03,4.53310e-03,0.004017205018987086,0.004665849591322296]

centers=[4.13843682, 6.00919059, 7.87994435, 9.75069811]
q2err=[centers[1]-centers[0],centers[1]-centers[0],centers[1]-centers[0],centers[1]-centers[0]]
q2_borders=[ 3.20305994, 5.0738137 , 6.94456747, 8.81532123, 10.686075 ]
plt.errorbar(centers,FLlist, xerr=q2err,yerr=FLerr, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$F_L^{D*}$ - Rapidsim(total geometry)')


def power(x,c,d,e):
  res=c*x**2+d*x+e
  return res

sol,_=curve_fit(power, centers, FLlist, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#FF9848',label='parabolic fit')
plt.ylim(0.0,1.0)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$F_L^{D*}$ ($q^2$)')
plt.title(r'${D*}$ polarisation fraction',fontsize=14, color='black')
plt.legend()
