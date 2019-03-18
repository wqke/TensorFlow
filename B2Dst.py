import tensorflow as tf
import sys, os
import numpy as np
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU

import TensorFlowAnalysis as tfa
from tensorflow.python.client import timeline

from ROOT import TFile

if __name__ == "__main__" : 

  # Four body angular phase space is described by 3 angles. 
  phsp = tfa.FourBodyAngularPhaseSpace()

  # Fit parameters of the model 
  """
  FL  = tfa.FitParameter("FL" ,  0.600,  0.000, 1.000, 0.01)    #Taken from Belle measurement
  AT2 = tfa.FitParameter("AT2",  0.200, -1.000, 1.000, 0.01)
  S5  = tfa.FitParameter("S5" , -0.100, -1.000, 1.000, 0.01)    #Initial guess
  """
  I8  = tfa.FitParameter("I8" ,  0.,  0.000, 1.000, 0.01)   
  I7 = tfa.FitParameter("I7",  0., -1.000, 1.000, 0.01)
  I6s  = tfa.FitParameter("I6s" , -0.250, -1.000, 1.000, 0.01) 
  I6c  = tfa.FitParameter("I6c" ,  0.300,  0.000, 1.000, 0.01)    
  I5 = tfa.FitParameter("I5",  0.350, 0.000, 1.000, 0.01)
  I4  = tfa.FitParameter("I4" , -0.200, -1.000, 1.000, 0.01)  
  I3  = tfa.FitParameter("I3" ,  -0.100,  -1.000, 1.000, 0.01)    
  I2s = tfa.FitParameter("I2s",  0.100, -1.000, 1.000, 0.01)
  I2c  = tfa.FitParameter("I2c" , -0.200, -1.000, 1.000, 0.01) 
  I1s  = tfa.FitParameter("I1s" ,  0.450,  0.000, 1.000, 0.01)    
  I1c = tfa.FitParameter("I1c",  0.550, 0.000, 1.000, 0.01)
  I9  = tfa.FitParameter("I9" , -0.100, -1.000, 1.000, 0.01) 
  ### Start of model description

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

    # Decay density
    pdf  = (9.0/(32.np.pi)) * I1c* cosThetast*cosThetast
    pdf +=  (9.0/(32.np.pi)) * I1s * sinTheta2st
    pdf +=  (9.0/(32.np.pi)) * I2c * cosThetast*cosThetast*cos2Thetal
    pdf +=  (9.0/(32.np.pi)) * I2s * sinTheta2st *  cos2Thetal
    pdf +=  (9.0/(32.np.pi))* I6c *cosThetast*cosThetast *cosThetal
    pdf +=  (9.0/(32.np.pi))* I6s * sinTheta2st *  cosThetal
    pdf +=  (9.0/(32.np.pi))* I3 * np.cos(2*chi) * sinTheta2l * sinTheta2st
    pdf +=  (9.0/(32.np.pi))* I9 * np.sin(2*chi) * sinTheta2l * sinTheta2st
    pdf +=  (9.0/(32.np.pi))* I4 * np.cos(chi) * 2 * sinThetal * cosThetal * sin2Thetast 
    pdf +=  (9.0/(32.np.pi))* I8 * np.sin(chi) * 2 * sinThetal * cosThetal * sin2Thetast 
    pdf +=  (9.0/(32.np.pi))* I5 * np.cos(chi) * sinThetal  * sin2Thetast 
    pdf +=  (9.0/(32.np.pi))* I7 * np.sin(chi) * sinThetal  * sin2Thetast 

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

  # Create toy MC data sample (with the model parameters set to their initial values)
  data_sample = tfa.RunToyMC( sess, model(data_ph), data_ph, phsp, 10000, majorant, chunk = 1000000)

  # TF graph for the PDF integral
  norm = tfa.Integral( model(norm_ph) )

  # TF graph for unbinned negalite log likelihood (the quantity to be minimised)
  nll = tfa.UnbinnedNLL( model(data_ph), norm )

  # Options for profiling
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  # Run MINUIT minimisation of the neg. log likelihood
  result = tfa.RunMinuit(sess, nll, { data_ph : data_sample, norm_ph : norm_sample }, options = options, run_metadata = run_metadata )
  print result
  tfa.WriteFitResults(result, "result.txt")

  # Run toy MC corresponding to fitted result
  fit_data = tfa.RunToyMC( sess, model(data_ph), data_ph, phsp, 1000000, majorant, chunk = 1000000)
  f = TFile.Open("toyresult.root", "RECREATE")
  tfa.FillNTuple("toy", fit_data, ["cos1", "cos2", "phi" ])
  f.Close()

  # Store timeline profile 
  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
  chrome_trace = fetched_timeline.generate_chrome_trace_format()
  with open('timeline.json', 'w') as f:
    f.write(chrome_trace)
