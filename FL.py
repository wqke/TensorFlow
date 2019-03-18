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
  phsp = tfa.RectangularPhaseSpace( ( (-1., 1.) ) )
  FL  = tfa.FitParameter("FL" ,  0.600,  0.000, 1.000, 0.01)    #Taken from Belle measurement 
  def model(x) : 
    # Get phase space variables
    cosThetast = phsp.Coordinate(x, 0) #D* angle costhetast

    # Derived quantities
    sinThetast = tfa.Sqrt( 1.0 - cosThetast * cosThetast )
    sinTheta2st =  (1.0 - cosThetast * cosThetast)
    sin2Thetast = (2.0 * sinThetast * cosThetast)
 
    # Decay density
    pdf  =  0.5 *FL* cosThetast*cosThetast
    pdf +=  0.25*(1-FL)*sinThetast*sinThetast
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
  tree.Add("/home/ke/pythonap/model_tree_vars.root")
  branch_names = ["costheta_X_true"]
  data_sample = tree2array(tree,branch_names)
  data_sample = rec2array(data_sample)
  data_sample = sess.run(phsp.Filter(data_sample))
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
  result = tfa.RunMinuit(sess, nll, { data_ph : data_sample, norm_ph : norm_sample }, options = options, run_metadata = run_metadata, runHesse=True)
  print result
  tfa.WriteFitResults(result, "result.txt")

  # Run toy MC corresponding to fitted result
  fit_data = tfa.RunToyMC( sess, model(data_ph), data_ph, phsp, 1000000, majorant, chunk = 1000000)
  f = TFile.Open("toyresult.root", "RECREATE")
  tfa.FillNTuple("toy", fit_data, ["cos*"])
  f.Close()

  # Store timeline profile 
  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
  chrome_trace = fetched_timeline.generate_chrome_trace_format()
  with open('timeline.json', 'w') as f:
    f.write(chrome_trace)


