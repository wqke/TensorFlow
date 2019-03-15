import tensorflow as tf
import sys, os

sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU

import TensorFlowAnalysis as tfa
from tensorflow.python.client import timeline

from ROOT import TFile

if __name__ == "__main__" : 

  # Four body angular phase space is described by 3 angles. 
  phsp = tfa.FourBodyAngularPhaseSpace()

  # Fit parameters of the model 
  FL  = tfa.FitParameter("FL" ,  0.600,  0.000, 1.000, 0.01)    #Taken from Belle measurement
  AT2 = tfa.FitParameter("AT2",  0.200, -1.000, 1.000, 0.01)
  S5  = tfa.FitParameter("S5" , -0.100, -1.000, 1.000, 0.01)    #Initial guess

  ### Start of model description

  def model(x) : 
    # Get phase space variables
    cosThetaK = phsp.CosTheta1(x)
    cosThetaL = phsp.CosTheta2(x)
    phi = phsp.Phi(x)

    # Derived quantities
    sinThetaK = tfa.Sqrt( 1.0 - cosThetaK * cosThetaK )
    sinThetaL = tfa.Sqrt( 1.0 - cosThetaL * cosThetaL )

    sinTheta2K =  (1.0 - cosThetaK * cosThetaK)
    sinTheta2L =  (1.0 - cosThetaL * cosThetaL)

    sin2ThetaK = (2.0 * sinThetaK * cosThetaK)
    cos2ThetaL = (2.0 * cosThetaL * cosThetaL - 1.0)

    # Decay density
    pdf  = (3.0/4.0) * (1.0 - FL ) * sinTheta2K
    pdf +=  FL * cosThetaK * cosThetaK
    pdf +=  (1.0/4.0) * (1.0 - FL) * sin2ThetaK *  cos2ThetaL
    pdf +=  (-1.0) * FL * cosThetaK * cosThetaK *  cos2ThetaL
    pdf +=  (1.0/2.0) * (1.0 - FL) * AT2 * sinTheta2K * sinTheta2L * tfa.Cos(2.0 * phi )
    pdf +=  S5 * sin2ThetaK * sinThetaL * tfa.Cos( phi )

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
