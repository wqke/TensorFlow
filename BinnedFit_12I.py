
import tensorflow as tf
import numpy as np

import sys, os, math
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU

import TensorFlowAnalysis as tfa

def MakeHistogram(phsp, sample, bins, weights = None, normed = False) : 
  hist = np.histogramdd(sample, bins = bins, range = phsp.Bounds(), weights = weights, normed = normed )
  return hist[0]  # Only return the histogram itself, not the bin boundaries

def HistogramNorm(hist) : 
  return np.sum( hist )

def BinnedChi2(hist1, hist2, err) : 
  return tf.reduce_sum( ((hist1-hist2)/err)**2 )

if __name__ == "__main__" : 

  # Four body angular phase space is described by 3 angles. 
  phsp = tfa.RectangularPhaseSpace( ( (-1., 1.), (-1., 1.), (-math.pi, math.pi) ) )

  # Fit parameters
  I1c = tfa.FitParameter("I1c" , 0., -100.000, 100.000, 0.01)
  I1s = tfa.FitParameter("I1s" , 0., -100.000, 100.000, 0.01)
  I2c = tfa.FitParameter("I2c" , 0., -100.000, 100.000, 0.01)
  I2s = tfa.FitParameter("I2s" , 0., -100.000, 100.000, 0.01)
  I6c = tfa.FitParameter("I6c" , 0., -100.000, 100.000, 0.01)
  I6s = tfa.FitParameter("I6s" , 0., -100.000, 100.000, 0.01)
  I3  = tfa.FitParameter("I3"  , 0., -100.000, 100.000, 0.01)
  I9  = tfa.FitParameter("I9"  , 0., -100.000, 100.000, 0.01)
  I4  = tfa.FitParameter("I4"  , 0., -100.000, 100.000, 0.01)
  I8  = tfa.FitParameter("I8"  , 0., -100.000, 100.000, 0.01)
  I5  = tfa.FitParameter("I5"  , 0., -100.000, 100.000, 0.01)
  I7  = tfa.FitParameter("I7"  , 0., -100.000, 100.000, 0.01)

  params = [ I1c, I1s, I2c, I2s, I6c, I6s, I3, I9, I4, I8, I5, I7 ]

  # Start of model description
  def model(x) : 
    # Get phase space variables
    cosThetaD = phsp.Coordinate(x, 0)
    cosThetaL = phsp.Coordinate(x, 1)
    chi       = phsp.Coordinate(x, 2)

    # Angular terms
    cosSqThetaD = cosThetaD**2
    cosSqThetaL = cosThetaL**2
    sinSqThetaD = 1.0 - cosSqThetaD
    sinSqThetaL = 1.0 - cosSqThetaL
    sinThetaL   = tfa.Sqrt(sinSqThetaL)
    sinThetaD   = tfa.Sqrt(sinSqThetaD)
    cos2ThetaL  = cosSqThetaL - sinSqThetaL
    sin2ThetaL  = 2*cosThetaL*sinThetaL
    cos2ThetaD  = cosSqThetaD - sinSqThetaD
    sin2ThetaD  = 2*cosThetaD*sinThetaD
    cosChi      = tfa.Cos(chi)
    sinChi      = tfa.Sin(chi)
    cos2Chi     = tfa.Cos(2*chi)
    sin2Chi     = tfa.Sin(2*chi)

    # Total PDF
    pdf  =  I1c*cosSqThetaD + I1s*sinSqThetaD
    pdf += (I2c*cosSqThetaD + I2s*sinSqThetaD)*cos2ThetaL
    pdf += (I6c*cosSqThetaD + I6s*sinSqThetaD)*cosThetaL
    pdf += ( I3*cos2Chi     + I9*sin2Chi     )*sinSqThetaL*sinSqThetaD
    pdf += ( I4*cosChi      + I8*sinChi      )*sin2ThetaL*sin2ThetaD
    pdf += ( I5*cosChi      + I7*sinChi      )*sinThetaL*sin2ThetaD
    return pdf
  ### End of model description

  # Placeholder for data sample (will be used to compile the model)
  data_ph = phsp.data_placeholder

  data_model = model(data_ph)

  # TF initialiser
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)

  # List fo keep template histograms
  histos = []

  uniform_sample = sess.run( phsp.UniformSample(2000000) )

  binning = (4, 4, 8)

  # Fill template histograms by setting each I coeff to 1
  # Templates are created as weighted histograms from the uniform sample
  # to allow negative values in bins
  for p in params : p.update(sess, 0.)  # Make sure all parameters are set to 0

  norm = None
  for p in params : 
    print "Creating template for term ", p.par_name
    p.update(sess, 1.)
    weight_sample = sess.run( data_model, feed_dict = { data_ph : uniform_sample } )
    hist = MakeHistogram(phsp, uniform_sample, binning, weights = weight_sample)
    if not norm : norm = HistogramNorm( hist )    # Norm is calculated from the 1st term
    histos += [ hist/norm ]                       # ... and then all histogrames are normalised 
    p.update(sess, 0.)
    print histos[-1]

  # Fitting model for templates. 
  def fit_model(histos) : 
    pdf = 0.
    for p, h in zip(params, histos) : pdf += p*h
    return pdf

  # Set parameters to their "true" values
  init_params = [ 1.0, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]
  for i,p in zip(init_params, params) : p.update(sess, i)

  # Generate sample to fit
  fit_sample = tfa.RunToyMC( sess, data_model, data_ph, phsp, 100000, 2., chunk = 1000000)
  fit_hist = MakeHistogram(phsp, fit_sample, binning)
  print fit_hist
  err_hist = np.sqrt(fit_hist + 0.001)
  norm = HistogramNorm(fit_hist)

  # Define binned Chi2 to be minimised
  chi2 = BinnedChi2( fit_model(histos), fit_hist/norm, err_hist/norm )

  # Run Minuit minimisation
  result = tfa.RunMinuit(sess, chi2)
  tfa.WriteFitResults(result, "result.txt")
