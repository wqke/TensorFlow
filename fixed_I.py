import matplotlib
matplotlib.use('Agg')  #fix python2 tkinter problem

import tensorflow as tf
import numpy as np
import collections

import sys, os, math
sys.path.append("../../TensorFlowAnalysis")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU

import TensorFlowAnalysis as tfa

from ROOT import TFile, TChain, TH3F
from root_numpy import root2array, rec2array, tree2array

import matplotlib.pyplot as plt
import rootplot.root2matplotlib as r2m
from scipy.stats import norm as sci_norm
from scipy.stats import sem as sem
import matplotlib.mlab as mlab
from root_pandas import to_root, read_root
from uncertainties import *
import pandas as pd
import random
import math


def MakeHistogram(phsp, sample, bins, weights = None, normed = False) : 
  hist = np.histogramdd(sample, bins = bins, range = phsp.Bounds(), weights = weights, normed = normed )
  return hist[0]  # Only return the histogram itself, not the bin boundaries

def MakeHistogram_1D(sample, bins, weights = None, normed = False, density = None) : 
  hist = np.histogram(sample, bins = bins, normed = normed, weights = weights, density = density)
  return hist[0]  # Only return the histogram itself, not the bin boundaries
  
def HistogramNorm(hist) : 
  return np.sum( hist )

def BinnedChi2(hist1, hist2, err) :
  return tf.reduce_sum( ((hist1 - hist2)/err)**2 )


if __name__ == "__main__" :

  #Read RapidSim signal sample for either 3pi mode or 3pipi0 mode
  mode = "Bd2DstTauNu"
  #3pi or 3pipi0
  sub_mode = sys.argv[1]
  #Geometry (all or LHCb)
  geom = sys.argv[2]
  #True or reco angles
  var_type = sys.argv[3]
  #Number of events to run on (in k) - 5, 10, 20, 40, 80
  num_sig = sys.argv[4]
  #Run a toy (Y/N)
  toy = sys.argv[5]
  #Hammer weight (SM / T1 / T2)
  ham = sys.argv[6]
	
  #The bkg that varies - Ds, Dplus, feed, D0, prompt, none
  syst = sys.argv[7] 

  #The filenumber : if we calculate the syst of internal bkg fraction, it selects the i-th file of the 100 random files 
  filenumber=sys.argv[8]


  #background fractions
  frac={}
  yielderr={}
  Yield={}
  
  #Fractions defined using Run 1 R(D*) fit

  feed_frac = 0.11
  Yield['signal']=1296.
  n_signal = Yield['signal']
  Yield['feed']=feed_frac*Yield['signal']
  Yield['Ds']=6835.
  Yield['D0']=1.41*445.
  Yield['Dplus']=0.245*Yield['Ds']
  Yield['prompt']=424.

  yielderr['signal'] = 86.
  yielderr['Ds'] = 166.
  yielderr['Dplus']= 0.245*166.
  yielderr['feed'] = feed_frac*86.
  yielderr['D0'] = 1.41 * 22.
  yielderr['prompt'] = 21.
  
  if syst!='none':
    Yield[syst]=Yield[syst]+ random.gauss(0,yielderr[syst]) 


  total_yield = Yield['D0'] +Yield['Dplus']+Yield['Ds']+Yield['prompt']+Yield['feed']+Yield['signal']
	
	
  frac['signal'] =Yield['signal']/total_yield  #floating
  frac['Ds'] = Yield['Ds']/total_yield
  frac['Dplus']= Yield['Dplus']/total_yield
  frac['feed'] = Yield['feed']/total_yield
  frac['D0'] = Yield['D0']/total_yield
  frac['prompt'] = Yield['prompt']/total_yield
  

  print "Variation on the fraction of : %s" % syst 	
  print "Initial component fractions: %s" % frac
  

  #frac[syst]=frac[syst]+ random.gauss(0,fracerr[syst])  


  #Number of angle bins depending on the signal yield, requiring roughly 25 events per bin
  #bdt_bins = 3
  #num_bins = int(math.ceil((float(num_sig)*(1.0/bdt_bins)*1000.0/25)**(1.0/3.0)))
  num_bins = int(math.ceil((float(num_sig)*1000.0/25)**(1.0/4.0)))
  
  print "NUMBER OF BINS IN EACH ANGLE : %s" % num_bins
  
  #Binning scheme
  var_bins = {"costheta_D_%s" % var_type: num_bins,
              "costheta_L_%s" % var_type: num_bins,
              "chi_%s" % var_type: num_bins,
              "BDT": num_bins
              }
	
  var_range = {"costheta_D_%s" % var_type: (-1.,1.),
               "costheta_L_%s" % var_type: (-1.,1.),
               "chi_%s" % var_type: (-math.pi,math.pi),
#               "BDT": [random.uniform(-0.05,0.05),3.6]
	"BDT": [0.,3.6]
               }
  
  var_titles = {"costheta_D_%s" % var_type: "$\\cos(\\theta_D)$",
                "costheta_L_%s" % var_type: "$\\cos(\\theta_L)$",
                "chi_%s" % var_type: "$\\chi$ [rad]",
                "BDT": "BDT"
                }
  print 'BDT CUT : ', var_range['BDT']
  # Four body angular phase space is described by 3 angles + q2 for background rejection
  phsp = tfa.RectangularPhaseSpace( ( var_range["costheta_D_%s" % var_type], var_range["costheta_L_%s" % var_type], var_range["chi_%s" % var_type], var_range["BDT"]))

  # TF initialiser
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  
  branch_names = ["costheta_D_%s" % var_type, "costheta_L_%s" % var_type, "chi_%s" % var_type, "BDT"]
  
  #Add tau flight which must be cut on
  branch_names.append("Tau_FD")
  #Cut to mimic the z flight distance significance cut
  Tau_FD_cut = 4000.

  #Read RapidSim sample used to determine bins
  print "Loading tree"
  bin_file = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bd2DstTauNu/%s_%s_Total/model_vars_weights_hammer_BDT.root" % (sub_mode,geom)
  
  cuts = "Tau_FD > %s and BDT > %s and costheta_D_%s>=-1 and costheta_D_%s<=1 and costheta_L_%s>=-1 and costheta_L_%s<=1 and chi_%s>=-%s and chi_%s<=%s" % (Tau_FD_cut,var_range["BDT"][0],var_type,var_type,var_type,var_type,var_type,math.pi,var_type,math.pi)
  
  bin_sample = read_root(bin_file,"DecayTree",columns=branch_names)
  bin_sample = bin_sample.query(cuts)
  
  #Reorder the columns to required order
  bin_sample = bin_sample.drop(columns=["Tau_FD"])
  branch_names.remove("Tau_FD")
  bin_sample = bin_sample[branch_names]
  qc_bin_vals={}
  #determine angle binnning
  for b in branch_names:
    print "%s" % b
    qc = pd.qcut(bin_sample[b], q=var_bins[b], precision=5)
    qc_bins = qc.unique()
    qc_bin_vals["%s" % b] = []
    for j in range(0,var_bins[b]):
      qc_bin_vals["%s" % b].append(qc_bins[j].left)
      qc_bin_vals["%s" % b].append(qc_bins[j].right)
    #Retain unique values then sort
    qc_bin_vals["%s" % b] = list(set(qc_bin_vals["%s" % b]))
    qc_bin_vals["%s" % b].sort()
    print qc_bin_vals["%s" % b]

  binning=(qc_bin_vals["costheta_D_%s" % var_type],qc_bin_vals["costheta_L_%s" % var_type],qc_bin_vals["chi_%s" % var_type],qc_bin_vals["BDT"])
  
  coeffs = ["I1c","I1s","I2c","I2s","I6c","I6s","I3","I4","I5","I7","I8","I9"]
  
  # Signal fit parameters
  init_vals = {"I1s": 0.415527,
  			   "I2c": -0.183184,
  			   "I2s": 0.074795,
  			   "I6c": 0.360249,
  			   "I6s": -0.262996,
  			   "I3": -0.121441,
  			   "I4": -0.150924,
  			   "I5": 0.298113,
  			   "I7": 0.0,
  			   "I8": 0.0,
  			   "I9": 0.0
  			   }
  
  #If running a toy, assign starting values using data fit results
  if(toy=="Y"):
  	default_results = pd.read_csv("results/result_Lifetime_%s_%s_%s_%s_Hammer_%s.txt" % (sub_mode,geom,var_type,num_sig,ham), header = None, sep=" ")
  	for c in init_vals:
		df = default_results[default_results[0].str.contains("%s" % c)]
		init_vals[c] = df.iat[0,1]

  Rate = tfa.FitParameter("Rate" , 0.75, 0., 10.)
  I1s =  init_vals["I1s"]
  I2c = init_vals["I2c"]
  I2s = init_vals["I2s"]
  I6c = init_vals["I6c"]
  I6s = init_vals["I6s"]
  I3  = init_vals["I3"]
  I4  = init_vals["I4"]
  I5  = init_vals["I5"]
  I7  = init_vals["I7"]
  I8  = init_vals["I8"]
  I9  = init_vals["I9"]
	
	
	
  #The floating fractions of the bkg
  frac_signal = tfa.FitParameter("frac_signal", frac['signal'] , 0., 1.)
  frac_Ds = tfa.FitParameter("frac_Ds", frac['Ds'] , 0., 1.)
  frac_Dplus = tfa.FitParameter("frac_Dplus", frac['Dplus'] , 0., 1.)
  	
  #File used to create templates (flat sample with unbinned weights)
  template_file = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bd2DstTauNu/3pi_%s_Total/model_vars_weights_hammer_BDT.root" % geom
  branch_names.append("Tau_FD")
  template_sample = read_root(template_file,"DecayTree",columns=branch_names)
  #Keep 1M events
  #template_sample = template_sample.sample(n=1000000,random_state=9289)
  template_sample = template_sample.query(cuts)
  template_sample = template_sample.drop(columns=["Tau_FD"])
  branch_names.remove("Tau_FD")
  #Reorder the columns to required order
  template_sample = template_sample[branch_names]
  template_sample= template_sample.values
  
  #Arrays containing each of the angular weights
  w = {}
  	
  print "Creating weight arrays for each angular term"
  for c in coeffs:
    weight = "w_%s" % c
    branch_names.append(weight)
    branch_names.append("Tau_FD")
    w[c] = read_root(template_file,"DecayTree",columns=branch_names)
    w[c] = w[c].query(cuts)
    w[c] = w[c][[weight]]
    w[c] = w[c].values 
    w[c] = np.reshape(w[c], len(w[c]))
    branch_names.remove(weight)
    branch_names.remove("Tau_FD")
  
  # List to keep template histograms
  histos = {}
  #Make histogram templates for each angular term
  hist_norm = None
  for c in coeffs:
    print "Creating template for term %s " % c
    weight_sample = w[c]
    hist = MakeHistogram(phsp, template_sample, binning, weights = weight_sample)
    if not hist_norm:
      hist_norm = float(HistogramNorm( hist ))
    histos[c] = hist/hist_norm

  #BACKGROUND
  bkg_names = list(frac)
  bkg_names.remove('signal')
  
  bkg_files = {}
  
  for bkg in bkg_names:
     bkg_files[bkg] = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Merged_Bkg/%s_BDT.root" % bkg

  bkg_samples={}
  for bkg in bkg_names:
  	branch_names.append("Tau_FD")
  	bkg_sample = read_root(bkg_files[bkg],"DecayTree",columns=branch_names)
  	bkg_sample = bkg_sample.query(cuts)
  	#Reorder the columns to required order
  	bkg_sample = bkg_sample.drop(columns=["Tau_FD"])
  	branch_names.remove("Tau_FD")
  	bkg_sample = bkg_sample[branch_names]
  	bkg_sample = bkg_sample.values
  	bkg_samples[bkg] = bkg_sample

  histos_bkg = {}
  for bkg in bkg_names:
    hist_bkg_norm = None
    hist_bkg = MakeHistogram(phsp, bkg_samples[bkg], binning)
    if not hist_bkg_norm:
      hist_bkg_norm = float(HistogramNorm( hist_bkg ))
    histos_bkg[bkg] = hist_bkg/hist_bkg_norm
  
  #Fit model
  def fit_model(histos,histos_bkg):
    #pdf = I1c*histos["I1c"]
    pdf = (1.0/3.0)*(4.0 - 6.0*I1s + I2c + 2.0*I2s)*histos["I1c"]
    pdf += I1s*histos["I1s"]
    pdf += I2c*histos["I2c"]
    pdf += I2s*histos["I2s"]
    pdf += I3*histos["I3"]
    pdf += I4*histos["I4"]
    pdf += I5*histos["I5"]
    pdf += I6c*histos["I6c"]
    pdf += I6s*histos["I6s"]
    pdf += I7*histos["I7"]
    pdf += I8*histos["I8"]
    pdf += I9*histos["I9"]  
    pdf = Rate*pdf
    pdf = frac_signal*pdf
    pdf += frac_Ds*histos_bkg["Ds"]
    pdf += frac_Dplus*histos_bkg["Dplus"]
    pdf += feed_frac*frac_signal*histos_bkg["feed"]    
    pdf += frac["D0"]*histos_bkg["D0"]
    pdf += np.abs(1.0 - frac_signal - frac_Ds - frac_Dplus - frac['feed'] - frac["D0"])*histos_bkg["prompt"]
    
    return pdf

  
  if(toy=="N"):
  	data_file_fit= "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bd2DstTauNu/%s_%s_Total/model_vars_weights_hammer_BDT.root" % (sub_mode,geom)
	branch_names.append('hamweight_%s' % ham)
	branch_names.append("Tau_FD")
	data_sample_fit = read_root(data_file_fit,"DecayTree",columns=branch_names)
  	data_sample_fit = data_sample_fit.query(cuts)
  	data_sample_fit = data_sample_fit.drop(columns=["Tau_FD"])
  	#data_sample_fit = data_sample_fit[branch_names]
  
  	#Randomly sample down to required size
  	data_sample_fit = data_sample_fit.sample(n=int(num_sig)*1000,random_state=int(10))
  	
	background_sample_fit = {}
	for bkg in bkg_names:
		background_sample_fit[bkg] = read_root(bkg_files[bkg],"DecayTree",columns=branch_names)
		background_sample_fit[bkg] = background_sample_fit[bkg].query(cuts)
		#background_sample_fit[bkg] = background_sample_fit[bkg][branch_names]
		background_sample_fit[bkg] = background_sample_fit[bkg].sample(n=int(len(data_sample_fit)*(frac[bkg]/frac['signal'])),random_state=int(10))
	
	
	#Add backgrounds to the signal
	for bkg in bkg_names:
		data_sample_fit = pd.concat([data_sample_fit,background_sample_fit[bkg]], ignore_index=True)

	#Peel off Hammer weight into a separate array
	weight = data_sample_fit[['hamweight_%s' % ham]]
	data_sample_fit = data_sample_fit.drop(columns=['hamweight_%s' % ham,"Tau_FD"])
	branch_names.remove('hamweight_%s' % ham)
	branch_names.remove("Tau_FD")
	data_sample_fit = data_sample_fit[branch_names]
	
	data_sample_fit_a = data_sample_fit.values
	weight_a = weight.values
	weight_a = np.reshape(weight_a, len(weight_a))
	fit_hist=MakeHistogram(phsp, data_sample_fit_a, binning,weight_a)
	err_hist=np.sqrt(fit_hist + 0.001)
	norm=HistogramNorm(fit_hist)
  	
  else:
  	init_op = tf.initialize_all_variables()
  	sess.run(init_op)
  	#Create an instance of the fit PDF, then Poisson vary the values in each bin
  	fit_hist=sess.run(fit_model(histos,histos_bkg))
  	#Convert density to number of events
  	fit_hist = fit_hist* (int(num_sig)*1000.0/n_signal) * total_yield
  	fit_hist = np.random.poisson(fit_hist)
  	err_hist=np.sqrt(fit_hist + 0.001)
  	norm=HistogramNorm(fit_hist)
  
  
  # Define binned Chi2 to be minimised
  chi2=BinnedChi2( fit_model(histos,histos_bkg), fit_hist.astype(float)/norm, err_hist.astype(float)/norm )
  # Run Minuit minimisation
  r, c = tfa.RunMinuit(sess, chi2, runHesse=True)
  result=r
  covmat=c
  print result
  
  #Save covariance matrix
  results_dir = ""
  toy_suf = ""
  if(toy=="N"):
    results_dir = "results"
  else:
    results_dir = "toys"
    random.seed(a=None)
    toy_rand = random.randint(1,1e10)
    toy_suf = "_%s" % toy_rand
  
  if syst!="none":
    syst_name="_"+syst
  else:
    syst_name=""

  if syst!="none":	
    filenumber_name="_"+filenumber
  else:
    filenumber_name=""		



  """
  #Write final results
  tfa.WriteFitResults(result,"%s/result_Lifetime_%s_%s_%s_%s_Hammer_%s%s%s%s.txt" % (results_dir,sub_mode,geom,var_type,num_sig,ham,syst_name,filenumber_name,toy_suf))
  #Write initial values
  init_vals = tfa.InitialValues()
  tfa.WriteFitResults(init_vals,"%s/init_Lifetime_%s_%s_%s_%s_Hammer_%s%s%s%s.txt" % (results_dir,sub_mode,geom,var_type,num_sig,ham,syst_name,filenumber_name,toy_suf))
  """
	
