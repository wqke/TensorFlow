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
# ===============================================================================

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


from matplotlib.ticker import AutoMinorLocator


from matplotlib import rc 
rc('font',**{'family':'serif','serif':['Roman']}) 
rc('text', usetex=True)


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



def BinnedLikelihood(pdf,data,norm):
	#data = observed number of events in each bin
	#pdf = normalised PDF
	#norm = total number of events
	
	#Convert PDF to expected events in each bin
	pdf = tf.scalar_mul(norm,pdf)
	return tf.reduce_sum(pdf - data + data*tf.log(data/pdf))

frac={}
#Branching fractions from PDG
frac['B2D0TauNu']=7.7e-3
frac['B2DstTauNu']=1.57e-2
frac['Dst2D0Pi']=0.677
#D0 and D* fractions in the sample
frac['D0']=frac['B2D0TauNu']+frac['B2DstTauNu']*frac['Dst2D0Pi']   #0.0183
frac['Dst']=frac['B2DstTauNu']                                     #0.0157
tot=frac['D0']+frac['Dst']*0.5					   #total sample=D0 sample + D* whose pion is reconstructed
#initial values used in the fit
frac['Dst']=frac['Dst']/tot
frac['D0']=frac['D0']/tot			                   #redefine to have total fraction=1
frac['DstinD0']=frac['Dst']*0.5/frac['D0']     		           #the fraction of D* reco as D0 in D0 sample



if __name__ == "__main__" :

  #Read RapidSim signal sample for either 3pi mode or 3pipi0 mode
  mode = "Bu2D0TauNu"
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
  
  #Number of angle bins depending on the signal yield, requiring roughly 25 events per bin
  #bdt_bins = 3
  #num_bins = int(math.ceil((float(num_sig)*(1.0/bdt_bins)*1000.0/25)**(1.0/3.0)))
  num_bins = int(math.ceil((float(num_sig)*1000.0/25)**(1.0/3.0)))
  
  print "NUMBER OF BINS IN EACH ANGLE : %s" % num_bins
  
  #Binning scheme
  var_bins = {"costheta_D_%s" % var_type: num_bins,
              "costheta_L_%s" % var_type: num_bins,
              "chi_%s" % var_type: num_bins
              }
	
  var_range = {"costheta_D_%s" % var_type: (-1.,1.),
               "costheta_L_%s" % var_type: (-1.,1.),
               "chi_%s" % var_type: (-math.pi,math.pi)
               }
  
  var_titles = {"costheta_D_%s" % var_type: "$\\cos(\\theta_D)$",
                "costheta_L_%s" % var_type: "$\\cos(\\theta_L)$",
                "chi_%s" % var_type: "$\\chi$ [rad]"
                }

  # Four body angular phase space is described by 3 angles + q2 for background rejection
  phsp = tfa.RectangularPhaseSpace( ( var_range["costheta_D_%s" % var_type], var_range["costheta_L_%s" % var_type], var_range["chi_%s" % var_type]))

  # TF initialiser
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  
  branch_names = ["costheta_D_%s" % var_type, "costheta_L_%s" % var_type, "chi_%s" % var_type]
  
  #Add tau flight which must be cut on
  branch_names.append("Tau_FD")
  #Cut to mimic the z flight distance significance cut
  Tau_FD_cut = 4000.

  #Read RapidSim sample used to determine bins
  print "Loading tree"
  bin_file_D0 = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bu2D0TauNu/%s_%s_Total/model_vars_weights.root" % (sub_mode,geom)
  bin_file_Dst = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bd2DstTauNu/3pi_LHCb_Total/merged_signal.root"
  
  ###
  
  bin_file_Dst_in_D0 = ""
  
  ###
  
  cuts = "Tau_FD > %s  and costheta_D_%s>=-1 and costheta_D_%s<=1 and costheta_L_%s>=-1 and costheta_L_%s<=1 and chi_%s>=-%s and chi_%s<=%s" % (Tau_FD_cut,var_type,var_type,var_type,var_type,var_type,math.pi,var_type,math.pi)
  
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

  binning=(qc_bin_vals["costheta_D_%s" % var_type],qc_bin_vals["costheta_L_%s" % var_type],qc_bin_vals["chi_%s" % var_type])
  
  coeffs = ["A","B","C"]
  Icoeffs = ["I1c","I1s","I2c","I2s","I6c","I6s","I3","I4","I5","I7","I8","I9"]
  
  # Signal fit parameters
  init_vals = {"A": 0.544970,
  			   "B": 0.365668
  			   }
  
  
  Iinit_vals = {"I1s": 0.415527,
  			   "I2c": -0.183184,
  			   "I2s": 0.074795,
  			   "I6c": -0.360249,
  			   "I6s": 0.262996,
  			   "I3": -0.121441,
  			   "I4": -0.150924,
  			   "I5": -0.298113,
  			   "I7": 0.0,
  			   "I8": 0.0,
  			   "I9": 0.0
           }  
  
  
  
  #If running a toy, assign starting values using truth-level unbinned fit results
  if(toy=="Y"):
  	default_results = pd.read_csv("../3D_Unbinned_SignalOnly_D0TauNu/results/result_3pi_all_true_100.txt", header = None, sep=" ")
  	for c in init_vals:
		  df = default_results[default_results[0].str.contains("%s" % c)]
		  init_vals[c] = df.iat[0,1]
      
  	Idefault_results = pd.read_csv("../3D_Unbinned_SignalOnly/results/result_3pi_all_true_100_Hammer_N.txt", header = None, sep=" ")
  	for c in Iinit_vals:
		  dg = Idefault_results[Idefault_results[0].str.contains("%s" % c)]
      Iinit_vals[c] = dg.iat[0,1]		
      
  A = tfa.FitParameter("A" , init_vals["A"], -1, 1)
  B = tfa.FitParameter("B" , init_vals["B"], -1, 1)
  
  #Rate = tfa.FitParameter("Rate" , 0.75, 0., 10.)
  I1s = tfa.FitParameter("I1s" , Iinit_vals["I1s"], -1, 1)
  I2c = tfa.FitParameter("I2c" , Iinit_vals["I2c"], -1, 1)
  I2s = tfa.FitParameter("I2s" , Iinit_vals["I2s"], -1, 1)
  I6c = tfa.FitParameter("I6c" , Iinit_vals["I6c"], -1, 1)
  I6s = tfa.FitParameter("I6s" , Iinit_vals["I6s"], -1, 1)
  I3  = tfa.FitParameter("I3"  , Iinit_vals["I3"], -1, 1)
  I4  = tfa.FitParameter("I4"  , Iinit_vals["I4"], -1, 1)
  I5  = tfa.FitParameter("I5"  , Iinit_vals["I5"], -1, 1)
  I7  = tfa.FitParameter("I7"  , Iinit_vals["I7"], -1, 1)
  I8  = tfa.FitParameter("I8"  , Iinit_vals["I8"], -1, 1)
  I9  = tfa.FitParameter("I9"  , Iinit_vals["I9"], -1, 1)

  frac_Dst_in_D0  = tfa.FitParameter("frca_Dst_in_D0"  ,frac['DstinD0'] , 0., 1.)
  frac_D0  = tfa.FitParameter("frac_D0"  , frac['D0'] , 0., 1.)
  #frac_Dst_not_in_D0=1-frac_D0
	
	
  #File used to create templates (flat sample with unbinned weights)
  template_file_D0 = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bu2D0TauNu/3pi_%s_Total/model_vars_weights.root" % geom
  branch_names.append("Tau_FD")
  template_sample_D0 = read_root(template_file_D0,"DecayTree",columns=branch_names)
  #Keep 1M events
  #template_sample = template_sample.sample(n=1000000,random_state=9289)
  template_sample_D0 = template_sample_D0.query(cuts)
  template_sample_D0 = template_sample_D0.drop(columns=["Tau_FD"])
  branch_names_D0.remove("Tau_FD")
  #Reorder the columns to required order
  template_sample_D0 = template_sample_D0[branch_names]
  template_sample_D0= template_sample_D0.values
  
  template_file_Dst = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bd2DstTauNu/3pi_%s_Total/merged_signal.root" % geom
  branch_names.append("Tau_FD")
  template_sample_Dst = read_root(template_file_Dst,"DecayTree",columns=branch_names)
  #Keep 1M events
  #template_sample = template_sample.sample(n=1000000,random_state=9289)
  template_sample_Dst = template_sample_Dst.query(cuts)
  template_sample_Dst = template_sample_Dst.drop(columns=["Tau_FD"])
  branch_names_Dst.remove("Tau_FD")
  #Reorder the columns to required order
  template_sample_Dst = template_sample_Dst[branch_names]
  template_sample_Dst= template_sample_Dst.values
   
  template_file_Dst_in_D0 = "    "
  branch_names.append("Tau_FD")
  template_sample_Dst_in_D0 = read_root(template_file_Dst_in_D0,"DecayTree",columns=branch_names)
  #Keep 1M events
  #template_sample = template_sample.sample(n=1000000,random_state=9289)
  template_sample_Dst_in_D0 = template_sample_Dst_in_D0.query(cuts)
  template_sample_Dst_in_D0 = template_sample_Dst_in_D0.drop(columns=["Tau_FD"])
  branch_names_Dst_in_D0.remove("Tau_FD")
  #Reorder the columns to required order
  template_sample_Dst_in_D0 = template_sample_Dst_in_D0[branch_names]
  template_sample_Dst_in_D0= template_sample_Dst_in_D0.values
    
  
  
  #Arrays containing each of the angular weights
  w_D0 = {}
  w_Dst = {}
  w_Dst_in_D0 = {}
  
  print "Creating weight arrays for each angular term"
  for c in coeffs:
    weight = "w_%s" % c
    branch_names.append(weight)
    branch_names.append("Tau_FD")
    w_D0[c] = read_root(template_file,"DecayTree",columns=branch_names)
    #w[c] = w[c].sample(n=1000000,random_state=9289)
    w_D0[c] = w_D0[c].query(cuts)
    w_D0[c] = w_D0[c][[weight]]
    w_D0[c] = w_D0[c].values 
    w_D0[c] = np.reshape(w_D0[c], len(w_D0[c]))
    branch_names.remove(weight)
    branch_names.remove("Tau_FD")
  
  # List to keep template histograms
  histos_D0 = {}
  histos_Dst = {}
  histos_Dst_in_D0 = {}
  
  #Make histogram templates for each angular term
  hist_norm = None
  for c in coeffs:
    print "Creating template for term %s " % c
    weight_sample = w_D0[c]
    hist = MakeHistogram(phsp, template_sample, binning, weights = weight_sample)
    if not hist_norm:
      hist_norm = float(HistogramNorm( hist ))
    histos_D0[c] = hist/hist_norm

  #Fit model
  def fit_model(histos_D0,histos_Dst,histos_Dst_in_D0):
    pdf = A * histos_D0["A"]
    pdf += B * histos_D0["B"]
    pdf += 3.0 * (1.0/2.0 - A) * histos_D0["C"] 
    #Normalise the PDF at each iteration
    pdf_array = sess.run(pdf)
    pdf = (1.0/np.sum(pdf_array))*pdf
    pdf = frac_D0 * pdf
    
    pdf_Dst = (1.0/3.0)*(4.0 - 6.0*I1s + I2c + 2.0*I2s)*histos_Dst["I1c"]
    pdf_Dst += I1s*histos_Dst["I1s"]
    pdf_Dst += I2c*histos_Dst["I2c"]
    pdf_Dst += I2s*histos_Dst["I2s"]
    pdf_Dst += I3*histos_Dst["I3"]
    pdf_Dst += I4*histos_Dst["I4"]
    pdf_Dst += -I5*histos_Dst["I5"]
    pdf_Dst += -I6c*histos_Dst["I6c"]
    pdf_Dst += -I6s*histos_Dst["I6s"]
    pdf_Dst += I7*histos_Dst["I7"]
    pdf_Dst += -I8*histos_Dst["I8"]
    pdf_Dst += -I9*histos_Dst["I9"]
    
    pdf_Dst_in_D0 = (1.0/3.0)*(4.0 - 6.0*I1s + I2c + 2.0*I2s)*histos_Dst_in_D0["I1c"]
    pdf_Dst_in_D0 += I1s*histos_Dst_in_D0["I1s"]
    pdf_Dst_in_D0 += I2c*histos_Dst_in_D0["I2c"]
    pdf_Dst_in_D0 += I2s*histos_Dst_in_D0["I2s"]
    pdf_Dst_in_D0 += I3*histos_Dst_in_D0["I3"]
    pdf_Dst_in_D0 += I4*histos_Dst_in_D0["I4"]
    pdf_Dst_in_D0 += -I5*histos_Dst_in_D0["I5"]
    pdf_Dst_in_D0 += -I6c*histos_Dst_in_D0["I6c"]
    pdf_Dst_in_D0 += -I6s*histos_Dst_in_D0["I6s"]
    pdf_Dst_in_D0 += I7*histos_Dst_in_D0["I7"]
    pdf_Dst_in_D0 += -I8*histos_Dst_in_D0["I8"]
    pdf_Dst_in_D0 += -I9*histos_Dst_in_D0["I9"]
    
    
    #normalise separately the 2 pdf's
    pdf_Dst_array = sess.run(pdf_Dst)
    pdf_Dst = (1.0/np.sum(pdf_Dst_array))*pdf_Dst
    pdf_Dst = (1.-frac_D0) * pdf_Dst
    
    pdf_Dst_in_D0_array = sess.run(pdf_Dst_in_D0)
    pdf_Dst_in_D0 = (1.0/np.sum(pdf_Dst_in_D0_array))*pdf_Dst_in_D0
    pdf_Dst_in_D0 = frac_Dst_in_D0 * pdf_Dst_in_D0
    
    pdf += pdf_Dst
    pdf += pdf_Dst_in_D0
    
    return pdf

  if(toy=="N"):
  	data_file_fit= "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bu2D0TauNu/%s_%s_Total/model_vars_weights.root" % (sub_mode,geom)
	
	  branch_names.append("Tau_FD")
	  data_sample_fit = read_root(data_file_fit,"DecayTree",columns=branch_names)
  	data_sample_fit = data_sample_fit.query(cuts)
  	#Randomly sample down to required size
  	data_sample_fit = data_sample_fit.sample(n=int(num_sig)*1000,random_state=int(10))
  	branch_names.remove("Tau_FD")
  	data_sample_fit = data_sample_fit[branch_names]
  	data_sample_fit_a = data_sample_fit.values
  	
  	fit_hist=MakeHistogram(phsp, data_sample_fit_a, binning)
  	err_hist=np.sqrt(fit_hist + 0.001)
	norm=HistogramNorm(fit_hist)

  else:
  	init_op = tf.initialize_all_variables()
  	sess.run(init_op)
  	#Create an instance of the fit PDF, then Poisson vary the values in each bin
  	fit_hist=sess.run(fit_model(histos))
  	#Convert density to number of events
  	fit_hist_default = fit_hist * (1.0/np.sum(fit_hist)) * (int(num_sig)*1000.0)
  	#Poisson vary each bin
  	fit_hist = np.random.poisson(fit_hist_default)
  	fit_hist = fit_hist.astype(float)
  	err_hist=np.sqrt(fit_hist + 0.001)
  	norm=HistogramNorm(fit_hist)
  
  
  # Define binned Chi2 to be minimised
  init_op = tf.initialize_all_variables()
  sess.run(init_op)
  #chi2=BinnedChi2( fit_model(histos), fit_hist.astype(float)/norm, err_hist.astype(float)/norm )
  logL = BinnedLikelihood(fit_model(histos), fit_hist.astype(float), norm)
  # Run Minuit minimisation
  #r, c = tfa.RunMinuit(sess, chi2, runHesse=True)
  r, c = tfa.RunMinuit(sess, logL, runHesse=True)
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


  np.save("%s/cov_Lifetime_%s_%s_%s_%s%s.npy" % (results_dir,sub_mode,geom,var_type,num_sig,toy_suf),covmat)
  
  #Derived results
  a=result['A'][0]
  b=result['B'][0]

  (a,b) = correlated_values([a,b],covmat)
  
  c = 3.0*(1.0/2.0 - a)
  
  para={'C':(c.n,c.s)}
  p = open( "%s/param_Lifetime_%s_%s_%s_%s%s.txt" % (results_dir,sub_mode,geom,var_type,num_sig,toy_suf), "w")
  slist=["C"]
  for s in slist:
    v=s+" "
    v += str(para[s][0])
    v += " "
    v += str(para[s][1])
    p.write(v + "\n")
  p.close()
  print para
  
  #Write final results
  tfa.WriteFitResults(result,"%s/result_Lifetime_%s_%s_%s_%s%s.txt" % (results_dir,sub_mode,geom,var_type,num_sig,toy_suf))
  #Write initial values
  init_vals = tfa.InitialValues()
tfa.WriteFitResults(init_vals,"%s/init_Lifetime_%s_%s_%s_%s%s.txt" % (results_dir,sub_mode,geom,var_type,num_sig,toy_suf))




