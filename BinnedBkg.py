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
  
  #background fractions
  frac={}
  
  #Fractions defined using Run 1 R(D*) fit
  n_signal = 1296.
  feed_frac = 0.11
  n_feed = feed_frac*n_signal
  n_ds = 6835.
  n_d0 = 1.41 * 445.
  n_dplus = 0.245 * n_ds
  n_prompt = 424.
  total_yield = n_signal + n_ds + n_dplus + n_feed + n_d0 + n_prompt
  
  frac['signal'] = n_signal/total_yield  #floating
  frac['Ds'] = n_ds/total_yield
  frac['Dplus']= n_dplus/total_yield
  frac['feed'] = n_feed/total_yield
  frac['D0'] = n_d0/total_yield
  frac['prompt'] = n_prompt/total_yield
  
  print "Initial component fractions: %s" % frac
  
  #Number of angle bins depending on the signal yield, requiring roughly 50 events per bin
  bdt_bins = 4
  num_bins = int((float(num_sig)*(1.0/bdt_bins)*1000.0/50)**(1.0/3.0))
  
  print "NUMBER OF BINS IN BDT : %s" % bdt_bins
  print "NUMBER OF BINS IN EACH VARIABLE : %s" % num_bins
  
  #Binning scheme
  var_bins = {"costheta_D_%s" % var_type: num_bins,
              "costheta_L_%s" % var_type: num_bins,
              "chi_%s" % var_type: num_bins,
              "BDT" : bdt_bins
              }
	
  var_range = {"costheta_D_%s" % var_type: (-1.,1.),
               "costheta_L_%s" % var_type: (-1.,1.),
               "chi_%s" % var_type: (-math.pi,math.pi),
               "BDT" : (-2.,4.)
               }
  
  var_titles = {"costheta_D_%s" % var_type: "$\\cos(\\theta_D)$",
                "costheta_L_%s" % var_type: "$\\cos(\\theta_L)$",
                "chi_%s" % var_type: "$\\chi$ [rad]",
                "BDT": "BDT"
                }
  	
  # Four body angular phase space is described by 3 angles + q2 for background rejection
  phsp = tfa.RectangularPhaseSpace( ( var_range["costheta_D_%s" % var_type], var_range["costheta_L_%s" % var_type], var_range["chi_%s" % var_type], var_range["BDT"] ))

  # TF initialiser
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  
  branch_names = ["costheta_D_%s" % var_type, "costheta_L_%s" % var_type, "chi_%s" % var_type, "BDT"]
  

  #Read RapidSim sample used to determine bins
  print "Loading tree"
  bin_file = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bd2DstTauNu/%s_%s_Total/model_vars_weights_hammer_BDT.root" % (sub_mode,geom)
  	
  bin_sample = read_root(bin_file,"DecayTree",columns=branch_names)
  bin_sample = bin_sample.query("costheta_D_%s>=-1 and costheta_D_%s<=1 and costheta_L_%s>=-1 and costheta_L_%s<=1 and chi_%s>=-%s and chi_%s<=%s and BDT > %s and BDT <= %s" % (var_type,var_type,var_type,var_type,var_type,math.pi,var_type,math.pi,var_range["BDT"][0],var_range["BDT"][1]))
  #Reorder the columns to required order
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
  
  # Signal fit parameters
  Rate = tfa.FitParameter("Rate" , 0.75, 0., 10.)
  #I1c = tfa.FitParameter("I1c" , 0.576159, -1, 1)
  I1s = tfa.FitParameter("I1s" , 0.415527, -1, 1)
  I2c = tfa.FitParameter("I2c" , -0.183184, -1, 1)
  I2s = tfa.FitParameter("I2s" , 0.074795, -1, 1)
  I6c = tfa.FitParameter("I6c" , 0.360249, -1, 1)
  I6s = tfa.FitParameter("I6s" , -0.262996, -1, 1)
  I3  = tfa.FitParameter("I3"  , -0.121441, -1, 1)
  I4  = tfa.FitParameter("I4"  , -0.150924, -1, 1)
  I5  = tfa.FitParameter("I5"  , 0.298113, -1, 1)
  I7  = tfa.FitParameter("I7"  , 0., -1, 1)
  I8  = tfa.FitParameter("I8"  , 0., -1, 1)
  I9  = tfa.FitParameter("I9"  , 0., -1, 1)

  #The floating fractions of the bkg
  frac_signal = tfa.FitParameter("frac_signal", frac['signal'] , 0., 1.)
  frac_Ds = tfa.FitParameter("frac_Ds", frac['Ds'] , 0., 1.)
  frac_Dplus = tfa.FitParameter("frac_Dplus", frac['Dplus'] , 0., 1.)
  
  coeffs = ["I1c","I1s","I2c","I2s","I6c","I6s","I3","I4","I5","I7","I8","I9"]
  	
  #File used to create templates (flat sample with unbinned weights)
  template_file = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bd2DstTauNu/3pi_%s_Total/model_vars_weights_hammer_BDT.root" % geom
  
  template_sample = read_root(template_file,"DecayTree",columns=branch_names)
  #Keep 1M events
  #template_sample = template_sample.sample(n=1000000,random_state=9289)
  template_sample = template_sample.query("costheta_D_%s>=-1 and costheta_D_%s<=1 and costheta_L_%s>=-1 and costheta_L_%s<=1 and chi_%s>=-%s and chi_%s<=%s and BDT > %s and BDT <= %s" % (var_type,var_type,var_type,var_type,var_type,math.pi,var_type,math.pi,var_range["BDT"][0],var_range["BDT"][1]))  
  #Reorder the columns to required order
  template_sample = template_sample[branch_names]

  template_sample= template_sample.values
  
  #Arrays containing each of the angular weights
  w = {}
  	
  print "Creating weight arrays for each angular term"
  for c in coeffs:
    weight = "w_%s" % c
    branch_names.append(weight)
    w[c] = read_root(template_file,"DecayTree",columns=branch_names)
    #w[c] = w[c].sample(n=1000000,random_state=9289)
    w[c] = w[c].query("costheta_D_%s>=-1 and costheta_D_%s<=1 and costheta_L_%s>=-1 and costheta_L_%s<=1 and chi_%s>=-%s and chi_%s<=%s and BDT > %s and BDT <= %s" % (var_type,var_type,var_type,var_type,var_type,math.pi,var_type,math.pi,var_range["BDT"][0],var_range["BDT"][1]))
    w[c] = w[c][[weight]]
    w[c] = w[c].values
    w[c] = np.reshape(w[c], len(w[c]))
    branch_names.remove(weight)
  
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
    bkg_sample = read_root(bkg_files[bkg],"DecayTree",columns=branch_names)
    bkg_sample = bkg_sample.query("costheta_D_%s>=-1 and costheta_D_%s<=1 and costheta_L_%s>=-1 and costheta_L_%s<=1 and chi_%s>=-%s and chi_%s<=%s and BDT > %s and BDT <= %s" % (var_type,var_type,var_type,var_type,var_type,math.pi,var_type,math.pi,var_range["BDT"][0],var_range["BDT"][1]))
    #Reorder the columns to required order
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
    #pdf += (1.0 - I1c - I1s - I2c - I2s - I3 - I4 - I5 - I6c - I6s - I7 - I8)*histos["I9"]
    
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
	  data_sample_fit = read_root(data_file_fit,"DecayTree",columns=branch_names)
  	data_sample_fit = data_sample_fit.query("costheta_D_%s>=-1 and costheta_D_%s<=1 and costheta_L_%s>=-1 and costheta_L_%s<=1 and chi_%s>=-%s and chi_%s<=%s and BDT > %s and BDT <= %s" % (var_type,var_type,var_type,var_type,var_type,math.pi,var_type,math.pi,var_range["BDT"][0],var_range["BDT"][1]))
  	#data_sample_fit = data_sample_fit[branch_names]
  
  	#Randomly sample down to required size
  	data_sample_fit = data_sample_fit.sample(n=int(num_sig)*1000,random_state=int(num_sig))
  	
	background_sample_fit = {}
	for bkg in bkg_names:
		background_sample_fit[bkg] = read_root(bkg_files[bkg],"DecayTree",columns=branch_names)
		background_sample_fit[bkg] = background_sample_fit[bkg].query("costheta_D_%s>=-1 and costheta_D_%s<=1 and costheta_L_%s>=-1 and costheta_L_%s<=1 and chi_%s>=-%s and chi_%s<=%s and BDT > %s and BDT <= %s" % (var_type,var_type,var_type,var_type,var_type,math.pi,var_type,math.pi,var_range["BDT"][0],var_range["BDT"][1]))
		#background_sample_fit[bkg] = background_sample_fit[bkg][branch_names]
		background_sample_fit[bkg] = background_sample_fit[bkg].sample(n=int(len(data_sample_fit)*(frac[bkg]/frac['signal'])),random_state=int(num_sig))
	
	#Add backgrounds to the signal
	for bkg in bkg_names:
		data_sample_fit = pd.concat([data_sample_fit,background_sample_fit[bkg]], ignore_index=True)

	#Peel off Hammer weight into a separate array
	weight = data_sample_fit[['hamweight_%s' % ham]]
	data_sample_fit = data_sample_fit.drop(columns=['hamweight_%s' % ham])
	branch_names.remove('hamweight_%s' % ham)
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
    toy_rand = random.randint(1,1e10)
    toy_suf = "_%s" % toy_rand
  	
  np.save("%s/cov_Lifetime_%s_%s_%s_%s_Hammer_%s%s.npy" % (results_dir,sub_mode,geom,var_type,num_sig,ham,toy_suf),covmat)
  
  #Derived results
  i9=result['I9'][0]
  i8=result['I8'][0]
  i7=result['I7'][0]
  i6s=result['I6s'][0]
  i6c=result['I6c'][0]
  i4=result['I4'][0]
  i5=result['I5'][0]
  i3=result['I3'][0]
  i2s=result['I2s'][0]
  i2c=result['I2c'][0]
  i1s=result['I1s'][0]
  #i1c=result['I1c'][0]
  rate=result['Rate'][0]
  (rate,i1s,i2c,i2s,i6c,i6s,i3,i4,i5,i7,i8,i9) = correlated_values([rate,i1s,i2c,i2s,i6c,i6s,i3,i4,i5,i7,i8,i9],covmat)
  #(rate,i1c,i1s,i2c,i2s,i6c,i6s,i3,i4,i5,i7,i8) = correlated_values([rate,i1c,i1s,i2c,i2s,i6c,i6s,i3,i4,i5,i7,i8],covmat)
  
  #i9 = 1.0 - i1c - i1s - i2c - i2s - i6c - i6s - i3 - i4 - i5 - i7 - i8
  i1c=(4 - 6*i1s + i2c + 2*i2s)/3
  rab=(i1c+2*i1s-3*i2c-6*i2s)/(2*(i1c+2*i1s+i2c+2*i2s))
  rlt= (3*i1c-i2c)/(2*(3*i1s-i2s))
  Gammaq=(3*i1c+6*i1s-i2c-2*i1s)/4.
  afb1=i6c+2*i6s
  afb=(3/8.)*(afb1/Gammaq)
  a3=(1/(np.pi*2))*i3/Gammaq
  a9=(1/(2*np.pi))*i9/Gammaq
  a6s=(-27/8.)*(i6s/Gammaq)
  a4=(-2/np.pi)*i4/Gammaq
  a8=(2/np.pi)*i8/Gammaq
  a5=(-3/4.)*(1-i8-i7-i9-i4-i3-i2s-i1s-i1c-i2c-i6s-i6c)/Gammaq
  a7=(-3/4.)*i7/Gammaq
  para={'RAB':(rab.n,rab.s),'RLT':(rlt.n,rlt.s),'AFB':(afb.n,afb.s),'A6s':(a6s.n,a6s.s),'A3':(a3.n,a3.s),'A9':(a9.n,a9.s),'A4':(a4.n,a4.s),'A8':(a8.n,a8.s),'A5':(a5.n,a5.s),'A7':(a7.n,a7.s), 'I1c': (i1c.n,i1c.s)}
  p = open( "%s/param_Lifetime_%s_%s_%s_%s_Hammer_%s%s.txt" % (results_dir,sub_mode,geom,var_type,num_sig,ham,toy_suf), "w")
  slist=['RAB','RLT','AFB','A6s','A3','A9','A4','A8','A5','A7','I1c']
  for s in slist:
    a=s+" "
    a += str(para[s][0])
    a += " "
    a += str(para[s][1])
    p.write(a + "\n")
  p.close()
  print para
  
  tfa.WriteFitResults(result,"%s/result_Lifetime_%s_%s_%s_%s_Hammer_%s%s.txt" % (results_dir,sub_mode,geom,var_type,num_sig,ham,toy_suf))
    
	
  #Get final fit PDF
  fit_result = sess.run(fit_model(histos,histos_bkg))
    	
  #1D projections
  fit_hist_proj = {}
  err_hist_proj  = {}
  norm_proj = {}
  fit_result_proj = {}
  data_vals = {}
  
  ds_dplus_feed_d0_prompt = {}
  ds_dplus_d0_prompt = {}
  dplus_d0_prompt = {}
  d0_prompt = {}
  d0 = {}
  
  for b in branch_names:
  	
    axis = [0,1,2,3]
    if(b=="costheta_D_%s" % var_type):
      axis.remove(0)
    elif(b=="costheta_L_%s" % var_type):
      axis.remove(1)
    elif(b=="chi_%s" % var_type):
      axis.remove(2)
    elif(b=="BDT"):
      axis.remove(3)
    	
    if(toy=="N"):
      data_vals["%s" % b] = data_sample_fit[b].values
      #For equi-populated bins
      fit_hist_proj["%s" % b] = MakeHistogram_1D(data_vals["%s" % b], (qc_bin_vals["%s" % b]),weight_a)
      #For equal sized bins
      #fit_hist_proj["%s" % b] = MakeHistogram_1D(data_vals["%s" % b], var_bins[b])
    else:
      fit_hist_proj["%s" % b] = np.sum(fit_hist, axis=tuple(axis), keepdims=False)
   
    err_hist_proj["%s" % b] = np.sqrt(fit_hist_proj["%s" % b])
    norm_proj["%s" % b] = HistogramNorm(fit_hist_proj["%s" % b])
    fit_hist_proj["%s" % b] = fit_hist_proj["%s" % b].astype(float)/norm_proj["%s" % b]
    err_hist_proj["%s" % b] = err_hist_proj["%s" % b].astype(float)/norm_proj["%s" % b]
      
    #Binning for equi-populated bins
    bin_centres = []
    bin_width = []
    for j in range(0,len(qc_bin_vals["%s" % b])-1):
      bin_centres.append(0.5*(qc_bin_vals["%s" % b][j]+qc_bin_vals["%s" % b][j+1]))
      bin_width.append(0.5*(qc_bin_vals["%s" % b][j+1]-qc_bin_vals["%s" % b][j]))
    
    #Binning for equal sized bins
    #bin_width = 0.5*float(var_range[b][1] - var_range[b][0])/var_bins[b]    
    #bin_centres = []
    #for j in range(0,var_bins[b]):
    #	bin_centres.append(var_range[b][0]+bin_width + j*2*bin_width)
      
    fit_result_proj["%s" % b] = np.sum(fit_result, axis=tuple(axis), keepdims=False)
  		
    fig,ax = plt.subplots(figsize=(7,7))
    
    #Plot data
    plt.errorbar(bin_centres,fit_hist_proj["%s" % b],yerr=err_hist_proj["%s" % b],ls='none',color='k',markersize='3',fmt='o',label="Data")
    
    #Components (use ordered dictionary to keep the order)
    comps = collections.OrderedDict()
    
    comps["d0"] = (np.sum(frac["D0"]*histos_bkg["D0"], axis=tuple(axis), keepdims=False), "magenta", "$B \\to D^{*} D^{0} (X)$")
    comps["prompt"] = (comps["d0"][0] + (np.sum((1.0 - frac_signal.fitted_value - frac_Ds.fitted_value - frac_Dplus.fitted_value - frac['feed'] - frac["D0"])*histos_bkg["prompt"], axis=tuple(axis), keepdims=False)), "lightgrey", "$B \\to D^{*} 3\\pi (X)$")
    comps["dplus"] = (comps["prompt"][0] + (np.sum(frac_Dplus.fitted_value*histos_bkg["Dplus"], axis=tuple(axis), keepdims=False)), "blue", "$B \\to D^{*} D^{+} (X)$")
    comps["ds"] = (comps["dplus"][0] + (np.sum(frac_Ds.fitted_value*histos_bkg["Ds"], axis=tuple(axis), keepdims=False)), "orange", "$B \\to D^{*} D_{s} (X)$")
    comps["feed"] = (comps["ds"][0] + (np.sum(feed_frac*frac_signal.fitted_value*histos_bkg["feed"], axis=tuple(axis), keepdims=False)), "aquamarine", "$B \\to D^{**} \\tau \\nu$")
    comps["sig"] = (fit_result_proj["%s" % b], "red", "$B^0 \\to D^{*} \\tau \\nu$")
    
    comp_list = comps.keys()
    comp_list.reverse()
  	
    #Loop over components
    for c in comp_list:
      #Loop over variable bins
      for k in range(0,len(bin_width)):
      	if(k==0):
      		label = comps[c][2]
      	else:
      		label = ""
        plt.bar(bin_centres[k],comps[c][0][k], 2*bin_width[k], color=comps[c][1],label=label)
  			
    plt.ylabel("Density")
    plt.xlabel(var_titles[b])
    plt.legend(loc='lower right')
      
    y_min,y_max = ax.get_ylim()
    plt.ylim(0.0,y_max*1.05)
    plt.show()
      
    if(toy=="N"):
      fig.savefig('figs/%s_%s_%s_%s_%s_Hammer_%s.pdf' % (b,sub_mode,geom,var_type,num_sig,ham))


  #Unrolled 1D plot of all bins
  fit_result_1d = fit_result.ravel()
    
  data_norm = fit_hist.astype(float)/norm
  data_norm_1d = data_norm.ravel()
    
  err_norm = err_hist.astype(float)/norm
  err_norm_1d = err_norm.ravel()
    
  x_max = num_bins**3 * bdt_bins
  x = np.linspace(0,x_max-1,x_max)
    
  fig,ax = plt.subplots(figsize=(15,5))
    
  plt.bar(x,fit_result_1d,edgecolor=None,color='r',alpha=0.5,label="Fit")
  plt.errorbar(x,data_norm_1d,yerr=err_norm_1d,ls='none',color='k',markersize='3',fmt='o',alpha=0.8,label="Data")
    
  plt.ylabel("Density")
  plt.xlabel("Bin number")
  plt.xlim(-1,x_max)
    
  plt.legend()
    
  plt.tight_layout()
  plt.show()
  if(toy=="N"):
    fig.savefig('figs/Fit_Lifetime_%s_%s_%s_%s_Hammer_%s.pdf' % (sub_mode,geom,var_type,num_sig,ham))
      
  		
    #Pull plot	
  pull = (data_norm_1d - fit_result_1d)/err_norm_1d
    
  fig,ax = plt.subplots(figsize=(15,5))
    
  plt.bar(x,pull,edgecolor='navy',color='royalblue',fill=True)
    
  plt.ylabel("Pull ($\sigma$)")
  plt.xlabel("Bin number")
  plt.xlim(-1,x_max)
  plt.ylim(-5,5)
    
  plt.tight_layout()
  plt.show()
  if(toy=="N"):
    fig.savefig('figs/Pull_Lifetime_%s_%s_%s_%s_Hammer_%s.pdf' % (sub_mode,geom,var_type,num_sig,ham))
  	
    #Histogram of the pull values with a fit
  fig,ax = plt.subplots(figsize=(7,7))
      
  pull_bins = int(np.sqrt(x_max))
  n, hist_bins, patches = plt.hist(pull,bins=pull_bins,range=(-5,5),histtype='step',color='navy',normed=True)
    
  plt.xlabel("Pull value ($\\sigma$)")
  plt.ylabel("Fraction of bins")
    
  mu = pull.mean()
  mu_err = sem(pull)
  sigma = pull.std()
    
  plt.title("$\\mu_{Pull} = %.3f \\pm %.3f$, $\\sigma_{Pull} = %.3f$" % (mu,mu_err,sigma))
    
  plt.show()
  if(toy=="N"):
    fig.savefig('figs/Pull_Hist_Lifetime_%s_%s_%s_%s_Hammer_%s.pdf' % (sub_mode,geom,var_type,num_sig,ham))
