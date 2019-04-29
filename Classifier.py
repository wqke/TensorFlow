import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from root_numpy import root2array, rec2array,array2root
from sklearn.metrics import accuracy_score, log_loss, classification_report, roc_auc_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
from root_pandas import *
import sys, os, math
sys.path.append("../../TensorFlowAnalysis")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU
from sklearn.model_selection import train_test_split


import pandas.core.common as com
from pandas.core.index import Index

from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix


branch_names=[ 'Tau_FD_z',  'Tau_E', '3pi_M', '3pi_PZ', 'Tau_m12', 'Tau_m13','Tau_m23',
         'Tau_FD', 'costheta_D_reco','costheta_L_reco','q2_reco','Tau_PZ_reco',
'chi_reco', 'Tau_life_reco']


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
  #Hammer weight (SM / T1 / T2)
  ham = sys.argv[5]

  var_range = {"costheta_D_%s" % var_type: (-1.,1.),
               "costheta_L_%s" % var_type: (-1.,1.),
               "chi_%s" % var_type: (-math.pi,math.pi),
               "Tau_life_%s" % var_type: (0.,3.0)
  }  
  

  frac={}
  M_B = 5.27963
  M_Dst = 2.01026
  
  q2_max = (M_B - M_Dst)**2
  q2_min = M_B - M_Dst
  
  if(var_type=="reco"):
    q2_min = 0.0  
  
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

  bkg_names=list(frac)
  bkg_names.remove('signal')
  data_file_fit= "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bd2DstTauNu/%s_%s_Total/model_vars_weights_hammer.root" % (sub_mode,geom)

  data_sample_fit = read_root(data_file_fit,"DecayTree",columns=branch_names)

  data_sample_fit = data_sample_fit.query("costheta_D_%s>=-1 and costheta_D_%s<=1 and costheta_L_%s>=-1 and costheta_L_%s<=1 and chi_%s>=-%s and chi_%s<=%s and Tau_life_%s > %s and Tau_life_%s <= %s and q2_%s > %s and q2_%s <= %s" % (var_type,var_type,var_type,var_type,var_type,math.pi,var_type,math.pi,var_type,var_range["Tau_life_%s" % var_type][0],var_type,var_range["Tau_life_%s" % var_type][1],var_type,q2_min,var_type,q2_max))
  #data_sample_fit = data_sample_fit[branch_names]

  #Randomly sample down to required size
  data_sample_fit = data_sample_fit.sample(n=int(num_sig)*1000,random_state=int(num_sig))
  
  
  data_sample_fit.to_root("/home/ke/tmps/signal.root","DecayTree")
  signal = root2array("/home/ke/tmps/signal.root","DecayTree",branch_names)
  signal = rec2array(signal)
  
  
  bkg_names.remove('signal')
  bkg_files={}
  for bkg in bkg_names:
  	bkg_files[bkg] = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Merged_Bkg/%s.root" % bkg
    
    
  background_sample_fit={}
  for bkg in bkg_names:
    background_sample_fit[bkg] = read_root(bkg_files[bkg],"DecayTree",columns=branch_names)
    background_sample_fit[bkg] = background_sample_fit[bkg].query("costheta_D_%s>=-1 and costheta_D_%s<=1 and costheta_L_%s>=-1 and costheta_L_%s<=1 and chi_%s>=-%s and chi_%s<=%s and Tau_life_%s > %s and Tau_life_%s <= %s and q2_%s > %s and q2_%s <= %s" % (var_type,var_type,var_type,var_type,var_type,math.pi,var_type,math.pi,var_type,var_range["Tau_life_%s" % var_type][0],var_type,var_range["Tau_life_%s" % var_type][1],var_type,q2_min,var_type,q2_max))
    background_sample_fit[bkg] = background_sample_fit[bkg][branch_names]
    background_sample_fit[bkg] = background_sample_fit[bkg].sample(n=int(len(data_sample_fit)*(frac[bkg]/frac['signal'])),random_state=int(num_sig))

  bkg_samp=background_sample_fit[bkg_names[0]]
  for bkg in bkg_names[1:] :   
    bkg_samp = pd.concat([bkg_samp,background_sample_fit[bkg]], ignore_index=True)
    
#  #Add backgrounds to the signal
#  for bkg in bkg_names:
#    data_sample_fit = pd.concat([data_sample_fit,background_sample_fit[bkg]], ignore_index=True)

  
  bkg_samp.to_root("/home/ke/tmps/bkg.root","DecayTree")
  backgr = root2array("/home/ke/tmps/bkg.root","DecayTree",branch_names)
  backgr = rec2array(backgr)
    


  # for sklearn data is usually organised
  # into one 2D array of shape (n_samples x n_features)
  # containing all the data and one array of categories
  # of length n_samples
  X = np.concatenate((signal, backgr))
  y = np.concatenate((np.ones(signal.shape[0]),
                      np.zeros(backgr.shape[0])))     #signal=1,bkg=0
    
  #df = pd.DataFrame(np.hstack((X, y.reshape(y.shape[0], -1))),columns=branch_names+['y'])  
  
    
  X_dev,X_eval, y_dev,y_eval = train_test_split(X, y,
                                              test_size=0.33, random_state=42)
  X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev,
                                                  test_size=0.33, random_state=492)  
  
  
  
  

  bdt = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=len(branch_names), max_depth = 2, random_state = 0)

  bdt.fit(X_train, y_train)


  def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30):
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.savefig('ouput.pdf')
    
    
  compare_train_test(bdt, X_train, y_train, X_test, y_test)
