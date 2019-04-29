import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from root_numpy import root2array, rec2array,array2root
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, log_loss, classification_report, roc_auc_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
from root_pandas import *
import sys, os, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU
from sklearn.model_selection import  train_test_split,KFold
from sklearn.utils.class_weight import compute_sample_weight

import joblib

import pandas.core.common as com
from pandas.core.index import Index

from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix


branch_names=['3pi_M',  'Tau_m12', 'Tau_m13','Tau_m23','Tau_FD','Tau_life_reco']


if __name__ == "__main__" :

  #Read RapidSim signal sample for either 3pi mode or 3pipi0 mode
  mode = "Bd2DstTauNu"
  sub_mode = '3pi'
  geom = 'LHCb'
  var_type = 'reco'
  ham = 'SM'

  var_range = {"Tau_life_%s" % var_type: (0.,3.0)}  
  
  data_file_fit= "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bd2DstTauNu/%s_%s_Total/model_vars_weights_hammer.root" % (sub_mode,geom)

  data_sample_fit = read_root(data_file_fit,"DecayTree",columns=branch_names)

  data_sample_fit = data_sample_fit.query("Tau_life_%s > %s and Tau_life_%s <= %s and Tau_FD>4000" % (var_type,var_range["Tau_life_%s" % var_type][0],var_type,var_range["Tau_life_%s" % var_type][1]))
  #data_sample_fit = data_sample_fit[branch_names]

  #Randomly sample down to required size
  #data_sample_fit = data_sample_fit.sample(n=int(num_sig)*1000,random_state=int(num_sig))
  
  
  data_sample_fit.to_root("/home/ke/tmps/signal.root","DecayTree")
  signal = root2array("/home/ke/tmps/signal.root","DecayTree",branch_names)
  signal = rec2array(signal)
  
  

  bkg_names=['Ds','Dplus','D0']
  bkg_files={}
  for bkg in bkg_names:
    bkg_files[bkg] = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Merged_Bkg/%s.root" % bkg


  #keep the same number of D backgrounds  (equal to Dplus numbers)
  background_sample_fit={}     
  background_sample_fit['Dplus']= read_root(bkg_files['Dplus'],"DecayTree",columns=branch_names)     
  background_sample_fit['Dplus'] = background_sample_fit['Dplus'].query("Tau_life_%s > %s and Tau_life_%s <= %s and Tau_FD>4000" % (var_type,var_range["Tau_life_%s" % var_type][0],var_type,var_range["Tau_life_%s" % var_type][1]))

  sample_length=len(background_sample_fit['Dplus'])       
  background_sample_fit['Ds']= read_root(bkg_files['Ds'],"DecayTree",columns=branch_names)      
  background_sample_fit['Ds'] = background_sample_fit['Ds'].query("Tau_life_%s > %s and Tau_life_%s <= %s and Tau_FD>4000" % (var_type,var_range["Tau_life_%s" % var_type][0],var_type,var_range["Tau_life_%s" % var_type][1]))
  background_sample_fit['Ds'] = background_sample_fit['Ds'].sample(n=int(sample_length),random_state=int(sample_length/1000.))

  background_sample_fit['D0']= read_root(bkg_files['D0'],"DecayTree",columns=branch_names)      
  background_sample_fit['D0'] = background_sample_fit['D0'].query("Tau_life_%s > %s and Tau_life_%s <= %s and Tau_FD>4000" % (var_type,var_range["Tau_life_%s" % var_type][0],var_type,var_range["Tau_life_%s" % var_type][1]))
  background_sample_fit['D0'] = background_sample_fit['D0'].sample(n=int(sample_length),random_state=int(sample_length/1000.))

  bkg_samp=background_sample_fit[bkg_names[0]]
  for bkg in bkg_names[1:] :   
    bkg_samp = pd.concat([bkg_samp,background_sample_fit[bkg]], ignore_index=True)


#  #Add backgrounds to the signal

  bkg_samp.to_root("/home/ke/tmps/bkg.root","DecayTree")
  backgr = root2array("/home/ke/tmps/bkg.root","DecayTree",branch_names)
  backgr = rec2array(backgr)

  # for sklearn data is usually organised
  # into one 2D array of shape (n_samples x n_features)
  # containing all the data and one array of categories
  # of length n_samples
  X = np.concatenate((signal, backgr))
  y = np.concatenate((np.ones(signal.shape[0]),np.zeros(backgr.shape[0])))     #signal=1,bkg=0
    
  #df = pd.DataFrame(np.hstack((X, y.reshape(y.shape[0], -1))),columns=branch_names+['y'])  
  
    
  X_dev,X_eval, y_dev,y_eval = train_test_split(X, y,test_size=0.33, random_state=42)
  X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev,test_size=0.33, random_state=492)  
  
  
  #weight the samples such that they are treated as having equal size statistically
  weights = compute_sample_weight(class_weight='balanced', y=y_train)
  bdt = GradientBoostingClassifier(n_estimators=1000, max_depth=1, learning_rate=0.1, min_samples_split=2,verbose=1)
  bdt.fit(X_train, y_train,sample_weight=weights)

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
    
    hist, bins = np.histogram(decisions[3],bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.savefig('ouput.pdf')
    
    
  compare_train_test(bdt, X_train, y_train, X_test, y_test)

  y_predicted = bdt.decision_function(X)
  y_predicted.dtype = [('y', np.float64)]

  y_predicted = bdt.predict(X_test)
  print classification_report(y_test, y_predicted,
                              target_names=["background", "signal"])
  print "Area under ROC curve: %.4f"%(roc_auc_score(y_test,
                                                  bdt.decision_function(X_test)))


  
  y_predicted = bdt.decision_function(X)
  y_predicted.dtype = [('y', np.float64)]
  array2root(y_predicted, "/home/ke/tmps/test-prediction.root", "BDToutput")
  
  
  joblib.dump(bdt, 'bdt.joblib')
  #To load in other scripts : bdt = joblib.load('bdt.joblib')
  
