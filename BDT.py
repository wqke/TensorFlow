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
bdt = joblib.load('bdt.joblib')

feed = root2array("/home/ke/tmps/signal.root","DecayTree",branch_names)
feed = rec2array(signal)

X_test = 


y_predicted = bdt.predict(X_test)

