
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numpy import *
import tensorflow as tf
import sys, os
import numpy as np
import math
from math import cos,sin,pi
import root_pandas
import pandas as pd
import TensorFlowAnalysis as tfa
from tensorflow.python.client import timeline
from root_numpy import root2array, rec2array, tree2array
from ROOT import TFile,TChain,TTree
from uncertainties import *

#Bd2DstDs1
files=['dsgamma_5pi_LHCb_Total/model_vars.root',
'dsgamma_etapi_LHCb_Total/model_vars.root',
'dsgamma_etappi_etapipi_LHCb_Total/model_vars.root',
'dsgamma_etappi_rhogamma_LHCb_Total/model_vars.root',
'dsgamma_etaprho_etapipi_LHCb_Total/model_vars.root',
'dsgamma_etaprho_rhogamma_LHCb_Total/model_vars.root',
'dsgamma_etarho_LHCb_Total/model_vars.root',
'dsgamma_omega3pi_LHCb_Total/model_vars.root',
'dsgamma_omegapi_LHCb_Total/model_vars.root',
'dsgamma_omegarho_LHCb_Total/model_vars.root',
'dsstpi0_dsgamma_5pi_LHCb_Total/model_vars.root',
'dsstpi0_dsgamma_etappi_etapipi_LHCb_Total/model_vars.root',
'dsstpi0_dsgamma_etappi_rhogamma_LHCb_Total/model_vars.root',
'dsstpi0_dsgamma_etaprho_etapipi_LHCb_Total/model_vars.root',
'dsstpi0_dsgamma_etaprho_rhogamma_LHCb_Total/model_vars.root',
'dsstpi0_dsgamma_etarho_LHCb_Total/model_vars.root',
'dsstpi0_dsgamma_omega3pi_LHCb_Total/model_vars.root',
'dsstpi0_dsgamma_omegapi_LHCb_Total/model_vars.root',
'dsstpi0_dsgamma_omegarho_LHCb_Total/model_vars.root']

#Bd2DstDs
"""
5pi_LHCb_Total/model_vars.root
etapi_LHCb_Total/model_vars.root
etappi_etapipi_LHCb_Total/model_vars.root
etappi_rhogamma_LHCb_Total/model_vars.root
etaprho_etapipi_LHCb_Total/model_vars.root
etaprho_rhogamma_LHCb_Total/model_vars.root
etarho_LHCb_Total/model_vars.root
omega3pi_LHCb_Total/model_vars.root
omegapi_LHCb_Total/model_vars.root
omegarho_LHCb_Total/model_vars.root

"""


for file in files:
  df=root_pandas.read_root(file,key='DecayTree')


