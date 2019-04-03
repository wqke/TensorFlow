
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
from pandas import * 
from tensorflow.python.client import timeline
from root_numpy import root2array, rec2array, tree2array
from ROOT import TFile,TChain,TTree
from uncertainties import *

#Bd2DstDs1

files=['/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsgamma_5pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsgamma_etapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsgamma_etappi_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsgamma_etappi_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsgamma_etaprho_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsgamma_etaprho_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsgamma_etarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsgamma_omega3pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsgamma_omegapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsgamma_omegarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsstpi0_dsgamma_5pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsstpi0_dsgamma_etappi_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsstpi0_dsgamma_etappi_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsstpi0_dsgamma_etaprho_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsstpi0_dsgamma_etaprho_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsstpi0_dsgamma_etarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsstpi0_dsgamma_omega3pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsstpi0_dsgamma_omegapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs1/dsstpi0_dsgamma_omegarho_LHCb_Total/model_vars.root']


dg=root_pandas.read_root(files,columns=['q2_reco','costheta_L_reco','costheta_D_reco','chi_reco'],key='DecayTree')



plt.hist(qlist[~np.isnan(qlist)],bins=100,density=True,histtype='step')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)
plt.savefig('/home/ke/graphs/Bd2DstDs1_q.pdf')
plt.close()



plt.hist(chilist[~np.isnan(chilist)],density=True,histtype='step')  
plt.title(r'$\chi$')
plt.savefig('/home/ke/graphs/Bd2DstDs1_chi.pdf')
plt.close()
"""
plt.hist(qlist[~np.isnan(qlist)]bins=100,,density=True,histtype='step')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)
plt.savefig('/home/ke/graphs/Bd2DstDs1_q.pdf')
plt.close()
plt.hist(qlist1[~np.isnan(qlist1)],bins=100,density=True,histtype='step')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)
plt.savefig('/home/ke/graphs/Bd2DstDs_q.pdf')
plt.close()
plt.hist(qlist2[~np.isnan(qlist2)],bins=100,density=True,histtype='step')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)
plt.savefig('/home/ke/graphs/Bd2DstDsst_q.pdf')
plt.close()
plt.hist(qlist3[~np.isnan(qlist3)],bins=100,density=True,histtype='step')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)
plt.savefig('/home/ke/graphs/Bu2DststDs_q.pdf')
plt.close()
"""



#Bd2DstDs

files1=['/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs/5pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs/etapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs/etappi_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs/etappi_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs/etaprho_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs/etaprho_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs/etarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs/omega3pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs/omegapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDs/omegarho_LHCb_Total/model_vars.root']



dg1=root_pandas.read_root(files1,columns=['q2_reco','costheta_L_reco','costheta_D_reco','chi_reco'],key='DecayTree')

qlist=dg['q2_reco']
Llist=dg['costheta_L_reco']
Dlist=dg['costheta_D_reco']
chilist=dg['chi_reco']

for file in files[1:]:
  df=root_pandas.read_root(file,key='DecayTree')
  qlist.append(df['q2_reco'],ignore_index=True)#[~np.isnan(df['q2_reco'])]
  Llist.append(df['costheta_L_reco'],ignore_index=True)#[~np.isnan(df['costheta_L_reco'])]
  Dlist.append(df['costheta_D_reco'],ignore_index=True)#[~np.isnan(df['costheta_D_reco'])]
  chilist.append(df['chi_reco'],ignore_index=True)#[~np.isnan(df['chi_reco'])]

  


qlist1=dg1['q2_reco']
Llist1=dg1['costheta_L_reco']
Dlist1=dg1['costheta_D_reco']
chilist1=dg1['chi_reco']

for file in files1[1:]:
  df=root_pandas.read_root(file,key='DecayTree')
  qlist1.append(df['q2_reco'],ignore_index=True)#[~np.isnan(df['q2_reco'])]
  Llist1.append(df['costheta_L_reco'],ignore_index=True)#[~np.isnan(df['costheta_L_reco'])]
  Dlist1.append(df['costheta_D_reco'],ignore_index=True)#[~np.isnan(df['costheta_D_reco'])]
  chilist1.append(df['chi_reco'],ignore_index=True)#[~np.isnan(df['chi_reco'])]

plt.hist(qlist1[~np.isnan(qlist1)],bins=100,density=True,histtype='step')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)
plt.savefig('/home/ke/graphs/Bd2DstDs_q.pdf')
plt.close()

plt.hist(Llist1[~np.isnan(Llist1)],density=True,histtype='step')  
plt.title(r'cos($\theta_L$)')
plt.savefig('/home/ke/graphs/Bd2DstDs_L.pdf')
plt.close()


plt.hist(Dlist1[~np.isnan(Dlist1)],density=True,histtype='step')  
plt.title(r'cos($\theta_D$)')
plt.savefig('/home/ke/graphs/Bd2DstDs_D.pdf')
plt.close()

plt.hist(chilist1[~np.isnan(chilist1)],density=True,histtype='step')  
plt.title(r'$\chi$')
plt.savefig('/home/ke/graphs/Bd2DstDs_chi.pdf')
plt.close()


#Bd2DstDsst

files2=['/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dsgamma_5pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dsgamma_etapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dsgamma_etappi_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dsgamma_etappi_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dsgamma_etaprho_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dsgamma_etaprho_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dsgamma_etarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dsgamma_omega3pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dsgamma_omegapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dsgamma_omegarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dspi0_5pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dspi0_etapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dspi0_etappi_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dspi0_etappi_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dspi0_etaprho_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dspi0_etaprho_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dspi0_etarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dspi0_omega3pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dspi0_omegapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bd2DstDsst/dspi0_omegarho_LHCb_Total/model_vars.root']

dg2=root_pandas.read_root(files2,columns=['q2_reco','costheta_L_reco','costheta_D_reco','chi_reco'],key='DecayTree')

qlist2=dg2['q2_reco']
Llist2=dg2['costheta_L_reco']
Dlist2=dg2['costheta_D_reco']
chilist2=dg2['chi_reco']

for file in files2[1:]:
  df=root_pandas.read_root(file,key='DecayTree')
  qlist2.append(df['q2_reco'],ignore_index=True)#[~np.isnan(df['q2_reco'])]
  Llist2.append(df['costheta_L_reco'],ignore_index=True)#[~np.isnan(df['costheta_L_reco'])]
  Dlist2.append(df['costheta_D_reco'],ignore_index=True)#[~np.isnan(df['costheta_D_reco'])]
  chilist2.append(df['chi_reco'],ignore_index=True)#[~np.isnan(df['chi_reco'])]

  
plt.hist(chilist2[~np.isnan(chilist2)],density=True,histtype='step')  
plt.title(r'$\chi$')
plt.savefig('/home/ke/graphs/Bd2DstDsst_chi.pdf')
plt.close()
  
  
"""
plt.hist(qlist2[~np.isnan(qlist2)],bins=100,density=True,histtype='step')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)

plt.hist(Llist2[~np.isnan(Llist2)],density=True,histtype='step')  

plt.hist(Dlist2[~np.isnan(Dlist2)],density=True,histtype='step')  

plt.hist(chilist2[~np.isnan(chilist2)],density=True,histtype='step')  

plt.hist(qlist4[~np.isnan(qlist4)],bins=100,density=True,histtype='step')  

plt.hist(qlist5[~np.isnan(qlist5)],bins=100,density=True,histtype='step')  

plt.savefig('/home/ke/graphs/Bu2DststDs1_q.pdf')
plt.close()
"""

#Bu2DststDs
files3=['/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs/5pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs/etapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs/etappi_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs/etappi_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs/etaprho_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs/etaprho_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs/etarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs/omega3pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs/omegapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs/omegarho_LHCb_Total/model_vars.root']


dg3=root_pandas.read_root(files3,columns=['q2_reco','costheta_L_reco','costheta_D_reco','chi_reco'],key='DecayTree')

qlist3=dg3['q2_reco']
Llist3=dg3['costheta_L_reco']
Dlist3=dg3['costheta_D_reco']
chilist3=dg3['chi_reco']

for file in files3[1:]:
  df=root_pandas.read_root(file,key='DecayTree')
  qlist3.append(df['q2_reco'],ignore_index=True)#[~np.isnan(df['q2_reco'])]
  Llist3.append(df['costheta_L_reco'],ignore_index=True)#[~np.isnan(df['costheta_L_reco'])]
  Dlist3.append(df['costheta_D_reco'],ignore_index=True)#[~np.isnan(df['costheta_D_reco'])]
  chilist3.append(df['chi_reco'],ignore_index=True)#[~np.isnan(df['chi_reco'])]


plt.hist(qlist3[~np.isnan(qlist3)],bins=100,density=True,histtype='step')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)
plt.savefig('/home/ke/graphs/Bu2DststDs_q.pdf')
plt.close()

plt.hist(Llist3[~np.isnan(Llist3)],density=True,histtype='step')  
plt.title(r'cos($\theta_L$)')
plt.savefig('/home/ke/graphs/Bu2DststDs.pdf')
plt.close()


plt.hist(Dlist3[~np.isnan(Dlist3)],density=True,histtype='step')  
plt.title(r'cos($\theta_D$)')
plt.savefig('/home/ke/graphs/Bu2DststDs.pdf')
plt.close()

plt.hist(chilist3[~np.isnan(chilist3)],density=True,histtype='step')  
plt.title(r'$\chi$')
plt.savefig('/home/ke/graphs/Bu2DststDs.pdf')
plt.close()



#Bu2DststDsst



files4=['/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dsgamma_5pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dsgamma_etapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dsgamma_etappi_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dsgamma_etappi_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dsgamma_etaprho_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dsgamma_etaprho_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dsgamma_etarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dsgamma_omega3pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dsgamma_omegapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dsgamma_omegarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dspi0_5pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dspi0_etapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dspi0_etappi_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dspi0_etappi_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dspi0_etaprho_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dspi0_etaprho_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dspi0_etarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dspi0_omega3pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dspi0_omegapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDsst/dspi0_omegarho_LHCb_Total/model_vars.root']


dg4=root_pandas.read_root(files4,columns=['q2_reco','costheta_L_reco','costheta_D_reco','chi_reco'],key='DecayTree')

qlist4=dg4['q2_reco']
Llist4=dg4['costheta_L_reco']
Dlist4=dg4['costheta_D_reco']
chilist4=dg4['chi_reco']

for file in files4[1:]:
  df=root_pandas.read_root(file,key='DecayTree')
  qlist4.append(df['q2_reco'],ignore_index=True)#[~np.isnan(df['q2_reco'])]
  Llist4.append(df['costheta_L_reco'],ignore_index=True)#[~np.isnan(df['costheta_L_reco'])]
  Dlist4.append(df['costheta_D_reco'],ignore_index=True)#[~np.isnan(df['costheta_D_reco'])]
  chilist4.append(df['chi_reco'],ignore_index=True)#[~np.isnan(df['chi_reco'])]l


plt.hist(qlist4[~np.isnan(qlist4)],bins=100,density=True,histtype='step')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)
plt.savefig('/home/ke/graphs/Bu2DststDsst_q.pdf')
plt.close()

plt.hist(Llist4[~np.isnan(Llist4)],density=True,histtype='step')  
plt.title(r'cos($\theta_L$)')
plt.savefig('/home/ke/graphs/Bu2DststDsst_L.pdf')
plt.close()


plt.hist(Dlist4[~np.isnan(Dlist4)],density=True,histtype='step')  
plt.title(r'cos($\theta_D$)')
plt.savefig('/home/ke/graphs/Bu2DststDsst_D.pdf')
plt.close()

plt.hist(chilist4[~np.isnan(chilist4)],density=True,histtype='step')  
plt.title(r'$\chi$')
plt.savefig('/home/ke/graphs/Bu2DststDsst_chi.pdf')
plt.close()





#Bu2DststDs1

files5=['/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsgamma_5pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsgamma_etapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsgamma_etappi_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsgamma_etappi_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsgamma_etaprho_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsgamma_etaprho_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsgamma_etarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsgamma_omega3pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsgamma_omegapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsgamma_omegarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsstpi0_dsgamma_5pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsstpi0_dsgamma_etapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsstpi0_dsgamma_etappi_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsstpi0_dsgamma_etappi_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsstpi0_dsgamma_etaprho_etapipi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsstpi0_dsgamma_etaprho_rhogamma_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsstpi0_dsgamma_etarho_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsstpi0_dsgamma_omega3pi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsstpi0_dsgamma_omegapi_LHCb_Total/model_vars.root',
'/data/lhcb/users/hill/bd2dsttaunu_angular/RapidSim_tuples/Bu2DststDs1/dsstpi0_dsgamma_omegarho_LHCb_Total/model_vars.root']

dg5=root_pandas.read_root(files5,columns=['q2_reco','costheta_L_reco','costheta_D_reco','chi_reco'],key='DecayTree')




qlist=dg['q2_reco']
qlist1=dg1['q2_reco']
qlist2=dg2['q2_reco']
qlist3=dg3['q2_reco']
qlist4=dg4['q2_reco']
qlist5=dg5['q2_reco']
Dlist=dg['costheta_D_reco']
Dlist1=dg1['costheta_D_reco']
Dlist2=dg2['costheta_D_reco']
Dlist3=dg3['costheta_D_reco']
Dlist4=dg4['costheta_D_reco']
Dlist5=dg5['costheta_D_reco']
Llist=dg['costheta_L_reco']
Llist1=dg1['costheta_L_reco']
Llist2=dg2['costheta_L_reco']
Llist3=dg3['costheta_L_reco']
Llist4=dg4['costheta_L_reco']
Llist5=dg5['costheta_L_reco']
chilist=dg['chi_reco']
chilist1=dg1['chi_reco']
chilist2=dg2['chi_reco']
chilist3=dg3['chi_reco']
chilist4=dg4['chi_reco']
chilist5=dg5['chi_reco']

plt.hist(qlist[~np.isnan(qlist)],bins=1000,density=True,histtype='step',label='Bd2DstDs1') 
plt.hist(qlist1[~np.isnan(qlist1)],bins=1000,density=True,histtype='step',label='Bd2DstDs') 
plt.hist(qlist2[~np.isnan(qlist2)],bins=1000,density=True,histtype='step',label='Bd2DstDsst')  
plt.hist(qlist3[~np.isnan(qlist3)],bins=1000,density=True,histtype='step',label='Bu2DststDs')  
plt.hist(qlist4[~np.isnan(qlist4)],bins=1000,density=True,histtype='step',label='Bu2DststDsst')  
plt.hist(qlist5[~np.isnan(qlist5)],bins=1000,density=True,histtype='step',label='Bu2DststDs1')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)
plt.legend()
plt.savefig('/home/ke/graphs/Q2.pdf')
plt.close()
plt.close()
plt.close()
plt.close()
plt.close()


plt.hist(chilist[~np.isnan(chilist)],bins=100,density=True,histtype='step',label='Bd2DstDs1') 
plt.hist(chilist1[~np.isnan(chilist1)],bins=100,density=True,histtype='step',label='Bd2DstDs') 
plt.hist(chilist2[~np.isnan(chilist2)],bins=100,density=True,histtype='step',label='Bd2DstDsst')  
plt.hist(chilist3[~np.isnan(chilist3)],bins=100,density=True,histtype='step',label='Bu2DststDs')  
plt.hist(chilist4[~np.isnan(chilist4)],bins=100,density=True,histtype='step',label='Bu2DststDsst')  
plt.hist(chilist5[~np.isnan(chilist5)],bins=100,density=True,histtype='step',label='Bu2DststDs1')  
plt.title(r'$\chi$')
plt.legend()
plt.savefig('/home/ke/graphs/chi.pdf')
plt.close()
plt.close()
plt.close()
plt.close()
plt.close()


plt.hist(Dlist[~np.isnan(Dlist)],bins=100,density=True,histtype='step',label='Bd2DstDs1') 
plt.hist(Dlist1[~np.isnan(Dlist1)],bins=100,density=True,histtype='step',label='Bd2DstDs') 
plt.hist(Dlist2[~np.isnan(Dlist2)],bins=100,density=True,histtype='step',label='Bd2DstDsst')  
plt.hist(Dlist3[~np.isnan(Dlist3)],bins=100,density=True,histtype='step',label='Bu2DststDs')  
plt.hist(Dlist4[~np.isnan(Dlist4)],bins=100,density=True,histtype='step',label='Bu2DststDsst')  
plt.hist(Dlist5[~np.isnan(Dlist5)],bins=100,density=True,histtype='step',label='Bu2DststDs1')  
plt.title(r'cos($\theta_D$)')
plt.legend()
plt.savefig('/home/ke/graphs/D.pdf')
plt.close()
plt.close()
plt.close()
plt.close()
plt.close()

plt.hist(Llist[~np.isnan(Llist)],bins=100,density=True,histtype='step',label='Bd2DstDs1') 
plt.hist(Llist1[~np.isnan(Llist1)],bins=100,density=True,histtype='step',label='Bd2DstDs') 
plt.hist(Llist2[~np.isnan(Llist2)],bins=100,density=True,histtype='step',label='Bd2DstDsst')  
plt.hist(Llist3[~np.isnan(Llist3)],bins=100,density=True,histtype='step',label='Bu2DststDs')  
plt.hist(Llist4[~np.isnan(Llist4)],bins=100,density=True,histtype='step',label='Bu2DststDsst')  
plt.hist(Llist5[~np.isnan(Llist5)],bins=100,density=True,histtype='step',label='Bu2DststDs1')  
plt.title(r'cos($\theta_L$)')
plt.legend()
plt.savefig('/home/ke/graphs/L.pdf')
plt.close()
plt.close()
plt.close()
plt.close()
plt.close()








qlist5=dg5['q2_reco']
Llist5=dg5['costheta_L_reco']
Dlist5=dg5['costheta_D_reco']
chilist5=dg5['chi_reco']

for file in files5[1:]:
  df=root_pandas.read_root(file,key='DecayTree')
  qlist5.append(df['q2_reco'],ignore_index=True)#[~np.isnan(df['q2_reco'])]
  Llist5.append(df['costheta_L_reco'],ignore_index=True)#[~np.isnan(df['costheta_L_reco'])]
  Dlist5.append(df['costheta_D_reco'],ignore_index=True)#[~np.isnan(df['costheta_D_reco'])]
  chilist5.append(df['chi_reco'],ignore_index=True)#[~np.isnan(df['chi_reco'])]l


plt.hist(qlist5[~np.isnan(qlist5)],bins=100,density=True,histtype='step')  
plt.title(r'$q^2$')
plt.xlim(0.,13.)
plt.savefig('/home/ke/graphs/Bu2DststDs1_q.pdf')
plt.close()

plt.hist(Llist5[~np.isnan(Llist5)],density=True,histtype='step')  
plt.title(r'cos($\theta_L$)')
plt.savefig('/home/ke/graphs/Bu2DststDs1_L.pdf')
plt.close()


plt.hist(Dlist5[~np.isnan(Dlist5)],density=True,histtype='step')  
plt.title(r'cos($\theta_D$)')
plt.savefig('/home/ke/graphs/Bu2DststDs1_D.pdf')
plt.close()

plt.hist(chilist5[~np.isnan(chilist5)],density=True,histtype='step')  
plt.title(r'$\chi$')
plt.savefig('/home/ke/graphs/Bu2DststDs1_chi.pdf')
plt.close()
