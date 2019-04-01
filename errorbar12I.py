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
# ==============================================================================

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numpy import *
import tensorflow as tf
import sys, os
import numpy as np
import math
from math import cos,sin,pi

sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU
import root_pandas
import pandas as pd
import TensorFlowAnalysis as tfa
from tensorflow.python.client import timeline
from root_numpy import root2array, rec2array, tree2array
from ROOT import TFile,TChain,TTree
from uncertainties import *
[I1s,I2c,I2s,I6c,I6s ,I3 ,I4 ,I5 ,I7 ,I8 ,I9]=[0.39581512500000005, -0.16444825, 0.0676855, 0.324758125, -0.252814625, -0.11448449999999999, -0.13989274999999998, 0.27817987499999997, -0.005926125, 0.00234375, -0.003685375]
[I1serr,I2cerr,I2serr,I6cerr,I6serr ,I3err ,I4err ,I5err ,I7err ,I8err ,I9err]=[0.00033787499999999996, 0.0009574999999999999, 0.0005718749999999999, 0.001021875, 0.000597125, 0.000588375, 0.000589875, 0.000532625, 0.0006163750000000001, 0.0006617500000000001, 0.000595]

[RAB,RLT,AFB,A6s,A3,A9,A4,A8,A5,A7,I1c]=[0.5449327872909999, 0.7860995595493749, -0.0678267719625125, 0.8532498338810001, -0.018220747200399998, -0.000586532579182, 0.08905846535782501, 0.001492111785675125, -0.20863486991425, 0.00444456061909625, 0.53201086622]
[RABerr,RLTerr,AFBerr,A6serr,A3err,A9err,A4err,A8err,A5err,A7err,I1cerr]=[0.0019113358779025, 0.0013271029985137501, 0.000488612975543875, 0.0020149376234124996, 9.364096372086249e-05, 9.471892186500001e-05, 0.000375589397310375, 0.0004212582904085, 0.0003995231617145, 0.000462318621479125, 0.00060464188557475]

Ilist=[I1s,I2c,I2s,I6c,I6s ,I3 ,I4 ,I5 ,I7 ,I8 ,I9,I1c]
Ierrlist=[I1serr,I2cerr,I2serr,I6cerr,I6serr ,I3err ,I4err ,I5err ,I7err ,I8err ,I9err,I1cerr]

Iname=["I1s","I2c","I2s","I6c","I6s" ,"I3" ,"I4" ,"I5" ,"I7" ,"I8" ,"I9","I1c"]

label1=r"3$\pi$-unbinned-all-true"
label2=r"3$\pi$-binned(4D)-LHCb-true"
label3=r"3$\pi$-binned(4D)-LHCb-reco"
label4=r"3$\pi$-binned(4D)-all-true"





def result(binned,dec,geom,retrue,num):
  result=[]
  err=[]
  result1=[]
  err1=[]
  if binned=='UnbinnedResult':
    f=open("/home/ke/TensorFlowAnalysis/"+binned+"/result_"+dec+"_"+geom+"_"+retrue+"_"+num+'_bintotal'+".txt", "r")
    linesf=f.readlines()  #I
    f.close()
    g=open("/home/ke/TensorFlowAnalysis/"+"ParamResult"+"/param_"+dec+"_"+geom+"_"+retrue+"_"+num+'_bintotal'+".txt", "r")
    linesg=g.readlines()   #Parameters
    g.close()
    for x in linesf:
      result.append(float(x.split(' ')[1]))
      err.append(float(x.split(' ')[2]))
    for x in linesg:
      result1.append(float(x.split(' ')[1]))
      err1.append(float(x.split(' ')[2]))  
    
    total_unbin=sum(result[:-1])+result1[-1]
    for i in range(len(result)):
      result[i]=result[i]/total_unbin
    for i in range(len(err)):
      err[i]=err[i]/total_unbin
    result[11]=result1[-1]/total_unbin
    err[11]=err1[-1]/total_unbin
  if binned=='BinnedResult':
    f=open("/home/ke/TensorFlowAnalysis/"+binned+"/result_"+dec+"_"+geom+"_"+retrue+"_"+num+".txt", "r")
    linesf=f.readlines()  #I
    f.close()
    g=open("/home/ke/TensorFlowAnalysis/"+binned+"/param_"+dec+"_"+geom+"_"+retrue+"_"+num+".txt", "r")
    linesg=g.readlines() #the parameters
    g.close()
    for x in linesf:
      result.append(float(x.split(' ')[1]))
      err.append(float(x.split(' ')[2]))  
    for x in linesg:
      result1.append(float(x.split(' ')[1]))
      err1.append(float(x.split(' ')[2]))  
    total_bin=0.733511
    for i in range(len(result)):
      result[i]=result[i]/total_bin
    result=result[1:]
    for i in range(len(err)):
      err[i]=err[i]/total_bin
    err=err[1:]
    result[11]=result1[-1]/total_bin
    err[11]=err1[-1]/total_bin
  return result, err

liste=["5","10","25","50","75","100","150","200"]

def xlist(n):
  ind=n
  return [ind-0.18,ind-0.06,ind+0.06,ind+0.18]




for j in range(8):
  plt.errorbar(xlist(0)[0],Ilist[0],yerr=Ierrlist[0],ecolor='#6059f7',color='#6059f7', fmt='.', elinewidth=2.5, capsize=0,label='Theory')
  plt.errorbar(xlist(0)[1],result("BinnedResult","3pi","LHCb","true",liste[j])[0][0],yerr=result("BinnedResult","3pi","LHCb","true",liste[j])[1][0],ecolor='#f2a026',color='#f2a026', fmt='.', elinewidth=2.5, capsize=0,label=label2)
  plt.errorbar(xlist(0)[2],result("BinnedResult","3pi","LHCb","reco",liste[j])[0][0],yerr=result("BinnedResult","3pi","LHCb","reco",liste[j])[1][0],ecolor='#960311',color='#960311',fmt='.',  elinewidth=2.5, capsize=0,label=label3)
  plt.errorbar(xlist(0)[3],result("BinnedResult","3pi","all","true",liste[j])[0][0],yerr=result("BinnedResult","3pi","all","true",liste[j])[1][0],ecolor='#0a4207',color='#0a4207', fmt='.', elinewidth=2.5, capsize=0,label=label4)

  for i in range(1,12):
    plt.errorbar(xlist(i)[0],Ilist[i],yerr=Ierrlist[i],fmt='.', ecolor='#6059f7',color='#6059f7', elinewidth=2.5, capsize=0)
    plt.errorbar(xlist(i)[1],result("BinnedResult","3pi","LHCb","true",liste[j])[0][i],yerr=result("BinnedResult","3pi","LHCb","true",liste[j])[1][i],fmt='.', ecolor='#f2a026',color='#f2a026', elinewidth=2.5, capsize=0)
    plt.errorbar(xlist(i)[2],result("BinnedResult","3pi","LHCb","reco",liste[j])[0][i],yerr=result("BinnedResult","3pi","LHCb","reco",liste[j])[1][i],fmt='.', ecolor='#960311',color='#960311', elinewidth=2.5, capsize=0)
    plt.errorbar(xlist(i)[3],result("BinnedResult","3pi","all","true",liste[j])[0][i],yerr=result("BinnedResult","3pi","all","true",liste[j])[1][i],fmt='.', ecolor='#0a4207',color='#0a4207', elinewidth=2.5, capsize=0)  
  plt.title("Fit results for "+liste[j]+"k events")
  plt.xlabel("Coefficients")
  plt.yticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1],['-1','-0.75','-0.5','-0.25','0','0.25','0.5','0.75','1'])
  plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],Iname) 
  plt.legend()
  plt.savefig('number'+liste[j]+'.pdf')
  plt.close()
  plt.close()
  plt.close()
  plt.close()



