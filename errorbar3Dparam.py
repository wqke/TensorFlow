import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numpy import *
import tensorflow as tf
import sys, os
import numpy as np
import math
from math import cos,sin,pi
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm,rc



#Fedele prediction
val = {'I1c':3.03,
          'I1s':2.04,
          'I2c':-0.89,
          'I2s':0.35,
          'I3': -0.56,
          'I4':-0.74,
          'I5': 1.61,
          'I6c':1.96,
          'I6s':-1.38,
          'I7': 0.000,
          'I8': 0.000,
          'I9': 0.000}
tot_rate = 0.
for v in val:
  tot_rate += val[v]
#  tot_rate=vals["I1c"]
for v in val:
  val[v] = val[v]/tot_rate
  
Iname=["RAB","RLT","AFB","A6s","A3","A9","A4","A8","A5","A7","I5"]

[RAB,RLT,AFB,A6s,A3,A9,A4,A8,A5,A7,I5]=[0.5549132947976879,0.8648180242634314,-0.06615214994487321,1.0270121278941566,-0.019653091098447942,0.,
                                        0.10388062437751054,0.,-0.26626240352811464,0.,val["I5"]]
[RABerr,RLTerr,AFBerr,A6serr,A3err,A9err,A4err,A8err,A5err,A7err,I5err]=[0.0062529608287506065,0.025602667902659606,
                                                                         0.009427987350140766,
                                                                         0.03054962092295636,0.0004794192172993514,
                                                                         0.,0.0024274130560811504,0.,0.,0.03213740467719109,
                                                                         0.06/tot_rate]
Ilist=[RAB,RLT,AFB,A6s,A3,A9,A4,A8,A5,A7,I5]
Ierrlist=[RABerr,RLTerr,AFBerr,A6serr,A3err,A9err,A4err,A8err,A5err,A7err,I5err]

label1=r"3$\pi$-unbinned-all-true"
label2=r"3$\pi$-binned(3D)-LHCb-true"
label3=r"3$\pi$-binned(3D)-LHCb-reco"



#define readfile 
def result(binned,dec,geom,retrue,num):
  if binned=='UnbinnedResult':
    f=open("/home/ke/TensorFlowAnalysis/"+"ParamResult"+"/param_"+dec+"_"+geom+"_"+retrue+"_"+num+'_bintotal'+".txt", "r")
  if binned=='BinnedResult':
    f=open("/home/ke/TensorFlowAnalysis/"+binned+"/param_"+dec+"_"+geom+"_"+retrue+"_"+num+".txt", "r")
  lines=f.readlines()
  result=[]
  err=[]
  for x in lines:
    result.append(float(x.split(' ')[1]))
    err.append(float(x.split(' ')[2]))  
  f.close()
  return result,err



##End of readfile
def xlist(n):
  return [n*2**(-0.15),n,n*2**(0.15)]

Xrange=[xlist(10)[0],50,xlist(100)[2]]


for i in range(11):
  plt.errorbar([xlist(10)[0],xlist(50)[0],xlist(100)[0]],
               [result("UnbinnedResult","3pi","all","true","10")[0][i],
              result("UnbinnedResult","3pi","all","true","50")[0][i],result("UnbinnedResult","3pi","all","true","100")[0][i]],
               yerr=[result("UnbinnedResult","3pi","all","true","10")[1][i],
               result("UnbinnedResult","3pi","all","true","50")[1][i],result("UnbinnedResult","3pi","all","true","100")[1][i]], fmt='o', color='black',
  ecolor='#6059f7', elinewidth=3, capsize=0,label=label1)

  plt.errorbar([xlist(10)[1],xlist(50)[1],xlist(100)[1]],
               [result("BinnedResult","3pi","LHCb","true","10")[0][i],
              result("BinnedResult","3pi","LHCb","true","50")[0][i],result("BinnedResult","3pi","LHCb","true","100")[0][i]],
               yerr=[result("BinnedResult","3pi","LHCb","true","10")[1][i],
               result("BinnedResult","3pi","LHCb","true","50")[1][i],result("BinnedResult","3pi","LHCb","true","100")[1][i]], fmt='o', color='black',
  ecolor='#f2a026', elinewidth=3, capsize=0,label=label2)

  plt.errorbar([xlist(10)[2],xlist(50)[2],xlist(100)[2]],
               [result("BinnedResult","3pi","LHCb","reco","10")[0][i],
              result("BinnedResult","3pi","LHCb","reco","50")[0][i],result("BinnedResult","3pi","LHCb","reco","100")[0][i]],
               yerr=[result("BinnedResult","3pi","LHCb","reco","10")[1][i],result("BinnedResult","3pi","LHCb","reco","50")[1][i],
               result("BinnedResult","3pi","LHCb","reco","100")[1][i]], fmt='o', color='black',
  ecolor='#960311', elinewidth=3, capsize=0,label=label3)
  plt.plot(Xrange,[Ilist[i]]*3,linestyle=':')
  plt.fill_between(Xrange,[Ilist[i]-Ierrlist[i]]*3,[Ilist[i]+Ierrlist[i]]*3 ,alpha=0.5,color='lightgray',label='Theory')
  plt.title(r"Fit results for "+Iname[i])
  plt.xlabel("N (1000's)")
  plt.ylabel(Iname[i])
  plt.xscale("log",basex=2.0)
  plt.xticks([10,50,100],('10','50','100')) 
  plt.legend()
  plt.savefig(Iname[i]+'.pdf')
  plt.close()
  plt.close()
  plt.close()
  plt.close()
  plt.close()

          
