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

vals = {'I1c':3.03,'I1s':2.04,'I2c':-0.89,'I2s':0.35,'I3': -0.56,'I4':-0.74,'I5': 1.61,'I6c':1.96,'I6s':-1.38,'I7': 0.000,'I8': 0.000,'I9': 0.000}
tot_rate = 0.
for v in vals:
  tot_rate += vals[v]
#  tot_rate=vals["I1c"]
for v in vals:
  vals[v] = vals[v]/tot_rate

Iname=["I1s","I2c","I2s","I6c","I6s" ,"I3" ,"I4" ,"I5" ,"I7" ,"I8" ,"I9"]
[I1s,I2c,I2s,I6c,I6s ,I3 ,I4 ,I5 ,I7 ,I8 ,I9]=[vals["I1s"],vals["I2c"],vals["I2s"],vals["I6c"],vals["I6s"] ,vals["I3"] ,vals["I4"] ,vals["I5"] ,vals["I7"] ,vals["I8"] ,vals["I9"]]
[I1serr,I2cerr,I2serr,I6cerr,I6serr ,I3err ,I4err ,I5err ,I7err ,I8err ,I9err]=[0.05/tot_rate,0.024/tot_rate,0.009/tot_rate,0.12/tot_rate,0.05/tot_rate,0.014/tot_rate,0.019/tot_rate,0.06/tot_rate,0/tot_rate,0/tot_rate,0/tot_rate]

Ilist=[I1s,I2c,I2s,I6c,I6s ,I3 ,I4 ,I5 ,I7 ,I8 ,I9]
Ierrlist=[I1serr,I2cerr,I2serr,I6cerr,I6serr ,I3err ,I4err ,I5err ,I7err ,I8err ,I9err]

label1=r"3$\pi$-unbinned-all-true"
label2=r"3$\pi$-binned(4D)-LHCb-true"
label3=r"3$\pi$-binned(4D)-LHCb-reco"
label4=r"3$\pi$-binned(4D)-all-reco"


#define readfile 
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

  return result, err


"""
def xlist(n):
  return [n*2**(-0.15),n,n*2**(0.15)]
"""
def xlist(n):
  liste=[5,10,25,50,75,100,150,200]
  ind=liste.index(n)+1
  return [ind-0.15,ind,ind+0.15]

Xrange=[xlist(5)[0],xlist(10)[1],xlist(25)[1],xlist(50)[1],xlist(75)[1],xlist(100)[1],xlist(150)[1],xlist(200)[2]]


for i in range(11):
  plt.errorbar([xlist(5)[0],xlist(10)[0],xlist(25)[0],xlist(50)[0],xlist(75)[0],xlist(100)[0],xlist(150)[0],xlist(200)[0]],
     [result("UnbinnedResult","3pi","all","true","5")[0][i],result("UnbinnedResult","3pi","all","true","10")[0][i],result("UnbinnedResult","3pi","all","true","25")[0][i],
     result("UnbinnedResult","3pi","all","true","50")[0][i],
    result("UnbinnedResult","3pi","all","true","75")[0][i],result("UnbinnedResult","3pi","all","true","100")[0][i],
      result("UnbinnedResult","3pi","all","true","150")[0][i],result("UnbinnedResult","3pi","all","true","200")[0][i]],
     yerr=[result("UnbinnedResult","3pi","all","true","5")[1][i],result("UnbinnedResult","3pi","all","true","10")[1][i],result("UnbinnedResult","3pi","all","true","25")[1][i],
     result("UnbinnedResult","3pi","all","true","50")[1][i],result("UnbinnedResult","3pi","all","true","75")[1][i],
           result("UnbinnedResult","3pi","all","true","100")[1][i],result("UnbinnedResult","3pi","all","true","150")[1][i],
          result("UnbinnedResult","3pi","all","true","200")[1][i]],fmt='o', color='#6059f7',
ecolor='#6059f7', elinewidth=3, capsize=0,label=label1)

  plt.errorbar([xlist(5)[1],xlist(10)[1],xlist(25)[1],xlist(50)[1],xlist(75)[1],xlist(100)[1],xlist(150)[1],xlist(200)[1]],
     [result("BinnedResult","3pi","LHCb","true","5")[0][i],result("BinnedResult","3pi","LHCb","true","10")[0][i],result("BinnedResult","3pi","LHCb","true","25")[0][i],
      result("BinnedResult","3pi","LHCb","true","50")[0][i],result("BinnedResult","3pi","LHCb","true","75")[0][i],
    result("BinnedResult","3pi","LHCb","true","100")[0][i],result("BinnedResult","3pi","LHCb","true","150")[0][i],
     result("BinnedResult","3pi","LHCb","true","200")[0][i]],
     yerr=[result("BinnedResult","3pi","LHCb","true","5")[1][i],result("BinnedResult","3pi","LHCb","true","10")[1][i],result("BinnedResult","3pi","LHCb","true","25")[1][i],
           result("BinnedResult","3pi","LHCb","true","50")[1][i],result("BinnedResult","3pi","LHCb","true","75")[1][i],
           result("BinnedResult","3pi","LHCb","true","100")[1][i],result("BinnedResult","3pi","LHCb","true","150")[1][i],
     result("BinnedResult","3pi","LHCb","true","200")[1][i]], fmt='o', color='#f2a026',
  ecolor='#f2a026', elinewidth=3, capsize=0,label=label2)

  plt.errorbar([xlist(5)[2],xlist(10)[2],xlist(25)[2],xlist(50)[2],xlist(75)[2],xlist(100)[2],xlist(150)[2],xlist(200)[2]],
     [result("BinnedResult","3pi","LHCb","reco","5")[0][i],result("BinnedResult","3pi","LHCb","reco","10")[0][i],result("BinnedResult","3pi","LHCb","reco","25")[0][i],
      result("BinnedResult","3pi","LHCb","reco","50")[0][i],result("BinnedResult","3pi","LHCb","reco","75")[0][i],
      result("BinnedResult","3pi","LHCb","reco","100")[0][i],result("BinnedResult","3pi","LHCb","reco","150")[0][i],
    result("BinnedResult","3pi","LHCb","reco","200")[0][i]],
     yerr=[result("BinnedResult","3pi","LHCb","reco","5")[1][i],result("BinnedResult","3pi","LHCb","reco","10")[1][i],result("BinnedResult","3pi","LHCb","reco","25")[1][i],
           result("BinnedResult","3pi","LHCb","reco","50")[1][i],result("BinnedResult","3pi","LHCb","reco","75")[1][i],
           result("BinnedResult","3pi","LHCb","reco","100")[1][i],result("BinnedResult","3pi","LHCb","reco","150")[1][i],
     result("BinnedResult","3pi","LHCb","reco","200")[1][i]], fmt='o', color='#960311',
  ecolor='#960311', elinewidth=3, capsize=0,label=label3)
  plt.plot(Xrange,[Ilist[i]]*8,linestyle=':')
  plt.fill_between(Xrange,[Ilist[i]-Ierrlist[i]]*8,[Ilist[i]+Ierrlist[i]]*8 ,alpha=0.5,color='lightgray',label='Theory')
  plt.title(r"Fit results for "+Iname[i])
  plt.xlabel("N (1000's)")
  plt.ylabel(Iname[i])
  #plt.xscale("log",basex=2.0)
  plt.xticks([1,2,3,4,5,6,7,8],('5','10','25','50','75','100','150','200')) 
  plt.legend()
  plt.savefig(Iname[i]+'.pdf')
  plt.close()
  plt.close()
  plt.close()
  plt.close()
  plt.close()

