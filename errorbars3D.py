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

Iname=["I8","I7","I6s","I6c","I4" ,"I3" ,"I2s" ,"I2c" ,"I1s" ,"I1c" ,"I9"]
[I8,I7,I6s,I6c,I4 ,I3 ,I2s ,I2c ,I1s ,I1c ,I9]=[vals["I8"],vals["I7"],vals["I6s"],vals["I6c"],vals["I4"] ,vals["I3"] ,vals["I2s"] ,vals["I2c"] ,vals["I1s"] ,vals["I1c"] ,vals["I9"]]
[I8err,I7err,I6serr,I6cerr,I4err ,I3err ,I2serr ,I2cerr ,I1serr ,I1cerr ,I9]=[0/tot_rate,0/tot_rate,0.05/tot_rate,0.12/tot_rate,0.019/tot_rate,0.014/tot_rate,0.009/tot_rate,0.024/tot_rate,0.05/tot_rate,0.12/tot_rate,0/tot_rate]
Ilist=[I8,I7,I6s,I6c,I4 ,I3 ,I2s ,I2c ,I1s ,I1c ,I9]
Ierrlist=[I8err,I7err,I6serr,I6cerr,I4err ,I3err ,I2serr ,I2cerr ,I1serr ,I1cerr ,I9]

label1=r"3$\pi$-unbinned-all-true"
label2=r"3$\pi$-binned(3D)-LHCb-true"
label3=r"3$\pi$-binned(3D)-LHCb-reco"

def shuffle(listI):
  result=[0]*11
  result[0]=listI[9] #I8
  result[1]=listI[8] #I7
  result[2]=listI[5] #I6s
  result[3]=listI[4] #I6c
  result[4]=listI[7] #I4
  result[5]=listI[6] #I3
  result[6]=listI[3] #I2s
  result[7]=listI[2] #I2c
  result[8]=listI[1] #I1s
  result[9]=listI[0] #I1c
  result[10]=listI[10] #I9
  return result

#define readfile 
def result(binned,dec,geom,retrue,num):
  if binned=='UnbinnedResult':
    f=open("/home/ke/TensorFlowAnalysis/"+binned+"/result_"+dec+"_"+geom+"_"+retrue+"_"+num+'_bintotal'+".txt", "r")
  if binned=='BinnedResult':
    f=open("/home/ke/TensorFlowAnalysis/"+binned+"/result_"+dec+"_"+geom+"_"+retrue+"_"+num+".txt", "r")
  lines=f.readlines()
  result=[]
  err=[]
  for x in lines:
    result.append(float(x.split(' ')[1]))
    err.append(float(x.split(' ')[2]))  
  f.close()
  if binned=='UnbinnedResult':
    return result,err
  if binned=='BinnedResult':
    return shuffle(result),shuffle(err)
"""


>>> l=[1,2,3,4,5,6,7,8,9,10,11]
>>> shuffle(I)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'I' is not defined
>>> shuffle(l)
[10, 9, 6, 5, 8, 7, 5, 6, 9, 10, 11]

"""

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
