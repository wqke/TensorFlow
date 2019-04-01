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
from uncertainties import *


#Fedele prediction
val = {'I1c':3.03,
          'I1s':2.04,
          'I2c':-0.893,
          'I2s':0.346,
          'I3': -0.563,
          'I4':-0.744,
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

I9th=val['I9']
I8th=val['I8']
I7th=val['I7']
I6sth=val['I6s']
I6cth=val['I6c']
I5th=val['I5']
I4th=val['I4']
I3th=val['I3']
I2sth=val['I2s']
I2cth=val['I2c']
I1sth=val['I1s']
I1cth=val['I1c']
rabth=(I1cth+2*I1sth-3*I2cth-6*I2sth)/(2*I1cth+4*I1sth+2*I2cth+4*I2sth)
rltth=(3*I1cth-I2cth)/(6*I1sth-2*I2sth)
Gammaqth=(3*I1cth+6*I1sth-I2cth-2*I1sth)/4.
afb1th=I6cth+2*I6sth
afbth=(3/8.)*(afb1th/Gammaqth)
a3th=(1/(np.pi*2))*I3th/Gammaqth
a9th=(1/(2*np.pi))*I9th/Gammaqth
a6sth=(-27/8.)*(I6sth/Gammaqth)
a4th=(-2/np.pi)*I4th/Gammaqth
a8th=(2/np.pi)*I8th/Gammaqth
a5th=(-3/4.)*(1-I8th-I7th-I9th-I4th-I3th-I2sth-I1sth-I1cth-I2cth-I6sth-I6cth)/Gammaqth
a7th=(-3/4.)*I7th/Gammaqth


Iname=["RAB","RLT","AFB","A6s","A3","A9","A4","A8","A5","A7","I1c"]

[RAB,RLT,AFB,A6s,A3,A9,A4,A8,A5,A7,I1c]=[rabth,rltth,afbth,a6sth,a3th,a9th,a4th,a8th,a5th,a7th,I1cth]
[RABerr,RLTerr,AFBerr,A6serr,A3err,A9err,A4err,A8err,A5err,A7err,I1cerr]=[0.0062529608287506065,0.025602667902659606,
                                                                         0.009427987350140766,
                                                                         0.03054962092295636,0.0004794192172993514,
                                                                         0.,0.0024274130560811504,0.,0.,0.03213740467719109,
                                                                         0.12/tot_rate]
Ilist=[RAB,RLT,AFB,A6s,A3,A9,A4,A8,A5,A7,I1c]
Ierrlist=[RABerr,RLTerr,AFBerr,A6serr,A3err,A9err,A4err,A8err,A5err,A7err,I1cerr]

label1=r"3$\pi$-unbinned-all-true"
label2=r"3$\pi$-binned(4D)-LHCb-true"
label3=r"3$\pi$-binned(4D)-LHCb-reco"


#define readfile 
def result(binned,dec,geom,retrue,num):
  result=[]
  err=[]
  result1=[]
  err1=[]
  if binned=='UnbinnedResult':
    f=open("/home/ke/TensorFlowAnalysis/"+"ParamResult"+"/param_"+dec+"_"+geom+"_"+retrue+"_"+num+'_bintotal'+".txt", "r")
    linesf=f.readlines()  #the parameters
    f.close()
    g=open("/home/ke/TensorFlowAnalysis/"+"UnbinnedResult"+"/result_"+dec+"_"+geom+"_"+retrue+"_"+num+'_bintotal'+".txt", "r")
    linesg=g.readlines()
    g.close()
    for x in linesf:
      result.append(float(x.split(' ')[1]))
      err.append(float(x.split(' ')[2]))  
    for x in linesg:
      result1.append(float(x.split(' ')[1]))
      err1.append(float(x.split(' ')[2]))  
    total_unbin=sum(result1[:-1])+result[-1]
    result[-1]=result[-1]/total_unbin
    err[-1]=err[-1]/total_unbin
  if binned=='BinnedResult':
    #f=open("/home/ke/TensorFlowAnalysis/"+binned+"/param_"+dec+"_"+geom+"_"+retrue+"_"+num+".txt", "r")
    #linesf=f.readlines()  #the parameters
    #f.close()
    g=open("/home/ke/TensorFlowAnalysis/"+binned+"/result_"+dec+"_"+geom+"_"+retrue+"_"+num+".txt", "r")
    linesg=g.readlines() 
    g.close()
    for x in linesg:
      result1.append(float(x.split(' ')[1]))
      err1.append(float(x.split(' ')[2]))  
    rate1=result1[0]
    i1s=result1[1]
    i2c=result1[2]
    i2s=result1[3]
    i6c=result1[4]
    i6s=result1[5]
    i3=result1[6]
    i4=result1[7]
    i5=result1[8]
    i7=result1[9]
    i8=result1[10]
    i9=result1[11]
    total_bin=0.733511
    covmat = np.load("/home/ke/TensorFlowAnalysis/BinnedResult/cov_%s_%s_%s_%s.npy" % (dec,geom,retrue,num))
    (rate1,i1s,i2c,i2s,i6c,i6s,i3,i4,i5,i7,i8,i9)= correlated_values([rate1,i1s,i2c,i2s,i6c,i6s,i3,i4,i5,i7,i8,i9],covmat)
    i1c=(1.0/3.0)*(4*rate1 - 6.*i1s + i2c + 2.*i2s)
    rab=((1.0/3.0)*(4*rate1 - 6.*i1s + i2c + 2.*i2s)+2*i1s-3*i2c-6*i2s)/(2*((1.0/3.0)*(4*rate1 - 6.*i1s + i2c + 2.*i2s)+2*i1s+i2c+2*i2s))
    rlt= ((4*rate1 - 6.*i1s + i2c + 2.*i2s)-i2c)/(2*(3*i1s-i2s))
    Gammaq=rate1
    afb1=i6c+2*i6s
    afb=(3/8.)*(afb1/Gammaq)
    a3=(1/(np.pi*2))*i3/Gammaq
    a9=(1/(2*np.pi))*i9/Gammaq
    a6s=(-27/8.)*(i6s/Gammaq)
    a4=(-2/np.pi)*i4/Gammaq
    a8=(2/np.pi)*i8/Gammaq
    a5=(-3/4.)*i5/Gammaq
    a7=(-3/4.)*i7/Gammaq
    result=[rab.n,rlt.n,afb.n,a6s.n,a3.n,a9.n,a4.n,a8.n,a5.n,a7.n,i1c.n/total_bin]   
    err=[rab.s,rlt.s,afb.s,a6s.s,a3.s,a9.s,a4.s,a8.s,a5.s,a7.s,i1c.s/total_bin]  

  return result,err



##End of readfile


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

