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



"""
#theory

result1=[]
err1=[]
result2=[]
err2=[]
result3=[]
err3=[]
result4=[]
err4=[]
result5=[]
err5=[]
result6=[]
err6=[]
result7=[]
err7=[]
result8=[]
err8=[]



f1=open("/home/ke/TensorFlowAnalysis/ParamResult/param_3pi_all_true_5_bintotal"+".txt", "r")
lines1=f1.readlines()  #I
f1.close()
for x in lines1:
  result1.append(float(x.split(' ')[1]))
  err1.append(float(x.split(' ')[2]))  
f2=open("/home/ke/TensorFlowAnalysis/ParamResult/param_3pi_all_true_10_bintotal"+".txt", "r")
lines2=f2.readlines()  #I
f2.close()
for x in lines2:
  result2.append(float(x.split(' ')[1]))
  err2.append(float(x.split(' ')[2]))  

f3=open("/home/ke/TensorFlowAnalysis/ParamResult/param_3pi_all_true_25_bintotal"+".txt", "r")
lines3=f3.readlines()  #I
f3.close()
for x in lines3:
  result3.append(float(x.split(' ')[1]))
  err3.append(float(x.split(' ')[2]))  

f4=open("/home/ke/TensorFlowAnalysis/ParamResult/param_3pi_all_true_50_bintotal"+".txt", "r")
lines4=f4.readlines()  #I
f4.close()
for x in lines4:
  result4.append(float(x.split(' ')[1]))
  err4.append(float(x.split(' ')[2]))  

f5=open("/home/ke/TensorFlowAnalysis/ParamResult/param_3pi_all_true_75_bintotal"+".txt", "r")
lines5=f5.readlines()  #I
f5.close()
for x in lines5:
  result5.append(float(x.split(' ')[1]))
  err5.append(float(x.split(' ')[2]))  
f6=open("/home/ke/TensorFlowAnalysis/ParamResult/param_3pi_all_true_100_bintotal"+".txt", "r")
lines6=f6.readlines()  #I
f6.close()
for x in lines6:
  result6.append(float(x.split(' ')[1]))
  err6.append(float(x.split(' ')[2]))  
f7=open("/home/ke/TensorFlowAnalysis/ParamResult/param_3pi_all_true_150_bintotal"+".txt", "r")
lines7=f7.readlines()  #I
f7.close()
for x in lines7:
  result7.append(float(x.split(' ')[1]))
  err7.append(float(x.split(' ')[2]))  

f8=open("/home/ke/TensorFlowAnalysis/ParamResult/param_3pi_all_true_200_bintotal"+".txt", "r")
lines8=f8.readlines()  #I
f8.close()
for x in lines8:
  result8.append(float(x.split(' ')[1]))
  err8.append(float(x.split(' ')[2]))  

results=[result1,result2,result3,result4,result5,result6,result7,result8]
errs=[err1,err2,err3,err4,err5,err6,err7,err8]
[I1s,I2c,I2s,I6c,I6s ,I3 ,I4 ,I5 ,I7 ,I8 ,I9]=[0,0,0,0,0,0,0,0,0,0,0]
total=0
totalerr=0
for result in results:
  total+=result[-1]
  totalerr+=err[-1]
total=total/8.
totalerr=totalerr/8.
  
  
  
for result in results:  
  I1s+=result[0]
  I2c+=result[1]
  I2s+=result[2]        
  I6c+=result[3]
  I6s+=result[4]
  I3+=result[5]
  I4+=result[6]
  I5+=result[7]
  I7+=result[8]
  I8+=result[9]
  I9+=result[10]
for err in errs:  
  I1s+=err[0]
  I2c+=err[1]
  I2s+=err[2]        
  I6c+=err[3]
  I6s+=err[4]
  I3+=err[5]
  I4+=err[6]
  I5+=err[7]
  I7+=err[8]
  I8+=err[9]
  I9+=err[10]
I1s=I1s/8.
I2c=I2c/8.
I2s=I2s/8.        
I6c=I6c/8.
I6s=I6s/8.
I3=I3/8.
I4=I4/8.
I5=I5/8.
I7=I7/8.
I8=I8/8.
I9=I9/8.
"""
[I1s,I2c,I2s,I6c,I6s ,I3 ,I4 ,I5 ,I7 ,I8 ,I9,FL]=[0.39581512500000005, -0.16444825, 0.0676855, 0.324758125, -0.252814625, -0.11448449999999999, -0.13989274999999998, 0.27817987499999997, -0.005926125, 0.00234375, -0.003685375,0.440082972583625]
[I1serr,I2cerr,I2serr,I6cerr,I6serr ,I3err ,I4err ,I5err ,I7err ,I8err ,I9err,FLerr]=[0.00033787499999999996, 0.0009574999999999999, 0.0005718749999999999, 0.001021875, 0.000597125, 0.000588375, 0.000589875, 0.000532625, 0.0006163750000000001, 0.0006617500000000001, 0.000595,0.00022790563627750002]

[RAB,RLT,AFB,A6s,A3,A9,A4,A8,A5,A7,I1c]=[0.5449327872909999, 0.7860995595493749, -0.0678267719625125, 0.8532498338810001, -0.018220747200399998, -0.000586532579182, 0.08905846535782501, 0.001492111785675125, -0.20863486991425, 0.00444456061909625, 0.53201086622]
[RABerr,RLTerr,AFBerr,A6serr,A3err,A9err,A4err,A8err,A5err,A7err,I1cerr]=[0.0019113358779025, 0.0013271029985137501, 0.000488612975543875, 0.0020149376234124996, 9.364096372086249e-05, 9.471892186500001e-05, 0.000375589397310375, 0.0004212582904085, 0.0003995231617145, 0.000462318621479125, 0.00060464188557475]

Iname=["RAB","RLT","AFB","A6s","A3","A9","A4","A8","A5","A7","I1c","FL"]
Ilist=[RAB,RLT,AFB,A6s,A3,A9,A4,A8,A5,A7,I1c]
Ierrlist=[RABerr,RLTerr,AFBerr,A6serr,A3err,A9err,A4err,A8err,A5err,A7err,I1cerr]

label1=r"3$\pi$-unbinned-all-true"
label2=r"3$\pi$-binned(4D)-LHCb-true"
label3=r"3$\pi$-binned(4D)-LHCb-reco"
label4=r"3$\pi$-binned(4D)-all-true"

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
    fl=(3.*i1c-i2c)/(3.*i1c+6.*i1s-i2c-2.*i2s)
    result=[rab.n,rlt.n,afb.n,a6s.n,a3.n,a9.n,a4.n,a8.n,a5.n,a7.n,i1c.n/total_bin,fl.n]   
    err=[rab.s,rlt.s,afb.s,a6s.s,a3.s,a9.s,a4.s,a8.s,a5.s,a7.s,i1c.s/total_bin,fl.s]  

  return result,err



##End of readfile



def xlist(n):
  liste=[5,10,25,50,75,100,150,200]
  ind=liste.index(n)+1
  return [ind-0.18,ind-0.06,ind+0.06,ind+0.18]


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

  plt.errorbar([xlist(5)[3],xlist(10)[3],xlist(25)[3],xlist(50)[3],xlist(75)[3],xlist(100)[3],xlist(150)[3],xlist(200)[3]],
     [result("BinnedResult","3pi","all","true","5")[0][i],result("BinnedResult","3pi","all","true","10")[0][i],result("BinnedResult","3pi","all","true","25")[0][i],
      result("BinnedResult","3pi","all","true","50")[0][i],result("BinnedResult","3pi","all","true","75")[0][i],
      result("BinnedResult","3pi","all","true","100")[0][i],result("BinnedResult","3pi","all","true","150")[0][i],
    result("BinnedResult","3pi","all","true","200")[0][i]],
     yerr=[result("BinnedResult","3pi","all","true","5")[1][i],result("BinnedResult","3pi","all","true","10")[1][i],result("BinnedResult","3pi","all","true","25")[1][i],
           result("BinnedResult","3pi","all","true","50")[1][i],result("BinnedResult","3pi","all","true","75")[1][i],
           result("BinnedResult","3pi","all","true","100")[1][i],result("BinnedResult","3pi","all","true","150")[1][i],
     result("BinnedResult","3pi","all","true","200")[1][i]], fmt='o', color='#0a4207',
  ecolor='#0a4207', elinewidth=3, capsize=0,label=label4)
  plt.plot(Xrange,[Ilist[i]]*8,linestyle=':')
  plt.fill_between(Xrange,[Ilist[i]-Ierrlist[i]]*8,[Ilist[i]+Ierrlist[i]]*8 ,alpha=0.5,color='lightgray',label='Theory')
  plt.title(r"Fit results for "+Iname[i])
  plt.xlabel("N (1000's)")
  plt.ylabel(Iname[i])
  plt.xticks([1,2,3,4,5,6,7,8],('5','10','25','50','75','100','150','200')) 
  plt.legend()
  plt.savefig(Iname[i]+'.pdf')
  plt.close()
  plt.close()
  plt.close()
  plt.close()
  plt.close()
