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
listI=[0,0,-1.38,1.96,-0.744,-0.563,0.346,-0.893,2.04,3.03,0]
[I8,I7,I6s,I6c,I4 ,I3 ,I2s ,I2c ,I1s ,I1c ,I9]=[0/sum(listI),0/sum(listI),-1.38/sum(listI),1.96/sum(listI),-0.744/sum(listI),-0.563/sum(listI),0.346/sum(listI),-0.893/sum(listI),2.04/sum(listI),3.03/sum(listI),0/sum(listI)]
[I8err,I7err,I6serr,I6cerr,I4err ,I3err ,I2serr ,I2cerr ,I1serr ,I1cerr ,I9]=[0/sum(listI),0/sum(listI),0.05/sum(listI),0.12/sum(listI),0.019/sum(listI),0.014/sum(listI),0.009/sum(listI),0.024/sum(listI),0.05/sum(listI),0.12/sum(listI),0/sum(listI)]

title_I6c=r"Unbinned fit for $I_{6c}$"
title_I6s=r"Unbinned fit for $I_{6s}$"

label1=r"3$\pi$-all-true"
label2=r"3$\pi$-LHCb-reco"
label3=r"3$\pi\pi_0$-all-true"
label4=r"3$\pi\pi_0$-LHCb-reco"


#I6c,3


#define readfile 
def result(dec,geom,retrue,num):
  f=open("result_"+dec+"_"+geom+"_"+retrue+"_"+num+".txt", "r")
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
  return [n*2**(-0.1),n*2**(-0.05),n*2**(0.05),n*2**(0.1)]

Xrange=[5,10,20,40,80]

plt.errorbar([xlist(5)[0],xlist(10)[0],xlist(20)[0],xlist(40)[0],xlist(80)[0]],
             [result("3pi","all","true","5")[0][3],result("3pi","all","true","10")[0][3],
            result("3pi","all","true","20")[0][3],result("3pi","all","true","40")[0][3],
              result("3pi","all","true","80")[0][3]],
             yerr=[result("3pi","all","true","5")[1][3],
             result("3pi","all","true","10")[1][3],result("3pi","all","true","20")[1][3],
             result("3pi","all","true","40")[1][3],result("3pi","all","true","80")[1][3]], fmt='o', color='black',
ecolor='#6059f7', elinewidth=3, capsize=0,label=label1)

plt.errorbar([xlist(5)[1],xlist(10)[1],xlist(20)[1],xlist(40)[1],xlist(80)[1]],
             [result("3pi","LHCb","reco","5")[0][3],result("3pi","LHCb","reco","10")[0][3],
            result("3pi","LHCb","reco","20")[0][3],result("3pi","LHCb","reco","40")[0][3],
              result("3pi","LHCb","reco","80")[0][3]],
             yerr=[result("3pi","LHCb","reco","5")[1][3],
             result("3pi","LHCb","reco","10")[1][3],result("3pi","LHCb","reco","20")[1][3],
             result("3pi","LHCb","reco","40")[1][3],result("3pi","LHCb","reco","80")[1][3]], fmt='o', color='black',
ecolor='#f2a026', elinewidth=3, capsize=0,label=label2)

plt.errorbar([xlist(5)[2],xlist(10)[2],xlist(20)[2],xlist(40)[2],xlist(80)[2]],
             [result("3pipi0","all","true","5")[0][3],result("3pipi0","all","true","10")[0][3],
            result("3pipi0","all","true","20")[0][3],result("3pipi0","all","true","40")[0][3],
              result("3pipi0","all","true","80")[0][3]],
             yerr=[result("3pipi0","all","true","5")[1][3],
             result("3pipi0","all","true","10")[1][3],result("3pipi0","all","true","20")[1][3],
             result("3pipi0","all","true","40")[1][3],result("3pipi0","all","true","80")[1][3]], fmt='o', color='black',
ecolor='#d80622', elinewidth=3, capsize=0,label=label3)

plt.errorbar([xlist(5)[3],xlist(10)[3],xlist(20)[3],xlist(40)[3],xlist(80)[3]],
             [result("3pipi0","LHCb","reco","5")[0][3],result("3pipi0","LHCb","reco","10")[0][3],
            result("3pipi0","LHCb","reco","20")[0][3],result("3pipi0","LHCb","reco","40")[0][3],
              result("3pipi0","LHCb","reco","80")[0][3]],
             yerr=[result("3pipi0","LHCb","reco","5")[1][3],
             result("3pipi0","LHCb","reco","10")[1][3],result("3pipi0","LHCb","reco","20")[1][3],
             result("3pipi0","LHCb","reco","40")[1][3],result("3pipi0","LHCb","reco","80")[1][3]], fmt='o', color='black',
ecolor='#4e0366', elinewidth=3, capsize=0,label=label4)


plt.plot(Xrange,[I6c]*5,linestyle=':')
plt.fill_between(Xrange,[I6c-I6cerr]*5 ,[I6c+I6cerr]*5 ,alpha=0.5,color='lightgray',label='Theory')
plt.title(title_I6c)
plt.xlabel("N (1000's)")
plt.ylabel(r"$I_{6c}$")
plt.xscale("log",basex=2.0)
plt.xticks([5,10,20,40,80]) 
plt.legend()

plt.savefig('I6c.pdf')
