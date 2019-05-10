import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import os
import root_pandas
from root_pandas import *
import pandas as pd
import numpy as np
from numpy import cos,sin,tan,sqrt,absolute,real,conjugate,imag,abs,max,min
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm,rc
from skhep.visual import MplPlotter as skh_plt


import rootplot.root2matplotlib as r2m   #plot errorbars on the histogram 
from ROOT import TH1F,TChain,TFile       #divide two histograms

#####
df=read_root("/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Bd2DstTauNu/3pi_LHCb_Total/model_vars_weights_hammer_BDT.root",
             'DecayTree',columns=['hamweight_SM','hamweight_T2','BDT','Tau_FD'])
SM=df['hamweight_SM']
T2=df['hamweight_T2']

#skh_plt.ratio_plot(dict(x=df['Tau_FD'],bins=100,range=[0.,100000],normed=True,color='r',weights=SM.values,histtype='step',linewidth=0.8,label=label1),dict(x=df['Tau_FD'],bins=100,range=[0.,100000.],normed=True,color='b',weights=T2.values,histtype='step',linewidth=0.8,label=label2),ratio_range=[0.8,1.2])
bin_heights1, bin_borders1, _=plt.hist(df['BDT'],weights=SM.values,label='SM',range=[-4.,4.],histtype='step',color='red',bins=20)
bin_heights2, bin_borders2, _=plt.hist(df['BDT'],weights=T2.values,label='T2',range=[-4.,4.],histtype='step',color='blue',bins=20)
bin_centers = bin_borders1[:-1] + np.diff(bin_borders1) / 2
plt.yscale('log')
plt.xlabel('BDT')
plt.legend()
plt.xlim(-4.,4.)
plt.title('BDT')
plt.savefig('BDT.pdf')
plt.close()
plt.close()

heights=[]
for i in range(len(bin_centers)):
  heights.append(bin_heights1[i]/float(bin_heights2[i]))
  

plt.axhline(y=np.average(heights[:-1]),linestyle='--',color='red')


plt.scatter(bin_centers,heights,color='gray')

plt.ylim(0.6,0.9) 
plt.xlim(-4.,4.)
plt.title("BDT SM/T2 ratio")
plt.savefig("BDTratio.pdf")
plt.close()

  
##

Tau_FD=[element/1000. for element in df['Tau_FD']]
bin_heights1, bin_borders1, _=plt.hist(Tau_FD,weights=SM.values,label='SM',normed=True,histtype='step',color='red',bins=100,range=[0.,100.])
bin_heights2, bin_borders2, _=plt.hist(Tau_FD,weights=T2.values,label='T2',normed=True,histtype='step',color='blue',bins=100,range=[0.,100.])
bin_centers = bin_borders1[:-1] + np.diff(bin_borders1) / 2
plt.yscale('log')
plt.xlabel('Tau_FD [mm]')
plt.legend()
plt.xlim(0.,100.)
plt.title('Tau_FD')
plt.savefig('TauFD.pdf')
plt.close()
plt.close()

heights=[]
for i in range(len(bin_centers)):
  heights.append(bin_heights1[i]/float(bin_heights2[i]))

plt.axhline(y=np.average(heights),linestyle='--',color='red')  

plt.scatter(bin_centers,heights,color='gray')

plt.xlabel('Tau_FD [mm]')
plt.ylim(0.6,0.9)  
plt.xlim(0.,100.)
plt.title("Tau_FD SM/T2 ratio")
plt.savefig("FDratio.pdf")
plt.close()
  
###
df=root_pandas.read_root('result_DstTauNu.root',key='data')
dg=root_pandas.read_root('result_DstTauNu.root',key='fit_result')

bin_heights, bin_borders, _=plt.hist(df['costheta_X_true'],density=True)
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
plt.close()
xerr=[(bin_borders[1]-bin_borders[0])/2.]*len(bin_centers)
plt.errorbar(bin_centers,bin_heights, xerr=xerr, fmt='o', color='black',label='data')
plt.hist(dg['costheta_X'],label='fit result',color='b',density=True,histtype='step')

plt.legend()
plt.title(r'cos($\theta_D$) ')
plt.savefig('costhetast.pdf')
plt.close()
plt.close()

bin_heights, bin_borders, _= plt.hist(df['costheta_L'],density=True)
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
plt.close()
xerr=[(bin_borders[1]-bin_borders[0])/2.]*len(bin_centers)
plt.errorbar(bin_centers,bin_heights, xerr=xerr, fmt='o', color='black',label='data')
plt.hist(dg['costheta_L'],label='fit result',color='b',density=True,histtype='step')

plt.legend()
plt.title(r'cos($\theta_L$) ')
plt.savefig('costhetaL.pdf')
plt.close()
plt.close()


bin_heights, bin_borders, _=plt.hist(df['chi'],density=True)
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
plt.close()
xerr=[(bin_borders[1]-bin_borders[0])/2.]*len(bin_centers)
plt.errorbar(bin_centers,bin_heights, xerr=xerr, fmt='o', color='black',label='data')
plt.hist(dg['chi'],label='fit result',color='b',density=True,histtype='step')
plt.legend()
plt.title(r'$\chi$')
plt.savefig('chi.pdf')
plt.close()
plt.close()




borders=[ 3.23624062,  5.09804311,  6.95984559,  8.82164808, 10.68345056]
centers = borders[:-1] + np.diff(borders) / 2

bin1={'I8': (9.536740132598531e-07, 0.005357802336437234), 'I6c': (0.36162361623616235, 0.016738634014647286), 'I3': (-0.10332103321033215, 0.03571659134207933), 'I2s': (0.06457564575645747, 0.01146773718474603), 'I5': (0.2970479704797048, 0.024199481769456632), 'I4': (-0.13653136531365317, 0.029624138271542866), 'I7': (0.0, 0.014266352668174609), 'loglh': -1445.2657449963137, 'I1s': (0.3763837638376384, 0.05971715870005462), 'iterations': 285, 'I6s': (-0.25461254612546125, 0.03473090509935628), 'I2c': (-0.16420664206642066, 0.034760662592753455), 'I1c': (0.559040590405904, 0.045778813639044846)}
bin2={'I8': (1.9473311851925246e-13, 0.0026595546118925173), 'I6c': (0.37275380045087325, 0.007100804250770365), 'I3': (-0.06770846425110688, 0.005064456159967978), 'I2s': (0.03303495507669485, 0.004803719871526724), 'I5': (0.2913047449091781, 0.006727855779420533), 'I4': (-0.08923990264748982, 0.005324616070581545), 'I7': (0.002424603204271847, 0.005221286948234205), 'loglh': -3808.490874406116, 'I1s': (0.296410263277083, 0.005870046245991356), 'iterations': 334, 'I6s': (-0.23146661246756106, 0.006072362563596445), 'I2c': (-0.1205048242858906, 0.009462770190585057), 'I1c': (0.5109451016995504, 0.007955321870353804)}
bin3={'I8': (1.4268856596277857e-08, 0.0034942075172879894), 'I6c': (0.34202834233097557, 0.009693101302870877), 'I3': (-0.14526382617927724, 0.007470866614311622), 'I2s': (0.08557946783014758, 0.006756560849723092), 'I5': (0.33302245337718306, 0.008616307904346449), 'I4': (-0.19141170717619527, 0.008129998936831861), 'I7': (0.003926434707870463, 0.006732689830435068), 'loglh': -3348.493795146821, 'I1s': (0.4990591670029613, 0.010722256491258075), 'iterations': 209, 'I6s': (-0.33060718026435354, 0.009325736294684694), 'I2c': (-0.19563284623525445, 0.01301438189418369), 'I1c': (0.6096802446413863, 0.011506646547995103)}
bin4={'I8': (2.1618858159211385e-07, 0.013058719412219505), 'I6c': (0.24599656419991578, 0.018562094680268343), 'I3': (-0.3419728474567232, 0.01636073511732733), 'I2s': (0.19672219603864693, 0.013655194126967918), 'I5': (0.34045992655483814, 0.013872378970887972), 'I4': (-0.3575002143360302, 0.016283227601357253), 'I7': (0.0015714509288673462, 0.012699286587185388), 'loglh': -1648.391991133542, 'I1s': (0.8947836379736009, 0.024744142954427617), 'iterations': 240, 'I6s': (-0.37605154236807925, 0.016950797238051152), 'I2c': (-0.39941097985228136, 0.024765570006029858), 'I1c': (0.7931859050189987, 0.022251520976753703)}

RABlist=[]
RABerr=[]
RLTlist=[]
RLTerr=[]
AFBlist=[]
AFBerr=[]
q2err=[]
A6slist=[]
A6serr=[]
A3list=[]
A3err=[]
A9err=[]
A9list=[]
A4list=[]
A4err=[]
A8list=[]
A8err=[]
A5list=[]
A5err=[]
A7list=[]
A7err=[]
binlist=[bin1,bin2,bin3,bin4]
for binn in binlist:
  I8=binn['I8'][0]
  I8err=binn['I8'][1]
  I6c=binn['I6c'][0]
  I6cerr=binn['I6c'][1]
  I3=binn['I3'][0]
  I3err=binn['I3'][1]
  I2s=binn['I2s'][0]
  I2serr=binn['I2s'][1]
  I5=binn['I5'][0]
  I5err=binn['I5'][1]
  I4=binn['I4'][0]
  I4err=binn['I4'][1]
  I7=binn['I7'][0]
  I7err=binn['I7'][1]
  I1s=binn['I1s'][0]
  I1serr=binn['I1s'][1]
  I6s=binn['I6s'][0]
  I6serr=binn['I6s'][1]
  I2c=binn['I2c'][0]
  I2cerr=binn['I2c'][1]
  I1c=binn['I1c'][0]
  I1cerr=binn['I1c'][1]
  rlt1=3*I1c-I2c
  rlt2=6*I1s-2*I2s
  rlt=rlt1/rlt2
  RLTlist.append(rlt)
  rlterr1=np.sqrt((3*I1cerr)**2+(I2cerr)**2)
  rlterr2=np.sqrt((6*I1serr)**2+(2*I2serr)**2)
  rlterr=rlt*np.sqrt((rlterr1/rlt1)**2+(rlterr2/rlt2)**2)
  RLTerr.append(rlterr)
  rab1=I1c+2*I1s-3*I2c-6*I2s
  rab2=2*I1c+4*I1s+2*I2c+4*I2s
  rab=rab1/rab2
  raberr1=np.sqrt(I1cerr**2+(2*I1serr)**2+(3*I2cerr)**2+(6*I2serr)**2)
  raberr2=np.sqrt((2*I1cerr)**2+(4*I1serr)**2+(2*I2cerr)**2+(4*I2serr)**2)
  raberr=rab*np.sqrt((raberr1/rab1)**2+(raberr2/rab2)**2)
  RABlist.append(rab)  
  RABerr.append(raberr)
  Gammaq=(3*I1c+6*I1s-I2c-2*I1s)/4.
  Gammaqerr=(np.sqrt((3*I1c)**2+(6*I1s)**2+(I2c)**2+(2*I1s)**2))/4.
  afb1=I6c+2*I6s
  afb1err=np.sqrt(I6cerr**2+4*I6serr**2)
  afb=(3/8.)*(afb1/Gammaq)
  AFBlist.append(afb) 
  afberr=(3/8.)*afb*np.sqrt((afb1err/afb1)**2+(Gammaqerr/Gammaq)**2)
  AFBerr.append(afberr)
  a6s=(-27/8.)*(I6s/Gammaq)
  a6serr=(-27/8.)*a6s*np.sqrt((I6serr/I6s)**2+(Gammaqerr/Gammaq)**2)
  A6slist.append(a6s)
  A6serr.append(a6serr)
  a3=(1/(np.pi*2))*I3/Gammaq
  a3err=(1/(np.pi*2))*a3*np.sqrt((I3err/I3)**2+(Gammaqerr/Gammaq)**2)
  A3err.append(a3err)
  A3list.append(a3)
  I9=1-I8-I7-I6s-I6c-I5-I4-I3-I2s-I2c-I1s-I1c
  a9=(1/(np.pi*2))*I9/Gammaq
  I9err=np.sqrt(I8err**2+I7err**2+I6serr**2+I6cerr**2+I5err**2+I4err**2+I3err**2+I2serr**2+I2cerr**2+I1serr**2+I1cerr**2)
  a9err=(1/(np.pi*2))*a9*np.sqrt((I9err/I9)**2+(Gammaqerr/Gammaq)**2)
  A9list.append(a9)
  A9err.append(a9err)
  a4=(-2/np.pi)*I4/Gammaq
  a4err=(-2/np.pi)*a4*np.sqrt((I4err/I4)**2+(Gammaqerr/Gammaq)**2)
  A4list.append(a4)
  A4err.append(a4err)
  a8=(2/np.pi)*I8/Gammaq
  a8err=(2/np.pi)*a8*np.sqrt((I8err/I8)**2+(Gammaqerr/Gammaq)**2)
  A8list.append(a8)
  A8err.append(a8err)
  a5=(-3/4.)*I5/Gammaq
  a5err=(-3/4.)*a5*np.sqrt((I5err/I5)**2+(Gammaqerr/Gammaq)**2)
  A5list.append(a5)
  A5err.append(a5err)
  a7=(-3/4.)*I7/Gammaq
  q2err.append((borders[1]-borders[0])/2.)
  if I7!=0:
    a7err=(-3/4.)*a7*np.sqrt((I7err/I7)**2+(Gammaqerr/Gammaq)**2)
    A7err.append(a7err)
  A7list.append(a7)
  
  
  
  
  ###RAB
RAB1list=RABlist
RAB1err=RABerr

plt.fill_between(centers, listm(RAB1list,RAB1err), listp(RAB1list,RAB1err),
    alpha=0.8, edgecolor='#3F7F4C', facecolor='#7EFF99',
    linewidth=0,label='calculated with I')


def power(x,c,d,e):
  res=c*x**2+d*x+e
  return res


sol,_=curve_fit(power, centers, RAB1list, maxfev=2000)
plt.plot(np.linspace(3,12,50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$R_{AB}$ ($q^2$)')
plt.title(r'$R_{AB}$',fontsize=14, color='black')
plt.legend()
plt.ylim(0.4,1.0)
plt.xlim(3,11)
plt.grid(linestyle='-', linewidth='0.5', color='gray')

###RLT
RLT1list=RLTlist
RLT1err=RLTerr
plt.errorbar(centers,RLT1list, xerr=q2err,yerr=RLT1err, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)

plt.fill_between(centers, listm(RLT1list,RLT1err), listp(RLT1list,RLT1err),
    alpha=0.8, edgecolor='#3F7F4C', facecolor='#7EFF99',
    linewidth=0,label='calculated with I')
def power(x,c,d,e):
        res=c*x**2+d*x+e
        return res
sol,_=curve_fit(power, centers, RLT1list, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$R_{L,T}$ ($q^2$)')
plt.title(r'$R_{L,T}$',fontsize=14, color='black')
plt.legend()
plt.ylim(0,4)
plt.xlim(3,11)
plt.grid(linestyle='-', linewidth='0.5', color='gray')

###AFB

AFB1list=AFBlist
AFB1err=AFBerr
def listm(list1,list2):
  res=[]
  for i in range(len(list1)):
    a=list1[i]-list2[i]
    res.append(a)
  return res
def listp(list1,list2):
  res=[]
  for i in range(len(list1)):
    a=list1[i]+list2[i]
    res.append(a)
  return res



def power(x,c,d,e):
        res=c*x**2+d*x+e
        return res
sol,_=curve_fit(power, centers, AFB1list, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.fill_between(centers, listm(AFB1list,AFB1err), listp(AFB1list,AFB1err),
    alpha=0.8, edgecolor='#3F7F4C', facecolor='#7EFF99',
    linewidth=0,label='calculated with I')



plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{FB}$ ($q^2$)')
plt.title(r'$A_{FB}$',fontsize=14, color='black')
plt.legend()
plt.ylim(-0.25,0.4)
plt.xlim(3,11)
plt.grid(linestyle='-', linewidth='0.5', color='gray')


###A6s


plt.errorbar(centers,A6slist, xerr=q2err,yerr=A6serr, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
sol,_=curve_fit(power, centers, A6slist, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='r',label='parabolic fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{6s}$ ($q^2$)')
plt.title(r'$A_{6s}$',fontsize=14, color='black')
plt.legend()
plt.xlim(3,11)
plt.grid(linestyle='-', linewidth='0.5', color='gray')


###A3

A31list=A3list
A31err=A3err
  
plt.errorbar(centers,A31list, xerr=q2err,yerr=A31err, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)

plt.fill_between(centers, listm(A31list,A31err), listp(A31list,A31err),
    alpha=0.8, edgecolor='#3F7F4C', facecolor='#7EFF99',
    linewidth=0,label='calculated with I')

sol,_=curve_fit(power, centers, A31list, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{3}$ ($q^2$)')
plt.title(r'$A_{3}$',fontsize=14, color='black')
plt.legend()
plt.xlim(3,11)
plt.ylim(-0.04,0.06)
plt.grid(linestyle='-', linewidth='0.5', color='gray')


###A7


  
plt.errorbar(centers,A7list, xerr=q2err,yerr=A7err, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
sol,_=curve_fit(linear, centers, A7list, maxfev=2000)
plt.plot(np.linspace(3,12,50),linear(np.linspace(3,12,50),sol[0],sol[1]),color='r',label='linear fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{7}$ ($q^2$)')
plt.title(r'$A_{7}$',fontsize=14, color='black')
plt.legend()
plt.xlim(3,11)
plt.ylim(-0.04,0.02)
plt.grid(linestyle='-', linewidth='0.5', color='gray')




###A4
plt.errorbar(centers,A4list, xerr=q2err,yerr=A4err, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
sol,_=curve_fit(power, centers, A4list, maxfev=2000)
plt.plot(np.linspace(3,12,50),power(np.linspace(3,12,50),sol[0],sol[1],sol[2]),color='r',label='parabolic fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{4}$ ($q^2$)')
plt.title(r'$A_{4}$',fontsize=14, color='black')
plt.legend()
plt.xlim(3,11)
plt.ylim(-0.3,0.2)
plt.grid(linestyle='-', linewidth='0.5', color='gray')


###A5
plt.errorbar(centers,A5list, xerr=q2err,yerr=A5err, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
sol,_=curve_fit(power, centers, A5list, maxfev=2000)
plt.plot(np.linspace(3,12,50),power(np.linspace(3,12,50),sol[0],sol[1],sol[2]),color='r',label='parabolic fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{5}$ ($q^2$)')
plt.title(r'$A_{5}$',fontsize=14, color='black')
plt.legend()
plt.xlim(3,11)
plt.grid(linestyle='-', linewidth='0.5', color='gray')

###A9


def linear(x,d,e): 
  res=d*x+e
  return res
  
plt.errorbar(centers,A9list, xerr=q2err,yerr=A9err, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
sol,_=curve_fit(linear, centers, A9list, maxfev=2000)
plt.plot(np.linspace(3,12,50),linear(np.linspace(3,12,50),sol[0],sol[1]),color='r',label='linear fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{9}$ ($q^2$)')
plt.title(r'$A_{9}$',fontsize=14, color='black')
plt.legend()
plt.xlim(3,11)
plt.ylim(-0.004,0.004)
plt.grid(linestyle='-', linewidth='0.5', color='gray')


##A8

plt.errorbar(centers,A8list, xerr=q2err,yerr=A8err, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
sol,_=curve_fit(linear, centers, A8list, maxfev=2000)
plt.plot(np.linspace(3,12,50),linear(np.linspace(3,12,50),sol[0],sol[1]),color='r',label='linear fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{8}$ ($q^2$)')
plt.title(r'$A_{8}$',fontsize=14, color='black')
plt.legend()
plt.xlim(3,11)
plt.ylim(-0.020,0.010)
plt.grid(linestyle='-', linewidth='0.5', color='gray')

