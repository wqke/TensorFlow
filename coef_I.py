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


if __name__ == "__main__" : 
  # Four body angular phase space is described by 3 angles. 
  phsp = tfa.RectangularPhaseSpace( ( (-1., 1.), (-1., 1.), (-math.pi, math.pi) ) )
  vals = {'I1c': 3.03,
            'I1s': 2.04,
            'I2c': -0.89,
            'I2s': 0.35,
            'I3': -0.56,
            'I4': -0.74,
            'I5': 1.61,
            'I6c': 1.96,
            'I6s': -1.38,
            'I7': 0.,
            'I8': 0.,
            'I9': 0.}
  tot_rate = 0.
  for v in vals:
    tot_rate += vals[v]
  for v in vals:
    vals[v] = vals[v]/tot_rate
  # Fit parameters of the model 
  I8  = tfa.FitParameter("I8",vals["I8"] ,  0.000, 1.000, 0.01)   
  I7 = tfa.FitParameter("I7",vals["I7"], -1.000, 1.000, 0.01)
  I6s  = tfa.FitParameter("I6s",vals["I6s"] ,  -1.000, 1.000, 0.01) 
  I6c  = tfa.FitParameter("I6c",vals["I6c"] ,   0.000, 1.000, 0.01)    
  I5 = tfa.FitParameter("I5",vals["I5"],   0.000, 1.000, 0.01)
  I4  = tfa.FitParameter("I4",vals["I4"] ,  -1.000, 1.000, 0.01)  
  I3  = tfa.FitParameter("I3",vals["I3"] ,   -1.000, 1.000, 0.01)    
  I2s = tfa.FitParameter("I2s",vals["I2s"],  -1.000, 1.000, 0.01)
  I2c  = tfa.FitParameter("I2c",vals["I2c"] , -1.000, 1.000, 0.01) 
  I1s  = tfa.FitParameter("I1s",vals["I1s"] , 0.000, 1.000, 0.01)    
  I1c = tfa.FitParameter("I1c",vals["I1c"], 0.000, 1.000, 0.01)
  params = [ I1c, I1s, I2c, I2s, I6c, I6s, I3, I9, I4, I8, I5, I7 ]
  
  ### Start of model description

  def model(x) : 
    # Get phase space variables
    cosThetast = phsp.Coordinate(x, 0)     #D* angle costhetast
    cosThetal = phsp.Coordinate(x, 1)  #Lepton angle costhetal
    chi = phsp.Coordinate(x, 2)
    # Derived quantities
    sinThetast = tfa.Sqrt( 1.0 - cosThetast * cosThetast )
    sinThetal = tfa.Sqrt( 1.0 - cosThetal * cosThetal )
    sinTheta2st =  (1.0 - cosThetast * cosThetast)
    sinTheta2l =  (1.0 - cosThetal * cosThetal)
    sin2Thetast = (2.0 * sinThetast * cosThetast)
    cos2Thetal = (2.0 * cosThetal * cosThetal - 1.0)
    coschi=tf.cos(chi)
    sinchi=tf.sin(chi)
    cos2chi=2*coschi*coschi-1
    sin2chi=2*sinchi*coschi    
    # Decay density
    pdf  =  I1c* cosThetast*cosThetast
    pdf += I1s * sinTheta2st
    pdf +=  I2c * cosThetast*cosThetast*cos2Thetal
    pdf +=  I2s * sinTheta2st *  cos2Thetal
    pdf +=  I6c *cosThetast*cosThetast *cosThetal
    pdf +=  I6s * sinTheta2st *  cosThetal
    pdf += I3 * cos2chi * sinTheta2l * sinTheta2st
    pdf += (1.0 -I1c -I1s -I2c -I2s -I3 -I4 -I5 - I6c -I6s - I7 -I8) * sin2chi * sinThetal * sinThetal * sinThetast * sinThetast
    pdf +=  I4 * coschi * 2 * sinThetal * cosThetal * sin2Thetast 
    pdf +=  I8 * sinchi * 2 * sinThetal * cosThetal * sin2Thetast 
    pdf +=  I5 * coschi * sinThetal  * sin2Thetast 
    pdf +=  I7 * sinchi * sinThetal  * sin2Thetast 
    return pdf

  ### End of model description

 
  # Placeholders for data and normalisation samples (will be used to compile the model)
  data_ph = phsp.data_placeholder
  norm_ph = phsp.norm_placeholder
 
  # TF initialiser
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)

  # Create normalisation sample (uniform sample in the 3D phase space)
  norm_sample = sess.run( phsp.UniformSample(1000000) )

  tree = TChain("DecayTree")
  tree.Add("/home/ke/calculateI/model_total_new.root")
  branch_names = ["costheta_X_true","costheta_L_true","chi_true"]
  data_sample = tree2array(tree,branches=['costheta_X_true','costheta_L_true','chi_true'],selection='q2_true >= 3.20305994 & q2_true<5.0738137')
  data_sample = rec2array(data_sample)
  #array([ 3.20305994,  5.0738137 ,  6.94456747,  8.81532123, 10.686075  ])   borders
  #array([4.13843682, 6.00919059, 7.87994435, 9.75069811])   centers  
  data_sample = sess.run(phsp.Filter(data_sample))
  data_sample = sess.run(phsp.Filter(data_sample))
  
  
  
  # Estimate the maximum of PDF for toy MC generation using accept-reject method
  majorant = tfa.EstimateMaximum(sess, model(data_ph), data_ph, norm_sample )*1.1
  print "Maximum = ", majorant

  # TF graph for the PDF integral
  norm = tfa.Integral( model(norm_ph) )

  # TF graph for unbinned negalite log likelihood (the quantity to be minimised)
  nll = tfa.UnbinnedNLL( model(data_ph), norm )

  # Options for profiling
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()



  # Run MINUIT minimisation of the neg. log likelihood
  result = tfa.RunMinuit(sess, nll, { data_ph : data_sample, norm_ph : norm_sample }, options = options, run_metadata = run_metadata, runHesse=True)
  print result
  tfa.WriteFitResults(result, "result.txt")

  # Run toy MC corresponding to fitted result
  fit_data = tfa.RunToyMC( sess, model(data_ph), data_ph, phsp, 1000000, majorant, chunk = 1000000)
  f = TFile.Open("toyresult.root", "RECREATE")
  tfa.FillNTuple("toy", fit_data, ["cos*", "cosl", "chi" ])
  f.Close()

  # Store timeline profile 
  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
  chrome_trace = fetched_timeline.generate_chrome_trace_format()
  with open('timeline.json', 'w') as f:
    f.write(chrome_trace)

    
    
    
    
  """"""
  
  
bin1={'I9': (0.0, 0.010607627405155862),'I8': (0.015669054754884504, 0.011335544303172546), 'I6c': (0.30089077571560785, 0.01730526182048453), 'I3': (-0.07664697712810142, 0.011270441555538968), 'I2s': (-0.0003613276419075495, 0.009587158432862453), 'I5': (5.397449154287415e-11, 0.004335453833712172), 'I4': (0.00876995407580794, 0.011473610838036552), 'I7': (-0.005185044785479098, 0.010626871338991428), 'loglh': -402.30161967940535, 'I1s': (0.2548081911160103, 0.012996547048620383), 'iterations': 362, 'I6s': (-0.08100891029547885, 0.010387232699476334), 'I2c': (-0.00843456616903926, 0.019896499087183883), 'I1c': (0.5871393365296936, 0.021625616329128405)}
bin2={'I9': (0.0, 0.005681658591526306), 'I8': (0.0029117094738847493, 0.009397343723619223), 'I6c': (0.26078103147355874, 0.012436705411886129), 'I3': (-0.06474280093362195, 0.008244597890732785), 'I2s': (0.0027734696127090785, 0.007068300055282806), 'I5': (0.013646055631861498, 0.007278847585266174), 'I4': (0.01373083758936433, 0.008087940666354099), 'I7': (-0.000493788381376703, 0.007457786963463553), 'loglh': -696.0002940131169, 'I1s': (0.3768367129008818, 0.012467219239719135), 'iterations': 361, 'I6s': (-0.17310723212650447, 0.009115185671568826), 'I2c': (-0.055520666163171306, 0.014297542892824588), 'I1c': (0.6213092306506548, 0.016889172253060192)}
bin3={'I9': (0.0, 0.005038506691403211), 'I8': (0.002998316736955331, 0.011303795995435378), 'I6c': (0.2520912095138095, 0.014334825722033734), 'I3': (-0.1119322156906184, 0.010636718506512044), 'I2s': (0.05819583063682332, 0.008706772771723381), 'I5': (0.0029011056148955383, 0.00862105371815386), 'I4': (-0.018668273938552415, 0.00983915807908392), 'I7': (-0.0027087524968834042, 0.008871632573932864), 'loglh': -862.535977329059, 'I1s': (0.5730829334797811, 0.020646007736140448), 'iterations': 372, 'I6s': (-0.31394145291597564, 0.014214847560629906), 'I2c': (-0.14024241732992504, 0.01804389675629542), 'I1c': (0.7123624206728822, 0.022364154232234812)}
bin4={'I9': (0.0, 0.005926033724912738), 'I8': (0.0028101259361832387, 0.011427876490151978), 'I6c': (0.19946960495571386, 0.01800712905778673), 'I3': (-0.13335062192907388, 0.015597751493046264), 'I2s': (0.14155232686715502, 0.013202333355721962), 'I5': (3.5079383842173684e-11, 0.0041677085545615555), 'I4': (0.0017109855844870125, 0.013432819471393742), 'I7': (-0.0021419965410274244, 0.012304207715664095), 'loglh': -509.8724746438387, 'I1s': (0.7670004586185274, 0.03348489955439782), 'iterations': 341, 'I6s': (-0.3396012270894664, 0.020187787639947607), 'I2c': (-0.30517305907948655, 0.027898210679316915), 'I1c': (0.6786459157255944, 0.026987045943140886)}

centers=[4.13843682, 6.00919059, 7.87994435, 9.75069811]
borders=[ 3.20305994,  5.0738137 ,  6.94456747,  8.81532123, 10.686075  ]

binlist=[bin1,bin2,bin3,bin4]
I8list=[]
I9list=[]
I6clist=[]
I3list=[]
I2slist=[]
I5list=[]
I4list=[]
I7list=[]
I1slist=[]
I6slist=[]
I2clist=[]
I1clist=[]

I8errlist=[]
I9errlist=[]
I6cerrlist=[]
I3errlist=[]
I2serrlist=[]
I5errlist=[]
I4errlist=[]
I7errlist=[]
I1serrlist=[]
I6serrlist=[]
I2cerrlist=[]
I1cerrlist=[]
for binn in binlist:
  I9list.append(binn['I9'][0])
  I9errlist.append(binn['I9'][1])
  I8list.append(binn['I8'][0])
  I8errlist.append(binn['I8'][1])
  I6clist.append(binn['I6c'][0])
  I6cerrlist.append(binn['I6c'][1])
  I3list.append(binn['I3'][0])
  I3errlist.append(binn['I3'][1])
  I2slist.append(binn['I2s'][0])
  I2serrlist.append(binn['I2s'][1])
  I5list.append(binn['I5'][0])
  I5errlist.append(binn['I5'][1])
  I4list.append(binn['I4'][0])
  I4errlist.append(binn['I4'][1])
  I7list.append(binn['I7'][0])
  I7errlist.append(binn['I7'][1])
  I1slist.append(binn['I1s'][0])
  I1serrlist.append(binn['I1s'][1])
  I6slist.append(binn['I6s'][0])
  I6serrlist.append(binn['I6s'][1])
  I2clist.append(binn['I2c'][0])
  I2cerrlist.append(binn['I2c'][1])
  I1clist.append(binn['I1c'][0])
  I1cerrlist.append(binn['I1c'][1])

 

I8list_th=[0,0,0,0]
I9list_th=[0,0,0,0]
I6clist_th=[(7.87e-16)/(2.0198e-15),5.59/15.044,3.93/11.23,2.19/7.52]
I3list_th=[(-5.31e-17)/(2.0198e-15),-1.13/15.044,-1.67/11.23,-2.29/7.52]
I2slist_th=[(4.05e-17)/(2.0198e-15),0.774/15.044,1.03/11.23,1.25/7.52]
I5list_th=[(4.89e-16)/(2.0198e-15),4.66/15.044,3.96/11.23,2.52/7.52]
I4list_th=[(-9.86e-17)/(2.0198e-15),-1.72/15.044,-2.18/11.23,-2.55/7.52]
I7list_th=[0,0,0,0]
I1slist_th=[(3.65e-16)/(2.0198e-15),4.92/15.044,5.66/11.23,6.21/7.52]
I6slist_th=[(-3.07e-16)/(2.0198e-15),-3.88/15.044,-3.9/11.23,-2.94/7.52]
I2clist_th=[(-1.46e-16)/(2.0198e-15),-2.22/15.044,-2.54/11.23,-2.72/7.52]
I1clist_th=[(9.43e-16)/(2.0198e-15),8.05/15.044,6.94/11.23,5.85/7.52]

I8errlist_th=[0,0,0,0]
I9errlist_th=[0,0,0,0]
I6cerrlist_th=[(0.5e-16)/(2.0198e-15),0.35/15.044,0.24/11.23,0.14/7.52]
I3errlist_th=[(0.16e-17)/(2.0198e-15),0.03/15.044,0.04/11.23,0.06/7.52]
I2serrlist_th=[(0.12e-17)/(2.0198e-15),0.02/15.044,0.03/11.23,0.03/7.52]
I5errlist_th=[(0.21e-16)/(2.0198e-15),0.18/15.044,0.15/11.23,0.09/7.52]
I4errlist_th=[(0.29e-17)/(2.0198e-15),0.05/15.044,0.06/11.23,0.06/7.52]
I7errlist_th=[0,0,0,0]
I1serrlist_th=[(0.11e-16)/(2.0198e-15),0.13/15.044,0.14/11.23,0.15/7.52]
I6serrlist_th=[(0.11e-16)/(2.0198e-15),0.14/15.044,0.13/11.23,0.1/7.52]
I2cerrlist_th=[(0.05e-16)/(2.0198e-15),0.06/15.044,0.07/11.23,0.07/7.52]
I1cerrlist_th=[(0.47e-16)/(2.0198e-15),0.32/15.044,0.23/11.23,0.15/7.52]

q2err=[centers[1]-centers[0],centers[1]-centers[0],centers[1]-centers[0],centers[1]-centers[0]]
q2err_th=[(6.2-min(borders))/2.,0.9,1.3/2.,(max(borders)-8.9)/2.]
centers_th=[(6.2-min(borders))/2.+min(borders),0.9+6.2,7.6+0.65,8.9+(max(borders)-8.9)/2.]


""""""
plt.errorbar(centers,I5list, xerr=q2err,yerr=I5errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I5 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I5list_th, xerr=q2err_th,yerr=I5errlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I5 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{5}$ ($q^2$)')
plt.title(r'$I_5$',fontsize=14, color='black')
plt.legend()

""""""
plt.errorbar(centers,I4list, xerr=q2err,yerr=I4errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I4 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I4list_th, xerr=q2err_th,yerr=I4errlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I4 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{4}$ ($q^2$)')
plt.title(r'$I_4$',fontsize=14, color='black')
plt.legend()

""""""
plt.errorbar(centers,I3list, xerr=q2err,yerr=I3errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I3 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I3list_th, xerr=q2err_th,yerr=I3errlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I3 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{3}$ ($q^2$)')
plt.title(r'$I_{3}$',fontsize=14, color='black')
plt.legend()


""""""
plt.errorbar(centers,I1clist, xerr=q2err,yerr=I1cerrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I1c - Rapidsim(total geometry)')


plt.errorbar(centers_th,I1clist_th, xerr=q2err_th,yerr=I1cerrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I1c - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{1c}$ ($q^2$)')
plt.title(r'$I_{1c}$',fontsize=14, color='black')
plt.legend()


""""""

plt.errorbar(centers,I2clist, xerr=q2err,yerr=I2cerrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I2c - Rapidsim(total geometry)')


plt.errorbar(centers_th,I2clist_th, xerr=q2err_th,yerr=I2cerrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I2c - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{2c}$ ($q^2$)')
plt.title(r'$I_{2c}$',fontsize=14, color='black')
plt.legend()



""""""
plt.errorbar(centers,I6clist, xerr=q2err,yerr=I6cerrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I6c - Rapidsim(total geometry)')


plt.errorbar(centers_th,I6clist_th, xerr=q2err_th,yerr=I6cerrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I6c - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{6c}$ ($q^2$)')
plt.title(r'$I_{6c}$',fontsize=14, color='black')
plt.legend()

""""""
plt.errorbar(centers,I8list, xerr=q2err,yerr=I8errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I8 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I8list_th, xerr=q2err_th,yerr=I1cerrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I8 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{8}$ ($q^2$)')
plt.title(r'$I_{8}$',fontsize=14, color='black')
plt.legend()
""""""
plt.errorbar(centers,I9list, xerr=q2err,yerr=I9errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I9 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I9list_th, xerr=q2err_th,yerr=I9errlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I9 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{9}$ ($q^2$)')
plt.title(r'$I_{9}$',fontsize=14, color='black')
plt.legend()
""""""
plt.errorbar(centers,I2slist, xerr=q2err,yerr=I2serrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I2s - Rapidsim(total geometry)')


plt.errorbar(centers_th,I2slist_th, xerr=q2err_th,yerr=I2serrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I2s - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{2s}$ ($q^2$)')
plt.title(r'$I_{2s}$',fontsize=14, color='black')
plt.legend()
""""""
plt.errorbar(centers,I1slist, xerr=q2err,yerr=I1serrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I1s - Rapidsim(total geometry)')


plt.errorbar(centers_th,I1slist_th, xerr=q2err_th,yerr=I1serrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I1s - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{1s}$ ($q^2$)')
plt.title(r'$I_{1s}$',fontsize=14, color='black')
plt.legend()
""""""
plt.errorbar(centers,I6slist, xerr=q2err,yerr=I6serrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I6s - Rapidsim(total geometry)')


plt.errorbar(centers_th,I6slist_th, xerr=q2err_th,yerr=I6serrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I6s - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{6s}$ ($q^2$)')
plt.title(r'$I_{6s}$',fontsize=14, color='black')
plt.legend()


