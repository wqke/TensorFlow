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

q2err=[(centers[1]-centers[0])/2.,(centers[1]-centers[0])/2.,(centers[1]-centers[0])/2.,(centers[1]-centers[0])/2.]
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
plt.savefig('I5.pdf')
plt.close()
plt.close()


""""""
plt.errorbar(centers,I4list, xerr=q2err,yerr=I4errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I4 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I4list_th, xerr=q2err_th,yerr=I4errlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I4 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{4}$ ($q^2$)')
plt.title(r'$I_4$',fontsize=14, color='black')
plt.legend()
plt.savefig('I4.pdf')
plt.close()
plt.close()

""""""
plt.errorbar(centers,I3list, xerr=q2err,yerr=I3errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I3 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I3list_th, xerr=q2err_th,yerr=I3errlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I3 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{3}$ ($q^2$)')
plt.title(r'$I_{3}$',fontsize=14, color='black')
plt.legend()
plt.savefig('I3.pdf')
plt.close()
plt.close()


""""""
plt.errorbar(centers,I1clist, xerr=q2err,yerr=I1cerrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I1c - Rapidsim(total geometry)')


plt.errorbar(centers_th,I1clist_th, xerr=q2err_th,yerr=I1cerrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I1c - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{1c}$ ($q^2$)')
plt.title(r'$I_{1c}$',fontsize=14, color='black')
plt.legend()
plt.savefig('I1c.pdf')
plt.close()
plt.close()


""""""

plt.errorbar(centers,I2clist, xerr=q2err,yerr=I2cerrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I2c - Rapidsim(total geometry)')


plt.errorbar(centers_th,I2clist_th, xerr=q2err_th,yerr=I2cerrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I2c - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{2c}$ ($q^2$)')
plt.title(r'$I_{2c}$',fontsize=14, color='black')
plt.legend()
plt.savefig('I2c.pdf')
plt.close()
plt.close()



""""""
plt.errorbar(centers,I6clist, xerr=q2err,yerr=I6cerrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I6c - Rapidsim(total geometry)')


plt.errorbar(centers_th,I6clist_th, xerr=q2err_th,yerr=I6cerrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I6c - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{6c}$ ($q^2$)')
plt.title(r'$I_{6c}$',fontsize=14, color='black')
plt.legend()
plt.savefig('I6c.pdf')
plt.close()
plt.close()

""""""
plt.errorbar(centers,I8list, xerr=q2err,yerr=I8errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I8 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I8list_th, xerr=q2err_th,yerr=I1cerrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I8 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{8}$ ($q^2$)')
plt.title(r'$I_{8}$',fontsize=14, color='black')
plt.legend()
plt.savefig('I8.pdf')
plt.close()
plt.close()

""""""
plt.errorbar(centers,I9list, xerr=q2err,yerr=I9errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I9 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I9list_th, xerr=q2err_th,yerr=I9errlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I9 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{9}$ ($q^2$)')
plt.title(r'$I_{9}$',fontsize=14, color='black')
plt.legend()
plt.savefig('I9.pdf')
plt.close()
plt.close()

""""""
plt.errorbar(centers,I2slist, xerr=q2err,yerr=I2serrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I2s - Rapidsim(total geometry)')


plt.errorbar(centers_th,I2slist_th, xerr=q2err_th,yerr=I2serrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I2s - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{2s}$ ($q^2$)')
plt.title(r'$I_{2s}$',fontsize=14, color='black')
plt.legend()
plt.savefig('I2s.pdf')
plt.close()
plt.close()

""""""
plt.errorbar(centers,I1slist, xerr=q2err,yerr=I1serrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I1s - Rapidsim(total geometry)')


plt.errorbar(centers_th,I1slist_th, xerr=q2err_th,yerr=I1serrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I1s - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{1s}$ ($q^2$)')
plt.title(r'$I_{1s}$',fontsize=14, color='black')
plt.legend()
plt.savefig('I1s.pdf')
plt.close()
plt.close()

""""""
plt.errorbar(centers,I6slist, xerr=q2err,yerr=I6serrlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I6s - Rapidsim(total geometry)')


plt.errorbar(centers_th,I6slist_th, xerr=q2err_th,yerr=I6serrlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I6s - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{6s}$ ($q^2$)')
plt.title(r'$I_{6s}$',fontsize=14, color='black')
plt.legend()
plt.savefig('I6s.pdf')
plt.close()
plt.close()
""""""
plt.errorbar(centers,I7list, xerr=q2err,yerr=I7errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I7 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I7list_th, xerr=q2err_th,yerr=I7errlist_th, fmt='o', color='#FF9848',
ecolor='lightblue', elinewidth=3, capsize=0,label='I7 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{7}$ ($q^2$)')
plt.title(r'$I_{7}$',fontsize=14, color='black')
plt.legend()
plt.savefig('I7.pdf')
plt.close()
plt.close()
###################
A9=[]
A9err=[]

RAB=[0.5168393418628526,0.5754315323083263,0.5259999434732144,0.5201484731767204]
RABerr=[0.03946140519844339,0.02379610813627703,0.019884245355197148,0.023525703588658225]
RLT=[1.1570902218955816,0.8510179773014285,0.6855078437442366,0.5420620614672387]
RLTerr=[0.042529226245617065,0.019378658321486806,0.013418064659929589,0.012259147243750605]
AFB=[0.07468736423475036,-0.03739650061542381,-0.12335433492743794,-0.13303462538684768]
AFBerr=[0.011765244964950851,0.007097271810108241,0.006281432800353698,-0.00733833478663557]
A6s=[0.3921071446676685,0.6819630268771364,0.9274668801632266,0.8475727665912955]
A6serr=[0.04762095358976847,0.030341851539147487,0.027376545105978874,0.033216329649641316]
A3=[-0.01749497601535148,-0.0120277240600514,-0.015593772761077455,-0.015694559942848487]
A3err=[0.0023025727512969,0.0014181995568725706,0.0012588023753893438,0.0015154175920712013]
A4=[-0.008007106970732241,-0.010203495879492616,0.010403039726970405,-0.0008054905310112061]
A4err=[0.010558692909665393,0.006077050399157597,0.005383292291295269,0.0063347007450097245]
A8=[0.0143060951594621,0.0021637147352230866,0.0016708351415484562,0.001322939160368935]
A8err=[0.010503461523109263,0.006988473470092948,0.0063079439378505246,0.005387124745950753]
A5=[-5.805612051143785e-11,-0.011946487721087216,-0.0019045868079891296,-1.9455714553637213e-11]
A5err=[0.004663307111338274,0.006446542608531029,0.005674887062792913,0.0023114929368526667]
A7=[0.005577145357337208,0.00043228878689010664,0.0017783062585460241,0.001187993308679663]
A7err=[0.011367325736833415,0.00652606110702692,0.0058090992416271045,0.006809086070842314]

bin1={'I9': (0.0, 0.010607627405155862),'I8': (0.015669054754884504, 0.011335544303172546), 'I6c': (0.30089077571560785, 0.01730526182048453), 'I3': (-0.07664697712810142, 0.011270441555538968), 'I2s': (-0.0003613276419075495, 0.009587158432862453), 'I5': (5.397449154287415e-11, 0.004335453833712172), 'I4': (0.00876995407580794, 0.011473610838036552), 'I7': (-0.005185044785479098, 0.010626871338991428), 'loglh': -402.30161967940535, 'I1s': (0.2548081911160103, 0.012996547048620383), 'iterations': 362, 'I6s': (-0.08100891029547885, 0.010387232699476334), 'I2c': (-0.00843456616903926, 0.019896499087183883), 'I1c': (0.5871393365296936, 0.021625616329128405)}
bin2={'I9': (0.0, 0.005681658591526306), 'I8': (0.0029117094738847493, 0.009397343723619223), 'I6c': (0.26078103147355874, 0.012436705411886129), 'I3': (-0.06474280093362195, 0.008244597890732785), 'I2s': (0.0027734696127090785, 0.007068300055282806), 'I5': (0.013646055631861498, 0.007278847585266174), 'I4': (0.01373083758936433, 0.008087940666354099), 'I7': (-0.000493788381376703, 0.007457786963463553), 'loglh': -696.0002940131169, 'I1s': (0.3768367129008818, 0.012467219239719135), 'iterations': 361, 'I6s': (-0.17310723212650447, 0.009115185671568826), 'I2c': (-0.055520666163171306, 0.014297542892824588), 'I1c': (0.6213092306506548, 0.016889172253060192)}
bin3={'I9': (0.0, 0.005038506691403211), 'I8': (0.002998316736955331, 0.011303795995435378), 'I6c': (0.2520912095138095, 0.014334825722033734), 'I3': (-0.1119322156906184, 0.010636718506512044), 'I2s': (0.05819583063682332, 0.008706772771723381), 'I5': (0.0029011056148955383, 0.00862105371815386), 'I4': (-0.018668273938552415, 0.00983915807908392), 'I7': (-0.0027087524968834042, 0.008871632573932864), 'loglh': -862.535977329059, 'I1s': (0.5730829334797811, 0.020646007736140448), 'iterations': 372, 'I6s': (-0.31394145291597564, 0.014214847560629906), 'I2c': (-0.14024241732992504, 0.01804389675629542), 'I1c': (0.7123624206728822, 0.022364154232234812)}
bin4={'I9': (0.0, 0.005926033724912738), 'I8': (0.0028101259361832387, 0.011427876490151978), 'I6c': (0.19946960495571386, 0.01800712905778673), 'I3': (-0.13335062192907388, 0.015597751493046264), 'I2s': (0.14155232686715502, 0.013202333355721962), 'I5': (3.5079383842173684e-11, 0.0041677085545615555), 'I4': (0.0017109855844870125, 0.013432819471393742), 'I7': (-0.0021419965410274244, 0.012304207715664095), 'loglh': -509.8724746438387, 'I1s': (0.7670004586185274, 0.03348489955439782), 'iterations': 341, 'I6s': (-0.3396012270894664, 0.020187787639947607), 'I2c': (-0.30517305907948655, 0.027898210679316915), 'I1c': (0.6786459157255944, 0.026987045943140886)}

corr_matrix1=array([[1.000,  0.040,  0.081, -0.246 ,-0.001, -0.028,  0.033 , 0.005, -0.064, -0.183, -0.288],
                   [  0.040  ,1.000 , 0.087 ,-0.195 ,-0.000 ,-0.015 , 0.081  ,0.020 ,-0.007 ,-0.228, -0.323],
                   [  0.081  ,0.087 , 1.000 ,-0.467  ,0.003  ,0.081 , 0.143 ,-0.036  ,0.143 ,-0.310 ,-0.279],
                   [  -0.246 ,-0.195 ,-0.467 , 1.000 ,-0.007 ,-0.204 ,-0.193 ,-0.082 ,-0.192 , 0.091 , 0.105],
                   [  -0.001 ,-0.000  ,0.003 ,-0.007 , 1.000  ,0.001  ,0.002 , 0.000 ,-0.001 ,-0.006 ,-0.009],
                   [  -0.028 ,-0.015,  0.081 ,-0.204 , 0.001  ,1.000  ,0.068 ,-0.009 ,-0.027 ,-0.186 ,-0.279],
                   [  0.033  ,0.081 , 0.143 ,-0.193  ,0.002  ,0.068  ,1.000  ,0.131  ,0.126 ,-0.454 ,-0.477],
                   [  0.005  ,0.020 ,-0.036 ,-0.082  ,0.000 ,-0.009  ,0.131  ,1.000 ,-0.398  ,0.184 ,-0.200],
                   [  -0.064 ,-0.007  ,0.143 ,-0.192 ,-0.001 ,-0.027 , 0.126 ,-0.398 , 1.000 ,-0.556 ,-0.346],
                   [  -0.183 ,-0.228 ,-0.310 , 0.091 ,-0.006 ,-0.186 ,-0.454 , 0.184 ,-0.556 , 1.000  ,0.593],
                   [-0.288 ,-0.323 ,-0.279 , 0.105 ,-0.009 ,-0.279 ,-0.477 ,-0.200 ,-0.346  ,0.593 , 1.000]])

from numpy import array
import numpy as np
import uncertainties
from uncertainties import *

corr_matrix2=array([
        [  1.000 ,-0.061,  0.038 ,-0.159, -0.063 ,-0.093, -0.056 ,-0.030, -0.101 ,-0.044, -0.140],
        [  -0.061 , 1.000 , 0.115 ,-0.147 ,-0.028 ,-0.017 , 0.027 , 0.002 , 0.001 ,-0.200 ,-0.256],
        [   0.038 , 0.115 , 1.000 ,-0.408  ,0.092 , 0.101 , 0.209 ,-0.108 , 0.293 ,-0.549 ,-0.425],
        [  -0.159 ,-0.147 ,-0.408 , 1.000 ,-0.134 ,-0.154 ,-0.143 ,-0.015 ,-0.160 , 0.048 , 0.004],
        [  -0.063, -0.028 , 0.092 ,-0.134 , 1.000 ,-0.062 , 0.021 ,-0.010 ,-0.042 ,-0.160 ,-0.224],
        [  -0.093 ,-0.017 , 0.101 ,-0.154 ,-0.062 , 1.000 , 0.025 ,-0.004 ,-0.021 ,-0.174 ,-0.232],
        [  -0.056 , 0.027 , 0.209 ,-0.143 , 0.021 , 0.025 , 1.000 , 0.081 , 0.097 ,-0.399 ,-0.417],
        [  -0.030 , 0.002 ,-0.108 ,-0.015 ,-0.010 ,-0.004 , 0.081 , 1.000 ,-0.384 , 0.150 ,-0.182],
        [   -0.101 , 0.001 , 0.293 ,-0.160 ,-0.042 ,-0.021 , 0.097 ,-0.384 , 1.000 ,-0.574 ,-0.340],
       [  -0.044 ,-0.200 ,-0.549 , 0.048 ,-0.160 ,-0.174 ,-0.399 , 0.150 ,-0.574 , 1.000  ,0.649],
       [  -0.140, -0.256 ,-0.425,  0.004 ,-0.224, -0.232 ,-0.417, -0.182 ,-0.340,  0.649  ,1.000]                                                                         
])


corr_matrix3=array([
        [  1.000, -0.083 , 0.111, -0.160 ,-0.035, -0.033 , 0.013, -0.056 ,-0.014, -0.135 ,-0.187],
        [ -0.083 , 1.000 , 0.186 ,-0.142 ,-0.003 , 0.019 , 0.089, -0.056 , 0.068 ,-0.240 ,-0.278],
        [  0.111 , 0.186 , 1.000 ,-0.363 , 0.165 , 0.227 , 0.409, -0.243 , 0.512 ,-0.772 ,-0.651],
        [ -0.160 ,-0.142 ,-0.363 , 1.000 ,-0.141 ,-0.153 ,-0.163, -0.008 ,-0.176 , 0.102 , 0.041],
        [ -0.035 ,-0.003  ,0.165 ,-0.141 , 1.000 ,-0.074 , 0.085, -0.053 , 0.053 ,-0.215 ,-0.250],
        [ -0.033 , 0.019  ,0.227 ,-0.153 ,-0.074 , 1.000 , 0.111, -0.063 , 0.097 ,-0.293 ,-0.333],
        [  0.013 , 0.089  ,0.409 ,-0.163 , 0.085 , 0.111 , 1.000, -0.028 , 0.270 ,-0.556 ,-0.550],
        [  -0.056 ,-0.056 ,-0.243 ,-0.008 ,-0.053 ,-0.063 ,-0.028 , 1.000 ,-0.465 , 0.290 , 0.016],
        [ -0.014 , 0.068,  0.512 ,-0.176 , 0.053 , 0.097 , 0.270 ,-0.465 , 1.000 ,-0.688 ,-0.514],
       [  -0.135 ,-0.240 ,-0.772  ,0.102 ,-0.215 ,-0.293 ,-0.556  ,0.290 ,-0.688  ,1.000 , 0.787],
       [  -0.187 ,-0.278, -0.651  ,0.041, -0.250 ,-0.333, -0.550  ,0.016, -0.514  ,0.787,  1.000]
])

corr_matrix4=array([
        [1.000 ,-0.116,  0.082 ,-0.142, -0.000 ,-0.072, -0.014 ,-0.097,  0.002 ,-0.117, -0.153],
        [  -0.116 , 1.000 , 0.212 ,-0.109 , 0.000 , 0.011 , 0.131 ,-0.123,  0.153 ,-0.295, -0.316],
        [ 0.082 , 0.212  ,1.000 ,-0.293  ,0.001 , 0.231 , 0.445 ,-0.333,  0.569 ,-0.758, -0.662],
        [-0.142 ,-0.109, -0.293 , 1.000 ,-0.001 ,-0.135 ,-0.100 ,-0.067, -0.062 ,-0.012, -0.035],
        [   -0.000 , 0.000 , 0.001 ,-0.001 , 1.000 ,-0.001 , 0.000 ,-0.001,  0.000 ,-0.001, -0.001],
        [   -0.072 , 0.011 , 0.231 ,-0.135 ,-0.001 , 1.000 , 0.119 ,-0.141,  0.155 ,-0.306, -0.334],
        [   -0.014 , 0.131 , 0.445 ,-0.100 , 0.000 , 0.119 , 1.000 ,-0.166,  0.395 ,-0.618, -0.589],
        [   -0.097 ,-0.123 ,-0.333 ,-0.067 ,-0.001 ,-0.141 ,-0.166 , 1.000, -0.546 , 0.432,  0.176],
        [   0.002 , 0.153 , 0.569 ,-0.062 , 0.000 , 0.155 , 0.395 ,-0.546,  1.000, -0.785, -0.640],
       [ -0.117 ,-0.295 ,-0.758 ,-0.012 ,-0.001 ,-0.306 ,-0.618 , 0.432, -0.785,  1.000 , 0.806],
       [ -0.153, -0.316 ,-0.662, -0.035 ,-0.001, -0.334 ,-0.589,  0.176 ,-0.640,  0.806 , 1.000]
])
  
I8=bin4['I8']
I7=bin4['I7']
I6s=bin4['I6s']
I6c=bin4['I6c']
I5=bin4['I5']
I4=bin4['I4']
I3=bin4['I3']
I2s=bin4['I2s']
I2c=bin4['I2c']
I1s=bin4['I1s']
I1c=bin4['I1c']

(I8,I7,I6s,I6c,I5,I4,I3,I2s,I2c,I1s,I1c)= correlated_values_norm([I8,I7,I6s,I6c,I5,I4,I3,I2s,I2c,I1s,I1c], corr_matrix4)
rab=(I1c+2*I1s-3*I2c-6*I2s)/(2*I1c+4*I1s+2*I2c+4*I2s)
rlt=(3*I1c-I2c)/(6*I1s-2*I2s)
Gammaq=(3*I1c+6*I1s-I2c-2*I1s)/4.
afb1=I6c+2*I6s
afb=(3/8.)*(afb1/Gammaq)
a6s=(-27/8.)*(I6s/Gammaq)
a3=(1/(np.pi*2))*I3/Gammaq
#a9=(1/(np.pi*2))*I9/Gammaq
a4=(-2/np.pi)*I4/Gammaq
a8=(2/np.pi)*I8/Gammaq
a5=(-3/4.)*I5/Gammaq
a7=a7=(-3/4.)*I7/Gammaq


""""""

def power(x,c,d,e):
  res=c*x**2+d*x+e
  return res
centers=[4.13843682, 6.00919059, 7.87994435, 9.75069811]
q2_borders=[ 3.20305994,  5.0738137 ,  6.94456747,  8.81532123, 10.686075  ]
q2err=[(centers[1]-centers[0])/2.,(centers[1]-centers[0])/2.,(centers[1]-centers[0])/2.,(centers[1]-centers[0])/2.]
plt.errorbar(centers,A7, xerr=q2err,yerr=A7err, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$A_{7}$ - RapidSim')
sol,_=curve_fit(power, centers, A7, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.ylim(-0.04,0.02)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{7}$ ($q^2$)')
plt.title(r'$A_{7}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('A7.pdf')
plt.close()
plt.close()
""""""



