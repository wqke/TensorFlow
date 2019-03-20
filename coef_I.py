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
  """
  FL  = tfa.FitParameter("FL" ,  0.600,  0.000, 1.000, 0.01)    #Taken from Belle measurement
  """
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
    pdf  = (9.0/(32*np.pi)) * I1c* cosThetast*cosThetast
    pdf +=  (9.0/(32*np.pi)) * I1s * sinTheta2st
    pdf +=  (9.0/(32*np.pi)) * I2c * cosThetast*cosThetast*cos2Thetal
    pdf +=  (9.0/(32*np.pi)) * I2s * sinTheta2st *  cos2Thetal
    pdf +=  (9.0/(32*np.pi))* I6c *cosThetast*cosThetast *cosThetal
    pdf +=  (9.0/(32*np.pi))* I6s * sinTheta2st *  cosThetal
    pdf +=  (9.0/(32*np.pi))* I3 * cos2chi * sinTheta2l * sinTheta2st
    pdf += (1.0 -I1c -I1s -I2c -I2s -I3 -I4 -I5 - I6c -I6s - I7 -I8) * sin2chi * sinThetal * sinThetal * sinThetast * sinThetast
    pdf +=  (9.0/(32*np.pi))* I4 * coschi * 2 * sinThetal * cosThetal * sin2Thetast 
    pdf +=  (9.0/(32*np.pi))* I8 * sinchi * 2 * sinThetal * cosThetal * sin2Thetast 
    pdf +=  (9.0/(32*np.pi))* I5 * coschi * sinThetal  * sin2Thetast 
    pdf +=  (9.0/(32*np.pi))* I7 * sinchi * sinThetal  * sin2Thetast 
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
  tree.Add("/home/ke/pythonap/model_tree_vars.root")
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
  
  
bin1={'I9': (0.0, 0.0009757628116016104),'I8': (0.01575390796369075, 0.01129185470054736), 'I6c': (0.300886347692888, 0.017107596706314177), 'I3': (-0.07800445929054256, 0.011279507881430506), 'I2s': (-0.001169543090427272, 0.009601568117104387), 'I5': (2.0094889308097663e-08, 0.004406879139363523), 'I4': (0.008825337969216474, 0.011460821468293414), 'I7': (-0.005477299024893023, 0.010619485965988995), 'loglh': -403.03860490452234, 'I1s': (0.2545933710463001, 0.012698113603691108), 'iterations': 568, 'I6s': (-0.08042788774708043, 0.010343271766412476), 'I2c': (-0.004196483017134822, 0.019859253776980168), 'I1c': (0.5888471710047258, 0.020761502463508164)}
bin2={'I9': (0.0, 0.0005247614222594899),'I8': (9.536740132598531e-07, 0.8711048945615066), 'I6c': (0.36162361623616235, 0.0005541851212953752), 'I3': (-0.10332103321033215, 0.0005513870312759961), 'I2s': (0.06457564575645747, 0.0005511794434117645), 'I5': (0.2970479704797048, 0.000546917811930614), 'I4': (-0.13653136531365317, 0.0005508421189155399), 'I7': (0.0, 0.0005380257181550885), 'loglh': 849.3162218782222, 'I1s': (0.3763837638376384, 0.0005467860561387816), 'iterations': 283, 'I6s': (-0.25461254612546125, 0.0005533711409203557), 'I2c': (-0.16420664206642066, 0.0005556751290641815), 'I1c': (0.559040590405904, 0.0005526458859542172)}
bin3={'I9': (0.0, 0.0004641382009709094),'I8': (9.536740132598531e-07, 0.8317030215534756), 'I6c': (0.36162361623616235, 0.0005030488787721166), 'I3': (-0.10332103321033215, 0.0005006903629491966), 'I2s': (0.06457564575645747, 0.0004990957020570841), 'I5': (0.2970479704797048, 0.0004959993334389956), 'I4': (-0.13653136531365317, 0.00049887269982829), 'I7': (0.0, 0.0004875607072261645), 'loglh': 1176.50811426245, 'I1s': (0.3763837638376384, 0.0004956926887378366), 'iterations': 273, 'I6s': (-0.25461254612546125, 0.0005020541056828809), 'I2c': (-0.16420664206642066, 0.0005042439608666793), 'I1c': (0.559040590405904, 0.0005008261714464224)}
bin4={'I9': (0.0, 0.0005393949022374223),'I8': (9.536740132598531e-07, 0.8136807401117219), 'I6c': (0.36162361623616235, 0.0005208707637417986), 'I3': (-0.10332103321033215, 0.0005171554591967831), 'I2s': (0.06457564575645747, 0.0005180596389131598), 'I5': (0.2970479704797048, 0.0005138859289743891), 'I4': (-0.13653136531365317, 0.0005176344656264154), 'I7': (0.0, 0.000504517360304213), 'loglh': 1280.5226864658075, 'I1s': (0.3763837638376384, 0.0005116529206002918), 'iterations': 287, 'I6s': (-0.25461254612546125, 0.0005185882708106937), 'I2c': (-0.16420664206642066, 0.0005224314533861518), 'I1c': (0.559040590405904, 0.0005198002371440968)}
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
plt.errorbar(centers,I4list, xerr=q2err,yerr=I4errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I4 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I4list_th, xerr=q2err_th,yerr=I4errlist_th, fmt='o', color='#FF9848',
ecolor='lightgray', elinewidth=3, capsize=0,label='I4 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{4}$ ($q^2$)')
plt.title(r'$I_4$',fontsize=14, color='black')
plt.legend()

""""""
plt.errorbar(centers,I3list, xerr=q2err,yerr=I3errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I3 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I3list_th, xerr=q2err_th,yerr=I3errlist_th, fmt='o', color='#FF9848',
ecolor='lightgray', elinewidth=3, capsize=0,label='I3 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{3}$ ($q^2$)')
plt.title(r'$I_{3}$',fontsize=14, color='black')
plt.legend()


""""""
plt.errorbar(centers,I7list, xerr=q2err,yerr=I7errlist, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label='I7 - Rapidsim(total geometry)')


plt.errorbar(centers_th,I7list_th, xerr=q2err_th,yerr=I7errlist_th, fmt='o', color='#FF9848',
ecolor='lightgray', elinewidth=3, capsize=0,label='I7 - Theory')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$I_{7}$ ($q^2$)')
plt.title(r'$I_{7}$',fontsize=14, color='black')
plt.legend()



