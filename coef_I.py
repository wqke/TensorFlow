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
  
  
bin1={'I9': (-9.536740132598531e-07,0.7918978907785994),  'I8': (9.536740132598531e-07, 0.8232068457811121), 'I6c': (0.3884462151394422, 0.004123666443758167), 'I3': (-0.02589641434262946, 0.0033433798182763708), 'I2s': (0.019920318725099584, 0.002795408172666436), 'I5': (0.23904382470119523, 0.002108408037596571), 'I4': (-0.04980079681274896, 0.002209509345806715), 'I7': (0.0, 0.004259267488265939), 'loglh': 1673.6859330070388, 'I1s': (0.17928286852589637, 0.001929277811025698), 'iterations': 287, 'I6s': (-0.149402390438247, 0.0037183651931613215), 'I2c': (-0.06972111553784865, 0.004745414180768959), 'I1c': (0.4681274900398406, 0.0029526218074045207)}

bin2={'I9': (0.0005321802400515274,0.01937189102115804), 'I8': (0.015136756642085414, 0.008814838902486438), 'I6c': (0.25841055671053503, 0.014161486411722407), 'I3': (-0.08321603310061976, 0.009545083933419685), 'I2s': (0.018417665696870644, 0.008439172376322968), 'I5': (0.013701178767553668, 0.008822200022369692), 'I4': (-0.0032844548631916215, 0.009533934718218118), 'I7': (0.0060211234801577085, 0.008580111357092579), 'loglh': -600.5504115503347, 'I1s': (0.4438408787375673, 0.014589882421985223), 'iterations': 243, 'I6s': (-0.2332268403322908, 0.011171543039845022), 'I2c': (-0.07395485561363213, 0.016277349464227597), 'I1c': (0.637621843634913, 0.017940629522207097)}

bin3={'I9': (-0.01807320405560464,0.028202641816844665), 'I8': (0.007747071815710882, 0.012474831892402477), 'I6c': (0.23685281203754244, 0.018382557252344944), 'I3': (-0.11889014749182492, 0.014032499433078272), 'I2s': (0.08322731871654776, 0.011620931696731618), 'I5': (2.466375988718905e-06, 0.004399048074584494), 'I4': (-0.024985448049632653, 0.013028163951659066), 'I7': (-0.00945257566186919, 0.011734362522390407), 'loglh': -614.0437246707288, 'I1s': (0.6439461808662359, 0.028226830057596575), 'iterations': 380, 'I6s': (-0.35564520068963945, 0.019405707959718188), 'I2c': (-0.19457934740463434, 0.02479542704826082), 'I1c': (0.7498500735411797, 0.02880616896423821)}
bin4={'I9': (-0.00753207880177198,0.015567549247173037),'I8': (2.3961981843179103e-06, 0.015984107839973644), 'I6c': (0.1963221054502799, 0.01886868745886225), 'I3': (-0.13487834306570268, 0.01490964891765989), 'I2s': (0.1381330776741656, 0.013438732840722722), 'I5': (1.9879596857563797e-08, 0.007314856892917221), 'I4': (-0.0032272733746568916, 0.013617881904896545), 'I7': (-0.0001273067205527223, 0.012502437736883087), 'loglh': -459.64429312436806, 'I1s': (0.7645626191957862, 0.028509447266217514), 'iterations': 254, 'I6s': (-0.32919073439317836, 0.018303607596528382), 'I2c': (-0.3017481454229177, 0.02581260050730033), 'I1c': (0.6776836633807674, 0.024275327394433888)} 
      
      
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

#q2err_th=[(6.2-min(borders))/2.,0.9,1.3/2.,(max(borders)-8.9)/2.]
#q2err=q2err_th

q2_borders=[ 3.20305994, 6.2,  7.6,  8.9, 10.686075  ]
borders=q2_borders
q2err_th=[(q2_borders[1]-q2_borders[0])/2.,(q2_borders[2]-q2_borders[1])/2.,(q2_borders[3]-q2_borders[2])/2.,(q2_borders[4]-q2_borders[3])/2.]
q2err=q2err_th

centers_th=[(6.2-min(borders))/2.+min(borders),0.7+6.2,7.6+0.65,8.9+(max(borders)-8.9)/2.]
centers=centers_th

""""""
plt.errorbar(centers,I5list, xerr=q2err,yerr=I5errlist, fmt='o', color='black',
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


plt.errorbar(centers,I4list, xerr=q2err,yerr=I4errlist, fmt='o', color='black',
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



plt.errorbar(centers,I3list, xerr=q2err,yerr=I3errlist, fmt='o', color='black',
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



plt.errorbar(centers,I1clist, xerr=q2err,yerr=I1cerrlist, fmt='o', color='black',
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



plt.errorbar(centers,I2clist, xerr=q2err,yerr=I2cerrlist, fmt='o', color='black',
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




plt.errorbar(centers,I6clist, xerr=q2err,yerr=I6cerrlist, fmt='o', color='black',
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



plt.errorbar(centers,I8list, xerr=q2err,yerr=I8errlist, fmt='o', color='black',
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



plt.errorbar(centers,I9list, xerr=q2err,yerr=I9errlist, fmt='o', color='black',
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
plt.errorbar(centers,I2slist, xerr=q2err,yerr=I2serrlist, fmt='o', color='black',
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
plt.errorbar(centers,I1slist, xerr=q2err,yerr=I1serrlist, fmt='o', color='black',
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
plt.errorbar(centers,I6slist, xerr=q2err,yerr=I6serrlist, fmt='o', color='black',
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
plt.errorbar(centers,I7list, xerr=q2err,yerr=I7errlist, fmt='o', color='black',
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
###############################################################################################
A9=[-2.770710200760468e-07,9.005313726101284e-05,-0.002292023007510312,-0.0008891203747244368]
A9err=[0.23007018614467328,0.003278694013071172,0.003540773351170616,0.001830044104327186]
RAB=[0.5750000000000002,0.5498856623939414,0.5279901093799426,0.5233637516118604]
RABerr=[0.027146306049800235,0.024446135411145608,0.024026332926459316,0.023705635518962823]
RLT=[1.4230769230769234,0.7565352470404109,0.6610718187936241,0.5415773139523843]
RLTerr=[0.007018536472006611,0.01476522660491182,0.01467550928053812,0.012477272672414658]
AFB=[0.0613636363636364,-0.08294774899473933,-0.14176664036248526,-0.12851523624354616]
AFBerr=[0.007515460818806052,0.007047597798103426,0.007347641798766625,0.007230839653504665]
A6s=[0.9204545454545454,0.8368975113781187,0.9564327059221696,0.8240374618717381]
A6serr=[0.027568215541984145,0.02901886353202652,0.03210553144752514,0.03382299480969891]
A3=[-0.007523688218889593,-0.014081441374827293,-0.015077512132280107,-0.01592164475238609]
A3err=[0.001008805100091376,0.0014390353837524141,0.00149548664660652,0.0015340738433125933]
A4=[0.057874524760689175,0.002223122486426465,0.012674503448645964,0.0015238473137260007]
A4err=[0.002852482135423486,0.006433991351489434,0.006439553072364475,0.006415848576523933]
A8=[1.1082840803041872e-06,0.01024549444710149,0.0039298990456398545,1.1314319372514178e-06]
A8err=[0.9566655216811923,0.005900876171082866,0.00628784442764555,0.007547351453463896]
A5=[-0.3272727272727273,-0.010925445799344222,-1.47395503917178e-06,-1.1058453910035902e-08]
A5err=[0.0014003847030691367,0.007104662485584607,0.002628960781152184,0.0040690466941771366]
A7=[-0.0,-0.005056854041005077,0.005649046047193145,7.081710518362216e-05]
A7err=[0.005831324397571369,0.006875268289496952,0.006948067561474042,0.006954196406471324]

aRAB=[0.5211566414277087,0.5408623342102293,0.5280439503804069,0.5310634251624382]
aRABerr=[0.02184596403276382,0.021958460054919623,0.021637508480991247,0.022214932871534192]
aRLT=[]
aRLTerr=[]
aAFB=[]
aAFBerr=[]
aA6s=[]
aA6serr=[]
aA3=[]
aA3err=[]
aA4=[]
aA4err=[]
aA8=[]
aA8err=[]
aA5=[]
aA5err=[]
aA7=[]
aA7err=[]


from numpy import array
import numpy as np
import uncertainties
from uncertainties import *

corr_matrix1=array([[1.000, -0.982, -0.976 ,-0.980 ,-0.964, -0.960 ,-0.976 ,-0.970 ,-0.978, -0.952 ,-0.966],
                   [-0.982 , 1.000,  0.955 , 0.959,  0.943 , 0.939 , 0.955 , 0.949 , 0.957 , 0.931 , 0.945],
                   [-0.976 , 0.955,  1.000 , 0.952,  0.936 , 0.941 , 0.951 , 0.944 , 0.953 , 0.924 , 0.942],
                   [-0.980 , 0.959,  0.952 , 1.000,  0.940 , 0.944 , 0.954 , 0.949 , 0.961 , 0.932 , 0.952],
                   [-0.964 , 0.943,  0.936 , 0.940,  1.000 , 0.930 , 0.944  ,0.927 , 0.98 , 0.925  ,0.935],
                   [-0.960 , 0.939,  0.941 , 0.944,  0.930 , 1.000 , 0.932  ,0.929 , 0.937,  0.906 , 0.917 ],
                   [-0.976 , 0.955,  0.951 , 0.954,  0.944 , 0.932 , 1.000 , 0.949 , 0.954,  0.919 , 0.938],
                   [-0.970 , 0.949,  0.944 , 0.949,  0.927 , 0.929 , 0.949 , 1.000 , 0.944,  0.928 , 0.934],
                   [-0.978 , 0.957,  0.953 , 0.961,  0.938 , 0.937 , 0.954 , 0.944 , 1.000,  0.930 , 0.938],
                   [-0.952 , 0.931,  0.924 , 0.932,  0.925 , 0.906 , 0.919 , 0.928 , 0.930,  1.000 , 0.913],
                   [-0.966 , 0.945,  0.942 , 0.952,  0.935 , 0.917,  0.938 , 0.934 , 0.938,  0.913 , 1.000]])
                                                                               
I9=bin1['I9']
I8=bin1['I8']
I7=bin1['I7']
I6s=bin1['I6s']
I6c=bin1['I6c']
I5=bin1['I5']
I4=bin1['I4']
I3=bin1['I3']
I2s=bin1['I2s']
I2c=bin1['I2c']
I1s=bin1['I1s']
I1c=bin1['I1c']

(I8,I7,I6s,I6c,I5,I4,I3,I2s,I2c,I1s,I1c)= correlated_values_norm([I8,I7,I6s,I6c,I5,I4,I3,I2s,I2c,I1s,I1c], corr_matrix1)

corr_matrix2=array([
        [1.000 , 0.063 ,-0.158,  0.168 , 0.019 , 0.003 ,-0.053,  0.013 ,-0.016 , 0.212  ,0.266],
[0.063 , 1.000 , 0.146, -0.140 ,-0.021 , 0.001  ,0.040 ,-0.016  ,0.010 ,-0.198 ,-0.247],
[-0.158,  0.146 , 1.000, -0.396 , 0.120,  0.192 , 0.341 ,-0.145 , 0.427 ,-0.708 ,-0.580],
[0.168 ,-0.140 ,-0.396 , 1.000 ,-0.138, -0.161 ,-0.167 ,-0.005 ,-0.209  ,0.113  ,0.044],
[0.019 ,-0.021  ,0.120 ,-0.138 , 1.000, -0.074  ,0.046 ,-0.018 ,-0.001 ,-0.169 ,-0.215],
[0.003 , 0.001  ,0.192 ,-0.161 ,-0.074 , 1.000  ,0.083 ,-0.022 , 0.042 ,-0.264 ,-0.314],
[-0.053 , 0.040 , 0.341, -0.167 , 0.046,  0.083 , 1.000 , 0.066,  0.189 ,-0.504, -0.507],
[0.013 ,-0.016 ,-0.145 ,-0.005 ,-0.018 ,-0.022  ,0.066 , 1.000 ,-0.396  ,0.162 ,-0.125],
[-0.016 , 0.010,  0.427, -0.209, -0.001 , 0.042 , 0.189 ,-0.396 , 1.000 ,-0.624 ,-0.425],
[0.212 ,-0.198, -0.708 , 0.113 ,-0.169 ,-0.264 ,-0.504  ,0.162 ,-0.624  ,1.000  ,0.747],
[0.266 ,-0.247, -0.580 , 0.044 ,-0.215, -0.314, -0.507 ,-0.125, -0.425  ,0.747 , 1.000]                                                                        
])


I9=bin2['I9']
I8=bin2['I8']
I7=bin2['I7']
I6s=bin2['I6s']
I6c=bin2['I6c']
I5=bin2['I5']
I4=bin2['I4']
I3=bin2['I3']
I2s=bin2['I2s']
I2c=bin2['I2c']
I1s=bin2['I1s']
I1c=bin2['I1c']

(I8,I7,I6s,I6c,I5,I4,I3,I2s,I2c,I1s,I1c)= correlated_values_norm([I8,I7,I6s,I6c,I5,I4,I3,I2s,I2c,I1s,I1c], corr_matrix2)



corr_matrix3=array([
        [1.000 , 0.049, -0.195  ,0.152 , 0.056 ,-0.022 ,-0.086 , 0.080 ,-0.083 , 0.246 , 0.280],
[0.049 , 1.000 , 0.222 ,-0.128 ,-0.066  ,0.039 , 0.117 ,-0.083 , 0.115 ,-0.276 ,-0.308],
[-0.195 , 0.222 , 1.000 ,-0.327 , 0.028 , 0.275 , 0.448 ,-0.289 , 0.567 ,-0.801, -0.690],
[0.152 ,-0.128 ,-0.327 , 1.000 ,-0.114 ,-0.146 ,-0.145 ,-0.006 ,-0.156 , 0.073 , 0.008],
[0.056 ,-0.066  ,0.028 ,-0.114 , 1.000 ,-0.134 ,-0.010 ,-0.043 ,-0.061 ,-0.036 ,-0.077],
[-0.022 , 0.039  ,0.275 ,-0.146, -0.134,  1.000 , 0.140 ,-0.090 , 0.154 ,-0.341 ,-0.372],
[-0.086 , 0.117  ,0.448, -0.145, -0.010,  0.140 , 1.000 ,-0.071  ,0.321 ,-0.583 ,-0.576],
[0.080 ,-0.083 ,-0.289, -0.006, -0.043 ,-0.090 ,-0.071  ,1.000 ,-0.489 , 0.336 , 0.082],
[-0.083 , 0.115 , 0.567, -0.156, -0.061,  0.154 , 0.321 ,-0.489 , 1.000, -0.728 ,-0.576],
[0.246 ,-0.276 ,-0.801 , 0.073 ,-0.036, -0.341 ,-0.583 , 0.336 ,-0.728 , 1.000  ,0.815],
[0.280 ,-0.308, -0.690 , 0.008 ,-0.077, -0.372 ,-0.576,  0.082 ,-0.576 , 0.815  ,1.000]                                                                        
])
I9=bin3['I9']  
I8=bin3['I8']
I7=bin3['I7']
I6s=bin3['I6s']
I6c=bin3['I6c']
I5=bin3['I5']
I4=bin3['I4']
I3=bin3['I3']
I2s=bin3['I2s']
I2c=bin3['I2c']
I1s=bin3['I1s']
I1c=bin3['I1c']

(I8,I7,I6s,I6c,I5,I4,I3,I2s,I2c,I1s,I1c)= correlated_values_norm([I8,I7,I6s,I6c,I5,I4,I3,I2s,I2c,I1s,I1c], corr_matrix3)

corr_matrix4=array([
        [1.000 ,-0.101  ,0.126 ,-0.139 ,-0.002, -0.048  ,0.016 ,-0.096  ,0.039, -0.192 ,-0.225],
[-0.101 , 1.000 , 0.130 ,-0.122 ,-0.001 ,-0.034 , 0.049 ,-0.080 , 0.055, -0.207, -0.236],
[0.126  ,0.130  ,1.000 ,-0.347  ,0.005 , 0.169  ,0.331 ,-0.262 , 0.456 ,-0.681 ,-0.567],
[-0.139 ,-0.122 ,-0.347 , 1.000 ,-0.005 ,-0.147 ,-0.123 ,-0.067, -0.085 , 0.005 ,-0.021],
[-0.002 ,-0.001 , 0.005 ,-0.005 , 1.000 ,-0.005 , 0.002 ,-0.003 , 0.002 ,-0.008 ,-0.009],
[-0.048 ,-0.034 , 0.169 ,-0.147 ,-0.005 , 1.000 , 0.054 ,-0.105 , 0.079 ,-0.248 ,-0.285],
[0.016  ,0.049 , 0.331 ,-0.123  ,0.002 , 0.054 , 1.000 ,-0.092 , 0.265 ,-0.525 ,-0.499],
[-0.096 ,-0.080, -0.262, -0.067 ,-0.003 ,-0.105, -0.092 , 1.000, -0.518 , 0.378 , 0.076],
[0.039  ,0.055 , 0.456 ,-0.085  ,0.002  ,0.079 , 0.265 ,-0.518 , 1.000, -0.718 ,-0.533],
[-0.192 ,-0.207, -0.681 , 0.005 ,-0.008 ,-0.248, -0.525 , 0.378, -0.718 , 1.000,  0.739],
[-0.225, -0.236, -0.567 ,-0.021 ,-0.009, -0.285, -0.499  ,0.076, -0.533 , 0.739,  1.000]
])





I9=bin4['I9']
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
a9=(1/(2*np.pi))*(1-I8-I7-I5-I4-I3-I2s-I1s-I1c-I2c-I6s-I6c)/Gammaq

""""""

def power(x,c,d,e):
  res=c*x**2+d*x+e
  return res

def linear(x,d,e):
  res=d*x+e
  return res
centers=centers_th
q2_borders=[ 3.20305994, 6.2,  7.6,  8.9, 10.686075  ]
q2err=[(q2_borders[1]-q2_borders[0])/2.,(q2_borders[2]-q2_borders[1])/2.,(q2_borders[3]-q2_borders[2])/2.,(q2_borders[4]-q2_borders[3])/2.]
plt.errorbar(centers,A6s, xerr=q2err,yerr=A6serr, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$A_{6s}$ - RapidSim')
sol,_=curve_fit(power, centers, A6s, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.ylim(-1.5,1.5)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{6s}$ ($q^2$)')
plt.title(r'$A_{6s}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('A6s.pdf')
plt.close()
plt.close()

plt.errorbar(centers,A4, xerr=q2err,yerr=A4err, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$A_{4}$ - RapidSim')
sol,_=curve_fit(power, centers, A4, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.ylim(-0.3,0.2)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{4}$ ($q^2$)')
plt.title(r'$A_{4}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('A4.pdf')
plt.close()
plt.close()


plt.errorbar(centers,A5, xerr=q2err,yerr=A5err, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$A_{5}$ - RapidSim')
sol,_=curve_fit(power, centers, A5, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.ylim(-0.3,0.2)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{5}$ ($q^2$)')
plt.title(r'$A_{5}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('A5.pdf')
plt.close()
plt.close()
""""""
plt.errorbar(centers,A7, xerr=q2err,yerr=A7err, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$A_{7}$ - RapidSim')
sol,_=curve_fit(linear, centers, A7, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),linear(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1]),color='#3F7F4C')
plt.ylim(-0.04,0.02)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{7}$ ($q^2$)')
plt.title(r'$A_{7}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('A7.pdf')
plt.close()
plt.close()

plt.errorbar(centers,A8, xerr=q2err,yerr=A8err, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$A_{8}$ - RapidSim')
sol,_=curve_fit(linear, centers, A8, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),linear(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1]),color='#3F7F4C')
plt.ylim(-0.020,0.010)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{8}$ ($q^2$)')
plt.title(r'$A_{8}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('A8.pdf')
plt.close()
plt.close()


