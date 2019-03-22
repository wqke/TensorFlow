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
  
  
bin1={'I9': (0.0, 0.006188415204872955), 'I8': (9.536740132598531e-07, 0.5438200632855533), 'I6c': (0.3896425388652342, 0.0038674757743067945), 'I3': (-0.026289731656599646, 0.003656761087225069), 'I2s': (0.020051490246559167, 0.0030429304478095665), 'I5': (0.24210317853252794, 0.0023014671269239173), 'I4': (-0.04881671452619074, 0.002248899349584388), 'I7': (0.0, 0.004587773389557848), 'loglh': 1733.8016718556105, 'I1s': (0.1807109614813348, 0.0020335168543838267), 'iterations': 287, 'I6s': (-0.15199524705416378, 0.004013015795113373), 'I2c': (-0.07228438459253395, 0.004299322306980102), 'I1c': (0.46687790870383206, 0.0026344036531195936)}
bin2={'I9': (0.0, 0.006033113786298561),'I8': (0.015469593458582676, 0.009353896805830347), 'I6c': (0.2528298835033377, 0.014246896308276358), 'I3': (-0.08400793263041062, 0.009959389203668512), 'I2s': (0.015916306074060582, 0.00835407564044699), 'I5': (0.012574105379506761, 0.00850568676092403), 'I4': (-0.0044116311630206795, 0.009673345420913804), 'I7': (0.008130543440409532, 0.0086629509233081), 'loglh': -599.5124257302173, 'I1s': (0.44436851326101595, 0.016330254708434605), 'iterations': 351, 'I6s': (-0.23362930197825382, 0.011969251657062974), 'I2c': (-0.07029164345924177, 0.016842072516979945), 'I1c': (0.6418315194043966, 0.01960357926079148)}
bin3={'I9': (0.0, 0.006029437461817111), 'I8': (0.007747071815710882, 0.012474831892402477), 'I6c': (0.23685281203754244, 0.018382557252344944), 'I3': (-0.11889014749182492, 0.014032499433078272), 'I2s': (0.08322731871654776, 0.011620931696731618), 'I5': (2.466375988718905e-06, 0.004399048074584494), 'I4': (-0.024985448049632653, 0.013028163951659066), 'I7': (-0.00945257566186919, 0.011734362522390407), 'loglh': -614.0437246707288, 'I1s': (0.6439461808662359, 0.028226830057596575), 'iterations': 380, 'I6s': (-0.35564520068963945, 0.019405707959718188), 'I2c': (-0.19457934740463434, 0.02479542704826082), 'I1c': (0.7498500735411797, 0.02880616896423821)}
bin4={'I9': (-0.008914581413189238,0.015779316801162374),'I8': (2.3961981843179103e-06, 0.015984107839973644), 'I6c': (0.1963221054502799, 0.01886868745886225), 'I3': (-0.13487834306570268, 0.01490964891765989), 'I2s': (0.1381330776741656, 0.013438732840722722), 'I5': (1.9879596857563797e-08, 0.007314856892917221), 'I4': (-0.0032272733746568916, 0.013617881904896545), 'I7': (-0.0001273067205527223, 0.012502437736883087), 'loglh': -459.64429312436806, 'I1s': (0.7645626191957862, 0.028509447266217514), 'iterations': 254, 'I6s': (-0.32919073439317836, 0.018303607596528382), 'I2c': (-0.3017481454229177, 0.02581260050730033), 'I1c': (0.6776836633807674, 0.024275327394433888)} 
      
      
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
###############################################################################################
A9=[]
A9err=[]

RAB=[0.5376342560451569,0.5489983070056357,0.5259999434732144,0.5189182741407713]
RABerr=[0.02477864266333598,0.024526186528407413,0.01928099358261573,0.024237437398358902]
RLT=[0.9921261971131907,0.7575748629325549,0.6855078437442366,0.5380690782463855]
RLTerr=[0.02151727582499289,0.016340002890613572,0.01259771607193771,0.012541504368628017]
AFB=[0.024421017666857586,-0.0834127964466983,-0.12335433492743794,-0.12641454737537994]
AFBerr=[0.007356483361068994,0.007344379000407359,0.0062273583078599265,0.007514560871676504]
A6s=[0.5252215227836435,0.8324901322667959,0.9274668801632266,0.8154937899028142]
A6serr=[0.029875425841474654,0.030985413948756223,0.026021570791518324,0.03398140037087664]
A3=[-0.013761120880265671,-0.013920125791890853,-0.015593772761077455,-0.01633012772877743]
A3err=[0.0014502782072193677,0.0014853252925878864,0.0012455549355783137,0.0015718396163206152]
A4=[-0.015323972141456374,0.0015415831648845845,0.010403039726970405,-0.0008102852697284153]
A4err=[0.006465089322778664,0.006498980171041198,0.00536175545863742,0.006535165297547875]
A8=[1.595052616792824e-09,0.009352553918764635,0.0016708351415484562,9.675397703986898e-06]
A8err=[0.00443039275172179,0.00639895201055472,0.00631459872506381,0.004261446065604159]
A5=[-0.0016548644194183467,-0.007640513815224158,-0.0019045868079891296,-9.792241437632154e-09]
A5err=[0.006879046214155649,0.006828511614484466,0.0056601881878883395,0.004224773376029027]
A7=[0.009098788445593827,-0.005056854041005077,0.0017783062585460241,0.0021902123745440333]
A7err=[0.006925153010949783,0.006926541086779652,0.005807251587065241,0.007054754311084916]

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
1.000, -0.982, -0.976 ,-0.980 ,-0.964, -0.960 ,-0.976 ,-0.970 ,-0.978, -0.952 ,-0.966
-0.982 , 1.000,  0.955 , 0.959,  0.943 , 0.939 , 0.955 , 0.949 , 0.957 , 0.931 , 0.945
-0.976 , 0.955,  1.000 , 0.952,  0.936 , 0.941 , 0.951 , 0.944 , 0.953 , 0.924 , 0.942
-0.980 , 0.959,  0.952 , 1.000,  0.940 , 0.944 , 0.954 , 0.949 , 0.961 , 0.932 , 0.952
-0.964 , 0.943,  0.936 , 0.940,  1.000 , 0.930 , 0.944  ,0.927 , 0.98 , 0.925  ,0.935
-0.960 , 0.939,  0.941 , 0.944,  0.930 , 1.000 , 0.932  ,0.929 , 0.937,  0.906 , 0.917
-0.976 , 0.955,  0.951 , 0.954,  0.944 , 0.932 , 1.000 , 0.949 , 0.954,  0.919 , 0.938
-0.970 , 0.949,  0.944 , 0.949,  0.927 , 0.929 , 0.949 , 1.000 , 0.944,  0.928 , 0.934
-0.978 , 0.957,  0.953 , 0.961,  0.938 , 0.937 , 0.954 , 0.944 , 1.000,  0.930 , 0.938
-0.952 , 0.931,  0.924 , 0.932,  0.925 , 0.906 , 0.919 , 0.928 , 0.930,  1.000 , 0.913
-0.966 , 0.945,  0.942 , 0.952,  0.935 , 0.917,  0.938 , 0.934 , 0.938,  0.913 , 1.000

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
        [  1.000, -0.057,  0.160, -0.172, -0.015, -0.003,  0.056 ,-0.016,  0.020 ,-0.217, -0.271,],
        [ -0.057,  1.000,  0.149 ,-0.143, -0.018 , 0.001,  0.044 ,-0.017,  0.016, -0.204, -0.253],
        [  0.160,  0.149 , 1.000 ,-0.401 , 0.129 , 0.185,  0.338 ,-0.147 , 0.427, -0.705 ,-0.580],
        [ -0.172, -0.143 ,-0.401 , 1.000 ,-0.140 ,-0.166, -0.171 ,-0.009 ,-0.207,  0.118 , 0.055],
        [ -0.015, -0.018 , 0.129 ,-0.140 , 1.000 ,-0.068,  0.049 ,-0.020 , 0.009, -0.181 ,-0.227],
        [ -0.003 , 0.001 , 0.185 ,-0.166 ,-0.068 , 1.000 , 0.079 ,-0.025 , 0.039, -0.257 ,-0.307],
        [0.056 , 0.044 , 0.338 ,-0.171 , 0.049 , 0.079 , 1.000 , 0.060 , 0.191, -0.501 ,-0.505],
        [-0.016 ,-0.017 ,-0.147 ,-0.009 ,-0.020 ,-0.025  ,0.060 , 1.000 ,-0.400,  0.172 ,-0.117],
        [0.020 , 0.016 , 0.427 ,-0.207 , 0.009 , 0.039  ,0.191 ,-0.400 , 1.000, -0.629 ,-0.432],
       [-0.217 ,-0.204 ,-0.705 , 0.118 ,-0.181 ,-0.257 ,-0.501 , 0.172 ,-0.629,  1.000 , 0.747],
       [-0.271 ,-0.253 ,-0.580,  0.055 ,-0.227, -0.307 ,-0.505, -0.117 ,-0.432,  0.747 , 1.000]                                                                         
])
1.000 , 0.063 ,-0.158,  0.168 , 0.019 , 0.003 ,-0.053,  0.013 ,-0.016 , 0.212  ,0.266
0.063 , 1.000 , 0.146, -0.140 ,-0.021 , 0.001  ,0.040 ,-0.016  ,0.010 ,-0.198 ,-0.247
-0.158,  0.146 , 1.000, -0.396 , 0.120,  0.192 , 0.341 ,-0.145 , 0.427 ,-0.708 ,-0.580
0.168 ,-0.140 ,-0.396 , 1.000 ,-0.138, -0.161 ,-0.167 ,-0.005 ,-0.209  ,0.113  ,0.044
0.019 ,-0.021  ,0.120 ,-0.138 , 1.000, -0.074  ,0.046 ,-0.018 ,-0.001 ,-0.169 ,-0.215
0.003 , 0.001  ,0.192 ,-0.161 ,-0.074 , 1.000  ,0.083 ,-0.022 , 0.042 ,-0.264 ,-0.314
-0.053 , 0.040 , 0.341, -0.167 , 0.046,  0.083 , 1.000 , 0.066,  0.189 ,-0.504, -0.507
0.013 ,-0.016 ,-0.145 ,-0.005 ,-0.018 ,-0.022  ,0.066 , 1.000 ,-0.396  ,0.162 ,-0.125
-0.016 , 0.010,  0.427, -0.209, -0.001 , 0.042 , 0.189 ,-0.396 , 1.000 ,-0.624 ,-0.425
0.212 ,-0.198, -0.708 , 0.113 ,-0.169 ,-0.264 ,-0.504  ,0.162 ,-0.624  ,1.000  ,0.747
0.266 ,-0.247, -0.580 , 0.044 ,-0.215, -0.314, -0.507 ,-0.125, -0.425  ,0.747 , 1.000


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
        [  1.000, -0.055,  0.200, -0.158, -0.000,  0.017,  0.085, -0.085 , 0.081, -0.252, -0.289],
        [  -0.055,  1.000 , 0.216 ,-0.136 ,-0.000 , 0.027 , 0.107 ,-0.086 , 0.102 ,-0.269, -0.304],
        [ 0.200 , 0.216 , 1.000 ,-0.326 , 0.000 , 0.281 , 0.442 ,-0.291 , 0.566 ,-0.799, -0.686],
        [ -0.158 ,-0.136 ,-0.326 , 1.000 ,-0.000 ,-0.163 ,-0.146 ,-0.011 ,-0.163 , 0.068, -0.003],
        [-0.000 ,-0.000 , 0.000 ,-0.000 , 1.000 ,-0.000 , 0.000 ,-0.000 , 0.000 ,-0.001, -0.001],
        [0.017 , 0.027 , 0.281 ,-0.163 ,-0.000 , 1.000 , 0.138 ,-0.099 , 0.147 ,-0.351, -0.388],
        [  0.085 , 0.107 , 0.442 ,-0.146 , 0.000 , 0.138 , 1.000 ,-0.074 , 0.313 ,-0.576, -0.572],
        [-0.085 ,-0.086 ,-0.291 ,-0.011 ,-0.000 ,-0.099 ,-0.074 , 1.000 ,-0.496 , 0.339,  0.080],
        [ 0.081 , 0.102 , 0.566 ,-0.163 , 0.000  ,0.147 , 0.313 ,-0.496 , 1.000 ,-0.730, -0.576],
       [-0.252 ,-0.269 ,-0.799 , 0.068 ,-0.001 ,-0.351 ,-0.576 , 0.339 ,-0.730 , 1.000,  0.811],
       [ -0.289 ,-0.304 ,-0.686, -0.003 ,-0.001, -0.388 ,-0.572,  0.080 ,-0.576,  0.811,  1.000]
])
                                                                                                                                                                  
1.000 , 0.049, -0.195  ,0.152 , 0.056 ,-0.022 ,-0.086 , 0.080 ,-0.083 , 0.246 , 0.280
0.049 , 1.000 , 0.222 ,-0.128 ,-0.066  ,0.039 , 0.117 ,-0.083 , 0.115 ,-0.276 ,-0.308
-0.195 , 0.222 , 1.000 ,-0.327 , 0.028 , 0.275 , 0.448 ,-0.289 , 0.567 ,-0.801, -0.690
0.152 ,-0.128 ,-0.327 , 1.000 ,-0.114 ,-0.146 ,-0.145 ,-0.006 ,-0.156 , 0.073 , 0.008
0.056 ,-0.066  ,0.028 ,-0.114 , 1.000 ,-0.134 ,-0.010 ,-0.043 ,-0.061 ,-0.036 ,-0.077
-0.022 , 0.039  ,0.275 ,-0.146, -0.134,  1.000 , 0.140 ,-0.090 , 0.154 ,-0.341 ,-0.372
-0.086 , 0.117  ,0.448, -0.145, -0.010,  0.140 , 1.000 ,-0.071  ,0.321 ,-0.583 ,-0.576
0.080 ,-0.083 ,-0.289, -0.006, -0.043 ,-0.090 ,-0.071  ,1.000 ,-0.489 , 0.336 , 0.082
-0.083 , 0.115 , 0.567, -0.156, -0.061,  0.154 , 0.321 ,-0.489 , 1.000, -0.728 ,-0.576
0.246 ,-0.276 ,-0.801 , 0.073 ,-0.036, -0.341 ,-0.583 , 0.336 ,-0.728 , 1.000  ,0.815
0.280 ,-0.308 -0.690 , 0.008 ,-0.077, -0.372 ,-0.576,  0.082 ,-0.576 , 0.815  ,1.000

    
    
    
    
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
        [ 1.000, -0.074,  0.116 ,-0.122 ,-0.000, -0.036,  0.029, -0.097,  0.049, -0.163, -0.191],
        [-0.074,  1.000 , 0.203, -0.121 , 0.000 , 0.001,  0.123 ,-0.118 , 0.138, -0.286 ,-0.310],
        [0.116,  0.203 , 1.000, -0.307 , 0.001 , 0.225,  0.440 ,-0.316 , 0.555, -0.748 ,-0.652],
        [-0.122, -0.121 ,-0.307 , 1.000 ,-0.001 ,-0.144, -0.110 ,-0.068 ,-0.073, -0.003 ,-0.027],
        [  -0.000,  0.000 , 0.001 ,-0.001 , 1.000 ,-0.001,  0.001 ,-0.001 , 0.001, -0.002 ,-0.002],
        [    -0.036 , 0.001 , 0.225 ,-0.144 ,-0.001 , 1.000,  0.117 ,-0.138 , 0.147, -0.303 ,-0.334],
        [  0.029 , 0.123 , 0.440 ,-0.110 , 0.001 , 0.117,  1.000 ,-0.158 , 0.388, -0.620 ,-0.592],
        [ -0.097 ,-0.118 ,-0.316 ,-0.068 ,-0.001 ,-0.138, -0.158 , 1.000, -0.539,  0.416 , 0.154],
        [0.049 , 0.138 , 0.555 ,-0.073,  0.001 , 0.147,  0.388 ,-0.539,  1.000, -0.779 ,-0.628],
       [-0.163 ,-0.286 ,-0.748 ,-0.003 ,-0.002 ,-0.303, -0.620 , 0.416 ,-0.779,  1.000 , 0.800],
       [-0.191 ,-0.310, -0.652 ,-0.027, -0.002 ,-0.334, -0.592 , 0.154, -0.628 , 0.800 , 1.000]
])

1.000 ,-0.101  ,0.126 ,-0.139 ,-0.002, -0.048  ,0.016 ,-0.096  ,0.039, -0.192 ,-0.225
-0.101 , 1.000 , 0.130 ,-0.122 ,-0.001 ,-0.034 , 0.049 ,-0.080 , 0.055, -0.207, -0.236
0.126  ,0.130  ,1.000 ,-0.347  ,0.005 , 0.169  ,0.331 ,-0.262 , 0.456 ,-0.681 ,-0.567
-0.139 ,-0.122 ,-0.347 , 1.000 ,-0.005 ,-0.147 ,-0.123 ,-0.067, -0.085 , 0.005 ,-0.021
-0.002 ,-0.001 , 0.005 ,-0.005 , 1.000 ,-0.005 , 0.002 ,-0.003 , 0.002 ,-0.008 ,-0.009
-0.048 ,-0.034 , 0.169 ,-0.147 ,-0.005 , 1.000 , 0.054 ,-0.105 , 0.079 ,-0.248 ,-0.285
0.016  ,0.049 , 0.331 ,-0.123  ,0.002 , 0.054 , 1.000 ,-0.092 , 0.265 ,-0.525 ,-0.499
-0.096 ,-0.080, -0.262, -0.067 ,-0.003 ,-0.105, -0.092 , 1.000, -0.518 , 0.378 , 0.076
0.039  ,0.055 , 0.456 ,-0.085  ,0.002  ,0.079 , 0.265 ,-0.518 , 1.000, -0.718 ,-0.533
-0.192 ,-0.207, -0.681 , 0.005 ,-0.008 ,-0.248, -0.525 , 0.378, -0.718 , 1.000,  0.739
-0.225, -0.236, -0.567 ,-0.021 ,-0.009, -0.285, -0.499  ,0.076, -0.533 , 0.739,  1.000




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


