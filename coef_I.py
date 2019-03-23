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
  
  
bin1={'I9': (0.0011171225693240583,0.007197024682284227),  'I8': (-0.0040703372732217025, 0.007381693433498704), 'I6c': (0.294963766108564, 0.01135899698545928), 'I3': (-0.06670419913636494, 0.007522035174590935), 'I2s': (0.0023552240825619464, 0.0065378960336043845), 'I5': (0.0021395102851726744, 0.007385952480383762), 'I4': (0.01769416647366362, 0.007658459481277746), 'I7': (-0.008611696128999258, 0.007061008386601508), 'loglh': -801.1699601885375, 'I1s': (0.3088228234786152, 0.00885532351644236), 'iterations': 244, 'I6s': (-0.12190383759510115, 0.007650148643079613), 'I2c': (-0.02776892085360294, 0.012089551156116929), 'I1c': (0.6019663779893887, 0.014687525867038587)}

bin2={'I9': (0.03330761528749626,0.010064341963518865),  'I8': (-0.0022477412667292196, 0.009959010941589297), 'I6c': (0.29562277087996325, 0.013968330244940375), 'I3': (-0.016079451971054093, 0.00989920418247664), 'I2s': (-0.035510846520155126, 0.009846247788684759), 'I5': (0.06331844716731583, 0.00978844339525986), 'I4': (-0.02528369241917372, 0.011082138570424005), 'I7': (-0.008023399046132584, 0.009623122887842117), 'loglh': -465.35452292954426, 'I1s': (0.4961972473448717, 0.02149844117213151), 'iterations': 216, 'I6s': (-0.26783783925446425, 0.014580242330472504), 'I2c': (-0.1602997507398719, 0.021391792690733935), 'I1c': (0.6268366405379338, 0.019278223692487106)}

bin3={'I9': (-0.07995659441254155,0.033801389786184674),'I8': (0.006268878375645048, 0.014432379150508767), 'I6c': (0.28135284273829453, 0.02159215010531934), 'I3': (-0.03425715835990617, 0.01285475030460631), 'I2s': (-0.0019358996502600867, 0.018417123959542647), 'I5': (0.07719405980971272, 0.016581566051022567), 'I4': (-0.05279815034224278, 0.022527117312639122), 'I7': (0.003974616479737847, 0.01346659475356704), 'loglh': -497.01381592594055, 'I1s': (0.6828180639464891, 0.08917696976495904), 'iterations': 197, 'I6s': (-0.3662293858565322, 0.04515064999836915), 'I2c': (-0.2385930201305999, 0.05635731552140727), 'I1c': (0.7221617474022035, 0.06591609322770897)}

bin4={'I9': (-0.13990995829070585,1.234941750882434), 'I8': (0.006010420479328316, 0.014415919419093925), 'I6c': (0.22843024589260352, 0.021304609393842266), 'I3': (-0.1301435527701782, 0.01757932112211885), 'I2s': (0.059712096194822095, 0.012839480264098091), 'I5': (0.03155650210034633, 0.013151451116144242), 'I4': (-0.09082117200809314, 0.0174951960155445), 'I7': (0.00337594213382042, 0.01369403073798714), 'loglh': -379.31122565963847, 'I1s': (0.9745509260618408, 1.392646129395374), 'iterations': 190, 'I6s': (-0.42674889777498815, 0.02309417176790096), 'I2c': (-0.3813632780387739, 0.02764069384424095), 'I1c': (0.8653507260199778, 0.0036022698445408174)}

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
A9=[0.0002317340293486221,0.005267362259431925,-0.009910139262224437,-0.012954389084357577]
A9err=[0.0014944095811517013,0.0016347935632083314,0.0031498058250564514,0.10378963043112376]
RAB=[0.538541480479437,0.8333378896416951,0.7627870547927599,0.7052401183534124]
RABerr=[0.024644441621379798,0.033949173682306245,0.07169022117756123,0.1786384184072577]
RLT=[0.9921230228317532,0.6695119107070312,0.5864928513651227,0.5198109504888787]
RLTerr=[0.020732790881539148,0.016666172175935846,0.023611901200809863,0.7550918250871287]
AFB=[0.02500330817648041,-0.08944740688293581,-0.1317392360732825,-0.13636609338305275]
AFBerr=[0.007394237997152475,0.008082021552369707,0.008956041419178568,0.09650858312955601]
A6s=[0.5362409926386125,0.898204499198272,0.9625699592771888,0.8379041924236486]
A6serr=[0.029718061222135942,0.03524386484837346,0.030954258937450112,0.6376543124081426]
A3=[-0.013837007025733745,-0.0025428508685962116,-0.004245968860593382,-0.01205010879856367]
A3err=[0.0014417214906901387,0.0015240504046134602,0.001601337474458647,0.008215819877476249]
A4=[-0.014681792689546608,0.015993743902504565,0.026176053471202707,0.033636856551571434]
A4err=[0.006448503500574689,0.006876989977582726,0.008772037525584028,0.02103704713804849]
A8=[-0.0033773757080291585,-0.001421857123680877,0.003107959170949637,0.0022260409881054836]
A8err=[0.006102673868961059,0.006287568482009569,0.007363614736004823,0.007125743120467959]
A5=[-0.0020914356060666352,-0.04718685528878954,-0.04508691114674136,-0.013768869987306564]
A5err=[0.007228745741470095,0.007838015275448034,0.007191302535961949,0.016873182525549004]
A7=[0.008418191787921999,0.005979283868311576,-0.002321463341947025,-0.0014730057272327884]
A7err=[0.0068394505319516845,0.007124974486362972,0.008010029824092007,0.007155202276387396]

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

corr_matrix1=array([[1.000 , 0.033 , 0.106, -0.156 ,-0.026, -0.012  ,0.044 , 0.039 ,-0.042 ,-0.223 ,-0.304],
      [0.033 , 1.000 , 0.136 ,-0.209 , 0.016, -0.022  ,0.089 , 0.045 , 0.019 ,-0.263 ,-0.329],
      [0.106 , 0.136  ,1.000 ,-0.434 , 0.055,  0.106  ,0.185, -0.109 , 0.213 ,-0.513 ,-0.371],
      [-0.156 ,-0.209 ,-0.434 , 1.000 ,-0.148, -0.217 ,-0.192, -0.062 ,-0.162 , 0.129 , 0.093],
      [-0.026 , 0.016  ,0.055, -0.148 , 1.000, -0.014 , 0.020 , 0.048 ,-0.131 ,-0.137, -0.211],
      [-0.012 ,-0.022 , 0.106, -0.217, -0.014,  1.000 , 0.032 , 0.012, -0.032 ,-0.178, -0.262],
      [0.044 , 0.089  ,0.185, -0.192 , 0.020 , 0.032 , 1.000 , 0.153 , 0.049 ,-0.404 ,-0.437],
      [0.039 , 0.045 ,-0.109, -0.062 , 0.048,  0.012 , 0.153 , 1.000 ,-0.319 , 0.051 ,-0.246],
      [-0.042 , 0.019 , 0.213, -0.162, -0.131, -0.032,  0.049, -0.319,  1.000 ,-0.484 ,-0.310],
      [-0.223 ,-0.263 ,-0.513,  0.129, -0.137 ,-0.178, -0.404 , 0.051, -0.484 , 1.000 , 0.652],
      [-0.304 ,-0.329 ,-0.371,  0.093, -0.211 ,-0.262, -0.437 ,-0.246, -0.310  ,0.652,  1.000]
])


                                                                            
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
1-I8-I7-I5-I4-I3-I2s-I2c-I1c-I1s-I6c-I6s
corr_matrix2=array([
       [1.000 ,-0.053  ,0.156, -0.180 ,-0.023,  0.003  ,0.051 , 0.004  ,0.018, -0.225 ,-0.267],
[-0.053 , 1.000 , 0.150, -0.172 ,-0.010, -0.009 , 0.041 , 0.006 , 0.024 ,-0.211 ,-0.256],
[0.156  ,0.150  ,1.000 ,-0.407  ,0.116 , 0.202  ,0.297 ,-0.134  ,0.415 ,-0.713 ,-0.564],
[-0.180 ,-0.172 ,-0.407 , 1.000 ,-0.140 ,-0.165 ,-0.158 ,-0.008 ,-0.220 , 0.147 , 0.073],
[-0.023 ,-0.010 , 0.116 ,-0.140 , 1.000 ,-0.061 ,-0.024 ,-0.028 , 0.047 ,-0.188 ,-0.229],
[0.003 ,-0.009  ,0.202 ,-0.165 ,-0.061  ,1.000  ,0.025 ,-0.029 , 0.023 ,-0.256 ,-0.293],
[0.051 , 0.041  ,0.297 ,-0.158 ,-0.024  ,0.025  ,1.000  ,0.061 , 0.167 ,-0.442 ,-0.477],
[0.004 , 0.006 ,-0.134 ,-0.008 ,-0.028 ,-0.029  ,0.061  ,1.000 ,-0.360 , 0.098 ,-0.121],
[0.018 , 0.024  ,0.415 ,-0.220  ,0.047 , 0.023  ,0.167 ,-0.360 , 1.000 ,-0.622 ,-0.441],
[-0.225, -0.211 ,-0.713 , 0.147 ,-0.188 ,-0.256, -0.442 , 0.098, -0.622 , 1.000,  0.745],
[-0.267, -0.256 ,-0.564  ,0.073 ,-0.229 ,-0.293, -0.477 ,-0.121, -0.441  ,0.745,  1.000]                                                                   
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
[1.000  ,0.209  ,0.531, -0.423 ,-0.424,  0.407  ,0.002,  0.411  ,0.496, -0.549 ,-0.557],
[0.209  ,1.000  ,0.496 ,-0.400 ,-0.423 , 0.388 ,-0.006 , 0.383  ,0.459 ,-0.513 ,-0.523],
[0.531  ,0.496  ,1.000 ,-0.671 ,-0.661 , 0.805  ,0.133 , 0.707  ,0.920 ,-0.967 ,-0.947],
[-0.423 ,-0.400 ,-0.671,  1.000 , 0.313 ,-0.544 ,-0.220 ,-0.484 ,-0.583,  0.598 , 0.592],
[-0.424 ,-0.423 ,-0.661 , 0.313 , 1.000 ,-0.621 ,-0.107 ,-0.548 ,-0.688,  0.682 , 0.664],
[0.407  ,0.388  ,0.805 ,-0.544 ,-0.621  ,1.000  ,0.054  ,0.631  ,0.770 ,-0.828 ,-0.832],
[0.002 ,-0.006  ,0.133 ,-0.220 ,-0.107  ,0.054  ,1.000  ,0.086  ,0.086 ,-0.137 ,-0.159],
[0.411 , 0.383  ,0.707 ,-0.484 ,-0.548  ,0.631  ,0.086  ,1.000  ,0.652 ,-0.749 ,-0.781],
[0.496 , 0.459  ,0.920 ,-0.583 ,-0.688  ,0.770  ,0.086  ,0.652  ,1.000 ,-0.955 ,-0.933],
[-0.549 ,-0.513 ,-0.967 , 0.598 , 0.682 ,-0.828 ,-0.137 ,-0.749, -0.955 , 1.000 , 0.973],
[-0.557 ,-0.523, -0.947  ,0.592,  0.664 ,-0.832, -0.159 ,-0.781, -0.933  ,0.973 , 1.000]                                                                    
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
 [1.000,  0.960  ,0.974,  0.968  ,0.958,  0.966 , 0.966,  0.961  ,0.973, -0.980 ,-0.373],
 [0.960 , 1.000  ,0.973 , 0.968  ,0.958 , 0.966 , 0.966 , 0.961  ,0.973 ,-0.980 ,-0.373],
 [0.974 , 0.973  ,1.000 , 0.981  ,0.971 , 0.980 , 0.980 , 0.975  ,0.987 ,-0.994 ,-0.376],
 [0.968 , 0.968  ,0.981 , 1.000  ,0.966 , 0.974 , 0.974 , 0.969  ,0.981 ,-0.988 ,-0.376],
 [0.958 , 0.958  ,0.971 , 0.966  ,1.000 , 0.964 , 0.964 , 0.959  ,0.971 ,-0.978 ,-0.372],
 [0.966 , 0.966  ,0.980 , 0.974 , 0.964 , 1.000 , 0.973 , 0.968 , 0.980 ,-0.987 ,-0.375],
 [0.966 , 0.966  ,0.980 , 0.974 , 0.964 , 0.973 , 1.000 , 0.968 , 0.980 ,-0.987 ,-0.375],
 [0.961 , 0.961  ,0.975 , 0.969 , 0.959 , 0.968 , 0.968 , 1.000 , 0.975 ,-0.982 ,-0.375],
 [0.973 , 0.973  ,0.987 , 0.981 , 0.971 , 0.980 , 0.980  ,0.975 , 1.000 ,-0.993 ,-0.377],
 [-0.980, -0.980 ,-0.994 ,-0.988, -0.978, -0.987 ,-0.987 ,-0.982, -0.993 , 1.000,  0.377],
 [-0.373, -0.373, -0.376 ,-0.376, -0.372, -0.375, -0.375 ,-0.375, -0.377 , 0.377,  1.000]

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


