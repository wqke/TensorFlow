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
  
  
bin1={'I5':(0.2527342527078308,0.0012922618709802017),'I9': (6.815345142663354e-05, 0.0007363780329552316), 'I8': (-0.002386721459485708, 0.0008657955474773993), 'I6c': (0.38509936478046214, 0.0011481002936122309), 'I3': (-0.033882301770855494, 0.0007783495582233124), 'I2s': (0.02514509165410428, 0.0006945845529782169), 'I4': (-0.060431918916672656, 0.0008630068774557742), 'I7': (-0.002128790056209695, 0.0008239562981513515), 'loglh': -123591.21654238443, 'status': 3, 'I1s': (0.2140153332578847, 0.0009178869644652643), 'iterations': 390, 'I6s': (-0.1747480244741083, 0.0009727949072756981), 'I2c': (-0.07798719505525042, 0.0014891665287158906), 'I1c': (0.47450275588087365, 0.0014065402655303316)}

bin2={'I5':(0.3177376228829665,0.001998266655916547),'I9': (-0.013784997395315801, 0.001184959184759471), 'I8': (0.007829605172332776, 0.0012839893472408326), 'I6c': (0.3678928131752901, 0.001758238988066485), 'I3': (-0.10175300155622924, 0.0013841317226470085), 'I2s': (0.05801894308626143, 0.001126478227076011), 'I4': (-0.13438401515473408, 0.0014950856516396938), 'I7': (-0.009905838651591559, 0.0012284962388621201), 'loglh': -89024.65914450402, 'status': 3, 'I1s': (0.38474847511499966, 0.00203248772729836), 'iterations': 422, 'I6s': (-0.2819536269240339, 0.0018183168737017086), 'I2c': (-0.15729327028564932, 0.002394395100525193), 'I1c': (0.5628472905357038, 0.002374816138583258)}

bin3={'I5':(0.36110929217242516,0.002317268924449345),'I9': (0.0008070319660524738, 0.0016296149082979383), 'I8': (0.008217943121467286, 0.0017708662170631273), 'I6c': (0.32136925046894627, 0.002343782769085756), 'I3': (-0.1914257408428537, 0.0018743038205515372), 'I2s': (0.10849527948183835, 0.0016635208930805279), 'I4': (-0.2243944946949843, 0.001975486187406672), 'I7': (-0.013629077588286465, 0.001707050112066233), 'loglh': -67775.78420381197, 'status': 3, 'I1s': (0.5912466136406933, 0.00294117992341425), 'iterations': 395, 'I6s': (-0.3823382264717575, 0.002502704671141165), 'I2c': (-0.24702824435298698, 0.00308612850263823), 'I1c': (0.667570373099446, 0.002914434881364336)}


bin4={'I5':(0.345449316856836,0.003093634130602069),'I9': (-0.01616074833631287, 0.0025863573716347066), 'I8': (-0.0027137125030194387, 0.0027758760090514922), 'I6c': (0.2806517736387313, 0.0035499612689754434), 'I3': (-0.3626469130568928, 0.002815986091955991), 'I2s': (0.20832335273999214, 0.0025606498111447884), 'I4': (-0.3805415463471069, 0.0027692763163593703), 'I7': (-0.008850826708882265, 0.002560844316208455), 'loglh': -45012.534832562356, 'status': 3, 'I1s': (0.9837224763729211, 0.0032280044854778867), 'iterations': 417, 'I6s': (-0.42943171008539927, 0.003002099307596817), 'I2c': (-0.4496792236631042, 0.004038622763242294), 'I1c': (0.8318777610922372, 0.003704112641106372)}



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
A9=[0.0001437980330477631,-0.007737225862451445,-0.0023530793160551353,-0.0031457469162696292]
A9err=[0.0014597406354043118,0.0023058552810335223,0.001540728892139672,0.006430598859403867]
RAB=[0.5488346720365728,0.8558080113488992,0.5279549685279713,0.7356145385894016]
RABerr=[0.024950464229975322,0.0569485017909824,0.023064691451498698,0.04193290536886692]
RLT=[0.9915641308853663,0.6481616766561479,0.6545482385627722,0.5133693404799453]
RLTerr=[0.021478750828749053,0.01899882463791246,0.014374397432226248,0.051154413406348916]
AFB=[0.02388865484696414,-0.07436627354393552,-0.1440701685972974,-0.14548825857858244]
AFBerr=[0.007332143375631965,0.008011789350454929,0.007068328687260651,0.00648311743039475]
A6s=[0.5232668452996282,0.8518529745928569,0.968839418106823,0.8664560076681795]
A6serr=[0.03084686093950491,0.03051212288243445,0.031988452332250526,0.04045317108109509]
A3=[-0.01359254356941228,0.00040283304961478923,-0.014920000404160578,-0.013042946637209982]
A3err=[0.0014425599792156584,0.0016399829396156287,0.00147000628995977,0.001465971261523266]
A4=[-0.01524823408301919,0.01682810217574372,0.012568936002876032,0.03772850894780428]
A4err=[0.006479737945757953,0.007526481339039682,0.00636271886849096,0.007703043858683125]
A8=[-0.0026207657996902105,0.002410027971299482,0.004405230826654526,-0.0013568886150972367]
A8err=[0.006400740943372239,0.006728566507901355,0.006314493222994037,0.006443607547933649]
A5=[-0.0034567572909236636,-0.06606548784120057,0.004425953651450044,-0.010404410093561618]
A5err=[0.006819081773504645,0.007307932458182976,0.0067221909757895024,0.006821536551312253]
A7=[0.00972708387217058,2.4592370673080166e-05,0.006688668404144181,0.003784351240079097]
A7err=[0.006848058571071001,0.007687822487269767,0.0068624020671368465,0.007411270586011035]

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

corr_matrix1=array([[1.000 , 0.107  ,0.169, -0.308 , 0.079,  0.047 ,-0.031,  0.050 ,-0.228, -0.329 ,-0.002],
[0.107  ,1.000  ,0.200, -0.374 , 0.076  ,0.052 ,-0.039 , 0.068 ,-0.268, -0.373  ,0.264],
[0.169  ,0.200  ,1.000, -0.471 , 0.494 , 0.253 ,-0.250 , 0.372 ,-0.683, -0.546  ,0.160],
[-0.308 ,-0.374 ,-0.471,  1.000, -0.185, -0.369 ,-0.045, -0.068,  0.203,  0.391 ,-0.318],
[0.079  ,0.076  ,0.494 ,-0.185 , 1.000  ,0.163 ,-0.156 , 0.253 ,-0.500 ,-0.618  ,0.056],
[0.047  ,0.052  ,0.253 ,-0.369 , 0.163  ,1.000  ,0.050 , 0.131 ,-0.387 ,-0.421  ,0.060],
[-0.031 ,-0.039 ,-0.250, -0.045, -0.156 , 0.050 , 1.000, -0.373 , 0.350 ,-0.000 ,-0.044],
[0.050  ,0.068  ,0.372 ,-0.068 , 0.253  ,0.131 ,-0.373 , 1.000 ,-0.610 ,-0.525  ,0.056],
[-0.228 ,-0.268 ,-0.683,  0.203, -0.500 ,-0.387,  0.350, -0.610 , 1.000 , 0.578 ,-0.216],
[-0.329 ,-0.373 ,-0.546,  0.391, -0.618 ,-0.421, -0.000, -0.525 , 0.578 , 1.000 ,-0.319],
[-0.002 , 0.264 , 0.160, -0.318,  0.056 , 0.060, -0.044 , 0.056, -0.216 ,-0.319,  1.000]
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

(I8,I7,I6s,I6c,I9,I4,I3,I2s,I2c,I1s,I1c)= correlated_values_norm([I8,I7,I6s,I6c,I9,I4,I3,I2s,I2c,I1s,I1c], corr_matrix1)
1-I8-I7-I9-I4-I3-I2s-I2c-I1c-I1s-I6c-I6s
corr_matrix2=array([
[1.000  ,0.011  ,0.176, -0.282  ,0.090,  0.064 ,-0.056,  0.058 ,-0.203, -0.261 ,-0.034],
[0.011  ,1.000  ,0.251, -0.306  ,0.156 , 0.132 ,-0.074 , 0.114 ,-0.304 ,-0.382  ,0.228],
[0.176  ,0.251  ,1.000, -0.492  ,0.588 , 0.438 ,-0.302 , 0.520 ,-0.787 ,-0.665  ,0.254],
[-0.282 ,-0.306 ,-0.492,  1.000 ,-0.213, -0.360 , 0.003 ,-0.188 , 0.278 , 0.342 ,-0.300],
[0.090  ,0.156  ,0.588 ,-0.213  ,1.000  ,0.328 ,-0.206  ,0.406 ,-0.651 ,-0.705  ,0.158],
[0.064  ,0.132  ,0.438 ,-0.360  ,0.328  ,1.000 ,-0.030  ,0.315 ,-0.585 ,-0.569  ,0.127],
[-0.056 ,-0.074 ,-0.302 , 0.003 ,-0.206 ,-0.030 , 1.000 ,-0.441,  0.363 , 0.069 ,-0.065],
[0.058  ,0.114  ,0.520 ,-0.188 , 0.406  ,0.315 ,-0.441  ,1.000, -0.695 ,-0.577  ,0.124],
[-0.203 ,-0.304 ,-0.787 , 0.278, -0.651 ,-0.585 , 0.363 ,-0.695 , 1.000 , 0.725 ,-0.309],
[-0.261 ,-0.382 ,-0.665 , 0.342, -0.705 ,-0.569 , 0.069 ,-0.577 , 0.725 , 1.000 ,-0.365],
[-0.034 , 0.228 , 0.254 ,-0.300,  0.158 , 0.127 ,-0.065 , 0.124, -0.309 ,-0.365 , 1.000]                                                               
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

(I8,I7,I6s,I6c,I9,I4,I3,I2s,I2c,I1s,I1c)= correlated_values_norm([I8,I7,I6s,I6c,I9,I4,I3,I2s,I2c,I1s,I1c], corr_matrix2)



corr_matrix3=array([
[1.000 ,-0.051  ,0.131, -0.191  ,0.066,  0.044 ,-0.068,  0.016 ,-0.177, -0.231 ,-0.093],
[-0.051  ,1.000 , 0.221, -0.235 , 0.129,  0.108 ,-0.098,  0.089 ,-0.291, -0.347 , 0.138],
[0.131  ,0.221  ,1.000 ,-0.420  ,0.516 , 0.420 ,-0.317 , 0.460 ,-0.752 ,-0.601  ,0.164],
[-0.191 ,-0.235 ,-0.420 , 1.000 ,-0.157 ,-0.253, -0.024 ,-0.133 , 0.169,  0.194 ,-0.209],
[0.066  ,0.129  ,0.516 ,-0.157  ,1.000  ,0.301 ,-0.211  ,0.348 ,-0.612 ,-0.629  ,0.078],
[0.044  ,0.108  ,0.420 ,-0.253  ,0.301  ,1.000 ,-0.057  ,0.287 ,-0.585 ,-0.552  ,0.061],
[-0.068 ,-0.098 ,-0.317 ,-0.024 ,-0.211 ,-0.057 , 1.000 ,-0.494 , 0.415 , 0.069 ,-0.089],
[0.016  ,0.089  ,0.460 ,-0.133  ,0.348  ,0.287 ,-0.494  ,1.000 ,-0.657 ,-0.478  ,0.037],
[-0.177 ,-0.291 ,-0.752 , 0.169 ,-0.612 ,-0.585 , 0.415 ,-0.657 , 1.000 , 0.668 ,-0.210],
[-0.231 ,-0.347 ,-0.601 , 0.194 ,-0.629 ,-0.552 , 0.069 ,-0.478 , 0.668 , 1.000, -0.268],
[-0.093 , 0.138,  0.164 ,-0.209,  0.078 , 0.061, -0.089  ,0.037, -0.210 ,-0.268,  1.000]                                                                
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

(I8,I7,I6s,I6c,I9,I4,I3,I2s,I2c,I1s,I1c)= correlated_values_norm([I8,I7,I6s,I6c,I9,I4,I3,I2s,I2c,I1s,I1c], corr_matrix3)

corr_matrix4=array([
[1.000 ,-0.111  ,0.035 ,-0.167 ,-0.010 ,-0.034 ,-0.044 ,-0.068 ,-0.102 ,-0.154 ,-0.144],
[-0.111 , 1.000 , 0.033 ,-0.161 ,-0.008 ,-0.024, -0.047 ,-0.069 ,-0.115 ,-0.187 , 0.013],
[0.035 , 0.033  ,1.000 ,-0.441  ,0.223 , 0.153 ,-0.110  ,0.137 ,-0.426 ,-0.336  ,0.045],
[-0.167 ,-0.161 ,-0.441 , 1.000 ,-0.067 ,-0.188 ,-0.101 ,-0.049 , 0.065 , 0.124 ,-0.167],
[-0.010 ,-0.008 , 0.223 ,-0.067 , 1.000 , 0.101 ,-0.070 , 0.116 ,-0.435 ,-0.502 ,-0.009],
[-0.034 ,-0.024 , 0.153 ,-0.188 , 0.101  ,1.000 , 0.091 , 0.077 ,-0.439 ,-0.416 ,-0.028],
[-0.044 ,-0.047 ,-0.110 ,-0.101 ,-0.070  ,0.091 , 1.000 ,-0.434 , 0.261 ,-0.126 ,-0.041],
[-0.068 ,-0.069 , 0.137 ,-0.049,  0.116  ,0.077 ,-0.434  ,1.000 ,-0.468 ,-0.267 ,-0.066],
[-0.102 ,-0.115 ,-0.426 , 0.065, -0.435 ,-0.439 , 0.261 ,-0.468 , 1.000  ,0.496 ,-0.121],
[-0.154 ,-0.187 ,-0.336 , 0.124, -0.502 ,-0.416 ,-0.126 ,-0.267 , 0.496  ,1.000 ,-0.180],
[-0.144 , 0.013  ,0.045 ,-0.167, -0.009 ,-0.028 ,-0.041 ,-0.066 ,-0.121 ,-0.180 , 1.000]
])


I9=bin4['I9']
I8=bin4['I8']
I7=bin4['I7']
I6s=bin4['I6s']
I6c=bin4['I6c']
#I5=bin4['I5']
I4=bin4['I4']
I3=bin4['I3']
I2s=bin4['I2s']
I2c=bin4['I2c']
I1s=bin4['I1s']
I1c=bin4['I1c']

(I8,I7,I6s,I6c,I9,I4,I3,I2s,I2c,I1s,I1c)= correlated_values_norm([I8,I7,I6s,I6c,I9,I4,I3,I2s,I2c,I1s,I1c], corr_matrix4)
rab=(I1c+2*I1s-3*I2c-6*I2s)/(2*I1c+4*I1s+2*I2c+4*I2s)
rlt=(3*I1c-I2c)/(6*I1s-2*I2s)
Gammaq=(3*I1c+6*I1s-I2c-2*I1s)/4.
afb1=I6c+2*I6s
afb=(3/8.)*(afb1/Gammaq)

a3=(1/(np.pi*2))*I3/Gammaq
a9=(1/(2*np.pi))*(1-I8-I7-I5-I4-I3-I2s-I1s-I1c-I2c-I6s-I6c)/Gammaq
a6s=(-27/8.)*(I6s/Gammaq)
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

plt.errorbar(centers,A3, xerr=q2err,yerr=A3err, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$A_{3}$ - RapidSim')
sol,_=curve_fit(power, centers, A3, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.ylim(-0.04,0.06)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{3}$ ($q^2$)')
plt.title(r'$A_{3}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('A3.pdf')
plt.close()
plt.close()


plt.errorbar(centers,AFB, xerr=q2err,yerr=AFBerr, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$A_{FB}$ - RapidSim')
sol,_=curve_fit(power, centers, AFB, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.ylim(-0.3,0.4)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{FB}$ ($q^2$)')
plt.title(r'$A_{FB}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('AFB.pdf')
plt.close()
plt.close()

plt.errorbar(centers,RLT, xerr=q2err,yerr=RLTerr, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$R_{LT}$ - RapidSim')
sol,_=curve_fit(power, centers, RLT, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.ylim(-0,4)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$R_{LT}$ ($q^2$)')
plt.title(r'$R_{LT}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('RLT.pdf')
plt.close()
plt.close()


plt.errorbar(centers,RAB, xerr=q2err,yerr=RABerr, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$R_{AB}$ - RapidSim')
sol,_=curve_fit(power, centers, RAB, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),power(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1],sol[2]),color='#3F7F4C')
plt.ylim(-0.5,1.0)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$R_{AB}$ ($q^2$)')
plt.title(r'$R_{AB}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('RAB.pdf')
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

plt.errorbar(centers,A9, xerr=q2err,yerr=A9err, fmt='o', color='#3F7F4C',
ecolor='lightgray', elinewidth=3, capsize=0,label=r'$A_{9}$ - RapidSim')
sol,_=curve_fit(linear, centers, A9, maxfev=2000)
plt.plot(np.linspace(min(q2_borders),max(q2_borders),50),linear(np.linspace(min(q2_borders),max(q2_borders),50),sol[0],sol[1]),color='#3F7F4C')

plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{9}$ ($q^2$)')
plt.title(r'$A_{9}$ calculated with I',fontsize=14, color='black')
plt.legend()
plt.savefig('A9.pdf')
plt.close()
plt.close()

