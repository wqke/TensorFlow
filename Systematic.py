import matplotlib
matplotlib.use("Agg")

import sys,os 
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy import stats
import numpy as np
import json
from skhep.visual import MplPlotter as skh_plt
from scipy.stats import gaussian_kde

#3pi or 3pipi0
sub_mode = sys.argv[1]
#Geometry (all or LHCb)
geom = sys.argv[2]
#True or reco angles
var_type = sys.argv[3]
#Number of events to run on (in k) - 5, 10, 20, 40, 80
num_sig = sys.argv[4]
#Hammer (SM / T1 / T2)
ham = sys.argv[5]

#The systematic :prompt, feed, Ds, Dplus, D0
syst=sys.argv[6]

path = 'results/'
bins=10   #sqrt of the filenumber (100)
	
all_files = [f for f in listdir(path) if isfile(join(path, f))]

df = {}
df_init = {}
init_vals = {}
vals = {}
errs = {}
pulls = {}
out = {}

#Keep only relevant toys 
result_files = []
param_files = []
init_files = []

default_result = "result_Lifetime_%s_%s_%s_%s_Hammer_%s.txt" % (sub_mode,geom,var_type,num_sig,ham)
default_param = "param_Lifetime_%s_%s_%s_%s_Hammer_%s.txt" % (sub_mode,geom,var_type,num_sig,ham)

for f in all_files:
	if("%s_%s_%s_%s_Hammer_%s_" % (sub_mode,geom,var_type,num_sig,ham) in f and ".txt" in f and "result" in f):
		result_files.append(f)
	elif("%s_%s_%s_%s_Hammer_%s_" % (sub_mode,geom,var_type,num_sig,ham) in f and ".txt" in f and "param" in f):
		param_files.append(f)
	elif("%s_%s_%s_%s_Hammer_%s_" % (sub_mode,geom,var_type,num_sig,ham) in f and ".txt" in f and "init" in f):
		init_files.append(f)
 


    
    
    
print "Number of toys in sample: %s" % len(result_files)

coeffs = {"I1s": "$I_{1s}$",
		  "I2c": "$I_{2c}$",
		  "I2s": "$I_{2s}$",
		  "I6c": "$I_{6c}$",
		  "I6s": "$I_{6s}$",
		  "I3": "$I_{3}$",
		  "I4": "$I_{4}$",
		  "I5": "$I_{5}$",
		  "I7": "$I_{7}$",
		  "I8": "$I_{8}$",
		  "I9": "$I_{9}$",
		  "I1c": "$I_{1c}$",
		  "frac_signal": "$f_{D^{*} \\tau \\nu}}$",
		  "frac_Ds": "$f_{D^{*} D_{s} X}$",
		  "frac_Dplus": "$f_{D^{*} D^{+} X}$"
		  }

default_vals = {}
for c in coeffs:
	print "Collecting values for %s" % c
	df[c] = []
	df_init[c] = []
	vals[c] = []
	errs[c] = []
	init_vals[c] = []
	i = 0
	files = []
	if(c=="I1c"):
		files = param_files
    data_default=pd.read_csv(path+default_param, header = None, sep=" ") #
    default_vals[c]=(data_default[data_default[0].str.contains("%s" % c)]).iat[0,1] #
	else:
		files = result_files
    data_default=pd.read_csv(path+default_result, header = None, sep=" ") #
    defaut_vals[c]=(data_default[data_default[0].str.contains("%s" % c)]).iat[0,1] #
	for f in files:
		data = pd.read_csv(path+f, header = None, sep=" ")
		df[c].append(data[data[0].str.contains("%s" % c)])
		vals[c].append(df[c][i].iat[0,1])
		errs[c].append(df[c][i].iat[0,2])
		i += 1
    
    
	
		
	#Get initial values
	if(c!="I1c"):
		i = 0
		files = init_files
		for f in files:
			data = pd.read_csv(path+f, header = None, sep=" ")
			df_init[c].append(data[data[0].str.contains("%s" % c)])
			init_vals[c].append(df_init[c][i].iat[0,1])
			i += 1
	#Calculate I1c from the others 
	else:
		for i in range(0,len(files)):
			init_vals[c].append((1.0/3.0)*(4.0 - 6.0*init_vals["I1s"][i] + init_vals["I2c"][i] + 2.0*init_vals["I2s"][i]))
	
  
  
  
	#Plot pull
	fig,ax = plt.subplots(figsize=(7,7))
	
  skh_plt.hist(vals[c], bins=bins,range=(-10,10),errorbars=True,label="Result with systematic",color='k',histtype='marker',scale=(1.0/len(vals[c])))
  
	mu = np.mean(vals[c])
	mu_err = stats.sem(vals[c])
	sigma = np.std(vals[c])
	out["%s_mu" % c] = mu
	out["%s_mu_err" % c] = mu_err
	out["%s_sigma" % c] = sigma
	
	plt.xlabel("%s value ($\\sigma$)" % coeffs[c],fontsize=18)
	
	ymin, ymax = ax.get_ylim()
	plt.ylim(0.0,ymax)
  plt.axvline(x=default_vals[c] ,linestyle='-',color='grey',alpha=0.5,label="Defaut result")
  
	plt.axvline(x=mu,linestyle='--',color='b',label="Mean")
	plt.title("$mean = %.3f \\pm %.3f, \\sigma = %.3f$" % (mu,mu_err,sigma))
	plt.legend()
	#plt.show()
	plt.tight_layout()
	fig.savefig('figs/Syst_Dist_%s_%s_%s_%s_%s_%s.pdf' % (c,sub_mode,geom,var_type,num_sig,syst))

with open('results/syst_%s_%s_%s_%s_%s.json' % (sub_mode,geom,var_type,num_sig,syst), 'w') as outfile:  
  json.dump(out, outfile)


