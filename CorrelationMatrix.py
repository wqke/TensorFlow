
#Use uncertainties package : calculate the combined error 
#we need to know : - the correlation/covariance matrix  - the values with error 



from numpy import array
import numpy as np
import uncertainties
from uncertainties import *



val1={'I9': (1.0326646471270351e-07, 0.017157604203728694), 'I8': (-4.27737426589303e-08, 0.016997109462063065), 'I6c': (0.3616235141622446, 0.028887582943266432), 'I3': (-0.10332103144144655, 0.02404893594908497), 'I2s': (0.06457575059506793, 0.013763635107599925), 'I4': (-0.1365313701940487, 0.02430029471514289), 'I7': (-2.0492974339703096e-07, 0.015314046791704228), 'loglh': -518.4112350753132, 'status': 3, 'I1s': (0.3763838261090342, 0.030554763024308706), 'iterations': 276, 'I6s': (-0.2546126760989651, 0.02298326889848018), 'I2c': (-0.16420663579962236, 0.03918263578362252), 'I1c': (0.5590404502957198, 0.04465135098148654)}
#The values should be in the form (value, error)


mat1=np.array([
[1.000 ,-0.047  ,0.292, -0.367  ,0.217  ,0.236 ,-0.048,  0.226 ,-0.335 ,-0.339 , 0.072],
[-0.047 , 1.000 ,-0.038, -0.062 ,-0.089 ,-0.105, -0.041, -0.139 , 0.041 ,-0.047,  0.045],
[0.292 ,-0.038  ,1.000 ,-0.670  ,0.684  ,0.629 ,-0.084 , 0.667 ,-0.819 ,-0.736 , 0.489],
[-0.367 ,-0.062 ,-0.670 , 1.000 ,-0.494 ,-0.545 , 0.008 ,-0.550 , 0.556 , 0.519, -0.462],
[0.217 ,-0.089  ,0.684 ,-0.494  ,1.000  ,0.633 ,-0.008  ,0.603 ,-0.797 ,-0.824 , 0.499],
[0.236 ,-0.105  ,0.629 ,-0.545  ,0.633  ,1.000 , 0.158  ,0.564 ,-0.785 ,-0.798 , 0.462],
[-0.048 ,-0.041 ,-0.084,  0.008 ,-0.008 , 0.158 , 1.000 ,-0.217 , 0.069 ,-0.157,  0.097],
[0.226 ,-0.139  ,0.667 ,-0.550 , 0.603  ,0.564 ,-0.217  ,1.000 ,-0.793 ,-0.691 , 0.403],
[-0.335 , 0.041 ,-0.819 , 0.556, -0.797 ,-0.785 , 0.069 ,-0.793 , 1.000 , 0.848, -0.588],
[-0.339 ,-0.047 ,-0.736 , 0.519, -0.824 ,-0.798 ,-0.157 ,-0.691 , 0.848 , 1.000, -0.631],
[0.072  ,0.045 , 0.489 ,-0.462 , 0.499  ,0.462 , 0.097  ,0.403 ,-0.588 ,-0.631 , 1.000]
])



I9=val1['I9']
I8=val1['I8']
I7=val1['I7']
I6s=val1['I6s']
I6c=val1['I6c']
I5=val1['I5']
I4=val1['I4']
I3=val1['I3']
I2s=val1['I2s']
I2c=val1['I2c']
I1s=val1['I1s']
I1c=val1['I1c']

(I8,I7,I6s,I6c,I4,I3,I2s,I2c,I1s,I1c,I9)= correlated_values_norm([I8,I7,I6s,I6c,I4,I3,I2s,I2c,I1s,I1c,I9], mat1)
#define a correlation relation



"""
Test on this method : we obtain the combined error in +/-
>>> 1-I8-I7-I9-I4-I3-I2s-I2c-I1c-I1s-I6c-I6s
>>> 0.29704831680903765+/-0.022803505485894052

"""


#Calculate with any formula we want

rab=(I1c+2*I1s-3*I2c-6*I2s)/(2*I1c+4*I1s+2*I2c+4*I2s)
rlt=(3*I1c-I2c)/(6*I1s-2*I2s)
Gammaq=(3*I1c+6*I1s-I2c-2*I1s)/4.
a3=(1/(np.pi*2))*I3/Gammaq
a9=(1/(2*np.pi))*I9/Gammaq
a6s=(-27/8.)*(I6s/Gammaq)
a4=(-2/np.pi)*I4/Gammaq
a8=(2/np.pi)*I8/Gammaq
a5=(-3/4.)*(1-I8-I7-I9-I4-I3-I2s-I1s-I1c-I2c-I6s-I6c)/Gammaq
a7=(-3/4.)*I7/Gammaq
