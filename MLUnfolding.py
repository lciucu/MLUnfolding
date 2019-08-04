#!/Users/luizaadelinaciucu/anaconda2/bin/python

from __future__ import print_function

# Basic imports for the MLUnfolding example. The code depends on scipy, numpy and keras
import matplotlib.pyplot as plt
import scipy.integrate 
import numpy as np
import math
import keras

###############################################################################
### configurations  ###
###############################################################################

debug=True
verbose=True

outputFolder="./output"
probsFileName=outputFolder+"/nparray_binContent_particleLevel.npy"







###############################################################################
### functions ###
###############################################################################

def gen(probs,N=20000):
    if debug or verbose:
        print("start gen() with N",N)
# done function
    
def doItAll():
    if debug or verbose:
        print("start do it all for MLUnfolding")
    probs=np.load(probsFileName)
    print("probs",probs,type(probs),probs.shape,probs.dtype)
    gen(probs,N=20000)

# done function 


###############################################################################
### running  ###
###############################################################################

doItAll()



###############################################################################
### done all  ###
###############################################################################

print("")
print("")
print("All finished well for MLUnfolding.py")

