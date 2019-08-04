#!/Users/luizaadelinaciucu/anaconda2/bin/python

from __future__ import print_function

# Basic imports for the MLUnfolding example. The code depends on scipy, numpy and keras
import matplotlib.pyplot as plt
import scipy.integrate 
import numpy as np
import math

#
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras import backend as K

###############################################################################
### configurations  ###
###############################################################################

debug=True
verbose=True

outputFolder="./output"
extensions="png,pdf".split(",")
list_treeName=["particleLevel","nominal"]

NBins=500
supersample=1
NSample=NBins*supersample

batch_size=1000
epochs=50








###############################################################################
### functions ###
###############################################################################

def p(name,value):
    print(name,value,type(value),value.shape,value.dtype)
# done function
          
def gen(dict_treeName_probs,N=20000):
    if debug or verbose:
        print("start gen() with N",N)
    dict_treeName_asim={}
    for treeName in list_treeName:
        if debug or verbose:
            print("treeName",treeName)
        # simulated events
        a=np.random.choice(NSample,N,p=dict_treeName_probs[treeName]) 
        if debug:
            p("a",a)
            print("a.shape",a.shape)
            print("a/supersample",a/supersample)
        # output simulated bins
        asim = np.trunc(a/supersample)
        if debug:
            p("asim",asim)
        # plot histogram of asim
        plt.hist(asim,bins=np.linspace(0,NBins,NBins))
        for extension in extensions:
            plt.savefig(outputFolder+"/ML_asim_"+treeName+"."+extension)
        plt.close()
        dict_treeName_asim[treeName]=asim
    # done for loop
    return dict_treeName_asim["particleLevel"],dict_treeName_asim["nominal"]
# done function

# Definition of the model using 3-layer neural network.
def prepareModel(nvar=1, NBins=NBins, kappa=8):
    ''' Prepare KERAS-based sequential neural network with for ML unfolding. 
        Nvar defines number of inputvariables. NBins is number of truth bins. 
        kappa is an empirically tuned parameter for the intermediate layer'''
    model = Sequential()
    model.add(Dense(nvar,activation='linear',input_shape=(nvar,1)))
    if debug:
        print("model",model)
    model.add(Flatten())
    model.add(Dense(kappa*NBins**2,activation='relu'))
    #model.add(Dropout(0.25))
    #model.add(Dense(2*NBins,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(NBins,activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model
# done function 
    
def doItAll():
    if debug or verbose:
        print("start do it all for MLUnfolding")
    dict_treeName_probs={}
    for treeName in list_treeName:
        probsFileName=outputFolder+"/nparray_binContent_"+treeName+".npy"
        probs=np.load(probsFileName)
        print("probs",probs,type(probs),probs.shape,probs.dtype)
        dict_treeName_probs[treeName]=probs
    # done for loop over treeName
    gt,rt=gen(dict_treeName_probs,N=20000)
    gf,rf=gen(dict_treeName_probs,N=2000000)
    # create neural network model
    model = prepareModel(1)
    # Prepare inputs for keras
    # gcat = keras.utils.to_categorical(g,NBins)
    gtcat = keras.utils.to_categorical(gt,NBins)
    gfcat = keras.utils.to_categorical(gf)

    print("gt",gt,"NBins",NBins)
    print("gtcat",gtcat,gtcat.shape)
    print("gf",gf,"NBins",NBins)
    print("gfcat",gfcat,gfcat.shape)
    # print("r",r,r.shape)
    # r = r.reshape(r.shape[0],1,1)
    # print("r",r,r.shape)
    print("rt",rt,rt.shape)
    rt = rt.reshape(rt.shape[0],1,1)
    print("rt",rt,rt.shape)
    print("rf",rf,rf.shape)
    rf = rf.reshape(rf.shape[0],1,1)
    print("rf",rf,rf.shape)

    #return
    # Fit the model
    h = model.fit(rf,gfcat,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(rt,gtcat))
    # save model,if needed
    model.save(outputFolder+"/model.hdf5")
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

