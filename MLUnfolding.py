#!/usr/bin/env python

#########################################################################################################
#### import statements
#########################################################################################################

# import from basic Python to be able to read automatically the name of a file
import sys

# import to read a ROOT file as numpy array
import uproot

# import to use numpy arrays
import numpy as np

# imports for Keras to train NN
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras import backend as K

#########################################################################################################
#### configuration options
#########################################################################################################

debug=False
verbose=True
doTest=False

# initial input file, with the same number of reco and truth jets
inputFileName="/afs/cern.ch/user/l/lciucu/public/data/MLUnfolding/user.yili.18448069._000001.output.sync.root"
# the maximum value of jet pt we consider, meaning that if a jet has more than this (say 1098.5 GeV we overwrite its value to 500.0 GeV)
# by analogy with filling a ROOT histogram that has a max value of 500 GeV, all the values above that will go to the overflow bin
# the reason is is that truth and reco will go to different high values but we need to have the same maximum value
# as we treat both truth and reco at the same time to the NN and we need a function from reco to the truth
maxValue=500.0 # in GeV,
# the bin width for the leading jet pt
binWidth=10.0 # in GeV

# for DNN training with Keras
batch_size=1000
epochs=50

#########################################################################################################
#### Functions
#######################################################################################################

# a general function to print the values and other properties of a numpy array
# use to see the values of the numpy arrays in our code for debugging and understanding the code flow
def print_nparray(treeName,option,nparray_name,nparray):
    if verbose or debug:
        print("treeName",treeName,"option",option,nparray_name,nparray,"type",type(nparray),"shape",nparray.shape,"min",np.min(nparray),"max",np.max(nparray))
# done function

# read the .root file with uproot to have numpy arrays of leading jet pt for train and test samples
# for each of the reco (treeName="nominal") and truth (treeName="particleLevel")
# treeName is "nominal" for reco, which is input to the NN
# treeName is "particleLevel" for truth, which is output of the NN
# 
# option is "train" for the even event indices (index: 0, 2, 4, etc) 
# option is "test" for the odd event indices (index: 1, 3, 5, etc)
def read_root_file(uproot_file,treeName,option,maxValue,binWidth):
    if verbose or debug:
        print("")
        print("Start read_root_file() with treeName",treeName,"option",option,"maxValue",maxValue,"binWidth",binWidth)
    # get the tree
    tree=uproot_file[treeName]
    # get the number of events in the tree
    nrEvents=tree.numentries
    if debug or verbose:
        print("nrEvents",nrEvents)
    # get the array with the collection per jets
    # as many entries as number of events
    # one entry is the numpy array of jets
    array_jet_pt=tree.array("jet_pt")
    if debug or verbose:
        print(treeName,"array_jet_pt",array_jet_pt.shape,type(array_jet_pt))
    if debug:
        for jet_pt in array_jet_pt:
            print("jet_pt",jet_pt,jet_pt.shape)
    # we want to loop over the events and store the leading jet pt values into a list
    # so that we then can convert the list to a numpy array
    # we create a list called list_var, meaing list of our variable
    # the variable will be the leading jet pt
    # in the future we can try this for another variable, so it's good to use a generic name as var
    list_var=[]
    for i in range(nrEvents):
        # when we test our code we run only on 6 events, 3 for NN traininig, 3 for NN testing
        if doTest:
            if i>5:
                continue
        # now based on the option, keep either the even or add events
        if option=="train":
            # keep only even events for NN training, so skip (with continue) if the event is odd (i%2==1)
            if i%2==1:
                continue
            # done if
            # only even events with i%2==0 remain here
        elif option=="test":
            # keep only odd events for NN training, so skip (with continue) if the event is even (i%2==0)
            if i%2==0:
                continue
            # done if
            # only even events with i%2==1 remain here
        else:
            print("option",option,"is not known. Choose train or test. Will ABORT!!!")
            assert(False)
        # done if based on option
        # if here we are left with either even or odd events, based on our desired of NN train or test
        # 
        # for this event of index i, extract the collection of jets in the event
        jet_pt=array_jet_pt[i]
        if debug:
            print("i",i,"jet_pt",jet_pt,jet_pt.shape,type(jet_pt))
        # the leading jet_pt is the index number 0 of jet_pt
        leading_jet_pt=jet_pt[0]
        # convert from MeV to GeV by multiplying by 0.001
        leading_jet_pt*=0.001
        # if the value is larger than the max value, rewrite it to have the maxValue instead
        if leading_jet_pt>maxValue:
            leading_jet_pt=maxValue-0.001 # subtract just a tiny bit
        if debug:
            print("leading_jet_pt",type(leading_jet_pt),leading_jet_pt)
        # add the leading_jet_pt (the value of the pt in GeV for the leading jet in pt) to a list
        # we could call it list_leading_jet_pt, but to make the name shorter, let's use var from variable instead of leading_jet_pt
        list_var.append(leading_jet_pt)
    # done loop over events in the tree
    # the Python list of jets is now created, we convert it into a numpy array, to be able to do operations element-wise
    # meaning one element at a time, for example divide all jet values by the bin width
    nparray_var=np.array(list_var)
    print_nparray(treeName,option,"nparray_var",nparray_var)
    # divide each var by the desired bin width to find out in which bin it falls into
    # the bin is here as float, as only for the truth we need to make the bin width as integer by taking the floor
    nparray_varBinFloat=nparray_var/binWidth
    # notice how we divided a numpy array by a float number (binWidth) and that returns a numpy array 
    # where each element of the first array is divided by that float number
    print_nparray(treeName,option,"nparray_varBinFloat",nparray_varBinFloat)
    # all done, we can return the numpy array of the variable
    return nparray_varBinFloat
# done function

def gen(uproot_file,option,maxValue,binWidth):
    if verbose or debug:
        print("")
        print("Start gen() with option",option,"maxValue",maxValue,"binWidth",binWidth)
    # option must be either train or test, if another value will abort
    if (option=="train" or option=="test")==False:
        print("option",option,"not known. It should be either train or test. Will ABORT!!!")
        assert(False)
    # done if
    # we call the function read_root_file twice, once for reco (treeName=nominal) and once for truth (treName=particleLevel)
    nparray_varBinFloat_recon=read_root_file(uproot_file,"nominal",option,maxValue,binWidth)
    nparray_varBinFloat_truth=read_root_file(uproot_file,"particleLevel",option,maxValue,binWidth)
    if verbose or debug:
        print("")
    print_nparray("nominal",option,"nparray_varBinFloat_recon",nparray_varBinFloat_recon)
    print_nparray("particleLevel",option,"nparray_varBinFloat_truth",nparray_varBinFloat_truth)
    # our NN takes as input the bin value as float of the reco, but needs as output the bin value as integer (trucanted) of the float for the truth
    nparray_varBinInt_truth=np.trunc(nparray_varBinFloat_truth)
    print_nparray("particleLevel",option,"nparray_varBinInt_truth",nparray_varBinInt_truth)
    # done all, so return first the binInt for the truth and then the binFloat for the reco, to be the same order as in the gen() from the original example
    return nparray_varBinInt_truth, nparray_varBinFloat_recon
# done function

# 
# Definition of the model using 3-layer neural network
def prepareModel(NBins,nvar=1,kappa=8):
    ''' Prepare KERAS-based sequential neural network with for ML unfolding. 
        Nvar defines number of inputvariables. NBins is number of truth bins. 
        kappa is an empirically tuned parameter for the intermediate layer'''
    if verbose or debug:
        print("Start prepareModel with number of variables",nvar,"NBins",NBins,"kappa",kappa)
    # constructor of the model
    model = Sequential()
    # we will use three layers, with the same activation functions and kappa as from the code example
    # add the first layer, which is the input from reco, so take them as they are with a linear activation function
    model.add(Dense(nvar,activation='linear',input_shape=(nvar,1)))
    if debug:
        print("model",model)
    # here do we flatten the first layer?
    model.add(Flatten())
    # add the second layer with a relu activation function
    model.add(Dense(kappa*NBins**2,activation='relu'))
    # add the thir layer, or the output, with a softmax activation function
    model.add(Dense(NBins,activation='softmax'))
    # compile the model by choosing the optimiser and the metrics
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    # done all, so ready to return
    return model
# done function

# the function that runs everything
def doItAll():
    # open the .root file to be used for both type of trees and both types of options
    uproot_file=uproot.open(inputFileName)
    if debug:
        print("uproot_file",uproot_file)
    # call the gen function for the training of the NN
    # use the same naming convention from the original example
    # f means for training, g means the output as truth or generated, r means the input as reco or reconstructed or observed
    gf,rf=gen(uproot_file,"train",maxValue,binWidth)
    # call the gen function for the testing or validation of the NN
    gt,rt=gen(uproot_file,"test",maxValue,binWidth)
    # we need to further update these to prepare them to be used by the NN
    # like in the initial example, we use  keras.utils.to_categorical and the number of bins
    NBins=int(maxValue/binWidth)
    if verbose or debug:
        print("maxValue",maxValue,"binWidth",binWidth,"NBins",NBins)
    # for truth
    gfcat=keras.utils.to_categorical(gf,NBins)
    gtcat=keras.utils.to_categorical(gt,NBins)
    # for reco
    rf=rf.reshape(rf.shape[0],1,1)
    rt=rt.reshape(rt.shape[0],1,1)
    # these are what the NN will use, let's print those values
    print_nparray("particleLevel","train","gfcat",gfcat)
    print_nparray("particleLevel","test","gtcat",gtcat)
    print_nparray("nominal","train","rf",rf)
    print_nparray("nominal","test","rt",rt)
    # prepare the model of the DNN with Keras
    model=prepareModel(NBins,nvar=1,kappa=8)
    # fit the model
    h = model.fit(rf,gfcat,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(rt,gtcat))
    exit()
    # save model,if needed
    model.save(outputFolder+"/model.hdf5")
    # all done
# done function

#########################################################################################################
#### Run all
#######################################################################################################

doItAll()

#########################################################################################################
#### Done
#######################################################################################################

print("")
print("")
print("Finished well in "+sys.argv[0]) # prints out automatically the name of the file that we ran

exit()
