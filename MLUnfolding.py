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

# output
outputFolderName="./output5"

# for DNN training with Keras
doTrainNN=True
# the order is batch_size, epochs, kappa
list_list_option=[
    [1000,50,1],
    [1000,50,2],
    [1000,50,4],
    [1000,50,8],
    [1000,100,1],
    [1000,100,2],
    [1000,100,4],
    [1000,100,8],
]

# for test
if True:
    list_list_option=[
        [1000,300,8],
    ]
# done if

#########################################################################################################
#### Functions
#######################################################################################################

def get_NNFileNames(outputFolderName,batch_size,epochs,kappa):
    modelFileNameSuffix="batchSize_"+str(batch_size)+"_epochs_"+str(epochs)+"_"+str(kappa)+".hdf5"
    modelFileName=outputFolderName+"/model_full_"+modelFileNameSuffix
    weightsFileName=outputFolderName+"/model_weights_"+modelFileNameSuffix
    if debug:
        print("modelFileName",modelFileName)
        print("weightsFileName",weightsFileName)
    # done if
    return modelFileName,weightsFileName
# done function

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
    # constructor of the model with Sequential() 
    # Read more about the options and commands available here: https://keras.io/models/sequential/
    model = Sequential()
    # we will use three layers, with the same activation functions and kappa as from the code example
    # add the first layer, which is the input from reco, so take them as they are with a linear activation function
    model.add(Dense(nvar,activation='linear',input_shape=(nvar,1)))
    if debug:
        print("model",model)
    # here do we flatten the first layer?
    model.add(Flatten())
    # add the second layer with a relu activation function
    model.add(Dense(kappa*NBins,activation='relu'))
    # add the thir layer, or the output, with a softmax activation function
    model.add(Dense(NBins,activation='softmax'))
    # compile the model by choosing the optimiser and the metrics
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    # done all, so ready to return
    return model
# done function

# prepare bootstrap replica to determine statistical uncertainties
def prepareBootstrap(rt,N=10):
    ''' Prepare bootstrap weights, for error statistical and correlation estimation '''
    Nev = rt.shape[0]
    if verbose:
        print("Nev",Nev)
    p = np.random.poisson(size=(Nev,N))
    return p

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
    #
    # do the NN for different options, so we put all in a for loop
    for list_option in list_list_option:
        batch_size=list_option[0]
        epochs=list_option[1]
        kappa=list_option[2]
        if verbose:
            print("Start do NN part for batch_size",str(batch_size),"epochs",str(epochs),"kappa",str(kappa))
        modelFileName,weightsFileName=get_NNFileNames(outputFolderName,batch_size,epochs,kappa)
        # prepare the model of the DNN with Keras
        model=prepareModel(NBins,nvar=1,kappa=kappa)
        # let's train the NN only once and then save the weights to a file
        # if already trained, then we can simply read the weights and continue working from there
        # that way it is faster
        if doTrainNN:
            # fit the model
            h = model.fit(rf,gfcat,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(rt,gtcat))
            # save model to not spend the time every time to retrain, as it takes some time
            # a simple save
            model.save(modelFileName)
            # save also only the weights
            model.save_weights(weightsFileName)
        # done if
        out=model.predict(rt)
        # read as model2 with the same structure as before, but read the weights
        model2=prepareModel(NBins,nvar=1,kappa=8)
        model2.load_weights(weightsFileName)
        out2=model2.predict(rt)
        #
        if verbose:
            print_nparray("nominal","test","rt",rt)
            print_nparray("particleLevel","test","gt",gt)
            # by printing we see that these two should be the same, they have shape (21538,50)
            # while gtcat has terms strictly as zero and 1, out has terms that are very small like 10^-4
            # so let's loop over all the elements and for each get the list with 50 elements, and then loop one by one and compare them
            # but accepting that 10^-4 is close enough to zero and seeing if they get the element close to 1 at the same possition or close
            print_nparray("particleLevel","test","gtcat",gtcat)
            print_nparray("particleLevel","test","out",out)
            assert(gtcat.shape[0]==out.shape[0])
            assert(gtcat.shape[1]==out.shape[1])
            for i in range(gtcat.shape[0]):
                if i>5:
                    continue
                gtcat_current=gtcat[i]
                out_current=out[i]
                out2_current=out2[i]
                print_nparray("particleLevel","test","gtcat[i]",gtcat[i])
                print_nparray("particleLevel","test","out[i]",out[i])
                print_nparray("particleLevel","test","out2[i]",out2[i])
                for j in range(gtcat_current.shape[0]):
                    # if True:
                    if gtcat_current[j]>0.1:
                        print(i,j,"gtcat[i][j]",gtcat_current[j])
                # done for loop over j
                for j in range(out_current.shape[0]):
                    # if True:
                    if out_current[j]>0.1:
                        print(i,j,"out[i][j]",out_current[j])
                    # done if
                # done for loop over j
                for j in range(out2_current.shape[0]):
                    # if True:
                    if out2_current[j]>0.1:
                        print(i,j,"out2[i][j]",out2_current[j])
                    # done if
                # done for loop over j
            # done for loop over i
        # done loop over list_option
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
