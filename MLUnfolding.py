#!/usr/bin/env python

import uproot

import numpy as np

debug=False
verbose=True
doTest=False

#########################################################################################################
#### read the .root file with uproot to have numpy arrays of leading jet pt for train and test samples
#### for each of the reco (treeName="nominal") and truth (treeName="particleLevel")
#######################################################################################################

inputFileName="/afs/cern.ch/user/l/lciucu/public/data/MLUnfolding/user.yili.18448069._000001.output.sync.root"
file=uproot.open(inputFileName)
if debug:
    print("file",file)

def readROOTFile(treeName):
    if debug or verbose:
        print("Start for treeName=",treeName)
    # get the tree
    tree=file[treeName]
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
    # we create in fact two lists
    # half of events go to the training set, so those with i=even number, so i%2==0
    # have of the events go to the testing set, so those with i=odd number, so i%2==1
    list_leading_jet_pt_train=[]
    list_leading_jet_pt_test=[]
    for i in range(nrEvents):
        if doTest:
            # if we are testing we run only on a few events, so skip those with i above 5
            if i>5:
                continue
        jet_pt=array_jet_pt[i]
        if debug:
            print("i",i,"jet_pt",jet_pt,jet_pt.shape,type(jet_pt))
        # the leading jet_pt is the index number 0 of jet_pt
        leading_jet_pt=jet_pt[0]
        # convert from MeV to GeV by multiplying by 0.001
        leading_jet_pt*=0.001
        if debug:
            print("leading_jet_pt",type(leading_jet_pt),leading_jet_pt)
        # add the leading_jet_pt to the one of the two lists, training or testing, depending if the event is even or odd
        if i%2==0:
            list_leading_jet_pt_train.append(leading_jet_pt)
        else:
            list_leading_jet_pt_test.append(leading_jet_pt)
    # done loop over events in the tree
    # the two lists are now filled, each should have half of the number of entries as there are events
    # we create two numpy arrays from the two lists
    nparray_leading_jet_pt_train=np.array(list_leading_jet_pt_train)
    nparray_leading_jet_pt_test=np.array(list_leading_jet_pt_test)
    if verbose:
        print(treeName,"nparray_leading_jet_pt_train",nparray_leading_jet_pt_train,type(nparray_leading_jet_pt_train),nparray_leading_jet_pt_train.shape)
        print(treeName,"nparray_leading_jet_pt_test",nparray_leading_jet_pt_test,type(nparray_leading_jet_pt_test),nparray_leading_jet_pt_test.shape)
    # all done, we can return the numpy array of the leading jet pt
    return nparray_leading_jet_pt_train,nparray_leading_jet_pt_test
# done function


# run code that reads .root file and creates numpy array of jets for train and test saples for reco and truth trees
nparray_leading_jet_pt_train_recon,nparray_leading_jet_pt_test_recon=readROOTFile("nominal")
nparray_leading_jet_pt_train_truth,nparray_leading_jet_pt_test_truth=readROOTFile("particleLevel")

# from these we have to build the inputs to the DNN in the right format
# following example on toy data from here: https://github.com/aglazov/MLUnfold/blob/master/Unfold.ipynb
# this initial code with my printouts at  https://gitlab.cern.ch/lciucu/MLUnfolding/blob/master/initialMLUnfolding.py
# should the input to the NN be the numpy array of the jets pt, so nparray_leading_jet_pt_train_recon ?
# should the output to the NN be a numpy array of type integer with the bin value for each of the truth jet?

#################################################################
#### for truth jets find in which bins they are as
#### NN takes as input  the numpy array of reco jet pt
#### NN takes as output the numpy array of the bin value of the truth jet pt
#################################################################

jet_pt_bin_size=10.0 # in GeV
# only for the truth jets we find can not hope to predict the exact truth jet value, but rather in what truth jet pt bin it falls into
# so we divide the truth pt by 10 GeV, by the jet_pt_bin_size, and we obtain still a float value
# we truncte the float value to find the integer value smaller than it, to correspond to the index of the jet pt bin the jet falls into
# that is what we will aim as output to the NN
# so we create a new variable that stores the bin values as floats
nparray_leading_jet_pt_bin_as_float_train_truth=nparray_leading_jet_pt_train_truth/jet_pt_bin_size
if debug or verbose:
    print("nparray_leading_jet_pt_bin_as_float_train_truth",nparray_leading_jet_pt_bin_as_float_train_truth,nparray_leading_jet_pt_bin_as_float_train_truth.dtype,nparray_leading_jet_pt_bin_as_float_train_truth.shape)
# now we truncate the values in this array to get the bin values as integers
nparray_leading_jet_pt_bin_train_truth=np.trunc(nparray_leading_jet_pt_bin_as_float_train_truth)
if debug or verbose:
    print("nparray_leading_jet_pt_bin_train_truth",nparray_leading_jet_pt_bin_train_truth,nparray_leading_jet_pt_bin_train_truth.dtype,nparray_leading_jet_pt_bin_train_truth.shape)
# what we did so far was for the train sample, we do the same for the testing sample
#replace "train" with "test"
nparray_leading_jet_pt_bin_as_float_test_truth=nparray_leading_jet_pt_test_truth/jet_pt_bin_size
if debug or verbose:
    print("nparray_leading_jet_pt_bin_as_float_test_truth",nparray_leading_jet_pt_bin_as_float_test_truth,nparray_leading_jet_pt_bin_as_float_test_truth.dtype,nparray_leading_jet_pt_bin_as_float_test_truth.shape)
# now we truncate the values in this array to get the bin values as integers
nparray_leading_jet_pt_bin_test_truth=np.trunc(nparray_leading_jet_pt_bin_as_float_test_truth)
if debug or verbose:
    print("nparray_leading_jet_pt_bin_test_truth",nparray_leading_jet_pt_bin_test_truth,nparray_leading_jet_pt_bin_test_truth.dtype,nparray_leading_jet_pt_bin_test_truth.shape)

#################################################################
#### 
#################################################################
