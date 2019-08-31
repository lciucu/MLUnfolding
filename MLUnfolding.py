#!/usr/bin/env python

# import from basic Python to be able to read automatically the name of a file
import sys
import os

#########################################################################################################
#### command line arguments
#########################################################################################################

total = len(sys.argv)
# number of arguments plus 1                                                    
if total!=4:
    print("You need some arguments, will ABORT!")
    print("Ex: ",sys.argv[0]," stringNNs stringStage outputFolder")
    print("Ex: ",sys.argv[0]," B3_8_3_1000,B4_8_3_200 1110 output11_7")
    assert(False)
# done if     

my_stringNNs=sys.argv[1]
my_string_stage=sys.argv[2]
my_outputFolderName=sys.argv[3]

my_list_infoNN=[]
for stringNN in my_stringNNs.split(","):
    list_stringNN=stringNN.split("_")
    list_infoNN=[list_stringNN[0],int(list_stringNN[1]),int(list_stringNN[2]),int(list_stringNN[3])]
    my_list_infoNN.append(list_infoNN)
# done for loop

print("my_stringNNs",my_stringNNs)
print("my_list_infoNN",my_list_infoNN)
print("my_string_stage",my_string_stage)
print("my_outputFolderName",my_outputFolderName)

#########################################################################################################
#### import statements
#########################################################################################################



# to create a deep copy of a list
import copy

# import to use numpy arrays
import numpy as np
# to obtain reproducible results, meaning each new NN training to obtain the same result, set the seed of the random number now
# np.random.seed(2019)
# np.random.seed(20190825)
np.random.seed(98383822)

# plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab

#########################################################################################################
#### configuration options
#########################################################################################################

debug=False
verbose=True

overwriteSettings=True

# we split the code into stages, so that we can run only the last stage for instance
# the output of each stage is stored to files, that are read back at the next stage
# stage 1: read .root file and produce numpy arrays that are input and output to the NN training
# stage 1: input: .root file; output: rf, rt, gfcat, gtcat
# stage 2: use the rf, rt, gfcat, gtcat to produce a NN training and store the model weights to a file
# stage 3: use the NN to do further studies - not yet done
# stage 4: make plots of the NN training - done
string_stage="1111" # all steps
#string_stage="1000" # NN input
#string_stage="0100" # NN train
#string_stage="0010" # NN analyze
#string_stage="0001" # plots
#string_stage="1101"
#string_stage="0110"
#string_stage="0101"

if overwriteSettings:
    string_stage=my_string_stage

# output
outputFolderName="./output11_6"
if overwriteSettings:
    outputFolderName=my_outputFolderName
os.system("mkdir -p "+outputFolderName)

list_stage=list(string_stage)
doROOTRead=bool(int(list_stage[0]))
doNNTrain=bool(int(list_stage[1]))
doNNAnalyze=bool(int(list_stage[2]))
doPlot=bool(int(list_stage[3]))

doPlotMetrics=True
doPlotOutput1D=True
doPlotInputOutput2D=True

if debug:
    print("string_stage",string_stage)
    print("list_stage",list_stage)
    print("doROOTRead",doROOTRead)
    print("doNNTrain",doNNTrain)
    print("doNNAnalyze",doNNAnalyze)
    print("doPlot",doPlot)

# import to read a ROOT file as numpy array
if doROOTRead:
    import uproot

# imports for Keras for neural network
if doROOTRead or doNNTrain or doNNAnalyze:
    import keras

doROOTReadTest=False

# initial input file, with the same number of reco and truth jets
inputFileName="/afs/cern.ch/user/l/lciucu/public/data/MLUnfolding/user.yili.18448069._000001.output.sync.root"
# the maximum value of jet pt we consider, meaning that if a jet has more than this (say 1098.5 GeV we overwrite its value to 500.0 GeV)
# by analogy with filling a ROOT histogram that has a max value of 500 GeV, all the values above that will go to the overflow bin
# the reason is is that truth and reco will go to different high values but we need to have the same maximum value
# as we treat both truth and reco at the same time to the NN and we need a function from reco to the truth
maxValue=500.0 # in GeV,
# the bin width for the leading jet pt
binWidth=10.0 # in GeV
# number of bins is calculated from these two
NBins=int(maxValue/binWidth)
if verbose or debug:
    print("maxValue",maxValue,"binWidth",binWidth,"NBins",NBins)

# extensions="pdf,png"
extensions="png"

k=8
e=3
b=1000

# for DNN training with Keras
# the order is layer and kappa (from architecture), epochs, batchSize (from learning steps)
list_infoNN=[
    #["A1",k,e,b],
    #["B1",k,e,b],
    #["B2",k,e,b],
    #["B3",k,e,b],
    #["B4",k,e,b],
    #["B5",k,e,b],
    #["B10",k,e,b],
    #["C5",k,e,b],
    #["D5",k,e,b],
    #["B3",8,e,5000],
    #["B3",8,e,1000],
    ["B3",8,e,200],
    #["B3",8,e,60],
    ["B4",8,e,200],
    ["B5",8,e,200],
]

if overwriteSettings:
    list_infoNN=my_list_infoNN

#list_infoNN=[
#    ["A1",k,e,b],
#    ["B5",k,e,b],
#]

list_listInfoToPlot=[
    #["A1_B1_B2_B3_B4_B5_B10",[ ["A1",k,e,b],["B1",k,e,b],["B2",k,e,b],["B3",k,e,b],["B4",k,e,b],["B5",k,e,b],["B10",k,e,b] ]],
    #["B1_B2_B3_B4_B5_B10",[ ["B1",k,e,b],["B2",k,e,b],["B3",k,e,b],["B4",k,e,b],["B5",k,e,b],["B10",k,e,b] ]],
    #["B1_B2_B3_B4_B5",[ ["B1",k,e,b],["B2",k,e,b],["B3",k,e,b],["B4",k,e,b],["B5",k,e,b] ]],
    #["B2_B3_B4_B5",[ ["B2",k,e,b],["B3",k,e,b],["B4",k,e,b],["B5",k,e,b] ]],
    #["B3_B4_B5",[ ["B3",k,e,b],["B4",k,e,b],["B5",k,e,b] ]],
    #["B3_B4_B5_B10",[ ["B3",k,e,b],["B4",k,e,b],["B5",k,e,b],["B10",k,e,b] ]],
    #["B5_C5_D5",[ ["B5",k,e,b],["C5",k,e,b],["D5",k,e,b] ]],
    #["A1_B2",[ ["A1",k,e,b],["B2",k,e,b] ]],
    #["A1_B3",[ ["A1",k,e,b],["B3",k,e,b] ]],
    #["A1_B4",[ ["A1",k,e,b],["B4",k,e,b] ]],
    #["A1_B5",[ ["A1",k,e,b],["B5",k,e,b] ]],
    #["A1_B10",[ ["A1",k,e,b],["B10",k,e,b] ]],
    #["B3_B4",[ ["B3",k,e,b],["B4",k,e,b] ]],
    #["B3_B4",[ ["B3",k,e,b],["B4",k,e,b] ]],
    #["B3_B5",[ ["B3",k,e,b],["B5",k,e,b] ]],
    #["A1",[ ["A1",k,e,b] ]],
    #["B2",[ ["B2",k,e,b] ]],
    #["B3",[ ["B3",k,e,b] ]],
    #["B4",[ ["B4",k,e,b] ]],
    #["B5",[ ["B5",k,e,b] ]],
    #["B10",[ ["B10",k,e,b] ]],
    #["B3_b",[ ["B3",8,e,1000],["B3",8,e,5000],["B3",8,e,200],["B3",8,e,60] ]],
    ["B3_B4_B5_200",[ ["B3",8,e,200],["B4",8,e,200],["B5",8,e,200] ]],
]

list_metric=[
    "loss",
    "accuracy",
]

dict_metric_plotRange={
    #"loss":[1.70,3.8],
    #"accuracy":[0.05,0.45]
    #"loss":[1.70,2.0],
    #"accuracy":[0.25,0.35],
    "loss":[1.75,1.85],
    "accuracy":[0.30,0.35],
}

list_optionTrainTest=[
    "train",
    "test",
]

list_outputType=[
    "True",
    "Predicted",
]

# https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
# https://i.stack.imgur.com/lFZum.png
list_color="r,b,g,k,yellow,blueviolet,indianred,aqua,deeppink,slategrey".split(",")
list_optionPlot="r-,b-,g-,k-,r--,b--,g--,k--".split(",")

#########################################################################################################
#### Functions general
#######################################################################################################

def get_from_infoNN(infoNN):
    layer=infoNN[0]
    kappa=infoNN[1]
    nrEpoch=infoNN[2]
    batchSize=infoNN[3]
    if verbose:
        print("Start do NN part for","layer",layer,"kappa",str(kappa),"nrEpoch",str(nrEpoch),"batchSize",str(batchSize))
    nameNN="l_"+layer+"_k_"+str(kappa)+"_e_"+str(nrEpoch)+"_b_"+str(batchSize)
    if debug:
        print("nameNN",nameNN)
    # done if
    return nameNN,layer,kappa,nrEpoch,batchSize
# done function

# a general function to print the values and other properties of a numpy array
# use to see the values of the numpy arrays in our code for debugging and understanding the code flow
def print_nparray(option1,option2,nparray_name,nparray):
    if verbose or debug:
        print("")
        print("option1",option1,"option2",option2,nparray_name)
        print(nparray)
        print("type",type(nparray),"shape",nparray.shape,"min_value=%.3f"%np.min(nparray),"min_position=%.0f"%np.argmin(nparray),"max_value=%.3f"%np.max(nparray),"max_position=%.0f"%np.argmax(nparray))
# done function

#########################################################################################################
#### Functions for reading ROOT file
#######################################################################################################

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
        if doROOTReadTest:
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

# prepare inputs and outputs for the NN training for train and test
def get_input_and_output_for_NN(inputFileName):
    # if we run this stage, then compute and store the output in these files
    # and if we do not run this stage, we simply load from the files
    fileName_gfcat=outputFolderName+"/NN_train_output_gfcat_nparray.npy"
    fileName_gtcat=outputFolderName+"/NN_test_output_gtcat_nparray.npy"
    fileName_rf=outputFolderName+"/NN_train_input_rf_nparray.npy"
    fileName_rt=outputFolderName+"/NN_test_input_rt_nparray.npy"
    # start the if statement
    if doROOTRead:
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
        #
        # we need to further update these to prepare them to be used by the NN
        # like in the initial example, we use  keras.utils.to_categorical and the number of bins
        # for truth
        gfcat=keras.utils.to_categorical(gf,NBins)
        gtcat=keras.utils.to_categorical(gt,NBins)
        # for reco
        rf=rf.reshape(rf.shape[0],1,1)
        rt=rt.reshape(rt.shape[0],1,1)
        # write these four to files
        np.save(fileName_gfcat,gfcat)
        np.save(fileName_gtcat,gtcat)
        np.save(fileName_rf,rf)
        np.save(fileName_rt,rt)
        # done if we want to read the ROOT file again
    else:
        # else they are written to the file and we read the values from files
        gfcat=np.load(fileName_gfcat)
        gtcat=np.load(fileName_gtcat)
        rf=np.load(fileName_rf)
        rt=np.load(fileName_rt)
    # done if
    # now we have the inputs to the NN, so let's print them
    if verbose:
        print_nparray("particleLevel","train","gfcat",gfcat)
        print_nparray("particleLevel","test","gtcat",gtcat)
        print_nparray("nominal","train","rf",rf)
        print_nparray("nominal","test","rt",rt)
    # done, ready to return
    return gfcat,gtcat,rf,rt
# done function

#########################################################################################################
#### Functions for NN training and analyzing
#######################################################################################################

# definition of the model using 3-layer neural network
# https://keras.io/getting-started/sequential-model-guide/
# https://keras.io/layers/core/
def prepare_NN_model(NBins,nvar=1,layer="A",kappa=8):
    ''' Prepare KERAS-based sequential neural network with for ML unfolding. 
        Nvar defines number of inputvariables. NBins is number of truth bins. 
        kappa is an empirically tuned parameter for the intermediate layer'''
    if verbose or debug:
        print("Start prepareModel with number of variables",nvar,"NBins",NBins,"layer",layer,"kappa",kappa)
    # constructor of the model with Sequential() 
    # Read more about the options and commands available here: https://keras.io/models/sequential/
    model=keras.models.Sequential()
    # we will use three layers, with the same activation functions and kappa as from the code example
    # add the first layer, which is the input from reco, so take them as they are with a linear activation function
    model.add(keras.layers.Dense(nvar,activation='linear',input_shape=(nvar,1)))
    if debug:
        print("model",model)
    # here do we flatten the first layer
    # Flatten(): https://stackoverflow.com/questions/44176982/how-flatten-layer-works-in-keras?rq=1
    model.add(keras.layers.Flatten())
    # add the the inner layers with a relu activation function
    # the default from the initial code example with toy data was option A
    if layer=="A1":
        model.add(keras.layers.Dense(kappa*NBins**2,activation='relu'))
    elif layer=="B1":
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
    elif layer=="B2":
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
    elif layer=="B3":
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
    elif layer=="B4":
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
    elif layer=="B5":
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
    elif layer=="B10":
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
    elif layer=="C5":
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(int(0.5*kappa)*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
    elif layer=="D5":
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(2*kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
        model.add(keras.layers.Dense(kappa*NBins,activation='relu'))
    else:
        print("layer",layer,"not known. Choose A1,B1,B2,B3,B4,B5,B10,C5,D5. Will ABORT!!!")
        assert(False)
    # done if
    # add the thir layer, or the output, with a softmax activation function
    # softmax function: https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    model.add(keras.layers.Dense(NBins,activation='softmax'))
    # compile the model by choosing the optimiser and the metrics
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    # done all, so ready to return
    return model
# done function

def get_fileNameWeights(nameNN):
    fileNameWeights=outputFolderName+"/NN_model_weights_NN_"+nameNN+".hdf5"
    if debug:
        print("fileNameWeights",fileNameWeights)
    # done if
    return fileNameWeights
# done function

def get_fileNameLossAccuracyNrEpoch(nameNN):
    # loss and accuracy
    fileNameLossTrain=outputFolderName+"/NN_train_loss_NN_"+nameNN+"_nparray.npy"
    fileNameLossTest=outputFolderName+"/NN_test_loss_NN_"+nameNN+"_nparray.npy"
    fileNameAccuracyTrain=outputFolderName+"/NN_train_accuracy_NN_"+nameNN+"_nparray.npy"
    fileNameAccuracyTest=outputFolderName+"/NN_test_accuracy_NN_"+nameNN+"_nparray.npy"
    # nrEpoch
    fileNameNrEpoch=outputFolderName+"/NN_nrEpoch_NN_"+nameNN+"_nparray.npy"
    # 
    if debug:
        print("fileNameLossTrain",fileNameLossTrain)
        print("fileNameLossTest",fileNameLossTest)
        print("fileNameAccuracyTrain",fileNameAccuracyTrain)
        print("fileNameAccuracyTest",fileNameAccuracyTest)
        print("fileNameNrEpoch",fileNameNrEpoch)
    # done if
    return fileNameLossTrain,fileNameLossTest,fileNameAccuracyTrain,fileNameAccuracyTest,fileNameNrEpoch
# done function

# train the NN model
def train_NN_model(nameNN,nrEpoch,batchSize,model,gfcat,gtcat,rf,rt):
    # if we want to train, we train and store the weights to a file
    # we also store to files the numpy arrays of the loss and accuracy values for train and test
    # if we do not retrain, then we load these from the saved files to save time
    fileNameWeights=get_fileNameWeights(nameNN)
    fileNameLossTrain,fileNameLossTest,fileNameAccuracyTrain,fileNameAccuracyTest,fileNameNrEpoch=get_fileNameLossAccuracyNrEpoch(nameNN)
    # 
    if True:
        if verbose:
            print("Start train NN for",nameNN)
        # fit the model
        h=model.fit(rf,gfcat,batch_size=batchSize,epochs=nrEpoch,verbose=1,validation_data=(rt,gtcat),shuffle=False)
        if verbose:
            print("  End train NN for",nameNN)
        # the object h will remember the history of the loss and accuracy for each epoch
        # we retrieve these as numpy arrays from h and will store them in files
        # so that we can update the style of plots without having to do all the NN training again
        if debug:
            print("h.history")
            print(h.history)
            print("h.history.keys()")
            print(h.history.keys())
            print("print(h.history['val_loss'])")
            print(h.history['val_loss'],type(h.history['val_loss']))
        # losses and accuracy
        nparray_loss_train=np.array(h.history['loss'])
        nparray_loss_test=np.array(h.history['val_loss'])
        nparray_accuracy_train=np.array(h.history['acc'])
        nparray_accuracy_test=np.array(h.history['val_acc'])
        nparray_nrEpoch=np.array(range(nrEpoch))
        # store the weights of the trained NN model to a file
        model.save_weights(fileNameWeights)
        # save the numpy arrays of losses and accuracies to files
        np.save(fileNameLossTrain,nparray_loss_train)
        np.save(fileNameLossTest,nparray_loss_test)
        np.save(fileNameAccuracyTrain,nparray_accuracy_train)
        np.save(fileNameAccuracyTest,nparray_accuracy_test)
        np.save(fileNameNrEpoch,nparray_nrEpoch)
        # done if training
    #else:
    #    # if not training, so load the values from the files
    #    # load first the weights of the trained NN model into the empty NN model we have 
    #    model.load_weights(fileNameWeights)
    # done if
    # we do not need to return the model, as we passed it by argument
# done function

def analyze_NN_model(nameNN,model,gfcat,gtcat,rf,rt):
    if verbose:
        print("Start train NN for",nameNN)
    fileNameWeights=get_fileNameWeights(nameNN)
    model.load_weights(fileNameWeights)
    # now the model is loaded and it is ready to predict based on inputs
    # create a dictionary of inputs and outputs as a function of train and test
    # to loop over train and test and not write the same code twice
    dict_name_nparray={}
    dict_name_nparray["train_inputCategorical"]=rf
    dict_name_nparray["train_outputTrueCategorical"]=gfcat
    dict_name_nparray["test_inputCategorical"]=rt
    dict_name_nparray["test_outputTrueCategorical"]=gtcat
    # add the outputPredicted to the same dictionary
    for optionTrainTest in list_optionTrainTest:
        dict_name_nparray[optionTrainTest+"_outputPredictedCategorical"]=model.predict(dict_name_nparray[optionTrainTest+"_inputCategorical"])
        dict_name_nparray[optionTrainTest+"_input"]=dict_name_nparray[optionTrainTest+"_inputCategorical"].flatten()
        if debug or verbose:
            print_nparray(optionTrainTest,"none",optionTrainTest+"_inputCategorical",dict_name_nparray[optionTrainTest+"_inputCategorical"])
            print_nparray(optionTrainTest,"none",optionTrainTest+"_input",dict_name_nparray[optionTrainTest+"_input"])
        for outputType in list_outputType:
            dict_name_nparray[optionTrainTest+"_output"+outputType]=np.argmax(dict_name_nparray[optionTrainTest+"_output"+outputType+"Categorical"],axis=1)
            if debug or verbose:
                print_nparray(optionTrainTest,"none","output"+outputType+"Categorical",dict_name_nparray[optionTrainTest+"_output"+outputType+"Categorical"])
                print_nparray(optionTrainTest,"none","output"+outputType,dict_name_nparray[optionTrainTest+"_output"+outputType])
            if debug:
                print_nparray(optionTrainTest,"none","output"+outputType+"Categorical[0]",dict_name_nparray[optionTrainTest+"_output"+outputType+"Categorical"][0])
                print_nparray(optionTrainTest,"none","output"+outputType+"Categorical[1]",dict_name_nparray[optionTrainTest+"_output"+outputType+"Categorical"][1])
                print_nparray(optionTrainTest,"none","output"+outputType+"Categorical[2]",dict_name_nparray[optionTrainTest+"_output"+outputType+"Categorical"][2])
        # done for loop
    # done forl loop
    # save some of the nparrays from the dictionary to files as .npy
    for name in sorted(dict_name_nparray.keys()):
        if name.endswith("Categorical"):
            continue
        if verbose:
            print("name",name)
        outputFileName=outputFolderName+"/NN_"+name
        if "Predicted" in name:
            outputFileName+="_NN_"+nameNN
        outputFileName+="_nparray.npy"
        
        np.save(outputFileName,dict_name_nparray[name])
    # done for loop
    # nothing to return
# done function

########################################################################################################
#### Functions about plotting
#######################################################################################################

# overlay two or more numpy arrays as graphs
# info_legend="best", "uppler right", "lowerleft", etc
def overlayGraphsValues(list_tupleArray,outputFileName="overlay",extensions="pdf,png",info_x=["Procent of data reduced",[0.0,1.0],"linear"],info_y=["Figure of merit of performance",[0.0,100000.0],"log"],info_legend=["best"],title="Compared performance of 3D point cloud compression",debug=False):
    if debug:
        print("Start overlayGraphsValues")
        print("outputFileName",outputFileName)
        print("extensions",extensions)
        print("info_x",info_x)
        print("info_y",info_y)
        print("info_legend",info_legend)
        print("title",title)
    # x axis
    x_label=info_x[0]
    x_lim=info_x[1]
    x_lim_min=x_lim[0]
    x_lim_max=x_lim[1]
    if x_lim_min==-1 and x_lim_max==-1:
        x_set_lim=False
    else:
        x_set_lim=True
    x_scale=info_x[2]
    if debug:
        print("x_label",x_label,type(x_label))
        print("x_lim_min",x_lim_min,type(x_lim_min))
        print("x_lim_max",x_lim_max,type(x_lim_max))
        print("x_set_lim",x_set_lim,type(x_set_lim))
        print("x_scale",x_scale,type(x_scale))
    # y axis
    y_label=info_y[0]
    y_lim=info_y[1]
    y_lim_min=y_lim[0]
    y_lim_max=y_lim[1]
    if y_lim_min==-1 and y_lim_max==-1:
        y_set_lim=False
    else:
        y_set_lim=True
    y_scale=info_y[2]
    if debug:
        print("y_label",y_label,type(y_label))
        print("y_lim_min",y_lim_min,type(y_lim_min))
        print("y_lim_max",y_lim_max,type(y_lim_max))
        print("y_set_lim",y_set_lim,type(y_set_lim))
        print ("y_scale",y_scale,type(y_scale))
    # create empty figure
    plt.figure(1)
    # set x-axis
    plt.xlabel(x_label)
    if x_set_lim==True:
        plt.xlim(x_lim_min,x_lim_max)
    plt.xscale(x_scale)
    # set y-axis
    plt.ylabel(y_label)
    if y_set_lim==True:
        plt.ylim(y_lim_min,y_lim_max)
    plt.yscale(y_scale)
    # set title
    plt.title(title)
    # fill content of plot
    for i,tupleArray in enumerate(list_tupleArray):
        if debug:
            print("i",i,"len",len(tupleArray))
        x=tupleArray[0]
        y=tupleArray[1]
        c=tupleArray[2]
        l=tupleArray[3]
        plt.plot(x,y,c,label=l)
    # done loop over each element to plot
    # set legend
    plt.legend(loc=info_legend[0])
    # for each extension create a plot
    for extension in extensions.split(","):
        fileNameFull=outputFileName+"."+extension
        print("Saving plot at",fileNameFull)
        plt.savefig(fileNameFull)
    # close the figure
    plt.close()
# done function

# histtype: bar, barstacked, step, stepfilled
# nrBins: 100 or list of bins edges
# option A: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.hist.html
# only option A works if we want to add a text in the plot whose size is relative to the plot and not to the values plotted
# option B: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html
# to color different bins in different colors, like a rainbow gradient https://stackoverflow.com/questions/23061657/plot-histogram-with-colors-taken-from-colormap
# obtain the max value: # https://stackoverflow.com/questions/15558136/obtain-the-max-y-value-of-a-histogram
# plotting two histograms in one plt.hist did not work for me easily, but I loop over list of arrays anyway, as I need to give them different labels and colors etc

def overlay_histogram_from_nparray(list_tupleArray,outputFileName="./output_histo_from_nparray",extensions="png,pdf",nrBins=100,histtype="step",info_x=["x-axis"],info_y=["Number of points"],title="Title",text=None,info_legend=["best"],debug=False,verbose=False):
    if debug:
        print("Start draw_histogram_from_nparray()")
        print("outputFileName",outputFileName)
        print("extensions",extensions)
        print("info_x",info_x)
        print("info_y",info_y)
        print("title",title)
    # 
    max_y=np.NINF # negative infinity
    style="A"
    if style=="A":
        fig=pylab.figure()
        for i,(nparray,legendText) in enumerate(list_tupleArray):
            ax = fig.add_subplot(111)
            n,b,patches=ax.hist(nparray,bins=nrBins,alpha=1.0,color=list_color[i],histtype=histtype,label=legendText)
            if n.max()>max_y:
                max_y=n.max()
            if debug:
                print("n",n)
                print("b",b)
                print("patches",patches)
                print("max_y",max_y)
    if style=="B":
        for i,tupleArray in enumerate(list_tupleArray):
            print("tupleArray",tupleArray)
            nparray,legendText=tupleArray
            print("nparray",nparray)
            print("legendText",legendText)
            print("i",i)
            y,x,a=plt.hist(nparray,bins=nrBins,alpha=1,color=list_color[i],histtype=histtype,label=legendText)
            # note y (vertical) is returned before x (horizontal)
            print("x",type(x),x)
            print("y",type(y),y)
            #print(type(x),x,len(x),x.shape,np.min(x),np.max(x),np.sum(x)
            #print(type(y),y,len(y),y.shape,np.min(y),np.max(y),np.sum(y)
            #print(type(a),a
            if np.max(y)>max_y:
                max_y=np.max(y)
            #print_nparray("x",legendText,"x",x)
            #print_nparray("x",legendText,"y",y)
            #print_nparray("x",legendText,"a",a)
    # axes
    x_label=info_x[0]
    y_label=info_y[0]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0,max_y*1.2)
    # title
    plt.title(title)
    # text
    if text is not None:
        plt.text(0.2,0.9,text,bbox=dict(facecolor='red', alpha=0.5),horizontalalignment="left",fontstyle="oblique",transform=ax.transAxes)
    # legend
    # set legend
    plt.legend(loc=info_legend[0])
    # for each extension create a plot
    for extension in extensions.split(","):
        fileNameFull=outputFileName+"."+extension
        print("Saving plot at",fileNameFull)
        plt.savefig(fileNameFull)
    # close the figure
    plt.close()
# done function

# x=horizontal, y=vertical; nrBins=100, or nrBins=[0,1,2,3,4]
def draw_histogram_2d(x,y,outputFileName="./output_histo_2D",extensions="png,pdf",nrBins=100,info_x=["x-axis"],info_y=["y-axis"],title="Title",plotColorBar=True,debug=False,verbose=False):
    plt.hist2d(x,y,bins=nrBins)
    if plotColorBar:
        plt.colorbar()
    # axes
    x_label=info_x[0]
    y_label=info_y[0]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # title
    plt.title(title)
    # save plots
    for extension in extensions.split(","):
        plt.savefig(outputFileName+"."+extension)
    # done loop over extension
    plt.close()
# done function

def get_dict_name_nparray(list_infoNN):
    dict_name_nparray={}
    # dict_name_nparray["test"]=np.array(range(50))
    for infoNN in list_infoNN:
        nameNN,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
        # load the numpy arrays of the losses, accuracies and nrEpoch from files
        fileNameLossTrain,fileNameLossTest,fileNameAccuracyTrain,fileNameAccuracyTest,fileNameNrEpoch=get_fileNameLossAccuracyNrEpoch(nameNN)
        dict_name_nparray[nameNN+"_loss_train"]=np.load(fileNameLossTrain)
        dict_name_nparray[nameNN+"_loss_test"]=np.load(fileNameLossTest)
        dict_name_nparray[nameNN+"_accuracy_train"]=np.load(fileNameAccuracyTrain)
        dict_name_nparray[nameNN+"_accuracy_test"]=np.load(fileNameAccuracyTest)
        dict_name_nparray[nameNN+"_nrEpoch"]=np.load(fileNameNrEpoch)
    # done loop over infoNN
    if debug:
        print("dict_name_nparray")
        for name in sorted(dict_name_nparray.keys()):
            nparray=dict_name_nparray[name]
            print_nparray(name,name,name,nparray)
    # done all, ready to return
    return dict_name_nparray
# done function

# plot
def overlay_train_test(dict_name_nparray):
    # for each NN and  for each of "loss" and "accuracy", overlay train vs test
    # loop over the NN configurations
    for infoNN in list_infoNN:
        nameNN,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
        # loop over "loss" and "accuracy"
        for metric in list_metric:
            plotRange=dict_metric_plotRange[metric]
            # create the tuple of arrays to plot
            list_tupleArray=[]
            for i,optionTrainTest in enumerate(list_optionTrainTest):
                # color=list_color[i] # e.g. r-
                # optionPlot=color+"-" # e.g. r-
                optionPlot=list_optionPlot[i] # e.g. r--
                list_tupleArray.append((dict_name_nparray[nameNN+"_nrEpoch"],dict_name_nparray[nameNN+"_"+metric+"_"+optionTrainTest],optionPlot,optionTrainTest))
            # done for loop over optionTrainTest
            outputFileName=outputFolderName+"/NN_plot1D_optionTrainTest_"+metric+"_NN_"+nameNN
            overlayGraphsValues(list_tupleArray,outputFileName=outputFileName,extensions=extensions,
                                info_x=["Number of epochs",[-1,-1],"linear"],
                                info_y=["Value of the "+metric+" function",plotRange,"linear"],
                                info_legend=["best"],title="NN="+nameNN,debug=False)

        # done for loop over metric
    # done for loop over list_info
# done function


def overlay_infoNN(dict_name_nparray):
    for metric in list_metric:
        plotRange=dict_metric_plotRange[metric]
        for optionTrainTest in list_optionTrainTest:
            for listInfoToPlot in list_listInfoToPlot:
                name=listInfoToPlot[0]
                my_list_infoNN=listInfoToPlot[1]
                # create the tuple of arrays to plot
                list_tupleArray=[]
                # loop over the NN configurations
                for i,infoNN in enumerate(my_list_infoNN):
                    print("i",i)
                    nameNN,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
                    optionPlot=list_optionPlot[i] # e.g. r-
                    list_tupleArray.append((dict_name_nparray[nameNN+"_nrEpoch"],dict_name_nparray[nameNN+"_"+metric+"_"+optionTrainTest],
                                            optionPlot,nameNN))
                # done for loop over infoNN
                title="optionTrainTest="+optionTrainTest+" metric="+metric
                outputFileName=outputFolderName+"/NN_plot1D_"+optionTrainTest+"_"+metric+"_NN_"+name
                overlayGraphsValues(list_tupleArray,outputFileName=outputFileName,extensions=extensions,
                                    info_x=["Number of epochs",[-1,-1],"linear"],
                                    info_y=["Value of the "+metric+" function",plotRange,"linear"],
                                    info_legend=["best"],title=title,debug=False)
            # do for loop over listInfoToPlot
        # done for loop over metric
    # done for loop over list_info
# done function
#

def plot_outputPredictedMinusTrue():
    if debug or verbose:
        print("Start plot_outputPredictedMinusTrue()")
    nrBins=[-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5,5.5]
    for optionTrainTest in list_optionTrainTest:
        nparray_input=np.load(outputFolderName+"/NN_"+optionTrainTest+"_input_nparray.npy")
        nparray_outputTrue=np.load(outputFolderName+"/NN_"+optionTrainTest+"_outputTrue_nparray.npy")
        print("none","none","nparray_input",nparray_input)
        print("none","none","nparray_outputTrue",nparray_outputTrue)
        for listInfoToPlot in list_listInfoToPlot:
            name=listInfoToPlot[0]
            my_list_infoNN=listInfoToPlot[1]
            # create the tuple of arrays to plot
            list_tupleArray=[]
            for i,infoNN in enumerate(my_list_infoNN):
                nameNN,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
                nparray_outputPredicted=np.load(outputFolderName+"/NN_"+optionTrainTest+"_outputPredicted_NN_"+nameNN+"_nparray.npy")
                nparray_outputDiff=nparray_outputPredicted-nparray_outputTrue
                list_tupleArray.append((nparray_outputDiff,nameNN))
            # done for loop over infoNN
            title="optionTrainTest="+optionTrainTest+" NN output predicted minus true"
            outputFileName=outputFolderName+"/NN_plot1D_"+optionTrainTest+"_outputPredictedMinusTrue_NN_"+name
            overlay_histogram_from_nparray(list_tupleArray,outputFileName=outputFileName,extensions=extensions,nrBins=nrBins,histtype="step",info_x=["difference in NN output predicted minus true","linear"],info_y=["Number of jets","linear"],title=title,text=None,debug=False,verbose=False)
        # done for loop over my_list_infoNN
    # done for loop over optionTrainTest
# done function

# plot 2D matrix of input vs output (two for outputTrue and outputPredicted)
# and color code to see that it is almost diagonal
def plot_input_output():
    if debug or verbose:
        print("Start plot_input_output()")
    nrBins=np.arange(NBins)-0.5
    # 
    for optionTrainTest in list_optionTrainTest:
        list_list_dataType=[
            ["input","outputTrue"],
            ["outputTrue","input"],
            ]
        # we are now only for train or only for test
        dict_name_nparray={}
        # as we want to make the 2D plot as integers between input and output
        dict_name_nparray["input"]=np.trunc(np.load(outputFolderName+"/NN_"+optionTrainTest+"_input_nparray.npy")) # we truncate to get integers also for reco, as it is for truth
        dict_name_nparray["outputTrue"]=np.load(outputFolderName+"/NN_"+optionTrainTest+"_outputTrue_nparray.npy")
        # add each of the NNs trained
        for i,infoNN in enumerate(list_infoNN):
            nameNN,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
            dict_name_nparray["outputPredicted_NN_"+nameNN]=np.load(outputFolderName+"/NN_"+optionTrainTest+"_outputPredicted_NN_"+nameNN+"_nparray.npy")
            list_list_dataType.append(["outputTrue","outputPredicted_NN_"+nameNN])
        # done loop over NNs trained
        # now the dictionary contains the input and output of all the NNs
        #
        # we want to plot overlay in 1D of input, output_True and any combination of output_Predicted that is defined in list_listInfoToPlot
        # but we want also without no trained, so we create a deep copy of the list adn the we add one empty by hand
        my_list_listInfoToPlot=copy.deepcopy(list_listInfoToPlot)
        my_list_listInfoToPlot.append(["noTrained",[]])
        for listInfoToPlot in my_list_listInfoToPlot:
            name=listInfoToPlot[0]
            my_list_infoNN=listInfoToPlot[1]
            # create the tuple of arrays to plot
            list_tupleArray=[]
            # add the input and output_True that are to appear first
            list_tupleArray.append((dict_name_nparray["input"],"input"))
            list_tupleArray.append((dict_name_nparray["outputTrue"],"outputTrue"))
            # add our trained NNs
            for i,infoNN in enumerate(my_list_infoNN):
                nameNN,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
                list_tupleArray.append((dict_name_nparray["outputPredicted_NN_"+nameNN],nameNN))
            # done for loop over the trained NNs
            # make the plot of overlay of histograms
            title="optionTrainTest="+optionTrainTest
            outputFileName=outputFolderName+"/NN_plot1D_"+optionTrainTest+"_jetPtBin_NN_"+name
            overlay_histogram_from_nparray(list_tupleArray,outputFileName=outputFileName,extensions=extensions,nrBins=nrBins,histtype="step",info_x=["nr of bins of jet for "+name],info_y=['Number of jets'],title=title,text=None,info_legend=["best"],debug=False,verbose=False)
        # done loop over the listInfoToPlot
        #
        # now we want to plot 2D histograms of each of input with the various outputs
        # nrBins=np.arange(20)
        for list_dataType in list_list_dataType:
            name_horizontal=list_dataType[0]
            name_vertical=list_dataType[1]
            if debug:
                print_nparray(optionTrainTest,"horizontal",name_horizontal,dict_name_nparray[name_horizontal])
                print_nparray(optionTrainTest,"vertical",name_vertical,dict_name_nparray[name_vertical])
            title="optionTrainTest="+optionTrainTest+" "+name_vertical+" vs "+name_horizontal
            outputFileName=outputFolderName+"/NN_plot2D_"+optionTrainTest+"_"+name_horizontal+"_"+name_vertical
            draw_histogram_2d(dict_name_nparray[name_horizontal],dict_name_nparray[name_vertical],outputFileName=outputFileName,extensions=extensions,nrBins=nrBins,info_x=[name_horizontal],info_y=[name_vertical],title=title,plotColorBar=True,debug=debug,verbose=verbose)
        # done loop over list_dataType
    # done loop over optionTrainTest
# done function
            
########################################################################################################
#### Function doAll() putting all together
#######################################################################################################

# the function that runs everything
def doItAll():
    gfcat,gtcat,rf,rt=get_input_and_output_for_NN(inputFileName)    
    # loop over different NN that we compare (arhitecture and learning)
    for infoNN in list_infoNN:
        nameNN,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
        if doNNTrain or doNNAnalyze:
            # create empty train model architecture (bad initial weights)
            model=prepare_NN_model(NBins,nvar=1,layer=layer,kappa=kappa)
        if doNNTrain:
            # train the NN
            train_NN_model(nameNN,nrEpoch,batchSize,model,gfcat,gtcat,rf,rt)
            # now the model contains the trained data (with the good weights)
        if doNNAnalyze:
            # load the weights of the NN and make prediction of the NN
            analyze_NN_model(nameNN,model,gfcat,gtcat,rf,rt)
    # done for loop over infoNN
    if doPlot:
        if doPlotMetrics:
            # plot metrics of NN training (loss, accuracy)
            dict_name_nparray=get_dict_name_nparray(list_infoNN)
            overlay_train_test(dict_name_nparray)
            overlay_infoNN(dict_name_nparray)
        if doPlotOutput1D:
            # plot output of the NN training (predicted vs true)
            plot_outputPredictedMinusTrue()
        if doPlotInputOutput2D:
            # plot 2D matrix of input vs output (two for outputTrue and outputPredicted)
            # and color code to see that it is almost diagonal
            plot_input_output()
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
