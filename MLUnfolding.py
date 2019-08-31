#!/usr/bin/env python

#########################################################################################################
#### define the stages
#########################################################################################################

# exit()

#########################################################################################################
#### import statements
#########################################################################################################

# import from basic Python to be able to read automatically the name of a file
import sys

# import to use numpy arrays
import numpy as np

# plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#########################################################################################################
#### configuration options
#########################################################################################################

debug=False
verbose=True

# we split the code into stages, so that we can run only the last stage for instance
# the output of each stage is stored to files, that are read back at the next stage
# stage 1: read .root file and produce numpy arrays that are input and output to the NN training
# stage 1: input: .root file; output: rf, rt, gfcat, gtcat
# stage 2: use the rf, rt, gfcat, gtcat to produce a NN training and store the model weights to a file
# stage 3: use the NN to do further studies - not yet done
# stage 4: make plots of the NN training - done
#string_stage="1111"
#string_stage="0000"
#string_stage="1000"
#string_stage="0100"
#string_stage="0010"
#string_stage="0001"
string_stage="1101"
#string_stage="0101"

list_stage=list(string_stage)
doROOTRead=bool(int(list_stage[0]))
doNNTrain=bool(int(list_stage[1]))
doNNAnalyze=bool(int(list_stage[2]))
doPlot=bool(int(list_stage[3]))

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

# output
outputFolderName="./output8"
# extensions="pdf,png"
extensions="png"

# for DNN training with Keras
# the order is layer and kappa (from architecture), epochs, batchSize (from learning steps)
if True:
    list_infoNN=[
        ["A1",8,300,1000],
        #["B1",8,300,1000],
        #["B2",8,300,1000],
        #["B3",8,300,1000],
        #["B4",8,300,1000],
        ["B5",8,300,1000],
        #["B10",8,300,1000],
        #["C5",8,300,1000],
        #["D5",8,300,1000],
]

list_listInfoToPlot=[
    #["A1_B1_B2_B3_B4_B5_B10",[ ["A1",8,300,1000],["B1",8,300,1000],["B2",8,300,1000],["B3",8,300,1000],["B4",8,300,1000],["B5",8,300,1000],["B10",8,300,1000] ]],
    #["B1_B2_B3_B4_B5_B10",[ ["B1",8,300,1000],["B2",8,300,1000],["B3",8,300,1000],["B4",8,300,1000],["B5",8,300,1000],["B10",8,300,1000] ]],
    #["B1_B2_B3_B4_B5",[ ["B1",8,300,1000],["B2",8,300,1000],["B3",8,300,1000],["B4",8,300,1000],["B5",8,300,1000] ]],
    #["B5_C5_D5",[ ["B5",8,300,1000],["C5",8,300,1000],["D5",8,300,1000] ]],
    ["A1_B5",[ ["A1",8,300,1000],["B5",8,300,1000] ]],
]


list_metric=[
    "loss",
    "accuracy",
]

dict_metric_plotRange={
    "loss":[1.70,3.8],
    "accuracy":[0.05,0.45]
}

list_optionTrainTest=[
    "train",
    "test",
]

# https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
# https://i.stack.imgur.com/lFZum.png
list_color="r-,b-,g-,k-,r--,b--,g--,k--".split(",")

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
    fileNameStem="layer_"+layer+"_kappa_"+str(kappa)+"_nrEpoch_"+str(nrEpoch)+"_batchSize_"+str(batchSize)
    if debug:
        print("fileNameStem",fileNameStem)
    # done if
    return fileNameStem,layer,kappa,nrEpoch,batchSize
# done function

# a general function to print the values and other properties of a numpy array
# use to see the values of the numpy arrays in our code for debugging and understanding the code flow
def print_nparray(option1,option2,nparray_name,nparray):
    if verbose or debug:
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
    fileName_gfcat=outputFolderName+"/NN_output_train_nparray_gfcat.npy"
    fileName_gtcat=outputFolderName+"/NN_output_test_nparray_gtcat.npy"
    fileName_rf=outputFolderName+"/NN_input_train_nparray_rf.npy"
    fileName_rt=outputFolderName+"/NN_input_test_nparray_rt.npy"
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

def get_fileNameWeights(fileNameStem):
    fileNameWeights=outputFolderName+"/NN_model_"+fileNameStem+"_weights.hdf5"
    if debug:
        print("fileNameWeights",fileNameWeights)
    # done if
    return fileNameWeights
# done function

def get_fileNameLossAccuracyNrEpoch(fileNameStem):
    # loss and accuracy
    fileNameLossTrain=outputFolderName+"/NN_model_"+fileNameStem+"_train_nparray_loss.npy"
    fileNameLossTest=outputFolderName+"/NN_model_"+fileNameStem+"_test_nparray_loss.npy"
    fileNameAccuracyTrain=outputFolderName+"/NN_model_"+fileNameStem+"_train_nparray_accuracy.npy"
    fileNameAccuracyTest=outputFolderName+"/NN_model_"+fileNameStem+"_test_nparray_accuracy.npy"
    # nrEpoch
    fileNameNrEpoch=outputFolderName+"/NN_model_"+fileNameStem+"_nparray_nrEpoch.npy"
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
def train_NN_model(fileNameStem,nrEpoch,batchSize,model,gfcat,gtcat,rf,rt):
    # if we want to train, we train and store the weights to a file
    # we also store to files the numpy arrays of the loss and accuracy values for train and test
    # if we do not retrain, then we load these from the saved files to save time
    fileNameWeights=get_fileNameWeights(fileNameStem)
    fileNameLossTrain,fileNameLossTest,fileNameAccuracyTrain,fileNameAccuracyTest,fileNameNrEpoch=get_fileNameLossAccuracyNrEpoch(fileNameStem)
    # 
    if doNNTrain:
        if verbose:
            print("Start train NN for",fileNameStem)
        # fit the model
        h=model.fit(rf,gfcat,batch_size=batchSize,epochs=nrEpoch,verbose=1,validation_data=(rt,gtcat))
        if verbose:
            print("  End train NN for",fileNameStem)
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


def get_dict_name_nparray(list_infoNN):
    dict_name_nparray={}
    # dict_name_nparray["test"]=np.array(range(50))
    for infoNN in list_infoNN:
        fileNameStem,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
        # load the numpy arrays of the losses, accuracies and nrEpoch from files
        fileNameLossTrain,fileNameLossTest,fileNameAccuracyTrain,fileNameAccuracyTest,fileNameNrEpoch=get_fileNameLossAccuracyNrEpoch(fileNameStem)
        dict_name_nparray[fileNameStem+"_loss_train"]=np.load(fileNameLossTrain)
        dict_name_nparray[fileNameStem+"_loss_test"]=np.load(fileNameLossTest)
        dict_name_nparray[fileNameStem+"_accuracy_train"]=np.load(fileNameAccuracyTrain)
        dict_name_nparray[fileNameStem+"_accuracy_test"]=np.load(fileNameAccuracyTest)
        dict_name_nparray[fileNameStem+"_nrEpoch"]=np.load(fileNameNrEpoch)
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
        fileNameStem,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
        # loop over "loss" and "accuracy"
        for metric in list_metric:
            plotRange=dict_metric_plotRange[metric]
            # create the tuple of arrays to plot
            list_tupleArray=[]
            for i,optionTrainTest in enumerate(list_optionTrainTest):
                color=list_color[i] # e.g. r-
                # optionPlot=color+"-" # e.g. r-
                optionPlot=color # e.g. r--
                list_tupleArray.append((dict_name_nparray[fileNameStem+"_nrEpoch"],dict_name_nparray[fileNameStem+"_"+metric+"_"+optionTrainTest],optionPlot,optionTrainTest))
            # done for loop over optionTrainTest
            outputFileName=outputFolderName+"/NN_plot_"+fileNameStem+"_"+metric
            overlayGraphsValues(list_tupleArray,outputFileName=outputFileName,extensions=extensions,
                                info_x=["Number of epochs",[-1,-1],"linear"],
                                info_y=["Value of the "+metric+" function",plotRange,"linear"],
                                info_legend=["best"],title="NN="+fileNameStem,debug=False)

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
                    fileNameStem,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
                    color=list_color[i] # e.g. r-
                    optionPlot=color # e.g. r-
                    list_tupleArray.append((dict_name_nparray[fileNameStem+"_nrEpoch"],dict_name_nparray[fileNameStem+"_"+metric+"_"+optionTrainTest],
                                            optionPlot,fileNameStem))
                # done for loop over infoNN
                outputFileName=outputFolderName+"/NN_plot_"+metric+"_"+optionTrainTest+"_NN_"+name
                overlayGraphsValues(list_tupleArray,outputFileName=outputFileName,extensions=extensions,
                                    info_x=["Number of epochs",[-1,-1],"linear"],
                                    info_y=["Value of the "+metric+" function",plotRange,"linear"],
                                    info_legend=["best"],title="optionTrainTest="+optionTrainTest+" NN="+name,debug=False)
            # do for loop over listInfoToPlot
        # done for loop over metric
    # done for loop over list_info
# done function
#

########################################################################################################
#### Function doAll() putting all together
#######################################################################################################

# the function that runs everything
def doItAll():
    gfcat,gtcat,rf,rt=get_input_and_output_for_NN(inputFileName)    
    # loop over different NN that we compare (arhitecture and learning)
    for infoNN in list_infoNN:
        fileNameStem,layer,kappa,nrEpoch,batchSize=get_from_infoNN(infoNN)
        if doNNTrain or doNNAnalyze:
            # create empty train model architecture (bad initial weights)
            model=prepare_NN_model(NBins,nvar=1,layer=layer,kappa=kappa)
            # train the NN
            train_NN_model(fileNameStem,nrEpoch,batchSize,model,gfcat,gtcat,rf,rt)
            # now the model contains the trained data (with the good weights)
    # done for loop over infoNN
    if doPlot:
        dict_name_nparray=get_dict_name_nparray(list_infoNN)
        overlay_train_test(dict_name_nparray)
        overlay_infoNN(dict_name_nparray)
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
