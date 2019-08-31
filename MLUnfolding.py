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

#string_stage="1111"
#string_stage="0000"
#string_stage="1000"
#string_stage="0100"
#string_stage="0010"
#string_stage="0001"
string_stage="0101"

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
outputFolderName="./output6"

# for DNN training with Keras
doTrainNN=True
# the order is kappa (from architecture), epochs, batchSize, (from learning steps)
list_list_option=[
    #
    #[8,200,1000],
    #[4,200,1000],
    #[2,200,1000],
    #[2,200,1000],
    #[1,200,1000],
    #
    #[8,1000,1000],
    #[4,1000,1000],
    #[2,1000,1000],
    #[1,1000,1000],
    #
    #[8,5000,1000],
    #[4,5000,1000],
    #[2,5000,1000],
    #[1,5000,1000],
    #
]

# for test
if False:
    list_list_option=[
        [8,3,1000],
        #[8,50,1000],
        #[8,1000,1000],
    ]
# done if

#########################################################################################################
#### Functions general
#######################################################################################################

def get_fileNameStem(kappa,nrEpoch,batchSize):
    fileNameStem="kappa_"+str(kappa)+"_nrEpoch_"+str(nrEpoch)+"_batchSize_"+str(batchSize)
    if debug:
        print("fileNameStem",fileNameStem)
    # done if
    return fileNameStem
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
def prepare_NN_model(NBins,nvar=1,kappa=8):
    ''' Prepare KERAS-based sequential neural network with for ML unfolding. 
        Nvar defines number of inputvariables. NBins is number of truth bins. 
        kappa is an empirically tuned parameter for the intermediate layer'''
    if verbose or debug:
        print("Start prepareModel with number of variables",nvar,"NBins",NBins,"kappa",kappa)
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
    # add the second layer with a relu activation function
    model.add(keras.layers.Dense(kappa*NBins**2,activation='relu'))
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
        # fit the model
        h=model.fit(rf,gfcat,batch_size=batchSize,epochs=nrEpoch,verbose=1,validation_data=(rt,gtcat))
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

# plot
def plot_loss_accuracy_versus_nrEpoch(fileNameStem):
    # load the numpy arrays of the losses, accuracies and nrEpoch from files
    fileNameLossTrain,fileNameLossTest,fileNameAccuracyTrain,fileNameAccuracyTest,fileNameNrEpoch=get_fileNameLossAccuracyNrEpoch(fileNameStem)
    nparray_loss_train=np.load(fileNameLossTrain)
    nparray_loss_test=np.load(fileNameLossTest)
    nparray_accuracy_train=np.load(fileNameAccuracyTrain)
    nparray_accuracy_test=np.load(fileNameAccuracyTest)
    nparray_nrEpoch=np.load(fileNameNrEpoch)
    # 
    if verbose:
        print_nparray("loss","train","nparray_loss_train",nparray_loss_train)
        print_nparray("loss","test","nparray_loss_test",nparray_loss_test)
        print_nparray("accuracy","train","nparray_accuracy_train",nparray_accuracy_train)
        print_nparray("accuracy","test","nparray_accuracy_test",nparray_accuracy_test)
        print_nparray("nrEpoch","nrEpoch","nparray_nrEpoch",nparray_nrEpoch)
    #
    # do plot for loss
    list_tupleArray=[]
    list_tupleArray.append((nparray_nrEpoch,nparray_loss_train,"r-","train"))
    list_tupleArray.append((nparray_nrEpoch,nparray_loss_test,"b-","test"))
    outputFileName=outputFolderName+"/NN_plot_"+fileNameStem+"_loss"
    overlayGraphsValues(list_tupleArray,outputFileName=outputFileName,extensions="pdf,png",
                        info_x=["Number of epochs",[-1,-1],"linear"],
                        info_y=["Value of the loss function",[-1,-1],"linear"],
                        info_legend=["best"],title="Loss vs nrEpoch for NN "+fileNameStem,debug=False)
    #
    # do plot for accuracy
    list_tupleArray=[]
    list_tupleArray.append((nparray_nrEpoch,nparray_accuracy_train,"r-","train"))
    list_tupleArray.append((nparray_nrEpoch,nparray_accuracy_test,"b-","test"))
    outputFileName=outputFolderName+"/NN_plot_"+fileNameStem+"_accuracy"
    overlayGraphsValues(list_tupleArray,outputFileName=outputFileName,extensions="pdf,png",
                        info_x=["Number of epochs",[-1,-1],"linear"],
                        info_y=["Value of the accuracy function",[-1,-1],"linear"],
                        info_legend=["best"],title="Accuracy vs nrEpoch for NN "+fileNameStem,debug=False)
# done function


########################################################################################################
#### Function doAll() putting all together
#######################################################################################################

# the function that runs everything
def doItAll():
    gfcat,gtcat,rf,rt=get_input_and_output_for_NN(inputFileName)    
    # train and study several neural networks
    # different architectures coming from different kappa
    # different learning steps coming from different batchSize and number of nrEpoch
    # by doing a for loop over different options
    for list_option in list_list_option:
        kappa=list_option[0]
        nrEpoch=list_option[1]
        batchSize=list_option[2]
        if verbose:
            print("Start do NN part for","kappa",str(kappa),"nrEpoch",str(nrEpoch),"batchSize",str(batchSize))
        fileNameStem=get_fileNameStem(kappa,nrEpoch,batchSize)
        if doNNTrain or doNNAnalyze:
            # create the untrained deep neural network (DNN) model
            # architecture coming from the layers that appear with .add()
            # learning methods from .compile()
            model=prepare_NN_model(NBins,nvar=1,kappa=kappa)
            # get the trained DNN model either by training or by loading the trained model
            # we take the previous model as input and the function will change it to have the weights after training
            train_NN_model(fileNameStem,nrEpoch,batchSize,model,gfcat,gtcat,rf,rt)
            # now the model contains the trained data
        if doPlot:
            plot_loss_accuracy_versus_nrEpoch(fileNameStem)
        continue
    # done for loop over list_option
# done function

def bla():
    if True:
        # modelFileName,weightsFileName,figFileName,fig2FileName
        fileNameStem=get_fileNameStem(outputFolderName,batch_size,epochs,kappa)
        fileNameModel=outputFolderName+"/model_full_"+fileNameStem+".hdf5"
        weightsFileName=outputFolderName+"/model_weights_"+fileNameStem+".hdf5"
        # prepare the model of the DNN with Keras

        # let's train the NN only once and then save the weights to a file
        # if already trained, then we can simply read the weights and continue working from there
        # that way it is faster
        if doTrainNN:

            print("start plot")
            # Turn interactive plotting off
            plt.ioff()
            #
            fig=plt.figure()
            plt.plot(xc, train_loss,"r","train_loss")
            plt.plot(xc, val_loss,"b","val_loss")
            plt.savefig(figFileName)
            plt.close(fig)
            #
            fig=plt.figure()
            plt.plot(xc, train_acc,"r","train_acc")
            plt.plot(xc, val_acc,"b","val_acc")
            plt.savefig(fig2FileName)
            plt.close(fig)
            print("end plot")
            # save model to not spend the time every time to retrain, as it takes some time
            # a simple save
            model.save(modelFileName)
            # save also only the weights
            model.save_weights(weightsFileName)
        # done if
        out=model.predict(rt)
        # read as model2 with the same structure as before, but read the weights
        model2=prepareModel(NBins,nvar=1,kappa=kappa)
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
                indices_gtcat=np.where(gtcat_current==gtcat_current.max())
                index=np.argmax(gtcat_current)
                print("gtcat indices",indices_gtcat,"index",index)
                gtcat_max=0.0
                for j in range(gtcat_current.shape[0]):
                    # if True:
                    if gtcat_current[j]>0.1:
                        print(i,j,"gtcat[i][j]",gtcat_current[j])
                    if gtcat_current[j]>gtcat_max:
                        gtcat_max=gtcat_current[j]
                # done for loop over j
                print("gtcat_max",gtcat_max)
                indices_out=np.where(out_current==out_current.max())
                index=np.argmax(out_current)
                print("out indices",indices_out,"index",index)
                out_max=0.0
                for j in range(out_current.shape[0]):
                    # if True:
                    if out_current[j]>0.1:
                        print(i,j,"out[i][j]",out_current[j])
                    if out_current[j]>out_max:
                        out_max=out_current[j]
                    # done if
                # done for loop over j
                print("out_max",out_max)
                #
                indices_out2=np.where(out2_current==out2_current.max())
                index=np.argmax(out2_current)
                print("out2 indices",indices_out2,"index",index)
                out2_max=0.0
                for j in range(out2_current.shape[0]):
                    # if True:
                    if out2_current[j]>0.1:
                        print(i,j,"out2[i][j]",out2_current[j])
                    if out2_current[j]>out2_max:
                        out2_max=out2_current[j]
                    # done if
                # done for loop over j
                print("out2_max",out2_max)
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
