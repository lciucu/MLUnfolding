#!/usr/bin/env python

from __future__ import print_function

# Basic imports for the MLUnfolding example. The code depends on scipy, numpy and keras
import matplotlib.pyplot as plt
import scipy.integrate 
import numpy as np

# KERAS imports:
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras import backend as K

# Set basic parameters used for the followng examples
batch_size = 1000
num_classes = 10
NBins = num_classes
epochs = 50
smear = 0.5
#
# Define initial function:
#
fun = lambda x : 1

# A sample function used for testing. The function has a minimum at x = 0 and x ~ 0.6 which creates extra difficulties for unfolding due to smearing into these regions.
def ff(x):
    ''' Define function for testing. '''
    return np.sin(5*x)**2

#
# Plot the result
#
xi = np.linspace(0.,1.,100)
plt.plot(xi,ff(xi))

# Simple event generator to produce test (pseudo-data) and training samples. The sampling is based on the input function given by parameter fun where fun can be constant (flat prior) or any other function. The input function is modified during the unfolding from iteration to iteration.
def gen(fun,N=200000,bins=10, xmin=0., xmax=1.0,smear=1., supersample=50, smear2=0.25, shift1=0., shift2=-0.5):
    ''' Generate random events following function fun.
        A samples of N events is generated.  Truth is distributed in number of bins 'bins'.
        The function is varied between xmin and xmax which maped to bins between bin=0 and 'bins'.
        The response for the first variable is modeled by a single gaussian with the width given by 'smear'
        supersample parameter controls how many bins inside each truth-bin are sampled.
        The second variable uses log-normal distribution with smearing parameter of 
        The return parameters are binned truth and smeared reconstructed events.
    '''
    
    ifun = lambda x: scipy.integrate.quad(fun,0., x/xmax)[0]
    vfun = np.vectorize(ifun)

    NSample = bins*supersample  # finer sampling
    vals = vfun(np.linspace(xmin, xmax, NSample+1))
    probs = vals[1:]-vals[:-1]# want to param 
    probs = np.where(probs<0,0,probs)
    probs = probs/np.sum(probs)
    a = np.random.choice(NSample,N,p=probs)  # simulated events.
    asim = np.trunc(a/supersample)                   # output simulated bins

    # Reconstructed vars:
    # apply additional gaussian smearing
    g = np.random.normal(0.,smear,N) + shift1
    n = np.random.lognormal(0.,0.25,N)
    return asim,(a+(g)*supersample)/supersample,(a*n+shift2)/supersample

# Generate reference and two training samples
smear = 0.5
g,r,r2= gen(lambda x: ff(x),2000000,smear=smear)
gt,rt,rt2 = gen(lambda x: ff(x),20000,smear=smear)
gf,rf,rf2 = gen(lambda x: fun(x),2000000,smear=smear)

# Definition of the model using 3-layer neural network.
def prepareModel(nvar=1, NBins=NBins, kappa=8):
    ''' Prepare KERAS-based sequential neural network with for ML unfolding. 
        Nvar defines number of inputvariables. NBins is number of truth bins. 
        kappa is an empirically tuned parameter for the intermediate layer'''
    model = Sequential()
    model.add(Dense(nvar,activation='linear',input_shape=(nvar,1)))
    model.add(Flatten())
    model.add(Dense(kappa*NBins**2,activation='relu'))
    
    # model.add(Dropout(0.25))
    # model.add(Dense(2*NBins,activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(NBins,activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return model

# Prepare the model with one input variable
model = prepareModel(1)

# Prepare inputs for keras
gcat = keras.utils.to_categorical(g,NBins)
gtcat = keras.utils.to_categorical(gt,NBins)
gfcat = keras.utils.to_categorical(gf)

r = r.reshape(r.shape[0],1,1)
rt = rt.reshape(rt.shape[0],1,1)
rf = rf.reshape(rf.shape[0],1,1)

# Fit the model
h = model.fit(rf,gfcat,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(rt,gtcat))

# save model,if needed
model.save("model.hdf5")

# Prepare bootstrap replica to determine statistical uncertainties.
def prepareBootstrap(rt,N=10):
    ''' Prepare bootstrap weights, for error statistical and correlation estimation '''
    Nev = rt.shape[0]
    p = np.random.poisson(size=(Nev,N))
    return p

# Unfold the reference sample, prepare bootstrap replica of the result to estimate stat. uncertainties.
rt = rt.reshape(rt.shape[0],1,1)
out = model.predict(rt)

# get bootstrap errors:
bs = prepareBootstrap(out)
out2 = out[:,np.newaxis,:]*bs[:,:,np.newaxis]

ouproj = np.sum(out,axis=0)      # projected 
oustd  = np.std(np.sum(out2,axis=0),axis=0)
geproj = np.sum(gtcat,axis=0)

# The following cell defines a set of functions to parameterise the unfolded results with cubic splines The integral of the spline for each bin is required to be equal to the number of unfolded events in the bin
#
# Interpolation using integrated in bin splines
#
from scipy import optimize
from scipy import interpolate

#
# Some global parameters, needed for the spline interpolation:
#
xvals2 = np.zeros(12)
binb  = np.linspace(0.,10.,11)
afi = np.zeros(12)

def param(x):
    ''' Spline parameterisation'''
    tck,u = interpolate.splprep([x],u=xvals2)
    return tck

def funpar(x,tck):
    return interpolate.splev(x,tck)[0]

def funcfit(x):
    ''' Compare integral of the spline to the values given in bins. Uses global binb, xvals2, afi arrays '''
    tck,u = interpolate.splprep([x],u=xvals2)
    y = np.zeros(12)
    for i in range(10):
        y[i+1] = interpolate.splint(binb[i],binb[i+1],tck)[0]
#        y[i] = interpolate.splev(xvals[i],tck)[0]
    return afi-y

def getparam(vals):
    ''' Determine spline parameters such that integral for each bin equal to vals '''
    xvals = np.linspace(0.5,9.5,10)
    
#    afi = zeros(12)
#    xvals2 = zeros(12)
    afi[1:-1] = vals
    afi[-1] = vals[-1]
    xvals2[1:-1] = xvals
    xvals2[-1] = 10.
    u = optimize.least_squares(funcfit,afi) # ,args=(xvals2))
    tck = param(u.x)
    return tck

# More simple interpolation, splines drawn directly through unfolded cross sections.
from scipy.interpolate import interp1d
def GetInterpFuncReplica(unfRep):
    ''' Get interpolation function for a given unfolded result '''
    return interp1d(linspace(0.05,0.95,NBins),unfRep,fill_value='extrapolate',kind='cubic')

# Modify the sampling function fun, plot truth and unfolded results.
ou = np.sum(out,axis=0)

# 
useIntSpl = True
if useIntSpl:
    tck = getparam(ou)
    fun = lambda x : funpar(10*x,tck)
else:
    print (ou)
    fun = GetInterpFuncReplica(ou)
    
plt.figure()
plt.grid()

x = np.linspace(0.05,0.95,10)
plt.plot(x,fun(x),label='Next prior function')
plt.plot(x,ou,label='Unfolded result')
plt.plot(x,np.sum(gtcat,axis=0), label='Truth')
plt.legend()

# Following cells define iterative procedure
def ZeroIteration(nvar=2, NBs = 10):
    ''' Start unfolding. Prepare test/flat prior sampes. NBs defines number of bootstrap replica used for 
        stat. errors '''
    
    # generate events:
    g,r,r2= gen(lambda x: np.sin(5*x)**2,500000,smear=smear)
    gt,rt,rt2 = gen(lambda x: np.sin(5*x)**2,20000,smear=smear)  # test 
    gf,rf,rf2 = gen(lambda x: 1,500000,smear=smear)           # flat prior

    # prepare categorical representation:
    gcat = keras.utils.to_categorical(g,NBins)
    gtcat = keras.utils.to_categorical(gt,NBins)
    gfcat = keras.utils.to_categorical(gf)
    
    # prepare input:
    if nvar == 2:
        r  = transpose(array([r.reshape(r.shape[0]),r2]))
        rt = transpose(array([rt.reshape(rt.shape[0]),rt2]))
        rf = transpose(array([rf.reshape(rf.shape[0]),rf2]))
    r = r.reshape(r.shape[0],nvar,1)
    rt = rt.reshape(rt.shape[0],nvar,1)
    rf = rf.reshape(rf.shape[0],nvar,1)
    
    model = prepareModel(nvar)
    # fit
    model.fit(rf,gfcat,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(rt,gtcat))
         
    # prediction for test sample:
    out = model.predict(rt)
    
    # add bootstrap replica for it:
    bs = prepareBootstrap(out,NBs)
    out2 = out[:,np.newaxis,:]*bs[:,:,np.newaxis]
    return out2,rt,gtcat,r,gcat

def UnfoldingIteration(inpred,rt, gtcat, NBs=0, nvar=2, useIntSplines = True):
    ''' Perform unfolding iteration, including running over bootstrap replica. 
        Returns updated predictions '''
    pr = np.sum(inpred,axis=0)         # previous iteration results, for all Bs replica
    if NBs == 0:
        NBs = pr.shape[0]           # determine number of bootstrap replica automatically.
        
    out = np.zeros(inpred.shape)

    for i in range(NBs):        
        # two ways to get the function
        if useIntSplines:
            tck = getparam(pr[i])
            fun = lambda x : funpar(10*x,tck)
        else:
            fun = GetInterpFuncReplica(pr[i])
        gf,rf,rf2 = gen(lambda x: fun(x),500000,smear=smear)      # generate events  
        gfcat = keras.utils.to_categorical(gf)
        if nvar == 2:
            rf = transpose(array([rf.reshape(rf.shape[0]),rf2]))
        rf = rf.reshape(rf.shape[0],nvar,1)
        
        model = prepareModel(nvar)
        model.fit(rf,gfcat,batch_size=batch_size,epochs=epochs,verbose=0,validation_data=(rt,gtcat))
            
        out[:,i,:] = model.predict(rt)

    return out
            
# oo = UnfoldingIteration(out2,nvar=1)
# Sample unfolding sequence.

import os

NBootStrap = 1
NVar = 1

out,rt,gtcat,r,gcat = ZeroIteration(NVar,NBootStrap)

outDir = 'output'
os.system("mkdir -p "+outDir)
rt.tofile(outDir+"/rt.dat",sep=" ")
gtcat.tofile(outDir+"/gtcat.dat",sep=" ")
out.tofile(outDir+"/out0.dat")

for i in range(20):
    print ("Iteration",i)
    out = UnfoldingIteration(out,rt,gtcat, nvar=NVar)
    out.tofile(outDir+"/out"+str(i+1)+".dat")

# Plot results of the unfolding
outDir = "output/"

nit = 19
c = np.zeros((nit,10))
e = np.zeros((nit,10))

gtcat = np.fromfile(outDir+"gtcat.dat",sep=" ").reshape(20000,10)

#.reshape(20000,10)
#print (gcat[0:10])
for i in range(nit):
    out = np.fromfile(outDir+"/out"+str(i)+".dat").reshape(20000,1,10)
    c[i,:] = np.mean(np.sum(out,axis=0),axis=0)
    e[i,:] = np.std(np.sum(out,axis=0),axis=0)
plt.figure(figsize=(14,5))
#subplot(121)

x = np.linspace(0.5,9.5,10)
for i in range(0,19,2):
    plt.errorbar(x,c[i],e[i],label=str(i)+'iteration')
plt.plot(x,np.sum(gtcat,axis=0),label='truth')

plt.legend()

# Ratio plot
plt.figure(figsize=(14,5))
plt.subplot(121)
plt.grid()
x = np.linspace(0.5,9.5,10)
plt.ylim(0.8,2.4)
for i in range(0,15,2):
    plt.errorbar(x,c[i]/np.sum(gtcat,axis=0),e[i]/np.sum(gtcat,axis=0),label='Iter. {:d}'.format(i))
plt.xlabel("Bin number",size=15)
plt.ylabel("Unfolded/Generated",size=15)
plt.legend()
plt.subplot(122)
plt.grid()

plt.ylim(0.8,1.2)
for i in range(10,19,1):
    plt.errorbar(x,c[i]/np.sum(gtcat,axis=0),e[i]/np.sum(gtcat,axis=0),label='Iter. {:d}'.format(i))
#errorbar(x,c[0]/sum(gtcat,axis=0),e[0]/sum(gtcat,axis=0),label='zero')
plt.legend(ncol=4)
plt.xlabel("Bin number",size=15)
plt.ylabel("Unfolded/Generated",size=15)
plt.savefig("convergence.pdf")

# Extra functions to calculate conventional transfer matrix
def transferM(g,r):
    ''' Compute transfer matrx, for standard unfolding approaches '''
    tr = np.dot(np.transpose(r),g)/np.sum(g,axis=0)
    return tr

# 
# Prepare reco which is within 0-Nbins boundaries, get transfer / and inverted transfer matrix
# 
re1 = np.where(r>0, r,0 )
re2 = np.where(r>NBins-1,NBins-1,re1)
rcat = keras.utils.to_categorical(re2,NBins)
tr = transferM(gcat,rcat)
tri = np.linalg.inv(tr)

#  
#  Test that transfer matrix "works"
#
print ("Truth:",np.sum(gcat,axis=0))
print ("Reco:",np.sum(rcat,axis=0))
rr = np.sum(np.dot(tr,np.transpose(gcat)),axis=1)
gg = np.sum(np.dot(tri,np.transpose(rcat)),axis=1)
print ("Reco computed from truth by using transfer matrix:",rr)
print ("Truth computed from reco using inverse transfer matrix",gg)

def plot_rg(g,r,tr,plot2=0,r2=0,plotUnf=0,unfCent=0,unfStd=0,plotUnf2=0,unfCent2=0,unfStd2=0):
    ''' Helper function to produce plots of truth, reco and ML unfolded distributions '''
    plt.figure(figsize=(14.,5.))
    plt.subplot(121)
    plt.xlim(-1.,11.)
    ge = plt.hist(g.reshape(g.shape[0]),10,(0.,10),label='gen')
    re = plt.hist(r.reshape(r.shape[0]),13,(-2.,11.),alpha=0.8,label='rec')
    if plot2 != 0:
        re2 = plt.hist(r2,13,(-2.,11.),alpha=0.4,label='rec 2')
    if plotUnf != 0:
        xc = np.arange(0.5,10.5,1)
        xe = 0.5*np.ones(10)
        plt.errorbar(xc,unfCent,unfStd,xe,'s',color='g',label='unf 0it')
        
    if plotUnf2 != 0:
        plt.errorbar(xc,unfCent2,unfStd2,xe,'o',color='r',label='unf 3it')
    plt.xlabel("Bin number")
    plt.legend()
    plt.subplot(122)
    plt.imshow (tr,origin='lower')
    plt.xlabel("$x_g$")
    plt.ylabel('$x_r$')
    plt.colorbar()
    plt.savefig("third.pdf")
# done function

    
plot_rg(gt,rt,tr,1,rt2) # ,0,it0c,it0s,0,it3c,it3s)
