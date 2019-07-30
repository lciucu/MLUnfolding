#!/usr/bin/python
from __future__ import print_function

import ROOT

debug=True


# relative path
inputFileName="../data/MLUnfolding/user.yili.18448069._000001.output.root"
# absolute path
# inputFileName="/Users/luizaadelinaciucu/Work/ATLAS/data/MLUnfolding/user.yili.18448069._000001.output.root"

def start():
    print("start")

def end():
    print("All ended well!")
    
def doItAll():
    if debug:
        print("inputFileName",inputFileName)
    # done if
    inFile = ROOT.TFile.Open(inputFileName ,"READ")
    #
    treeNominal = inFile.Get("nominal")
    nrEvents=treeNominal.GetEntries()
    if debug:
        print("nominal",nrEvents)
    #
    treeParticleLevel = inFile.Get("particleLevel")
    nrEvents=treeParticleLevel.GetEntries()
    if debug:
        print("particleLevel",nrEvents)
        
# done function



    

    
start()
doItAll()
end()

