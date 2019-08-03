#!/usr/bin/python
from __future__ import print_function

import ROOT

import numpy as np

debug=True
verbose=True
#list_treeName=["nominal","particleLevel"]
list_treeName=["nominal"]


# relative path
inputFileName="../data/MLUnfolding/user.yili.18448069._000001.output.root"
# absolute path
# inputFileName="/Users/luizaadelinaciucu/Work/ATLAS/data/MLUnfolding/user.yili.18448069._000001.output.root"

def start():
    print("start")

def end():
    print("All ended well!")


def doItForOneTree(inFile,treeName,debug):
    if debug:
        print("start doItForOneTree() for tree",treeName)
    tree=inFile.Get(treeName)
    nrEvents=tree.GetEntries()
    if debug:
        print("nrEvents",nrEvents)
    # create a canvas
    c=ROOT.TCanvas("c_leading_jet_pt_"+treeName,"c_leading_jet_pt "+treeName,600,400)
    # create a histogram that is owned by the canvas
    h=ROOT.TH1F("leading_jet_pt_"+treeName,"leading jet pT "+treeName+" [GeV]",48,20.0,500.0)
    # tell the histogram to remember the weights
    h.Sumw2()
    # fill histogram
    # loop over all events in the tree
    for i in range(0,nrEvents):
        if debug:
            print("i",i)
        tree.GetEntry(i)
        jet_pt=getattr(tree,"jet_pt")
        # jet_pt is a collections of jets
        # I checked they appeared in decreasing order of pt in MeV
        # leading jet has index 0
        leading_jet_pt=jet_pt[0]*0.001 # convert from MeV to GeV
        if debug:
            print("i",i,"nrJet",len(jet_pt),"leading_jet_pt",leading_jet_pt)
        # fill histogram
        h.Fill(leading_jet_pt)
    # done loop over events
    # draw histogram
    if debug or verbose:
        print("h nrEntries "+treeName,h.GetEntries())
        print("h Integral "+treeName,h.Integral())
    # normalized histogram (integral becomes 1.0, interpret as probabillity distribution)
    h.Scale(1.0/h.Integral())
    if debug or verbose:
        print("after normalizing h Integral "+treeName,h.Integral())
    #  draw the histogram on the canvas
    h.Draw("hist")
    # save the canvas in a file in both .png and .pdf
    outputFileName="./output/histo_leading_jet_pt_"+treeName
    for extension in "png,pdf".split(","):
        c.SaveAs(outputFileName+"."+extension)
    # done for loop over extension
# done function
    
def doItAll():
    if debug:
        print("start doItAll()")
        print("inputFileName",inputFileName)
    # done if
    inFile = ROOT.TFile.Open(inputFileName ,"READ")
    for treeName in list_treeName:
        if debug:
            print("treeName",treeName)
        doItForOneTree(inFile,treeName,debug)
    # done for loop over treeName
# done function

# run
start()
doItAll()
end()

