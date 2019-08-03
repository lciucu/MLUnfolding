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

    #####################################################
    ####### do all for the nominal tree #################
    #####################################################
    #
    treeNominal = inFile.Get("nominal")
    nrEventsNominal=treeNominal.GetEntries()
    if debug:
        print("nominal",nrEventsNominal)

    # create a canvas
    c=ROOT.TCanvas("c_leading_jet_pt_nominal","c_leading_jet_pt nominal",600,400)
    # create a histogram that is owned by the canvas
    histo_leading_jet_pt=ROOT.TH1D("leading_jet_pt_nominal","leading jet pT nominal [MeV]",48,20e3,500e3)
    # tell the histogram to remember the weights
    histo_leading_jet_pt.Sumw2()
    # loop over all the elements (i.e. events) in the tree called "nominal"
    for i in range(0,nrEventsNominal):
        # if we want to run only on a few events
        if i!=4:
            continue
        treeNominal.GetEntry(i)
        if debug:
            print("i",i)
            met=getattr(treeNominal,"met_met")
            print("i",i,"met",met,type(met))
        jet_pt=getattr(treeNominal,"jet_pt")
        if debug:
            print("i",i,"jet_pt",type(jet_pt))
        # with code below I looked at the output and confirmed that the leading jet has index zero
        if debug:
            nrJet=len(jet_pt)
            print("i",i,"jet_pt",type(jet_pt),"nrJet",nrJet)
            for j in range(0,nrJet):
                print("jet",j,"with pt",jet_pt[j])
        leading_jet_pt=jet_pt[0]
        if debug:
            print("leading_jet_pt",leading_jet_pt)
        # fill the histogram for each event with the leading_jet_pt of the event
        histo_leading_jet_pt.Fill(leading_jet_pt)
    # done loop over all events
    # now we can print how many entries are filled in the histogram
    # it should be as many as events in the tree
    print("histo_leading_jet_pt for nrEntriesNominal",histo_leading_jet_pt.GetEntries())
    # normalized histogram (integral becomes 1.0, interpret as probabillity distribution)
    histo_leading_jet_pt.Scale(1.0/histo_leading_jet_pt.Integral())
    # draw the histogram on the canvas
    histo_leading_jet_pt.Draw("hist")
    # save the canvas in a file in both .png and .pdf
    outputFileName="./output/histo_leading_tree_nominal"
    c.SaveAs(outputFileName+".png")
    c.SaveAs(outputFileName+".pdf")
        
    #####################################################
    ####### do all for the particleLevel tree ###########
    #####################################################
    
    treeParticleLevel = inFile.Get("particleLevel")
    nrEventsParticleLevel=treeParticleLevel.GetEntries()
    if debug:
        print("particleLevel",nrEventsParticleLevel)
    # create a canvas
    c=ROOT.TCanvas("c_leading_jet_pt_particleLevel","leading_jet_pt particleLevel",600,400)
    # create a histogram that is owned by the canvas
    histo_leading_jet_pt=ROOT.TH1D("leading_jet_pt_particleLevel","leading jet pT particleLevel [MeV]",48,20e3,500e3)
    # tell the histogram to remember the weights
    histo_leading_jet_pt.Sumw2()
    # loop over all the elements (i.e. events) in the tree called "particleLevel"
    for i in range(0,nrEventsParticleLevel):
        if i!=8:
            continue
        treeParticleLevel.GetEntry(i)
        if debug:
            print("i",i)
            met=getattr(treeParticleLevel,"met_met")
            print("i",i,"met",met,type(met))
        jet_pt=getattr(treeParticleLevel,"jet_pt")
        if debug:
            print("i",i,"jet_pt",type(jet_pt))
        # with code below I looked at the output and confirmed that the leading jet has index zero
        if debug:
            nrJet=len(jet_pt)
            print("i",i,"jet_pt",type(jet_pt),"nrJet",nrJet)
            for j in range(0,nrJet):
                print("jet",j,"with pt",jet_pt[j])
        leading_jet_pt=jet_pt[0]
        if debug:
             print("leading_jet_pt",leading_jet_pt)
        # fill the histogram for each event with the leading_jet_pt of the event
        histo_leading_jet_pt.Fill(leading_jet_pt)
    # done loop over all events
    # now we can print how many entries are filled in the histogram
    # it should be as many as events in the tree
    print("histo_leading_jet_pt nrEntries for particleLevel",histo_leading_jet_pt.GetEntries())
    # normalized histogram (integral becomes 1.0, interpret as probabillity distribution)
    histo_leading_jet_pt.Scale(1.0/histo_leading_jet_pt.Integral())
    # draw the histogram on the canvas
    histo_leading_jet_pt.Draw("hist")
    # save the canvas in a file in both .png and .pdf
    outputFileName="./output/histo_leading_tree_particleLevel"
    c.SaveAs(outputFileName+".png")
    c.SaveAs(outputFileName+".pdf")
        
# done function



    

    
start()
doItAll()
end()

