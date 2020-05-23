The goal of this project is to adapt a code example of machine learning unfolding (MLUnfolding) using toy data to use it for the jet energy reconstruction of `ttbar e-mu` analysis. 

# The example

The code running on toy data by A. Glazov from DESY Hamburg can be found as a [Jupiter Notebook on GitHub.com](https://github.com/aglazov/MLUnfold/blob/master/Unfold.ipynb).

Unfolding is the procedure to infer the truth data (what happened in reality in nature) from the reconstructed data (observed and measured in our experiment by our detector).

The unfolding involves several iterative steps of machine learning using a neural network training in TensorFlow via Keras, in Python. The NN takes as input the reconstructed values, and has as output the truth values.

# The physics study

The goal is to adapt this example to use real data coming from the jet pt distribution from the `ttbar e-mu` analysis. We use this `.root` flat tree 

```
/afs/cern.ch/user/l/lciucu/public/data/MLUnfolding/user.yili.18448069._000001.output.sync.root
```

This file contains both the reconstructed and truth jets, already matched. Meaning the `i`-th event from reco, corresponds to the `i`-th event from truth. Each jet collection collects jets already the jets ranked by pt. We take index `0` to consider the leading reco jet and the leading truth jet. 

Given the leading jet pt as a continuous reco value, we want to find out in which jet pt bin (set by user, say 10 GeV or 20 GeV) does the truth jet falls. The input is a continuous value, but the output is a discrete value, that can take values from 0 to 49 if we consider 10 GeV bins from 0 to 500 GeV. Bin 0 contains jets with pt from 0 to 10 GeV. Jets with pt larger than 500 GeV have values set by hand to 499.999 GeV. It is equivalent to moving the overflow of a histogram bin to the last bin of the histogram. Given the output can take any value from 0 to 49, the problem we try to solve is a classification, and the possible labels are not only two (as in a simple signal to background classification, typically used in ATLAS), but a more complex one, with 50 labels. The NN will return the probability that for a given jet its pt falls in any of these bins. The total probability for all bins must be 1.0. This is ensured by the `softmax` activation function for the last layer of the NN. We then consider the bin with the largest probability as our choice by the NN. The predicted bin value can then be compared with the true bin value when making the plots of this project.

To optimize the NN performance, we take as input in fact not the jet pt, but the jet pt divided by the bin width. This is the jet pt bin as a real value. Making things more consistent with the output being the integer jet pt bin value for the truth.

# The technical implementation

We need to read a typical `.root` file, and then process it with `Keras`. It is not easy to run these both together in the same environment, either on `lxplus` at CERN, or on one's own laptop. If an environment has ROOT, it's Python version does not contain the ML software, and vice-versa.

One solution is to use the CERN server called [SWAN](https://swan003.cern.ch), that uses both ROOT and the ML stack. While the code can run, and one can create folders, the code can write out output, that can be then run by other code, it is not that easy to run with. As it requires the code to be written as a Jupiter Notebook, and the server is sometimes slow, or does not allows to connect to it, and even our wifi to CERN can be slow from outside CERN.

A better solution used in this project is to connect on `lxplus` with two terminals. In one a ML software is set up via a docker container, via the singularity command, which overwrites the entire ATLAS usual operating system. 

```
singularity exec '/cvmfs/unpacked.cern.ch/registry.hub.docker.com/atlasml/ml-base:latest' bash
```

This environment does not contain ROOT. However, it contains the [uproot package](https://github.com/scikit-hep/uproot) to read the `.root` file directl in `numpy` arrays used by the `ML` environment. 

We can thus have all of our code into only one environment and only one file.

We notice that NN training can take a long time, and code developing for the NN analysis and plot making requires many code iterations. For this we decide to store the NN weights into a text file, which is then read back any time needed. We generalize this further and divide the analysis in four stages, each of them writing its output to files, which are then read back by the subsequent stages that need it. Thus we can speed up development and running by running only the step we want:

* read .root file
* train NN
* analyse NN
* make plots

We also tried to run several NNs in parallel, but CERN IT complained about using up too many resources. For the future using the [CERN batch system](http://batchdocs.web.cern.ch/batchdocs/tutorial/exercise8c.html) is recommended, as per these instructions.

The code is also made configurable with options set up at the top that allow to chane there and then run the rest of the code unchanged:
* list of NN configurations to study
* jet pt range
* jet pt bin width
* plot colors and other options

We optimised (fine-tuned) the NN hyper-parameters for our specific data, thus changing the recommended NN configuration suggested in the toy data code example:
* number of hidden layers: `1` -> `3`.
* number of neurons (nodes) on each hidden layer: `k*NBins*Nbins` -> `k*Nbins`
* k: `8` -> `4`
* number of events (leading jets) per batch size: `1000` -> `200`

For this optimal choice many configurations were compared, by leaving all parameters constants, except one that was tried with values smaller and larger than the ones from the example.

The number of epochs in the training was also optimised with values up to 3000 being tested. Some NN configurations achieve the maximal performance faster than others, meaning using fewer epochs. We prefer these ones, as one NN training is just the first (zero-th) stage of a ML unfolding procedure. Many NN need to be trained iteratively. We find the best number of epochs is 150.

During the DESY summer school project of 5 weeks there was only time to perform this one first NN training. If there is more time, the next steps from the toy examples would be followed, with several NN traines recursively and statistical uncertainties measured via a boostrap method.

# Report

This repository contains also the `LaTex` report that will summarise the technique and the results. At the very end we may add the key plots in `.png` format. The `.pdf` of the report is [here](https://gitlab.cern.ch/lciucu/MLUnfolding/-/blob/master/report/reportMLUnfolding.pdf).

# Presentation

This repository contains also a `LaTex` `Beamer` presentation with the plots produced. The [DESY template](https://pr.desy.de/e223/index_ger.html) is used. The `.pdf` of the presentation is [here](https://gitlab.cern.ch/lciucu/MLUnfolding/-/blob/master/presentation/Presentation.pdf).

# Other bibliography sources

* [Unfolding tutorials](http://www.desy.de/~liyichen) by Yichen Li, DESY Zeuthen. 

# How to edit this README

The README is written for `GitLab` in `Markdown`. Take a look at this [command cheetsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).