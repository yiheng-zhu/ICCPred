# *ICCPRed: Integrating Unsupervised Language Model with Multi-View Multiple Sequence Alignments for Inter-Chain Contact Prediction*
This package contains deep learning models and related scripts to run ICCPred.This repository is the official implementation of ICCPred: Accurate prediction of inter-chain contact maps using Integrating unsupervised Language Model with multi-view Multiple Sequence Alignments.
Install virtual environment
conda create -n ICCPred pyhton=3.8
conda activate ICCPred
pip install ESM

#### Extract Multiple Sequence Alignments
cd MSA/
# download library to MSA folder 
# https://zhanggroup.org/cpxDeepMSA/download/package.tar.bz2
# tar -xvf package.tar.bz2 
# mv package/library ../
#
# ls MSA/library
##
python cpxMSA.py -t=targetID -iDir=/home/example
# targetID: target name ie. -t=16gsA-16gsB
# The directory that contains the target directory data. For complexes like 16gsA-16gsB if the directory 16gsA-16gsB full path is /home/user/targets/117eA-117eB the targetpath should store /home/user/targets/ i.e. -iDir=/home/user/targets/
# Extract Feature embedding
Python Extract_MSA_Feature_Embedding.py
# Training ICCPred model 
Python train_single_model.py
# Testing ICCPred model
Python test_model.py
