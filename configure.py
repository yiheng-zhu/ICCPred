#!/usr/bin/env python

import os
import sys

rootpath=os.path.dirname(os.path.abspath(__file__))

HHLIB=rootpath+'/hhsuite2.0'
os.environ["HHLIB"]=HHLIB

path_dict=dict(
### hhblits ###########
hhblits= os.path.join(HHLIB,"bin/hhblits"),
hhblitsdb=os.path.join(rootpath,'library','uniclust30_2018_08','uniclust30_2018_08'),
hhblits_stringdb=os.path.join(rootpath,'library','STRING','string_hhblits','stringdb','stringdb'),
hhblitsdbpl=os.path.join(HHLIB,"scripts/hhblitsdb.pl"),
# remove redundant sequence and calculate Necs file
remove_Necs=os.path.join(rootpath,'rmRedundantCpxSeq'),
### string links
stringlinkdb=os.path.join(rootpath,'library','STRING','protein_links'),
string_species=os.path.join(rootpath,'library','STRING','all_species'),
ENA_db =os.path.join(rootpath,'library','ENA','ENA_2019_10_08_r141'),
TAXID_db = os.path.join(rootpath,'library','TAXID','uniprot_ID_mapping_TAX_ID'),
)

bin_dict=dict(

   hhblitsdb     =os.path.join(HHLIB,"scripts/hhblitsdb.pl"),

)
