#!/usr/bin/env python

#coding=utf-8
doc='''
############################################################
# cpxMSA.py
python cpxMSA.py -t=target -iDir=XXX
#
#
# This is a python script used to build the protein-protein interaction MSA based on the 
# HHblits, uniref30, TAX, ENA database
############################################################
'''

import os
import sys
import numpy as np
import time
import math
import getpass
import collections
from operator import itemgetter
import string
from string import Template
import subprocess
import shutil
from configure import path_dict

def build_cpxMSA(target,targetpath,gene=True,phy=True,string=True,workdir=''):
    """
    """
    # step1. build workdir
    if not workdir:
       workdir=make_workdir(workdir)    
    # step2. copy file 2 workdir
    Copyfile2workdir(target,targetpath,workdir)
    # run hhblits to generate chainA and chainB *.a3m file
    seqA=target.split('-')[0]+'.fa'
    seqB=target.split('-')[1]+'.fa'
    sequenceA=read_one_sequence(os.path.join(workdir,seqA))
    sequenceB=read_one_sequence(os.path.join(workdir,seqB)) 
    chainAlength=len(sequenceA)
    chainBlength=len(sequenceB)
    
    if not os.path.exists(os.path.join(targetpath,target,'MSA')):
       os.makedirs(os.path.join(targetpath,target,'MSA'))
    if string:
      print("Starting STRING....")
      hhblitsAfile_string = os.path.join(targetpath,target,'HHblits',target.split('-')[0]+'.string.a3m')
      hhblitsBfile_string = os.path.join(targetpath,target,'HHblits',target.split('-')[1]+'.string.a3m')
      if not os.path.exists(hhblitsAfile_string) or not os.path.exists(hhblitsBfile_string):
         hhblitsfileA_string,hhblitsfileB_string=run_build_mono_deepMSA(seqA,seqB,sequenceA,sequenceB,path_dict['hhblits_stringdb'],workdir,dimer=True,mono=False)
      else:
          hhblitsfileA_string = os.path.join(targetpath,target,'HHblits',target.split('-')[0]+'.string')
          hhblitsfileB_string = os.path.join(targetpath,target,'HHblits',target.split('-')[1]+'.string')
      species=path_dict["string_species"]
      finalMSA1=os.path.join(targetpath,target,"MSA")+'/'+"PPIS.a3m"
      if not os.path.exists(finalMSA1):   
         build_string_MSA(species,hhblitsfileA_string+'.a3m',hhblitsfileB_string+'.a3m',finalMSA1)
    print("STRING step finished.")

    if gene or phy:
       print('Starting gene or phy...')
       hhblitsAfile=os.path.join(targetpath,target,'HHblits',target.split('-')[0]+'.a3m')
       hhblitsBfile=os.path.join(targetpath,target,'HHblits',target.split('-')[1]+'.a3m')
       if not os.path.exists(hhblitsAfile) or not os.path.exists(hhblitsBfile):
          hhblitsAfile=os.path.join(workdir,target.split('-')[0]+'.a3m')
          hhblitsBfile=os.path.join(workdir,target.split('-')[1]+'.a3m')
          hhblits_monofileA,hhblits_monofileB=run_build_mono_deepMSA(seqA,seqB,sequenceA,sequenceB,path_dict['hhblitsdb'],workdir,dimer=False,mono=True)
    
       if (not os.path.exists(hhblitsAfile) or not os.path.exists(hhblitsBfile)):
          print("hhlbits step not finihsed, please check it. exit()\n")
          exit()
    else:
       print("please gene or phy step, please check it. exit()\n")
       exit()

    ID_seqA,targetSeqA=Read_HHblits_file(hhblitsAfile,"A")
    ID_seqB,targetSeqB=Read_HHblits_file(hhblitsBfile,"B")
    TAXID_db=path_dict['TAXID_db']
    if phy:
       print("Matching phy...")
       phy_MSAfile = os.path.join(targetpath,target,'MSA',"PIS.a3m")
       ID_TAXA=looktable_TAXID(ID_seqA,TAXID_db,'A',workdir)
       ID_TAXB=looktable_TAXID(ID_seqB,TAXID_db,'B',workdir)
       phy_paired=taxID_match(ID_TAXA,ID_TAXB)
       txt ='>chainA-chainB\n'
       txt +=targetSeqA+targetSeqB+'\n'
       for key in phy_paired:
           for line in phy_paired[key]:
               txt +=">"+line[0]+'\n'+line[1]+'\n'
       fw = open(phy_MSAfile,'w')
       fw.write(txt)
       fw.close()
       print('Matching phy finished.')


    ENA_db=path_dict['ENA_db'] 
    if gene:
       print('Matching gene...')
       ID_geneA=looktable_gene(ID_seqA,ENA_db,"A",workdir)
       ID_geneB=looktable_gene(ID_seqB,ENA_db,"B",workdir)
       gene_pairing=gene_match(ID_geneA,ID_geneB)
       txt = '>chainA-chainB\n'
       txt += targetSeqA+targetSeqB+'\n'
       for keyA in gene_pairing:
           cout=0
           if len(gene_pairing[keyA])>=2:
              for line in gene_pairing[keyA]:
                 keyB=line[1]
                 msaA=ID_seqA[keyA][0][1]
                 msaB=ID_seqB[keyB][0][1]
                 txt +=">"+keyA+"-"+keyB+"\n"+msaA+msaB+'\n'
           else:
             keyB=gene_pairing[keyA][0][1]
             msaA=ID_seqA[keyA][0][1]
             msaB=ID_seqB[keyB][0][1]
             txt +=">"+keyA+"-"+keyB+"\n"+msaA+msaB+'\n'
       gene_MSAfile = os.path.join(targetpath,target,'MSA',"GDS.a3m")
       fg = open(gene_MSAfile,'w')
       fg.write(txt)
       fg.close()
       print('Matching gene finished.')
       
    cmd=['rm -rf',workdir]
    os.system(' '.join(cmd))



# **************************************************************************
## step 1. run HHblits
# *************************************************************************
def run_build_mono_deepMSA(seqA,seqB,sequenceA,sequenceB,hhblitsdb,workdir,dimer=True,mono=False):
    """
    seqA chainA sequence
    seqB chainB sequence
    workdir 
    """
    query_fastaA=os.path.join(workdir,seqA)
    query_fastaB=os.path.join(workdir,seqB)
    
    if dimer:
      
       hhblits_monofileA=os.path.join(workdir,target.split('-')[0]+'.string')
       hhblits_monofileB=os.path.join(workdir,target.split('-')[1]+'.string')
    if mono:
       hhblits_monofileA=os.path.join(workdir,target.split('-')[0])
       hhblits_monofileB=os.path.join(workdir,target.split('-')[1])
   
    run_hhblits_monomerdb(query_fastaA,hhblitsdb,1,8,100,75,hhblits_monofileA)
    
    if sequenceA==sequenceB:
       shutil.copyfile(hhblits_monofileA+".a3m",hhblits_monofileB+'.a3m')
    else:
       run_hhblits_monomerdb(query_fastaB,hhblitsdb,1,8,100,75,hhblits_monofileB)

    if not os.path.exists(targetpath+'/'+target+'/HHblits'):
       os.mkdir(targetpath+'/'+target+'/HHblits')
    
    cmd=['cd',workdir+';','cp -f','*.a3m',\
         targetpath+'/'+target+'/HHblits'+';',\
        'cd',workdir+';'\
        'cp -f','*.a3m',targetpath+'/'+target+'/HHblits']
    os.system(' '.join(cmd))

    return hhblits_monofileA,hhblits_monofileB





## run HHblits ###################
# hhblits command #####
# $infile: input/query: single sequence or multiple sequence alignment (MSA)
# $db: uniclust302017 hhblits database
# $ncpu: [1,8] number of CPUs to use (for shared memory SMPs) (default=2)
# $n: number of iterations, default=2
# $e: [0,1] E-value cutoff for inclusion in result alignment (def=0.001)
# $id: [0,100] maximum pairwise sequence identity (def=90)
# $diff: [0,inf] filter MSAs by selecting most diverse set of sequences, keeping
#        at least this many seqs in each MSA block of length 50 (def=1000)
#        set $diff=inf
# $neffmax [1,20] skip further search iterations when diversity Neff of query MSA
#                  becomes larger than neffmax (default=10.0)    
#hhblits_monomerdb=Template(path_dict["hhblits"]+\
#" -i $infile -d $db -cpu $ncpu -n $n -id $seqid -cov $cov -oa3m $outprefix.a3m;"+\
#" grep -v '^>' $outprefix.a3m|sed 's/[a-z]//g' > $outprefix.aln")

hhblits_monomerdb=Template(path_dict["hhblits"]+\
" -i $infile -d $db -cpu $ncpu -n $n -id $seqid -cov $cov -e 1E-20 -maxfilt 100000000 -neffmax 20 -nodiff -realign_max 10000000 -oa3m $outprefix.a3m;"+\
" grep -v '^>' $outprefix.a3m|sed 's/[a-z]//g' > $outprefix.aln")

def run_hhblits_monomerdb(query_fasta,db,ncpu,n,seqid,cov,hhblits_prefix):
    ""
    ""
    
    cmd=hhblits_monomerdb.substitute(
        infile = query_fasta,
        db = db,
        ncpu = ncpu,
        n = n,
        seqid=seqid,
        cov=cov,
        outprefix=hhblits_prefix,
    )
    sys.stdout.write(cmd+'\n')
    os.system(cmd)

    
def Copyfile2workdir(target,targetpath,workdir):
    """
    copy file from targetpath to workdir
    
    """
    if not os.path.exists(targetpath+'/'+target+'/'+target.split('-')[0]+'.fa') or \
      not os.path.exists(targetpath+'/'+target+'/'+target.split('-')[1]+'.fa'):
      sys.stderr.write("ERROR!, sequence file %s or %s not exists, please check \
                       it\n"%(target+'/'+target.split('-')[0]+'.fa',\
                      target+'/'+target.split('-')[1]+'.fa'))
      exit()
    cmd=['cp -f',targetpath+'/'+target+'/'+target.split('-')[0]+'.fa',workdir]
    os.system(' '.join(cmd))
    cmd=['cp -f',targetpath+'/'+target+'/'+target.split('-')[1]+'.fa',workdir]
    os.system(' '.join(cmd))

###########################################333
#############################################
#  read hhblits file *.a3m
#############################################
#############################################
def Read_HHblits_file(hhblitsfile,chainID):
    """
    read aligment sequence and sorted by seqID
    """
    spec_pro={}
    fs=open(hhblitsfile,'r')
    HHblits_Infor=fs.readlines()
    fs.close()
    homename=[line.strip() for line in HHblits_Infor \
               if line.strip().startswith('>')]
    try:
       homeseq=[line.strip().translate(line.maketrans('', '', string.ascii_lowercase)) for line in HHblits_Infor if line.strip()[0] !=">"]
    except:
       homeseq=[line.strip().translate(None,string.ascii_lowercase) for line in HHblits_Infor if line.strip()[0] !=">"]
    targetSeq=homeseq[0]
    targetname=homename[0]
    Length=len(targetSeq)
    temp_seqID = []
    ID_seq = {}
    for indx in range(len(homeseq)):
        if indx ==0:
           continue
        key = homename[indx].split('|')[1]
        if len(key) != 6 and len(key) != 10:
           continue
        tmp_cout = [homeseq[indx][i] for i in range(Length) if homeseq[indx][i]==targetSeq[i]]
        if key+"_"+chainID in ID_seq and len(ID_seq[key+"_"+chainID])>=1:
           ID_seq[key+"_"+chainID].append([len(tmp_cout),homeseq[indx]])
        elif key+"_"+chainID not in ID_seq:
             ID_seq[key+"_"+chainID] =[[len(tmp_cout),homeseq[indx]]]
    return ID_seq,targetSeq

######################################################################
## look table phy and matching
######################################################################
def looktable_TAXID(ID_seq,TAXID_db,chainID,workdir):
    """
    look up table:
    ID_seq: ID dict           TAXID
            ID_seq[unipID]=["00000"]
    TAXID_db: TAX database//taxonomy database        
    """
    tmp_uniprotID_file = os.path.join(workdir,"tmp_unportID"+chainID)
    tmp_ID_TAX_file = os.path.join(workdir,"tmp_ID_taxID"+chainID)
    fw = open(tmp_uniprotID_file,'w')
    txt = ''
    for key in ID_seq:
        txt +=key.split("_")[0]+"\n"
    fw.write(txt)
    fw.close()

    cmd = ["LC_ALL=C fgrep -wf",tmp_uniprotID_file,TAXID_db,">",tmp_ID_TAX_file] # LC_ALL=C use locate instead of UTF-8
    os.system(' '.join(cmd))
    if not os.path.isfile(tmp_ID_TAX_file):
       exit()
    fr = open(tmp_ID_TAX_file,"r")
    tmp_ID_TAX = fr.readlines()
    fr.close()
    ID_TAX = {}
    for line in tmp_ID_TAX:
        line = line.strip().split()
        if line[2] in ID_TAX and len(ID_TAX[line[2]])>=1:
           ID_TAX[line[2]].append([line[0]+"_"+chainID]+ID_seq[line[0]+"_"+chainID][0])
        elif line[2] not in ID_TAX:
             ID_TAX[line[2]]=[[line[0]+"_"+chainID]+ID_seq[line[0]+"_"+chainID][0]]
    #cmd = ["rm -rf",tmp_uniprotID_file]
    #os.system(' '.join(cmd))
    #cmd = ["rm -rf",tmp_ID_TAX_file]
    #os.system(' '.join(cmd))

    return ID_TAX
def taxID_gene_match(phy_paired):
    """
    phy and gen
    """
    

def taxID_match(ID_TAXA,ID_TAXB):
    """
    """
    phy_paired={}

    for keyA in ID_TAXA:
        
        if keyA not in ID_TAXB:
           continue
        valueA = sorted(ID_TAXA[keyA],key=itemgetter(2)) # sorted by seqID
        valueB = sorted(ID_TAXB[keyA],key=itemgetter(2)) # sorted by seqID
        minlen =min(len(valueA),len(valueB))
        pairing_AB=[]
        for indx in range(minlen): 
            pair_ID=valueA[indx][0]+"-"+valueB[indx][0]
            #pair_seqID=valueA[indx][1]+"-"+valueB[indx][1]
            pair_seq=valueA[indx][2]+valueB[indx][2]
            pairing_AB.append([pair_ID,pair_seq])
        phy_paired[keyA]=pairing_AB

    return phy_paired


#############################################################################
# gene
############################################################################

def looktable_gene(ID_seq,ENA_db,chainID,workdir):
    """
    look up table:

    ID_seq: ID dict 
            ID_seq[unipID]=['50w6','000o','F/R']
    ENA_db: ENA database
    extract geneID from ENA database
    """
    tmp_uniprotID_file = os.path.join(workdir,"tmp_unportID"+chainID)
    tmp_ID_gene_file = os.path.join(workdir,"tmp_ID_gene"+chainID)
    fw = open(tmp_uniprotID_file,'w')
    txt = ''
    for key in ID_seq:
        txt +=key.split("_")[0]+"\n"
    fw.write(txt)
    fw.close()
    cmd = ["LC_ALL=C fgrep -wf",tmp_uniprotID_file,ENA_db,">",tmp_ID_gene_file]
    os.system(' '.join(cmd))
    if not os.path.isfile(tmp_ID_gene_file):
       exit()
    fr = open(tmp_ID_gene_file,"r")
    tmp_ID_gene = fr.readlines()
    fr.close()
    ID_gene = {}
    for line in tmp_ID_gene:
        line = line.strip().split()
        ID_gene[line[0]+"_"+chainID]=line[1].split("_")
    #cmd = ["rm -rf","tmp_*"]
    #os.system(' '.join(cmd))
    return ID_gene

def base2num(contig_order):
    """
    change contig_order to number

    contig_order ="002T" 
    --- number = 179
    """
    Nums="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    Num_key={}
    for i in range(len(Nums)):
        Num_key[Nums[i]]=i
    numbers =0
    for line in contig_order:
        numbers *=62
        numbers +=Num_key[line]
    return numbers
  

def gene_match(ID_geneA,ID_geneB):
    """
    match gene ID
    """
    
    gene_paired = {}  # key:[delta,uniprot]


    for keyA in ID_geneA:
        contigA=ID_geneA[keyA][0]       # contig name
        contigA_loc = base2num(ID_geneA[keyA][1]) # loaction
        contigA_dir = ID_geneA[keyA][2] # direction
        for keyB in ID_geneB:
            contigB=ID_geneB[keyB][0]       # contig name
            contigB_loc = base2num(ID_geneB[keyB][1]) # loaction
            contigB_dir = ID_geneB[keyB][2] # direction
            if contigA == contigB and contigB_dir==contigA_dir:
               delta_gene = abs(contigA_loc-contigB_loc)
               if delta_gene>=1 and delta_gene<=20:

                  if keyA not in gene_paired:
                     gene_paired[keyA]=[[delta_gene,keyB]]
                  elif keyA in gene_paired:
                       if len(gene_paired[keyA])==1 and gene_paired[keyA][0][0]>delta_gene:
                          gene_paired[keyA]=[[delta_gene,keyB]]
                       if len(gene_paired[keyA])>1 and gene_paired[keyA][0][0]>delta_gene:
                          gene_paired[keyA]=[[delta_gene,keyB]]
                       if len(gene_paired[keyA])>1 and gene_paired[keyA][0][0]==delta_gene:
                          gene_paired[keyA].append([delta_gene,keyB])
                  #if keyB not in gene_paired:
                     #gene_paired[keyB]=[[delta_gene,keyA]]
                  #elif keyB in gene_paired:
                       #if len(gene_paired[keyB])==1 and gene_paired[keyB][0][0]>delta_gene:
                          #gene_paired[keyB]=[[delta_gene,keyA]]
                       #if len(gene_paired[keyB])>1 and gene_paired[keyB][0][0]>delta_gene:
                          #gene_paired[keyB]=[[delta_gene,keyA]]
                       #if len(gene_paired[keyB])>1 and gene_paired[keyB][0][0]==delta_gene:
                          #gene_paired[keyB].append([delta_gene,keyA])
    return gene_paired

#######################################################################################

def build_string_MSA(species,hhblitsfileA,hhblitsfileB,MSA1file):
    """
    combine method MSA
    """
    cpx_MSA1=''
    spec_proA,seqA=Read_string(species,hhblitsfileA)
    spec_proB,seqB=Read_string(species,hhblitsfileB)
    cpx_MSA1 +='>'+'chain1'+'-'+'chain2'+'\n'+seqA+seqB+'\n'
    for key in spec_proA.keys():
        if len(spec_proA[key])==0 or len(spec_proA[key])==0:
           continue
        for specA,proA,seqA in spec_proA[key]:
            for specB,proB,seqB in spec_proB[key]:
                if specA==specB and proA!=proB:
                   score=search_string(specA,proA,proB)
                   if score!='' and int(score)>=600:
                      cpx_MSA1 +='>'+proA+'-'+proB+'\n'+seqA+seqB+'\n'
                      #cpx_MSA1 +=seqB+seqA+'\t'+str(score)+'\n'
        
    fmsa=open(MSA1file,'w')
    fmsa.write(cpx_MSA1)
    fmsa.close()


def Read_string(species,hhblitsfile):
    """
    """
    spec_pro={}
    fs=open(species,'r')
    Spe=fs.readlines()
    fs.close()
    for line in Spe:
        line=line.strip()
        spec_pro[line]=[]
    ifr=open(hhblitsfile,'r')
    Homo_seq_nam=ifr.readlines()
    
    ifr.close()
    homename=[line.strip()[1:] for line in Homo_seq_nam \
             if line.strip().startswith('>')]
    homeseq=[line.strip() for line in Homo_seq_nam if line.strip()[0] !=">"]
    targetSeq=homeseq[0]
    targetname=homename[0]
    Length=len(targetSeq)
    for indx in range(1,len(homename)):
        cov_c=collections.Counter(homeseq[indx])['-']
        if float(cov_c)/Length <=0.5:
           species_protein=homename[indx].split()[0]
           pos=species_protein.find(".")
           spe_type=species_protein[0:pos]
           spec_pro[spe_type].append([species_protein[0:pos],species_protein[pos+1:],\
           homeseq[indx].translate(str.maketrans('','',string.punctuation))])#translate(None,string.ascii_lowercase)])
    return spec_pro,targetSeq

## search string database/string links ##############
def search_string(species,proteinA,proteinB):
    """
    search proteinA and proteinB in string link.
    input: species type
    proteinA is a single protein.
    proteinB is a single protein.
 
    return score
    """
    species_proteinfile=os.path.join(path_dict['stringlinkdb']+species,proteinA)
 
    if not os.path.exists(species_proteinfile):
       return ''

    cmd="LC_ALL=C fgrep"+" "+"'"+proteinB+"'"+" "+"%s"%species_proteinfile
    stdout,stderr=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE).communicate()
 
    if stdout.decode('utf-8')=='':
       return ''
    line=stdout.decode('utf-8').strip()
    link_score=float(line.split()[1])
    if float(link_score)>=150:
       return link_score
    else:
       return ''

##################################################################################



### basisc function ##########################################################
# read sequence
# mkdir workdir
##############################################################################

def read_sequences(infile):
    names=[];
    sequences=[];
    fp=file(infile,'r');
    blocks=('\n'+fp.read()).split('\n>');
    for block in blocks:
        lines=block.splitlines();
        if len(lines)>=2:
            names.append(lines[0]);
            sequence='';
            for line in lines[1:]:
                sequence+=line.strip('\n');
            sequences.append(sequence);
    fp.close();
    return names,sequences;

def read_one_sequence(query_fasta='seq.seq'):
    """
    check if input is legal single sequence fasta and read the sequence
    """
    fp=open(query_fasta,'r')
    txt=fp.read()
    fp.close()
    if ('\n'+txt).count('\n>')!=1:
       sys.stderr.write("ERROR! Input is not single sequence fasta.")
       exit()
    sequence=''
    for line in txt.splitlines():
        if not line.startswith('>'):
            sequence+=line.strip()
    sequence=sequence.upper().replace(' ','').replace('\t','')
    illegal_residues=set(sequence)-set("ABCDEFGHIKLMNOPQRSTUVWXYZ")
    if illegal_residues:
       sys.stderr.write("ERROR! %s contains illegal residues %s\n"%(
           query_fasta,' '.join(illegal_residues)))
       exit()
    return sequence

### make temp folder ###

def mkdir_if_not_exist(workdir):
    ''' create folder if not exists '''
    if not os.path.isdir(workdir):
        os.makedirs(workdir)


def make_workdir(workdir):
    ''' creat tmp folder '''
    if not workdir:
       import random
       if os.getenv('SLURM_JOBID'):
          
        workdir="/scratch/%s/%s/cpx_MSA_%s"%(os.getenv("USER"),
               os.getenv('SLURM_JOBID'),
               random.randint(0,10**10)) 
        workdir="/tmp/%s/cpx_MSA_%s"%(os.getenv("USER"),random.randint(0,10**10))
       else:
         workdir="/tmp/%s/cpx_MSA_%s"%(os.getenv("USER"),
              random.randint(0,10**10))
       while(os.path.isdir(workdir)):
            if os.getenv('SLURM_JOBID'):
                workdir="/scratch/%s/%s/cpx_MSA_%s"%(os.getenv("USER"),
                       os.getenv('SLURM_JOBID'),
                       random.randint(0,10**10)) 
                workdir="/tmp/%s/cpx_MSA_%s"%(os.getenv("USER"),random.randint(0,10**10)) 
            else:
  
                workdir="/tmp/%s/cpx_MSA_%s"%(os.getenv("USER"),
               random.randint(0,10**10))
    mkdir_if_not_exist(workdir)
    sys.stdout.write("created folder %s\n"%workdir)
    return workdir

#########################################################################

if __name__=="__main__":
   if len(sys.argv)<2:
      sys.stderr.write(doc)
      exit()
  
   for arg in sys.argv[1:]:
       if arg.startswith("-t="):
           target=arg[len("-t="):]
       elif arg.startswith("-iDir="):
          targetpath=arg[len("-iDir="):]
       else:
          sys.stderr.write("ERROR! Unknown argument %s\n"%arg)
          exit()
   build_cpxMSA(target,targetpath)   
   #build_cpxDeepMSA(target,targetpath)
