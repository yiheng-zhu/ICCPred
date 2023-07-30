#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import sys
import torch
from torch import nn, optim
from model_set import Resnet_Serial
import os
import Common_Methods as cm
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CUDA_LAUNCH_BLOCKING=1

# parameters for dataset
data_dir = sys.argv[1]
model_type = sys.argv[2]
stop_stype = sys.argv[3]
is_serial = sys.argv[4]

test_data_name = data_dir + "/test_dataset.pkl"

depth_threshold = 256

# read dataset
test_dataset = pickle.load(open(test_data_name, 'rb'))

attention_add1 = data_dir + "/Attention_Map/gene/" + str(depth_threshold) + "/"
attention_add2 = data_dir + "/Attention_Map/string/" + str(depth_threshold) + "/"
attention_add3 = data_dir + "/Attention_Map/phy/" + str(depth_threshold) + "/"
attention_add4 = data_dir + "/Attention_Map/Single/" + str(depth_threshold) + "/"
attention_add5 = data_dir + "/Attention_Map/cpxDeepMSA/" + str(depth_threshold) + "/"


index_list = [1, 1, 1, 0, 0]
dim_list = [144, 144, 144, 40, 144]
msa_type_list = ["gene", "string", "phy",  "Single", "cpxDeepMSA"]

feature_name = ""
for i in range(len(index_list)):
    if(index_list[i]==1):
        feature_name = feature_name + msa_type_list[i] + "_"

feature_name = feature_name + "Combine"

if(is_serial=="1"):

    model_dir = data_dir + "/Train_Model/Serial_" + feature_name + "/" + model_type + "/"
    log_dir = data_dir + "/log/Serial_" + feature_name + "/" + model_type + "/"
    result_dir = data_dir + "/Prediction_Result/Serial_" + feature_name + "/" + model_type + "/" + stop_stype + "/"

else:
    model_dir = data_dir + "/Train_Model/Parallel_" + feature_name + "/" + model_type + "/"
    log_dir = data_dir + "/log/Parallel_" + feature_name + "/" + model_type + "/"
    result_dir = data_dir + "/Prediction_Result/Parallel_" + feature_name + "/" + model_type + "/" + stop_stype + "/"

# parameters for machine learning models
stop_stype = "Acc"
epochs = 100
save_name_label = model_type+"_"+stop_stype
weight_figure = True  #loss 是否带权重
hidden_size1 = 256
hidden_size2 = 32
num_layer = 1
res_layer = 5
res_dim = 64
metric_number = 15

# loss function
class FocalLoss(nn.Module):

    def __init__(self, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduce=False)
        self.gamma = gamma

    def forward(self, pred, label, weight_mask):
        BCE_loss = self.criterion(pred, label) # = -log(pt)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        loss_weight = F_loss.mul(weight_mask)
        loss = torch.mean(loss_weight)
        return loss

# load data
def read_data(protein, msa_add):

    pdb_id = protein["Complex_code"]
    L = protein["l_length"]
    E = pickle.load(open(msa_add + "/" + pdb_id + "_row_attentions.pkl", 'rb'))
    Map_Label = protein["Map_Label"][:L, L:]
    p_num = np.sum(np.sum(Map_Label == 1))

    pscore = (Map_Label.shape[0] * Map_Label.shape[1] - np.sum(np.sum(Map_Label == 1))) / np.sum(np.sum(Map_Label == 1))

    if weight_figure:
        Label_weight = np.where(Map_Label > 0, pscore, 1)
    else:
        Label_weight = np.ones(Map_Label.shape)

    device = torch.device("cuda")

    E = torch.FloatTensor(E).to(device)
    Map_Label = torch.FloatTensor(Map_Label).to(device)
    Label_weight = torch.FloatTensor(Label_weight).to(device)

    l1_l2, l1_l2 = protein["Map_Label"].shape

    return l1_l2, E, Map_Label, p_num, Label_weight


# load model
if(is_serial=="1"):
    model = Resnet_Serial(res_dim, res_layer, torch.sigmoid, True, index_list, dim_list)
else:
    model = Resnet_Serial(res_dim, res_layer, torch.sigmoid, True, index_list, dim_list)

if torch.cuda.is_available():

    device = torch.device("cuda")
    model = model.to(device)
    print("Transferred model to GPU")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = FocalLoss()

#test

all_times = 10
result_array = [ dict() for i in range(all_times)]

for times in range(all_times):


    # model address
    model_file = model_dir + "/" + "train_" + model_type+"_" + stop_stype + "_" + str(times + 1) + ".tar"

    model.load_state_dict(torch.load(model_file))
    torch.no_grad()

    # predicted results address
    current_result_dir =  result_dir + "/" + str(times + 1) + "/"
    cm.create_dir(current_result_dir)

    number = 0
    all_pre = np.array([0 for i in range(metric_number)])

    for protein in test_dataset:

        pdb_id = protein["Complex_code"]

        l1_l2, E1, Map_Label, p_num, Label_weight = read_data(protein, attention_add1)
        l1_l2, E2, Map_Label, p_num, Label_weight = read_data(protein, attention_add2)
        l1_l2, E3, Map_Label, p_num, Label_weight = read_data(protein, attention_add3)
        l1_l2, E4, Map_Label, p_num, Label_weight = read_data(protein, attention_add4)
        l1_l2, E5, Map_Label, p_num, Label_weight = read_data(protein, attention_add5)

        protein_preds, protein_scores = model(E1, E2, E3, E4, E5)
        pr_l = Map_Label.cpu().numpy().flatten()
        pr_p = protein_scores.detach().cpu().numpy()

        met = cm.testset_evaluate(pr_p, pr_l, l1_l2, p_num)

        all_pre = all_pre + met
        number = number + 1

        result_file = current_result_dir + "/" + pdb_id + ".txt"
        np.savetxt(result_file, pr_p, fmt="%6f")

        result_array[times][pdb_id] = pr_p


    all_pre = all_pre/number

    # print results
    print("The results of " + str(number) + " examples for test dataset in the " + str(times + 1) + "-th round:")
    cm.print_result(all_pre)
    print()

# calculate average results
average_result_dir = result_dir + "/average/"
cm.create_dir(average_result_dir)

all_pre = np.array([0 for i in range(metric_number)])
number = 0
for protein in test_dataset:

    pdb_id = protein["Complex_code"]

    # average results
    pr_p = result_array[0][pdb_id]

    for times in range(1, all_times):
        pr_p = cm.max_matrix(pr_p, result_array[times][pdb_id])

    # save results
    result_file = average_result_dir + "/" + pdb_id + ".txt"
    np.savetxt(result_file, pr_p, fmt="%6f")

    #evaluation
    L = protein["l_length"]
    Map_Label = protein["Map_Label"][:L, L:]
    sum_l, sum_l = protein["Map_Label"].shape
    p_num = np.sum(np.sum(Map_Label == 1))

    pr_l = Map_Label.flatten()
    pr_p = pr_p.flatten()
    met = cm.testset_evaluate(pr_p, pr_l, sum_l, p_num)

    all_pre  = all_pre  + met

    number = number + 1

all_pre  = all_pre/number

print("The average results of " + str(number) + " examples for test dataset:")
cm.print_result(all_pre)


