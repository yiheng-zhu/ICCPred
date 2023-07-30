#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import sys
from torch import nn, optim
import torch
import numpy as np
import os
import Common_Methods as cm
from model_set import Resnet_Serial, Resnet_Serial_linear

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.manual_seed(6)
import random
random.seed(6)

CUDA_LAUNCH_BLOCKING=1

# parameters for dataset

data_dir = sys.argv[1]
model_type = sys.argv[2]
is_serial = sys.argv[3]
gpu_id = sys.argv[4]
current_times = int(sys.argv[5])


os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

depth_threshold = 256

train_dataset_add = data_dir + "/train_dataset.pkl"
val_dataset_add = data_dir + "/validation_dataset.pkl"
test_dataset_add = data_dir + "/test_dataset.pkl"

attention_add1 = data_dir + "/Attention_Map/string/" + str(depth_threshold) + "/"
attention_add2 = data_dir + "/Attention_Map/phy/" + str(depth_threshold) + "/"
attention_add3 = data_dir + "/Attention_Map/gene/" + str(depth_threshold) + "/"
attention_add4 = data_dir + "/Attention_Map/Single/" + str(depth_threshold) + "/"
attention_add5 = data_dir + "/Attention_Map/cpxDeepMSA/" + str(depth_threshold) + "/"


index_list = [int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])]
dim_list = [144, 144, 144, 40, 144]
msa_type_list = ["string", "phy", "gene", "Single", "cpxDeepMSA"]

feature_name = ""
for i in range(len(index_list)):
    if(index_list[i]==1):
        feature_name = feature_name + msa_type_list[i] + "_"

feature_name = feature_name + "Combine"

if(is_serial=="1"):
    model_dir = data_dir + "/Train_Model/Serial_" + feature_name + "/" + model_type + "/"
    log_dir = data_dir + "/log/Serial_" + feature_name + "/" + model_type + "/"
else:
    model_dir = data_dir + "/Train_Model/Parallel_" + feature_name + "/" + model_type + "/"
    log_dir = data_dir + "/log/Parallel_" + feature_name + "/" + model_type + "/"

cm.create_dir(log_dir)
cm.create_dir(model_dir)

print(log_dir)
print(model_dir)


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

# read_dataset
train_dataset = pickle.load(open(train_dataset_add, 'rb'))
val_dataset = pickle.load(open(val_dataset_add, 'rb'))
test_dataset = pickle.load(open(test_dataset_add, 'rb'))

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
    model = Resnet_Serial_linear(res_dim, res_layer, torch.sigmoid, True, index_list, dim_list)

if torch.cuda.is_available():

    device = torch.device("cuda")
    model = model.to(device)
    print("Transferred model to GPU")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = FocalLoss()

for times in range(current_times, current_times + 1):

    es = 0
    best_avg_loss = 99999.
    best_acc = 0.

    log_file = log_dir + "/log_" + stop_stype + "_" + str(times)
    f = open(log_file, "w")

    for e in range(epochs):  # 循环训练

        e_loss = 0.  # 当前epochs loss
        e_acc = 0.
        number = 0
        all_pre = [0 for i in range(metric_number)]

        # training

        for protein in train_dataset:

            l1_l2, E1, Map_Label, p_num, Label_weight = read_data(protein, attention_add1)
            l1_l2, E2, Map_Label, p_num, Label_weight = read_data(protein, attention_add2)
            l1_l2, E3, Map_Label, p_num, Label_weight = read_data(protein, attention_add3)
            l1_l2, E4, Map_Label, p_num, Label_weight = read_data(protein, attention_add4)
            l1_l2, E5, Map_Label, p_num, Label_weight = read_data(protein, attention_add5)

            optimizer.zero_grad()  # 梯度设为0
            protein_preds, protein_scores = model(E1, E2, E3, E4, E5)

            protein_loss = criterion(protein_preds, Map_Label, Label_weight)
            pr_loss = protein_loss.item()  # it_loss(batch_loss) iter_losses(all batch_loss list)
            protein_loss.backward()  # 反向传播计算
            optimizer.step()  # 优化器根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用

            pr_l = Map_Label.cpu().numpy().flatten()
            pr_p = protein_scores.detach().cpu().numpy().flatten()

            met = cm.testset_evaluate(pr_p, pr_l, l1_l2, p_num)
            e_acc += met[14]
            e_loss += pr_loss

            for k in range(len(all_pre)):
                all_pre[k] = all_pre[k] + met[k]
            number = number + 1

        e_loss = e_loss / number
        e_acc = e_acc / number

        for k in range(len(all_pre)):
            all_pre[k] = all_pre[k] / number

        f.write("The results of " + str(number) + " examples in the " + str(e + 1) + "-th iteration in training dataset:\n")
        f.write(cm.get_result(all_pre) + "\n")
        f.write(cm.get_msg("UPDATE\tEpoch {}: train avg loss:{} train avg acc:{}".format(e + 1, round(e_loss, 6), round(e_acc, 6))) + "\n")

        # validation
        val_avg_acc = 0.  # top real acc
        val_avg_loss = 0.  #
        number = 0
        all_pre = [0 for i in range(metric_number)]

        for protein in val_dataset:

            l1_l2, E1, Map_Label, p_num, Label_weight = read_data(protein, attention_add1)
            l1_l2, E2, Map_Label, p_num, Label_weight = read_data(protein, attention_add2)
            l1_l2, E3, Map_Label, p_num, Label_weight = read_data(protein, attention_add3)
            l1_l2, E4, Map_Label, p_num, Label_weight = read_data(protein, attention_add4)
            l1_l2, E5, Map_Label, p_num, Label_weight = read_data(protein, attention_add5)

            protein_preds, protein_scores = model(E1, E2, E3, E4, E5)
            pr_l = Map_Label.cpu().numpy().flatten()
            pr_p = protein_scores.detach().cpu().numpy().flatten()
            protein_loss = criterion(protein_preds, Map_Label, Label_weight)  #
            pr_loss = protein_loss.item()

            met = cm.testset_evaluate(pr_p, pr_l, l1_l2, p_num)
            val_avg_acc += met[14]
            val_avg_loss += pr_loss

            for k in range(len(all_pre)):
                all_pre[k] = all_pre[k] + met[k]
            number = number + 1

        for k in range(len(all_pre)):
            all_pre[k] = all_pre[k] / number

        val_avg_acc = val_avg_acc / number
        val_avg_loss = val_avg_loss / number

        f.write("The results of " + str(number) + " examples in the " + str(e + 1) + "-th iteration in validation dataset:\n")
        f.write(cm.get_result(all_pre) + "\n")

        # test
        number = 0
        all_pre = [0 for i in range(metric_number)]

        for protein in test_dataset:

            l1_l2, E1, Map_Label, p_num, Label_weight = read_data(protein, attention_add1)
            l1_l2, E2, Map_Label, p_num, Label_weight = read_data(protein, attention_add2)
            l1_l2, E3, Map_Label, p_num, Label_weight = read_data(protein, attention_add3)
            l1_l2, E4, Map_Label, p_num, Label_weight = read_data(protein, attention_add4)
            l1_l2, E5, Map_Label, p_num, Label_weight = read_data(protein, attention_add5)

            protein_preds, protein_scores = model(E1, E2, E3, E4, E5)
            pr_l = Map_Label.cpu().numpy().flatten()
            pr_p = protein_scores.detach().cpu().numpy().flatten()

            met = cm.testset_evaluate(pr_p, pr_l, l1_l2, p_num)

            for k in range(len(all_pre)):
                all_pre[k] = all_pre[k] + met[k]
            number = number + 1

        for k in range(len(all_pre)):
            all_pre[k] = all_pre[k] / number

        f.write("The results of " + str(number) + " examples in the " + str(e + 1) + "-th iteration in test dataset:\n")
        f.write(cm.get_result(all_pre) + "\n")

        # is stop ?
        Early_stopping_Flag = False
        if stop_stype == "Loss":
            if val_avg_loss < best_avg_loss:
                Early_stopping_Flag = False
            else:
                Early_stopping_Flag = True
        else:
            if val_avg_acc > best_acc:
                Early_stopping_Flag = False
            else:
                Early_stopping_Flag = True

        if Early_stopping_Flag == False:

            best_acc = val_avg_acc
            best_avg_loss = val_avg_loss
            es = 0

            train_model_file = os.path.join(model_dir, "train_" + save_name_label + "_" + str(times) + ".tar")
            torch.save(model.state_dict(), train_model_file)
            f.write(cm.get_msg("UPDATE\tEpoch {}: val avg loss:{} val avg acc:{}".format(e + 1, round(best_avg_loss, 6), round(best_acc, 6))) + "\n\n")
            f.flush()

        else:
            es += 1
            f.write("Counter {} of 5".format(es) + "\n\n")
            f.flush()

            if es > 2:
                f.write("Early stopping with best_acc: " + str(best_acc) + "\n\n")
                f.flush()
                break

    f.close()