import datetime
from sklearn import metrics
import numpy as np
from Bio import SeqIO
import itertools
from typing import List, Tuple
import random
import os

def printt(msg):   # print message

    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("{}| {}".format(time_str, msg))


def get_msg(msg):   # print message

    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return "{}| {}".format(time_str, msg)

def get_aupr(rec_ls, pre_ls):  # calcualte aupr

    pr_value = 0.0
    for ix in range(len(rec_ls[:-1])):
        x_right, x_left = rec_ls[ix], rec_ls[ix + 1]
        y_top, y_bottom = pre_ls[ix], pre_ls[ix + 1]
        temp_area = abs(x_right - x_left) * (y_top + y_bottom) * 0.5
        pr_value += temp_area
    return pr_value

def testset_evaluate_old(preds, labels,L, real_num):  # evaluate results

    labels = labels.reshape([-1])
    preds = preds.reshape([-1])
    metric = []

    # aupr
    precision_ls, recall_ls, thresholds = metrics.precision_recall_curve(labels, preds)
    aupr = get_aupr(recall_ls, precision_ls)

    # auc
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    # top 1, 5, 10, 20, 50, 100
    sort_result = np.zeros((2, len(labels)))
    sort_result[0, :] = preds
    sort_result[1, :] = labels
    sort_results = np.transpose(sort_result).tolist()
    sort_results.sort(reverse=True)
    sort_results = np.array(sort_results)

    metric.append(sum(sort_results[:1, 1] == 1) / 1)
    metric.append(sum(sort_results[:5, 1] == 1) / 5)
    metric.append(sum(sort_results[:10, 1] == 1) / 10)
    metric.append(sum(sort_results[:20, 1] == 1) / 20)
    metric.append(sum(sort_results[:50, 1] == 1) / 50)
    metric.append(sum(sort_results[:100, 1] == 1) / 100)

    RPFF = np.where(sort_results[:, 1] == 1)[0][0]
    metric.append(int(RPFF) + 1)

    metric.append(aupr)
    metric.append(auc)

    # top L/30, L/20, L/10, L/5, L/2

    metric.append(sum(sort_results[:int(L / 30), 1] == 1) / int(L / 30))
    metric.append(sum(sort_results[:int(L / 20), 1] == 1) / int(L / 20))
    metric.append(sum(sort_results[:int(L / 10), 1] == 1) / int(L / 10))
    metric.append(sum(sort_results[:int(L / 5), 1] == 1) / int(L / 5))
    metric.append(sum(sort_results[:int(L / 2), 1] == 1) / int(L / 2))

    metric.append(sum(sort_results[:real_num, 1] == 1) / real_num)

    return np.array(metric)


def testset_evaluate_old(preds, labels,L, real_num):  # evaluate results

    labels = labels.reshape([-1])
    preds = preds.reshape([-1])
    metric = []

    # aupr
    precision_ls, recall_ls, thresholds = metrics.precision_recall_curve(labels, preds)
    aupr = get_aupr(recall_ls, precision_ls)

    # auc
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    # top 1, 5, 10, 20, 50, 100
    sort_result = np.zeros((2, len(labels)))
    sort_result[0, :] = preds
    sort_result[1, :] = labels
    sort_results = np.transpose(sort_result).tolist()
    sort_results.sort(reverse=True)
    sort_results = np.array(sort_results)

    metric.append(sum(sort_results[:1, 1] == 1) / 1)
    metric.append(sum(sort_results[:5, 1] == 1) / 5)
    metric.append(sum(sort_results[:10, 1] == 1) / 10)
    metric.append(sum(sort_results[:20, 1] == 1) / 20)
    metric.append(sum(sort_results[:50, 1] == 1) / 50)
    metric.append(sum(sort_results[:100, 1] == 1) / 100)

    RPFF = np.where(sort_results[:, 1] == 1)[0][0]
    metric.append(int(RPFF) + 1)

    metric.append(aupr)
    metric.append(auc)

    # top L/30, L/20, L/10, L/5, L/2

    metric.append(sum(sort_results[:int(L / 30), 1] == 1) / int(L / 30))
    metric.append(sum(sort_results[:int(L / 20), 1] == 1) / int(L / 20))
    metric.append(sum(sort_results[:int(L / 10), 1] == 1) / int(L / 10))
    metric.append(sum(sort_results[:int(L / 5), 1] == 1) / int(L / 5))
    metric.append(sum(sort_results[:int(L / 2), 1] == 1) / int(L / 2))

    metric.append(sum(sort_results[:real_num, 1] == 1) / real_num)

    return np.array(metric)



def testset_evaluate(preds, labels,L, real_num):  # evaluate results

    labels = labels.reshape([-1])
    preds = preds.reshape([-1])
    metric = []

    # aupr
    precision_ls, recall_ls, thresholds = metrics.precision_recall_curve(labels, preds)
    aupr = get_aupr(recall_ls, precision_ls)

    # auc
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    # top 1, 5, 10, 20, 50, 100

    idx = np.argsort(-preds)
    select_labels = labels[idx]

    metric.append(sum(select_labels[:1] == 1) / 1)
    metric.append(sum(select_labels[:5] == 1) / 5)
    metric.append(sum(select_labels[:10] == 1) / 10)
    metric.append(sum(select_labels[:20] == 1) / 20)
    metric.append(sum(select_labels[:50] == 1) / 50)
    metric.append(sum(select_labels[:100] == 1) / 100)

    RPFF = 0
    metric.append(int(RPFF) + 1)

    metric.append(aupr)
    metric.append(auc)

    # top L/30, L/20, L/10, L/5, L/2

    metric.append(sum(select_labels[:int(L / 30)] == 1) / int(L / 30))
    metric.append(sum(select_labels[:int(L / 20)] == 1) / int(L / 20))
    metric.append(sum(select_labels[:int(L / 10)] == 1) / int(L / 10))
    metric.append(sum(select_labels[:int(L / 5)] == 1) / int(L / 5))
    metric.append(sum(select_labels[:int(L / 2)] == 1) / int(L / 2))

    metric.append(sum(select_labels[:real_num] == 1) / real_num)

    return np.array(metric)

def print_result(all_pre):  # print results

    print("top 1:" + str(round(all_pre[0], 3)) + ", top 5:" + str(round(all_pre[1], 3)) + ", top 10:" + str(
        round(all_pre[2], 3)) + ", top 20:" + str(round(all_pre[3], 3)) +
          ", top 50:" + str(round(all_pre[4], 3)) + ", top 100:" + str(round(all_pre[5], 3)) + ", P:" + str(
        round(all_pre[6], 3)))

    print("AUPR:" + str(round(all_pre[7], 3)) + ", AUC:" + str(round(all_pre[8], 3)) + ", top L/30:" + str(
        round(all_pre[9], 3)) + ", top L/20:" + str(round(all_pre[10], 3)) +
          ", top L/10:" + str(round(all_pre[11], 3)) + ", top L/5:" + str(round(all_pre[12], 3)) + ", top L/2:" + str(
        round(all_pre[13], 3)) + ", precision:" + str(round(all_pre[14], 3)))

def get_result(all_pre):  # print results

   line = "top 1:" + str(round(all_pre[0], 3)) + ", top 5:" + str(round(all_pre[1], 3)) + ", top 10:" \
          + str(round(all_pre[2], 3)) + ", top 20:" + str(round(all_pre[3], 3)) \
          + ", top 50:" + str(round(all_pre[4], 3)) + ", top 100:" + str(round(all_pre[5], 3)) + ", P:" + str(round(all_pre[6], 3))

   line = line + "\n"

   line = line + "AUPR:" + str(round(all_pre[7], 3)) + ", AUC:" + str(round(all_pre[8], 3)) + ", top L/30:"\
          + str(round(all_pre[9], 3)) + ", top L/20:" + str(round(all_pre[10], 3)) +", top L/10:"\
          + str(round(all_pre[11], 3)) + ", top L/5:" + str(round(all_pre[12], 3)) + ", top L/2:"\
          + str(round(all_pre[13], 3)) + ", precision:" + str(round(all_pre[14], 3))
   return line


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, str(record.seq).upper())
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def read_msa_random(msa_file, max_depth):

    f = open(msa_file, "r")
    text = f.read()
    f.close()

    sequence = ""
    name = ""
    name_list = []
    sequence_dict = dict()

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):

            if(len(sequence)>0):

                name_list.append(name)
                sequence_dict[name] = remove_gap_sequence(sequence)

            name = line
            sequence = ""

        else:
            sequence = sequence + line

    name_list.append(name)
    sequence_dict[name] = remove_gap_sequence(sequence)

    final_name_list = []
    sub_name_list = name_list[1:]
    random.shuffle(sub_name_list)
    sub_name_list = sub_name_list[:max_depth-1]

    final_name_list.append(name_list[0])
    final_name_list.extend(sub_name_list)

    final_sequence_list = []
    final_sequence_list.append(sequence_dict[name_list[0]])
    for name in sub_name_list:
        final_sequence_list.append(sequence_dict[name])

    return [(final_name_list[i], final_sequence_list[i]) for i in range(len(final_name_list))]

def read_length(length_file):

    f = open(length_file, "r")
    text = f.read()
    f.close()

    length_dict = dict()
    for line in text.splitlines():
        values = line.strip().split()
        length_dict[values[0]] = values[1:]

    return length_dict

def create_dir(workdir):
    if (os.path.exists(workdir) == False):
        os.makedirs(workdir)


def max_matrix(a, b):

    m, n =  a.shape
    c = np.array(np.zeros([m,n]))
    for i in range(m):
        for j in range(n):
            c[i, j] = max(a[i,j], b[i,j])

    return c

def is_lower(s):

    ap_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

    return s not in ap_list

def do_filter(sequence):

    sequence = "".join(list(filter(is_lower, sequence)))
    return sequence


def remove_gap_sequence(sequence):

    sequence = do_filter(sequence)

    return sequence