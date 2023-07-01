import pdb
import numpy as np


# 读取 .fa 文件，返回序列和标签
def read_seq_graphprot(seq_file, label):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    return seq_list, labels
 
def read_data_file(posifile, negafile):
    data = dict()
    seqs1, labels1 = read_seq_graphprot(posifile, label = 1)  # 正样本标签是1
    seqs2, labels2 = read_seq_graphprot(negafile, label = 0)  # 负样本标签是0
    seqs = seqs1 + seqs2
    labels = labels1 + labels2
    # print(labels)
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    return data
 
# 若长度不足501，使用N填充到501；若长度大于501，只取前501个核苷酸
def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

# 把处理好的等长（501）RNA序列 转变成 矩阵
def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)

    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def get_bag_data_1_channel(data, max_len):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len = max_len)
        bag_subt = []
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        # print tri_fea
        bag_subt.append(tri_fea.T)
        # print tri_fea.T
        bags.append(np.array(bag_subt))
        # print bags
    return bags, labels

# channel
def get_data(posi, nega, window_size):
    # 读取正、负样本文件，处理得到标准化的data和label
    data = read_data_file(posi, nega)
    # 得到处理好的样本矩阵和对应标签
    train_bags, label = get_bag_data_1_channel(data, max_len = window_size)
    return train_bags, label


# posi = r".\ALKBH5_Baltz2012.train.positives.fa"
# nega = r".\ALKBH5_Baltz2012.train.negatives.fa"
# train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 501)
#
# print(np.array(train_bags).shape)
# print(np.array(train_labels).shape)
# print(train_bags[0][0][0])
# print(train_labels[-5:])


# AUC计算
# from sklearn.metrics import roc_auc_score
# auc = roc_auc_score(X_labels, predict)    # X_labels测试集真实标签，predict是使用模型预测出来的概率