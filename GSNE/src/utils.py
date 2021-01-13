import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import math
from sklearn import svm

def get_neighborhood_list(node, in_neighboor, out_neighboor, length):
    walk = [node]
    for _ in range(length):
        now_node = int(walk[-1])
        pre_node = -1
        if len(walk) > 1:
            pre_node = int(walk[-2])
        if now_node == -1:
            continue
        else:
            pro = []
            for nei_node in in_neighboor[now_node]:
                p1 = (len(in_neighboor[nei_node])+1)/(len(in_neighboor[now_node])+len(out_neighboor[nei_node]))
                p2 = 1
                if pre_node != -1 and nei_node not in in_neighboor[pre_node]:
                    p2 = 2
                p = p1*p2
                pro.append(p)
            if sum(pro) > 0:
                pro = np.array(pro)/sum(pro)
            now_pro = np.random.random()
            sum_pro = 0
            next_node = -1
            for i in range(len(pro)):
                sum_pro += pro[i]
                if now_pro <= sum_pro:
                    next_node = in_neighboor[now_node][i]
                    break
            walk.append(next_node)

    if walk[-1] == -1:
        walk.pop()
    walk.reverse()
    walk.pop()
    valid = len(walk)
    walk += [0]*(length-valid)
    return walk, valid

def cosine_similarity(a, b):
    _sum = 0.0
    _sum_a = 0.0
    _sum_b = 0.0
    for i in range(len(a)):
        _sum = _sum + a[i] * b[i]
        _sum_a = _sum_a + a[i] * a[i]
        _sum_b = _sum_b + b[i] * b[i]
    return _sum/(np.sqrt(_sum_a) * np.sqrt(_sum_b))

def evaluate_ROC(X_test, Embeddings):
    y_true = [ X_test[i][2] for i in range(len(X_test))]
    y_predict = [ cosine_similarity(Embeddings[X_test[i][0],:], Embeddings[X_test[i][1], :]) for i in range(len(X_test))]
    roc = roc_auc_score(y_true, y_predict)
    if roc < 0.5:
        roc = 1 - roc
    return roc


def evaluate_CF(X_test, Embeddings, radio):
    data_pair = []
    label_pair = []
    for i in range(len(X_test)):
        label_pair.append(X_test[i][1])
        data_pair.append(Embeddings[X_test[i][0]])

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(label_pair)
    data = np.array(data_pair)

    MAF = []
    MIF = []
    ACC = []
    for _ in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=radio)

        lr = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
        lr.fit(X_train, Y_train)
        Y_pred = lr.predict_proba(X_test)
        Y_pred = lb.transform(np.argmax(Y_pred, 1))

        Macro_F1 = f1_score(Y_test, Y_pred, average='macro')
        Micro_F1 = f1_score(Y_test, Y_pred, average='micro')
        acc = 0
        for i in range(len(Y_test)):
            if np.argmax(Y_test[i]) == np.argmax(Y_pred[i]):
                acc += 1
        acc /= len(Y_test)

        MAF.append(Macro_F1)
        MIF.append(Micro_F1)
        ACC.append(acc)
        
    return sum(MAF)/len(MAF), sum(MIF)/len(MIF), sum(ACC)/len(ACC)