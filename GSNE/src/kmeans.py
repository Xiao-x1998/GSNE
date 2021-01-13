from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
import math
import argparse
from scipy.optimize import linear_sum_assignment

def getLabel(labelfile):
    f = open(labelfile)
    line = f.readline()
    label_array = []
    while line:
        items = line.strip().split('\t')
        items = [int(item) for item in items]
        label_array.append(items[1])
        line = f.readline()
    f.close()
    return np.array(label_array)

def cluster_acc(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def cluster_nmi(y_true,y_pred):
    #样本点数
    toty_truel = len(y_true)
    y_true_ids = set(y_true)
    y_pred_ids = set(y_pred)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idy_true in y_true_ids:
        for idy_pred in y_pred_ids:
            idy_trueOccur = np.where(y_true==idy_true)
            idy_predOccur = np.where(y_pred==idy_pred)
            idy_truey_predOccur = np.intersect1d(idy_trueOccur,idy_predOccur)
            px = 1.0*len(idy_trueOccur[0])/toty_truel
            py = 1.0*len(idy_predOccur[0])/toty_truel
            pxy = 1.0*len(idy_truey_predOccur)/toty_truel
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idy_true in y_true_ids:
        idy_trueOccurCount = 1.0*len(np.where(y_true==idy_true)[0])
        Hx = Hx - (idy_trueOccurCount/toty_truel)*math.log(idy_trueOccurCount/toty_truel+eps,2)
    Hy = 0
    for idy_pred in y_pred_ids:
        idy_predOccurCount = 1.0*len(np.where(y_pred==idy_pred)[0])
        Hy = Hy - (idy_predOccurCount/toty_truel)*math.log(idy_predOccurCount/toty_truel+eps,2)
    MIhy_truet = 2.0*MI/(Hx+Hy)
    return MIhy_truet

def getCluster(X, Y):
    y = np.array([item[1] for item in Y])
    n_clusters = len(set(y))
    model = KMeans(n_clusters = n_clusters, random_state=2020, n_jobs = 4, max_iter = 200)
    model.fit(X)
    pre_y = model.predict(X)

    ARI = metrics.adjusted_rand_score(y, pre_y)
    ACC = cluster_acc(y, pre_y)
    NMI = metrics.normalized_mutual_info_score(y, pre_y, average_method='arithmetic')
    
    return ARI, ACC, NMI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='k-means cluster')
    parser.add_argument('--emb', default='pubmed_embedding_node_classfication.npy', 
                        help='embedding file')
    parser.add_argument('--label', default='data_pubmed/group.txt', 
                        help='the label of data')
    parser.add_argument('--max_iter', default=200, type=int, 
                        help='the max iteration')
    args = parser.parse_args()
    X = np.load(args.emb)
    y = getLabel(args.label)
    n_clusters = len(set(y))
    print(n_clusters)
    model = KMeans(n_clusters = n_clusters, random_state=2020, n_jobs = 4, max_iter = args.max_iter)
    model.fit(X)
    pre_y = model.predict(X)

    ARI = metrics.adjusted_rand_score(y, pre_y)
    ACC = cluster_acc(y, pre_y)
    NMI = metrics.normalized_mutual_info_score(y, pre_y, average_method='arithmetic')
    print("ARI:",ARI,", ACC:",ACC,", NIM:",NMI)