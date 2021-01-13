import numpy as np
import os
import random
from random import shuffle

class LoadData(object):

    def __init__(self, dataset_name, undirected, data_type):
        path = 'dataset/data_%s/'%(dataset_name)
        self.linkfile = path + "graph.txt"
        self.attrfile = path + "feature.txt"
        self.labelfile = path + "group.txt"
        
        self.undirected = undirected
        self.node_set = set()
        self.label_set = set()
        self.nodes = {}
        self.X = {}
        self.X_test = [] # [id, label]

        if data_type == 0:
            self.readlink2()
            fp = open("{}train_links.txt".format(path), 'w')
            for link in self.links:
                fp.write('{}\t{}\n'.format(link[0],link[1]))
            fp.close()

            fp = open("{}test_links.txt".format(path), 'w')
            for link in self.X_test:
                fp.write('{}\t{}\t{}\n'.format(link[0],link[1],link[2]))
            fp.close()            
        else:
            self.readlink()
            self.readlabel()

        self.readattr()
        self.construct_X()

        self.get_neighboor()

    # generate train links and test links
    def readlink2(self):
        f = open(self.linkfile)
        self.links = []
        self.valid_links = []
        self.total_links = {}
        self.train_set = set()
        line = f.readline()
        while line:
            items = line.strip().split('\t')
            items = [int(item) for item in items]
            self.node_set.add(items[0])
            self.node_set.add(items[1])

            if items[0] not in self.total_links:
                self.total_links[items[0]] = []
            self.total_links[items[0]].append(items[1])

            p = random.random()
            if p < 0.85:
                self.links.append([items[0], items[1]])
                self.train_set.add(items[0])
            elif p < 0.95:
                self.X_test.append([items[0], items[1], 1])
            else:
                self.valid_links.append([items[0], items[1]])

            if self.undirected:
                if items[1] not in self.total_links:
                    self.total_links[items[1]] = []
                self.total_links[items[1]].append(items[0])

                if p < 0.85:
                    self.links.append([items[1], items[0]])
                    self.train_set.add(items[1])
                elif p < 0.95:
                    self.X_test.append([items[1], items[0], 1])
                else:
                    self.valid_links.append([items[1], items[0]])   

            line = f.readline()
        f.close()
        self.id_N = len(self.node_set)
        self.nodes['node_id'] = []
        num = 0
        for i in range(self.id_N):
            self.nodes['node_id'].append(i)
            if i not in self.train_set:
                self.links.append([i,i])
                num += 1
        print("id_N:%d"%(self.id_N))

        test_link_num = len(self.X_test)
        for i in range(test_link_num):
            node_id1 = self.X_test[i][0]
            while True:
                node_id2 = random.randint(0, self.id_N-1)
                if node_id2 not in self.total_links[node_id1]:
                    self.X_test.append([node_id1, node_id2, 0])
                    break


    def readlink(self):
        f = open(self.linkfile)
        self.links = []
        line = f.readline()
        while line:
            items = line.strip().split('\t')
            items = [int(item) for item in items]
            self.node_set.add(items[0])
            self.node_set.add(items[1])
            link = [items[0], items[1]]
            self.links.append(link)
            if self.undirected:
                self.links.append([items[1], items[0]])
            line = f.readline()
        f.close()
        self.id_N = len(self.node_set)
        self.nodes['node_id'] = []
        for i in range(self.id_N):
            self.nodes['node_id'].append(i)
        print("id_N:%d"%(self.id_N))

    def readattr(self):
        f = open(self.attrfile)
        line = f.readline()
        self.nodes['node_attr'] = []
        while line:
            items = line.strip().split('\t')
            self.nodes['node_attr'].append([float(item) for item in items])
            line = f.readline()
        f.close()
        self.attr_M = len(self.nodes['node_attr'][0])
        print("attr_M:%d"%(self.attr_M))  
    
    def readlabel(self):
        f = open(self.labelfile)
        line = f.readline()
        while line:
            items = line.strip().split('\t')
            items = [int(item) for item in items]
            self.X_test.append([items[0], items[1]])
            self.label_set.add(items[1])
            line = f.readline()
        f.close()
        self.label_T = len(self.label_set)
        print("label_T:%d"%(self.label_T))  

    def construct_X(self):
        self.X['data_id_list'] =  np.ndarray(shape=(len(self.links)), dtype=np.int32)
        self.X['data_attr_list'] = np.ndarray(shape=(len(self.links), self.attr_M), dtype=np.float32)
        self.X['data_label_list'] = np.ndarray(shape=(len(self.links), 1), dtype=np.int32)

        for i in range(len(self.links)):
            self.X['data_id_list'][i] = self.links[i][1]
            self.X['data_attr_list'][i] = self.nodes['node_attr'][self.links[i][1]]
            self.X['data_label_list'][i, 0] = self.links[i][0]

    def get_neighboor(self):
        self.in_neighboor = [[] for _ in range(self.id_N)]
        self.out_neighboor = [[] for _ in range(self.id_N)]
        self.node_neighboor_martrix = np.zeros(shape = [self.id_N, self.id_N])
        
        for link in self.links:
            self.in_neighboor[link[0]].append(link[1])
            self.out_neighboor[link[1]].append(link[0])
            self.node_neighboor_martrix[link[1]][link[0]] = 1