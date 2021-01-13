import math
import numpy as np
import tensorflow as tf
import utils
import time
import datetime
import os
import scipy
import scipy.sparse as sparse
from kmeans import getCluster

class Model( object ):
    def __init__(self, test_type, data, embedding_size,
                 batch_size, alpha, lamb, gamma, n_neg_samples, walk_length, max_epoch, learning_rate, output, agg):
        self.test_type = test_type
        self.batch_size = batch_size

        self.node_N = data.id_N
        self.attr_M = data.attr_M
        self.X_train = data.X
        self.X_test = data.X_test
        self.nodes = data.nodes
        self.in_neighboor = data.in_neighboor
        self.out_neighboor = data.out_neighboor

        self.embedding_size = embedding_size
        self.alpha = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.n_neg_samples = n_neg_samples
        self.walk_length = walk_length
        self.max_epoch = max_epoch
        self.lr = learning_rate

        if test_type == 0:
            self.output_file = "emb/"+output+"_embedding_link_prediction.npy"
        elif test_type == 1:
            self.output_file = "emb/"+output+"_embedding_node_classfication.npy"
        else:
            self.output_file = "emb/"+output+"_embedding_node_clustering.npy"

        self.agg = agg

        FORMAT = r'%m_%d_%H_%M_%S'
        time_now = datetime.datetime.now().strftime(FORMAT)

        self.txt_name = 'result/{}_task{}_{}.txt'.format(output, test_type, time_now)

        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        
        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data.
            self.train_data_id = tf.placeholder(tf.int32, shape=[None])  # batch_size * 1
            self.train_data_attr = tf.placeholder(tf.float32, (None, self.attr_M))  # batch_size * attr_M
            self.train_labels = tf.placeholder(tf.int32, (None,1))  # batch_size * 1

            if self.agg == 'LSTM':
                self.sequence_labels = tf.placeholder(tf.int32, (None, self.walk_length))  # batch_size * walk_length
            else:
                self.sequence_labels = tf.placeholder(tf.int32, (None, 1))  # batch_size * 1

            self.seqlen_placeholder = tf.placeholder(tf.int32, (None,))
            self.neighborhood_placeholder = tf.placeholder(tf.int32, (None, self.walk_length))


            # Variables.
            self.weights = self._initialize_weights()

            # Model.
                # self embed
                    # Look up embeddings for node_id.
            self.id_self_embed = tf.nn.embedding_lookup(self.weights['id_self_embeddings'], self.train_data_id)  # batch_size * id_dim
            self.attr_embed =  tf.matmul(self.train_data_attr, self.weights['attr_embeddings'])  # batch_size * attr_dim

                    # mix
            self.pre_embed_layer = tf.concat([self.id_self_embed, self.alpha * self.attr_embed], axis = 1) # batch_size * (id_dim + attr_dim)

                    # hidden layer 
            H_hidden_1 = tf.nn.softsign(tf.matmul(self.pre_embed_layer, self.weights['hidden_layer1_weight']) + self.weights['hidden_layer1_biases'])
            H_hidden_2 = tf.nn.softsign(tf.matmul(H_hidden_1, self.weights['hidden_layer2_weight']) + self.weights['hidden_layer2_biases'])
            self.node_self_embed = H_hidden_2

                # context embed
            self.context_embed = tf.matmul(self.nodes['node_attr'], self.weights['attr_embeddings'])
            self.node_context_embed = tf.nn.embedding_lookup(self.context_embed, self.neighborhood_placeholder)

                    # aggregate
            if self.agg == 'MEAN':
                self.mean_output = tf.reduce_mean(self.node_context_embed, axis = 1)
                self.predict_info = tf.layers.dense(self.mean_output, units=self.node_N, activation=None)
                self.out = self.mean_output
            elif self.agg == 'LINEAR':
                self.mean_output = tf.reduce_mean(self.node_context_embed, axis = 1)
                self.linear_output = tf.layers.dense(self.mean_output, units = self.embedding_size, activation = tf.nn.softsign)
                self.predict_info = tf.layers.dense(self.linear_output, units=self.node_N, activation=None)
                self.out = self.linear_output
            elif self.agg == 'LSTM':
                cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.embedding_size, layer_norm=True),
                                             input_keep_prob=1.0, output_keep_prob=1.0)
                outputs, states = tf.nn.dynamic_rnn(cell, self.node_context_embed, dtype=tf.float32, sequence_length=self.seqlen_placeholder)
                self.lstm_output = tf.reshape(outputs,shape=[-1,self.embedding_size])
                self.predict_info = tf.layers.dense(self.lstm_output, units=self.node_N, activation=None)
                self.out = states.h


            # Compute the loss, using a sample of the negative labels each time.
            self.loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_info, labels=tf.reshape(self.sequence_labels, shape=[-1])))
            
            self.loss_2 = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.context_embed, self.weights['biases'], self.train_labels,
                                                                   self.node_self_embed, self.n_neg_samples, self.node_N))

            self.loss_3 = tf.losses.mean_squared_error(tf.reshape(tf.nn.embedding_lookup(self.context_embed, self.train_labels), shape=[-1,self.embedding_size]), self.out)
            
            # Optimizer.
            if self.agg == 'LSTM':
                self.loss_lstm = (1-self.lamb)*self.loss_1 + self.lamb*self.loss_3
                self.loss_str = self.loss_2

                self.optimizer1 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss_lstm)
                self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss_str)
            else:
                self.loss= (1-self.lamb)*self.loss_1 + self.lamb*self.loss_3 + self.loss_2

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)

            # init
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.sess.run(tf.global_variables_initializer())

    def _initialize_weights(self):
        all_weights = dict()
        init_glorot_normal = tf.glorot_normal_initializer()

        all_weights['id_self_embeddings'] = tf.get_variable('id_self_embeddings', shape=[self.node_N, self.embedding_size], initializer=init_glorot_normal)
        all_weights['id_context_embeddings'] = tf.get_variable('id_context_embeddings', shape=[self.attr_M, self.embedding_size], initializer=init_glorot_normal)
        all_weights['attr_embeddings'] = tf.get_variable('attr_embeddings', shape=[self.attr_M, self.embedding_size], initializer=init_glorot_normal)
        all_weights['hidden_layer1_weight'] = tf.get_variable('hidden_layer1_weight', shape=[2*self.embedding_size, 2*self.embedding_size], initializer=init_glorot_normal)
        all_weights['hidden_layer2_weight'] = tf.get_variable('hidden_layer2_weight', shape=[2*self.embedding_size, self.embedding_size], initializer=init_glorot_normal)
        
        all_weights['biases'] = tf.Variable(tf.zeros([self.node_N]))
        all_weights['hidden_layer1_biases'] = tf.Variable(tf.zeros([2*self.embedding_size]))
        all_weights['hidden_layer2_biases'] = tf.Variable(tf.zeros([self.embedding_size]))

        return all_weights

    def partial_fit(self, X, walks, turn): # fit a batch
        feed_dict = {self.train_data_id: X['batch_data_id'],
                     self.train_data_attr: X['batch_data_attr'],
                     self.train_labels: X['batch_data_label'],
                     self.neighborhood_placeholder: walks[0],
                     self.seqlen_placeholder: walks[1],
                     self.sequence_labels:walks[2]}

        if self.agg != 'LSTM':
            if turn:
                loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                return loss
            else:
                return 0

        loss = 0
        if turn:
            loss_lstm, _ = self.sess.run((self.loss_lstm, self.optimizer1), feed_dict=feed_dict)
            loss = loss_lstm
        else:
            loss_str, _ = self.sess.run((self.loss_str, self.optimizer2), feed_dict=feed_dict)
            loss = loss_str
        return loss

    def write_training(self, epoch_id, loss):
        fw = open(self.txt_name, 'a')
        fw.write('epoch  %d\n'%(epoch_id))
        fw.write('Training:\n')
        fw.write('Loss: %f\n'%(loss))
        fw.close()

    def write_testing(self, **args):
        fw = open(self.txt_name, 'a')
        fw.write('Testing:\n')
        for key in args.keys():
            fw.write('%s : %f\n'%(key, args[key]))
        fw.write('\n')
        fw.close()

    def write_data(self):
        fw = open(self.txt_name, 'a')
        fw.write("embedding_size: %d\n"%(self.embedding_size))
        fw.write("walk_length: %d\n"%(self.walk_length))
        fw.write("Max_epoch: %d\n"%(self.max_epoch))
        fw.write("batch_size: %d\n"%(self.batch_size))
        fw.write("n_neg_samples: %d\n"%(self.n_neg_samples))
        fw.write("alpha: %.2f\n"%(self.alpha))
        fw.write("lamb: %.2f\n"%(self.lamb))
        fw.write("gamma: %.2f\n"%(self.gamma))
        fw.write("learning rate: %f\n\n"%(self.lr))
        fw.close()
        
    def train(self): # fit a dataset

        print('Training...')
        self.write_data()

        best_acc = 0
        bad_case = 0

        for epoch in range(self.max_epoch):
            total_batch = int( len(self.X_train['data_id_list']) / self.batch_size)

            sumLoss = 0
            for turn in [True, False]:
                # Loop over all batches
                for _ in range(total_batch):
                    # generate a batch data
                    batch_xs = {}
                
                    start_index = np.random.randint(0, len(self.X_train['data_id_list']) - self.batch_size)
                    batch_xs['batch_data_id'] = self.X_train['data_id_list'][start_index:(start_index + self.batch_size)]
                    batch_xs['batch_data_attr'] = self.X_train['data_attr_list'][start_index:(start_index + self.batch_size)]
                    batch_xs['batch_data_label'] = self.X_train['data_label_list'][start_index:(start_index + self.batch_size)]

                    walks = [[] for _ in range(3)]
                    for j in batch_xs['batch_data_label']:
                        walk, length = utils.get_neighborhood_list(j, self.in_neighboor, self.out_neighboor, self.walk_length)
                        walks[0].append(walk)
                        walks[1].append(length)
                        if self.agg == 'LSTM':
                            walks[2].append(walk[1:]+[j])
                        else:
                            walks[2].append(j)
                    loss = self.partial_fit(batch_xs, walks, turn)
                    sumLoss += loss

            print('epoch: %d    loss:  %f'%(epoch+1, sumLoss))

            self.write_training(epoch+1, sumLoss)

            # Display logs per epoch
            Embeddings_self = self.getEmbedding('self', self.nodes)
            Embeddings_context = self.getEmbedding('context', self.nodes)
            Embeddings = self.gamma*Embeddings_self + (1-self.gamma)*Embeddings_context
            
            if self.test_type == 0:
                roc = utils.evaluate_ROC(self.X_test, Embeddings)
                self.write_testing(AUC = roc)

                if roc > best_acc:
                    best_acc = roc
                    bad_case = 0
                    np.save(self.output_file, Embeddings)
                else:
                    bad_case += 1
            elif self.test_type == 1:
                test_ratio = [0.95, 0.9, 0.85, 0.8, 0.75]
                for ratio in test_ratio:
                    macro_F1, micro_F1, ACC = utils.evaluate_CF(self.X_test, Embeddings, ratio)
                    self.write_testing(Radio = ratio, MAF = macro_F1, MIF = micro_F1, ACC = ACC)

                    if ratio == 0.9:
                        if ACC > best_acc:
                            best_acc = ACC
                            bad_case = 0
                            np.save(self.output_file, Embeddings)
                        else:
                            bad_case += 1
            else:
                ARI, ACC, NMI = getCluster(Embeddings, self.X_test)
                self.write_testing(ARI = ARI, ACC = ACC, NMI = NMI)

                if ACC > best_acc:
                    best_acc = ACC
                    bad_case = 0
                    np.save(self.output_file, Embeddings)
                else:
                    bad_case += 1
            
            if bad_case == 10:
                break

    def getEmbedding(self, type, nodes):
        if type == 'self':
            feed_dict = {self.train_data_id: nodes['node_id'], self.train_data_attr: nodes['node_attr']}
            Embedding = self.sess.run(self.node_self_embed, feed_dict=feed_dict)
            return Embedding
        if type == 'context':
            Embedding = self.sess.run(self.context_embed)
            return Embedding
        
