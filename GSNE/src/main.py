import argparse
from Model import Model
import LoadData
import time
import tensorflow as tf
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run MODEL.")
    parser.add_argument('-t','--test_type', type=int, default=1,
                        help='0 for link prediction, 1 for node classfication, 2 for node clustering')
    parser.add_argument('-d','--dataset_name', type=str, default='citeseer')
    parser.add_argument('-agg','--aggregator', type=str, default='MEAN', 
                        help='LSTM, MEAN and LINEAR is provided')
    parser.add_argument('-L','--walk_length', type=int, default=10)
    parser.add_argument('--undirected', default=True, type=bool)
    
    parser.add_argument('-E','--embedding_size', type=int, default=128)
    
    parser.add_argument('-e', '--max_epoch', type=int, default=200)
    parser.add_argument('-B', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    
    parser.add_argument('-n','--n_neg_samples', type=int, default=10)
    
    parser.add_argument('-a', '--alpha', type=float, default=0.5,
                        help='the rate of attribute')
    parser.add_argument('-l', '--lamb', type=float, default=0.7,
                       help='the rate of different part loss in lstm loss [0,1]')                   
    parser.add_argument('-g', '--gamma', type=float, default=0.5,
                        help='the rate of self_embedding and context_embedding [0,1]')
    
    return parser.parse_args()

def run_model(data, args):
    model = Model(args.test_type, data, args.embedding_size, args.batch_size, args.alpha, args.lamb, args.gamma, args.n_neg_samples,
                  args.walk_length, args.max_epoch, args.learning_rate, args.dataset_name+'_'+args.aggregator, args.aggregator)
    model.train()

def main(args):
    if args.aggregator != 'LSTM' and args.aggregator != 'MEAN' and args.aggregator != 'LINEAR':
        print('UnKnown aggregator! Only LSTM/MEAN/LINEAR is provided')
        return

    print("Loading...")
    data = LoadData.LoadData(args.dataset_name, args.undirected, args.test_type)

    print("data load over!")
    if args.test_type == 0:
        print("\nLink Prediction...")
        print("Total nodes: ", data.id_N)
        print('Total attributes: ', data.attr_M)
        print("Total training links: ", len(data.links))
        print("Total testing links: ", len(data.X_test))
    elif args.test_type == 1:
        print("\nNode Classification...")
        print("Total nodes: ", data.id_N)
        print('Total attributes: ', data.attr_M)
        print("Total edges: ", len(data.links))
    else:
        print("\nNode Clustering...")
        print("Total nodes: ", data.id_N)
        print('Total attributes: ', data.attr_M)
        print("Total edges: ", len(data.links))

    run_model(data, args)

if __name__ == '__main__':
    random.seed(2021)
    np.random.seed(2021)
    tf.set_random_seed(2021)
    main(parse_args())
