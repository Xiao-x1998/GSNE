# GSNE
/*Source code and datasets for IJCAI 2021 paper "Strengthening Global Information for Attributed Network Embedding with a Smart Random Walk"*/

---

## Requirements

Please make sure your environment includes:

```
python (tested on 3.6.8)
tensorflow (tested on 1.12.0)
```

Then, run the command:
```
pip install -r requirements.txt
```

## Run

Run the following command for training GSNE:

```
python src/main.py --dataset_name [cora,citeseer,pubmed] --test_type [0,1,2] --aggregator [LSTM,MEAN,LINEAR]
```

For example, you can run GSNE on Cora dataset for node classfication task with LSTM aggregator like:

```
python src/main.py --dataset_name cora --test_type 1 --aggregator LSTM --walk_length 10
```
/*
## Cite

If you use the code, please cite our paper:

```
@inproceedings{cui2020adaptive,
  title={Adaptive Graph Encoder for Attributed Graph Embedding},
  author={Cui, Ganqu and Zhou, Jie and Yang, Cheng and Liu, Zhiyuan},
  booktitle={Proceedings of SIGKDD 2020},
  year={2020}
}
```
*/