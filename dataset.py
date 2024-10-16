from config import args
import joblib
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import random
from tqdm import tqdm


class MyDataLoader(object):

    def __init__(self, dataset, batch_size, mini_batch_size=0):
        self.total = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        if mini_batch_size == 0:
            self.mini_batch_size = self.batch_size

    def __getitem__(self, item):
        ceil = (item + 1) * self.batch_size
        sub_dataset = self.dataset[ceil - self.batch_size:ceil]
        #import pudb;pu.db
        # if ceil >= self.total:
        #     random.shuffle(self.dataset)
        return DataLoader(sub_dataset, batch_size=self.mini_batch_size)

    def __len__(self):
        if self.total == 0: return 0
        return (self.total - 1) // self.batch_size + 1


def split_train_valid_test(data, train_size, valid_part=0.1):
    # train_data = data[:train_size]
    # test_data = data[train_size:]
    # random.shuffle(train_data)
    # valid_size = round(valid_part * train_size)
    # valid_data = train_data[:valid_size]
    # train_data = train_data[valid_size:]
    # import pudb;pu.db



    with open(f"home/TextING-pytorch/corpus/DDoS2019.texts_new.txt", "r", encoding="latin1") as f:
        texts = f.read().strip().split("\n")

    c = list(zip(texts, data))
    random.shuffle(c)
    texts, data = zip(*c)
    train_data = data[:train_size]
    test_data = data[train_size:]

    txt_test = texts[train_size:]

    valid_size = round(valid_part * train_size)
    valid_data = train_data[:valid_size]
    train_data = train_data[valid_size:]

    # with open(f"home/TextING-pytorch/corpus/ids2012.texts_new_test.txt", "w") as f:
    #     f.write("\n".join(txt_test))

    
    # train_data = data[:train_size]
    # test_data = data[train_size:]
    #
    # train_data = random.sample(data, train_size)
    # # import pudb;pu.db
    # setA = set(data)
    # setB = set(train_data)
    #
    # test_data = list(setA - setB)
    # #import pudb;pu.db
    # random.shuffle(train_data)
    # valid_size = round(valid_part * train_size)
    # valid_data = train_data[:valid_size]
    # train_data = train_data[valid_size:]

    return train_data, valid_data, test_data


def get_data_loader(dataset, batch_size, mini_batch_size):
    # param
    train_size = args[dataset]["train_size"]

    # load data

    inputs = np.load(f"home/TextING-pytorch/temp/{dataset}.inputs.npy")
    graphs = np.load(f"home/TextING-pytorch/temp/{dataset}.graphs.npy")
    weights = np.load(f"home/TextING-pytorch/temp/{dataset}.weights.npy")
    targets = np.load(f"home/TextING-pytorch/temp/{dataset}.targets.npy")
    len_inputs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.inputs.pkl")
    len_graphs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.graphs.pkl")
    word2vec = np.load(f"home/TextING-pytorch/temp/{dataset}.word2vec.npy")


    # inputs = np.load(f"home/TextING-pytorch/temp/{dataset}.inputs_pac_count_new2.npy")
    # graphs = np.load(f"home/TextING-pytorch/temp/{dataset}.graphs_pac_count_new2.npy")
    # weights = np.load(f"home/TextING-pytorch/temp/{dataset}.weights_pac_count_new2.npy")
    # targets = np.load(f"home/TextING-pytorch/temp/{dataset}.targets_new.npy")
    # len_inputs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.inputs_pac_count_new2.pkl")
    # len_graphs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.graphs_pac_count_new2.pkl")
    # word2vec = np.load(f"home/TextING-pytorch/temp/{dataset}.word2vec_pac_count_new2.npy")

    # inputs = np.load(f"home/TextING-pytorch/temp/{dataset}.inputs_pac_4_packet4.npy")
    # graphs = np.load(f"home/TextING-pytorch/temp/{dataset}.graphs_pac_4_packet4.npy")
    # weights = np.load(f"home/TextING-pytorch/temp/{dataset}.weights_pac_4_packet4.npy")
    # targets = np.load(f"home/TextING-pytorch/temp/{dataset}.targets_4_packet4.npy")
    # len_inputs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.inputs_pac_4_packet4.pkl")
    # len_graphs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.graphs_pac_4_packet4.pkl")
    # word2vec = np.load(f"home/TextING-pytorch/temp/{dataset}.word2vec_pac_4_packet4.npy")

    # inputs = np.load(f"home/TextING-pytorch/temp/{dataset}.inputs_pac_new_4.npy")
    # graphs = np.load(f"home/TextING-pytorch/temp/{dataset}.graphs_pac_new_4.npy")
    # weights = np.load(f"home/TextING-pytorch/temp/{dataset}.weights_pac_new_4.npy")
    # targets = np.load(f"home/TextING-pytorch/temp/{dataset}.targets_new_4.npy")
    # len_inputs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.inputs_pac_new_4.pkl")
    # len_graphs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.graphs_pac_new_4.pkl")
    # word2vec = np.load(f"home/TextING-pytorch/temp/{dataset}.word2vec_pac_new_4.npy")

    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # inputs = np.load(f"home/TextING-pytorch/temp/{dataset}.inputs_pac_4_packet2.npy")
    # graphs = np.load(f"home/TextING-pytorch/temp/{dataset}.graphs_pac_4_packet2.npy")
    # weights = np.load(f"home/TextING-pytorch/temp/{dataset}.weights_pac_4_packet2.npy")
    # targets = np.load(f"home/TextING-pytorch/temp/{dataset}.targets_new_4.npy")
    # len_inputs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.inputs_pac_4_packet2.pkl")
    # len_graphs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.graphs_pac_4_packet2.pkl")
    # word2vec = (np.load(f"home/TextING-pytorch/temp/{dataset}.word2vec_pac_4_packet2.npy"))
    # # word2vec = torch.from_numpy(word2vec).to(device)

    # inputs = np.load(f"home/TextING-pytorch/temp/{dataset}.inputs_pac_packet.npy")
    # graphs = np.load(f"home/TextING-pytorch/temp/{dataset}.graphs_pac_packet.npy")
    # weights = np.load(f"home/TextING-pytorch/temp/{dataset}.weights_pac_packet.npy")
    # targets = np.load(f"home/TextING-pytorch/temp/{dataset}.targets_new.npy")
    # len_inputs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.inputs_pac_packet.pkl")
    # len_graphs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.graphs_pac_packet.pkl")
    # word2vec = np.load(f"home/TextING-pytorch/temp/{dataset}.word2vec_pac_packet.npy")

    # inputs = np.load(f"home/TextING-pytorch/temp/{dataset}.inputs_count2_norm_7.npy")
    # graphs = np.load(f"home/TextING-pytorch/temp/{dataset}.graphs_count2_norm_7.npy")
    # weights = np.load(f"home/TextING-pytorch/temp/{dataset}.weights_count2_norm_7.npy")
    # targets = np.load(f"home/TextING-pytorch/temp/{dataset}.targets_new.npy")
    # len_inputs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.inputs_count2_norm_7.pkl")
    # len_graphs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.graphs_count2_norm_7.pkl")
    # word2vec = np.load(f"home/TextING-pytorch/temp/{dataset}.word2vec_pac_packet.npy")


    # inputs = np.load(f"home/TextING-pytorch/temp/{dataset}.inputs_pac_count_packet4.npy")
    # graphs = np.load(f"home/TextING-pytorch/temp/{dataset}.graphs_pac_count_packet4.npy")
    # weights = np.load(f"home/TextING-pytorch/temp/{dataset}.weights_pac_count_packet4.npy")
    # targets = np.load(f"home/TextING-pytorch/temp/{dataset}.targets_new_packet4.npy")
    # len_inputs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.inputs_pac_count_packet4.pkl")
    # len_graphs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.graphs_pac_count_packet4.pkl")
    # word2vec = np.load(f"home/TextING-pytorch/temp/{dataset}.word2vec_pac_count_packet4.npy")

    # inputs = np.load(f"home/TextING-pytorch/temp/{dataset}.inputs_pac_count_norm.npy")
    # graphs = np.load(f"home/TextING-pytorch/temp/{dataset}.graphs_pac_count.npy")
    # weights = np.load(f"home/TextING-pytorch/temp/{dataset}.weights_pac_count_norm.npy")
    # targets = np.load(f"home/TextING-pytorch/temp/{dataset}.targets_new.npy")
    # len_inputs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.inputs_pac_count_norm.pkl")
    # len_graphs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.graphs_pac_count_norm.pkl")
    # word2vec = np.load(f"home/TextING-pytorch/temp/{dataset}.word2vec_pac_count_norm.npy")

    # inputs = np.load(f"home/TextING-pytorch/temp/{dataset}.inputs.npy")
    # graphs = np.load(f"home/TextING-pytorch/temp/{dataset}.graphs.npy")
    # weights = np.load(f"home/TextING-pytorch/temp/{dataset}.weights.npy")
    # targets = np.load(f"home/TextING-pytorch/temp/{dataset}.targets.npy")
    # len_inputs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.inputs.pkl")
    # len_graphs = joblib.load(f"home/TextING-pytorch/temp/{dataset}.len.graphs.pkl")
    # word2vec = np.load(f"home/TextING-pytorch/temp/{dataset}.word2vec.npy")



    # py graph dtype
    data = []
    for x, edge_index, edge_attr, y, lx, le in tqdm(list(zip(
            inputs, graphs, weights, targets, len_inputs, len_graphs))):
        #
        x = torch.tensor(x[:lx], dtype=torch.long)
        #import pudb;pu.db
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor([e[:le] for e in edge_index], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr[:le], dtype=torch.float)
        lens = torch.tensor(lx, dtype=torch.long)
        data.append(Data(x=x, y=y, edge_attr=edge_attr, edge_index=edge_index, length=lens))

    # split
    train_data, test_data, valid_data = split_train_valid_test(data, train_size, valid_part=0.1)

    # with open(f"home/TextING-pytorch/text", "w") as f:
    #     f.write("\n".join(str(x) for x in test_data.x))

    # import pudb;pu.db
    # return loader & word2vec
    return [MyDataLoader(data, batch_size=batch_size, mini_batch_size=mini_batch_size)
            for data in [train_data, test_data, valid_data]], word2vec

