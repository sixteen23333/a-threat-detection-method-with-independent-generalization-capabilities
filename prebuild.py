import joblib
from tqdm import tqdm
import scipy.sparse as sp
from collections import Counter
import numpy as np
import time
import random
from math import log
import torch

#python home/TextING-pytorch/prebuild-2.py

# 数据集
dataset = "ids2012"

# 参数
window_size = 7
embedding_dim = 300
max_text_len = 800

# normalize
def normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    return adj_normalized


def pad_seq(seq, pad_len):
    if len(seq) > pad_len: return seq[:pad_len]
    return seq + [0] * (pad_len - len(seq))



if __name__ == '__main__':
    #load data
    

   #  word2index = joblib.load(f"home/TextING-pytorch/temp/{dataset}.word2index_new.pkl")
   #  with open(f"home/TextING-pytorch/temp/{dataset}.texts_new.remove.txt", "r") as f:
   #      texts = f.read().strip().split("\n")
   #  # bulid graph
   #  inputs = []
   #  graphs = []
   #
   #  #import pudb;pu.db
   #  t = time.time()
   #
   #
   #  for doc_words in texts:
   #
   #      windows = []
   #      row, col, weight = [],[],[]
   #      edges = []
   #      words = doc_words.split()
   #      edges2=[]
   #      k1 = [word2index[w] for w in doc_words.split()]
   #      #import pudb;pu.db
   #      k1 = k1[:max_text_len]
   #      k2 = words[:max_text_len]
   #      words=words[:max_text_len]
   #      #import pudb;pu.db
   #      nodes = list(set(k1))
   #      nodes2=list(set(k2))
   #      node2index2 = {e: i for i, e in enumerate(nodes2)}
   #      node2index = {e: i for i, e in enumerate(nodes)}
   #      #import pudb;pu.db
   #      length = len(words)
   #      if length <= window_size:
   #          windows.append(words)
   #      else:
   #      # print(length, length - window_size + 1)
   #          for j in range(length - window_size + 1):
   #              window = words[j: j + window_size]
   #              windows.append(window)
   #      #print("calculating word frequency...")
   #
   #      word_window_freq = {}
   #      for window in windows:
   #          appeared = set()
   #          for i in range(len(window)):
   #              if window[i] in appeared:
   #                  continue
   #              if window[i] in word_window_freq:
   #                  word_window_freq[window[i]] += 1
   #              else:
   #                  word_window_freq[window[i]] = 1
   #              #import pudb;pu.db
   #              appeared.add(window[i])
   #      #print("calculating word pair frequency...")
   #
   #      edg=[]
   #      for i in range(len(words)):
   #          center = node2index2[words[i]]
   #          for j in range(i - window_size, i + window_size + 1):
   #              if i != j and 0 <= j < len(words):
   #                  neighbor = node2index2[words[j]]
   #                  edg.append((center, neighbor))
   #      edge_count = Counter(edg).items()
   #
   #
   #      num_window = len(windows)
   #      pmi_dict = {}
   #      #import pudb;pu.db
   #          #print("calculating pmi...")
   #      for key in edge_count:
   #
   #
   #          i = int(key[0][0])
   #          j = int(key[0][1])
   #          count = key[1]
   #
   #
   #          list_of_key = list(node2index2.keys())
   #          list_of_value = list(node2index2.values())
   #          position1 = list_of_value.index(i)
   #          vocab_i=list_of_key[position1]
   #
   #          position2 = list_of_value.index(j)
   #          vocab_j=list_of_key[position2]
   #          #import pudb;pu.db
   #          word_freq_i = word_window_freq[vocab_i]
   #          word_freq_j = word_window_freq[vocab_j]
   #          #import pudb;pu.db
   #          pmi = (1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j / (num_window * num_window))
   #
   #
   #          if pmi <= 0:
   #              continue
   #
   #
   #          edges2.append((i, j))
   #          weight.append(pmi)
   #
   #
   #
   #
   #      # word_pair_count = {}
   #      # for window in windows:
   #      #     for i in range(1, len(window)):
   #      #         for j in range(0, i):
   #      #             word_i = window[i]
   #
   #      #             word_i_id = node2index2[word_i]
   #      #             # word_i_id = word_id_map[word_i]
   #      #             word_j = window[j]
   #      #             word_j_id = node2index2[word_j]
   #      #             # word_j_id = word_id_map[word_j]
   #      #             if word_i_id == word_j_id:
   #      #                 continue
   #      #             word_pair_str = str(word_i_id) + ',' + str(word_j_id)
   #      #             if word_pair_str in word_pair_count:
   #      #                 word_pair_count[word_pair_str] += 1
   #      #             else:
   #      #                 word_pair_count[word_pair_str] = 1
   #      #         # two orders
   #      #             word_pair_str = str(word_j_id) + ',' + str(word_i_id)
   #
   #      #             if word_pair_str in word_pair_count:
   #      #                 word_pair_count[word_pair_str] += 1
   #      #             else:
   #      #                 word_pair_count[word_pair_str] = 1
   #
   #
   #      # num_window = len(windows)
   #      # pmi_dict = {}
   #      # #print("calculating pmi...")
   #      # for key in word_pair_count:
   #      #     temp = key.split(',')
   #      #     i = int(temp[0])
   #      #     j = int(temp[1])
   #      #     count = word_pair_count[key]
   #
   #
   #      #     list_of_key = list(node2index2.keys())
   #      #     list_of_value = list(node2index2.values())
   #      #     position1 = list_of_value.index(i)
   #      #     vocab_i=list_of_key[position1]
   #
   #      #     position2 = list_of_value.index(j)
   #      #     vocab_j=list_of_key[position2]
   #      #     #import pudb;pu.db
   #      #     word_freq_i = word_window_freq[vocab_i]
   #      #     word_freq_j = word_window_freq[vocab_j]
   #      #     pmi = log((1.0 * count / num_window) /
   #      #                 (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
   #      #     if pmi <= 0:
   #      #         continue
   #      #     # row.append(train_size + i)
   #      #     # col.append(train_size + j)
   #
   #      #     edges.append((i, j))
   #      #     weight.append(pmi)
   #          # pmi_dict[key] = pmi
   #          #import pudb;pu.db
   #
   #      #print("create pmi graph...")
   #
   #      edge_count2 = Counter(edges2).items()
   #      row = [x for (x, y), c in edge_count2]
   #      col = [y for (x, y), c in edge_count2]
   #      # import pudb;pu.db
   #      #import pudb;pu.db
   #      adj_part = sp.csr_matrix(
   #              (weight, (row, col)), shape=(len(nodes), len(nodes)))
   #      adj_normalized = normalize_adj(adj_part)
   #      weight_normalized = [adj_normalized[x][y] for (x, y), c in edge_count2]
   #      #import pudb;pu.db
   #      inputs.append(nodes)
   #      graphs.append([row, col, weight_normalized])
   #
   #
   #  len_inputs = [len(e) for e in inputs]
   #  len_graphs = [len(x) for x, y, c in graphs]
   #
   #
   #  #padding input
   #  pad_len_inputs = max(len_inputs)
   #  pad_len_graphs = max(len_graphs)
   #  inputs_pad = [pad_seq(e, pad_len_inputs) for e in tqdm(inputs)]
   #  graphs_pad = [[pad_seq(ee, pad_len_graphs) for ee in e] for e in tqdm(graphs)]
   #
   #  inputs_pad = np.array(inputs_pad)
   #  weights_pad = np.array([c for x, y, c in graphs_pad])
   #  graphs_pad = np.array([[x, y] for x, y, c in graphs_pad])
   #
   #  # word2vec
   #  all_vectors = np.load(f"home/TextING-pytorch/source/glove.6B.{embedding_dim}d.npy")
   #  all_words = joblib.load(f"home/TextING-pytorch/source/glove.6B.words.pkl")
   #  all_word2index = {w: i for i, w in enumerate(all_words)}
   #  index2word = {i: w for w, i in word2index.items()}
   #  word_set = [index2word[i] for i in range(len(index2word))]
   #  oov = np.random.normal(-0.1, 0.1, embedding_dim)
   #  word2vec = [all_vectors[all_word2index[w]] if w in all_word2index else oov for w in word_set]
   #  word2vec.append(np.zeros(embedding_dim))
   # #import pudb;pu.db
   #  # save
   #  joblib.dump(len_inputs, f"home/TextING-pytorch/temp/{dataset}.len.inputs_pminolog_norm_7.pkl")
   #  joblib.dump(len_graphs, f"home/TextING-pytorch/temp/{dataset}.len.graphs_pminolog_norm_7.pkl")
   #  np.save(f"home/TextING-pytorch/temp/{dataset}.inputs_pminolog_norm_7.npy", inputs_pad)
   #  np.save(f"home/TextING-pytorch/temp/{dataset}.graphs_pminolog_norm_7.npy", graphs_pad)
   #  np.save(f"home/TextING-pytorch/temp/{dataset}.weights_pminolog_norm_7.npy", weights_pad)
   #  np.save(f"home/TextING-pytorch/temp/{dataset}.word2vec_pminolog_norm_7.npy", word2vec)


    
   


    word2index = joblib.load(f"home/TextING-pytorch/temp/{dataset}.word2index.pkl")
    with open(f"home/TextING-pytorch/temp/{dataset}.texts.remove.txt", "r") as f:
        texts = f.read().strip().split("\n")
    # bulid graph
    inputs = []
    graphs = []
    for text in tqdm(texts):
        words = [word2index[w] for w in text.split()]
        words = words[:max_text_len]  # 限制最大长度
        nodes = list(set(words))
        node2index = {e: i for i, e in enumerate(nodes)}
        #import pudb;pu.db
        edges = []
        for i in range(len(words)):
            center = node2index[words[i]]
            for j in range(i - window_size, i + window_size + 1):
                if i != j and 0 <= j < len(words):
                    neighbor = node2index[words[j]]
                    edges.append((center, neighbor))
        edge_count = Counter(edges).items()
        row = [x for (x, y), c in edge_count]
        col = [y for (x, y), c in edge_count]
        weight = [c for (x, y), c in edge_count]
        adj = sp.csr_matrix((weight, (row, col)), shape=(len(nodes), len(nodes)))
        # import pudb;pu.db
        adj_normalized = normalize_adj(adj)
        weight_normalized = [adj_normalized[x][y] for (x, y), c in edge_count]
        #import pudb;pu.db
        inputs.append(nodes)
        graphs.append([row, col, weight_normalized])
        
    len_inputs = [len(e) for e in inputs]
    len_graphs = [len(x) for x, y, c in graphs]

    # padding input
    pad_len_inputs = max(len_inputs)
    pad_len_graphs = max(len_graphs)
    inputs_pad = [pad_seq(e, pad_len_inputs) for e in tqdm(inputs)]
    graphs_pad = [[pad_seq(ee, pad_len_graphs) for ee in e] for e in tqdm(graphs)]

    inputs_pad = np.array(inputs_pad)
    weights_pad = np.array([c for x, y, c in graphs_pad])
    graphs_pad = np.array([[x, y] for x, y, c in graphs_pad])

    # word2vec
    all_vectors = np.load(f"home/TextING-pytorch/source/glove.6B.{embedding_dim}d.npy")
    all_words = joblib.load(f"home/TextING-pytorch/source/glove.6B.words.pkl")
    all_word2index = {w: i for i, w in enumerate(all_words)}
    index2word = {i: w for w, i in word2index.items()}
    word_set = [index2word[i] for i in range(len(index2word))]
    oov = np.random.normal(-0.1, 0.1, embedding_dim)
    word2vec = [all_vectors[all_word2index[w]] if w in all_word2index else oov for w in word_set]
    word2vec.append(np.zeros(embedding_dim))


    # save
    joblib.dump(len_inputs, f"home/TextING-pytorch/temp/{dataset}.len.inputs.pkl")
    joblib.dump(len_graphs, f"home/TextING-pytorch/temp/{dataset}.len.graphs.pkl")
    np.save(f"home/TextING-pytorch/temp/{dataset}.inputs.npy", inputs_pad)
    np.save(f"home/TextING-pytorch/temp/{dataset}.graphs.npy", graphs_pad)
    np.save(f"home/TextING-pytorch/temp/{dataset}.weights.npy", weights_pad)
    np.save(f"home/TextING-pytorch/temp/{dataset}.word2vec.npy", word2vec)