




# #python home/TextING-pytorch/preprocess.py

import nltk

# nltk.download("stopwords")

from nltk.corpus import stopwords
from collections import Counter
import re
import joblib
import numpy as np

#DDoS2019

dataset = "ddos2019-confuse0.8"
# param
stop_words = set(stopwords.words('english'))
least_freq = 5
if dataset == "mr" or "SST" in dataset:
    stop_words = set()
    least_freq = 0


# func load texts & labels
def load_dataset(dataset):
    with open(f"home/TextING-pytorch/corpus/{dataset}.texts.txt", "r", encoding="latin1") as f:
        texts = f.read().strip().split("\n")
    # with open(f"home/TextING-pytorch/corpus/{dataset}.labels.txt", "r") as f:
    #     labels = f.read().strip().split("\n")

    labels=[]
    with open(f"home/TextING-pytorch/corpus/{dataset}.labels.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            # import pudb;pu.db
            temp = line.split()[2].split()
            # label1 = ",".join(temp)
            labels.append(",".join(temp))




    return texts, labels


def filter_text(text: str):
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", text)
    text = text.replace("'ll ", " will ")
    text = text.replace("'d ", " would ")
    text = text.replace("'m ", " am ")
    text = text.replace("'s ", " is ")
    text = text.replace("'re ", " are ")
    text = text.replace("'ve ", " have ")
    text = text.replace(" can't ", " can not ")
    text = text.replace(" ain't ", " are not ")
    text = text.replace("n't ", " not ")
    text = text.replace(",", " , ")
    text = text.replace("!", " ! ")
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    text = text.replace("?", " ? ")
    text = re.sub(r"\s{2,}", " ", text)
    return " ".join(text.strip().split())


if __name__ == '__main__':
    texts, labels = load_dataset(dataset)

    # handle texts
    texts_clean = [filter_text(t) for t in texts]

    word2count = Counter([w for t in texts_clean for w in t.split()])
    # word_count = [[w, c] for w, c in word2count.items() if c >= least_freq and w not in stop_words]
    word_count = [[w, c] for w, c in word2count.items()]
    word2index = {w: i for i, (w, c) in enumerate(word_count)}

    words_list = [[w for w in t.split() if w in word2index] for t in texts_clean]

    texts_remove = [" ".join(ws) for ws in words_list]

    # labels 2 targets
    label2index = {l: i for i, l in enumerate(set(labels))}
    targets = [label2index[l] for l in labels]
    #import pudb;pu.db
    # save
    with open(f"home/TextING-pytorch/temp/{dataset}.texts.clean.txt", "w") as f:
        f.write("\n".join(texts_clean))

    with open(f"home/TextING-pytorch/temp/{dataset}.texts.remove.txt", "w") as f:
        f.write("\n".join(texts_remove))

    np.save(f"home/TextING-pytorch/temp/{dataset}.targets.npy", targets)
    joblib.dump(word2index, f"home/TextING-pytorch/temp/{dataset}.word2index.pkl")




