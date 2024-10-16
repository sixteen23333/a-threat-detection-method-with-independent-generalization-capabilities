import time
from sklearn import metrics
from torch import nn
import torch
from config import *
from dataset import get_data_loader
from model import Model
import os
import psutil

import sys
import codecs
import time
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#python home/TextING-pytorch/train.py

def train_eval(cate, loader, model, optimizer, loss_func, device):
    model.train() if cate == "train" else model.eval()
    preds, labels, loss_sum = [], [], 0.

    for i in range(len(loader)):
        loss = torch.tensor(0., requires_grad=True).float().to(device)
        #import pudb;pu.db
        for j, graph in enumerate(loader[i]):
            graph = graph.to(device)
            #import pudb;pu.db
            targets = graph.y
            # import pudb;pu.db
            y = model(graph)
            loss += loss_func(y, targets)
            #import pudb;pu.db


            preds.append(y.max(dim=1)[1].data)
            labels.append(targets.data)

        loss = loss / len(loader[i])

        
        if cate == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.data
    
    #import pudb;pu.db
    
    
    preds = torch.cat(preds).tolist()
    labels = torch.cat(labels).tolist()
    loss = loss_sum / len(loader)
    acc = metrics.accuracy_score(labels, preds) * 100
    #import pudb;pu.db
    return loss, acc, preds, labels

#CUDA_VISIBLE_DEVICES=0 python home/TextING-pytorch/train.py


if __name__ == '__main__':
    dataset = "DDoS2019"
    #DDoS2019
    print("load dataset")
    # params
    batch_size = 500  # 反向传播时的batch
    mini_batch_size = 64  # 计算时的batch
    lr = 0.008
    dropout = 0.6
    weight_decay = 0.
    hid_dim = 96
    freeze = True
    start = 0

    num_classes = args[dataset]['num_classes']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    (train_loader, test_loader, valid_loader), word2vec = get_data_loader(dataset, batch_size, mini_batch_size)
    num_words = len(word2vec) - 1 #255




    #device = torch.device('cpu')
    #import pudb;pu.db
    model = Model(num_words, num_classes, word2vec=word2vec, hid_dim=hid_dim, freeze=freeze)

    # x = torch.load('home/TextING-pytorch/TFC2016-CNNparams.pkl')
    # del x['embed.weight']
    # del x['read.mlp.1.weight']
    # del x['read.mlp.1.bias']
    # model.load_state_dict(x, strict=False)


    # model.load_state_dict(torch.load('home/TextING-pytorch/TFC2016-CNNparams.pkl'),strict=False)


    loss_func = nn.CrossEntropyLoss()
    #import pudb;pu.db
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model = model.to(device)
    #import pudb;pu.db
    print("-" * 50)
    print(f"params: [start={start}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}]")
    print("-" * 50)
    print(model)
    print("-" * 50)
    print(dataset)

    t_begin = time.time()
    best_acc = 0.
    for epoch in range(start + 1, 100):
        t_epoch_begin = time.time()
        t1 = time.time()
        train_loss, train_acc, _, _ = train_eval("train", train_loader, model, optimizer, loss_func, device)
        t2 = time.time()
        with torch.no_grad():
            valid_loss, valid_acc, _, _ = train_eval("valid", valid_loader, model, optimizer, loss_func, device)
            test_loss, test_acc, preds, labels = train_eval("test", test_loader, model, optimizer, loss_func, device)
        #import pudb;pu.db
        if best_acc < test_acc:
            best_acc = test_acc

        cost = time.time() - t1
        t_train = t2-t1
        t_test = time.time()-t2
        print((f"epoch={epoch:03d}, cost={cost:.2f}, "
               f"train:[{train_loss:.4f}, {train_acc:.2f}%], "
               f"valid:[{valid_loss:.4f}, {valid_acc:.2f}%], "
               f"test:[{test_loss:.4f}, {test_acc:.2f}%], "
               f"t:[{t_train:.4f}, {t_test:.4f}%], "
               f"best_acc={best_acc:.2f}%"))



    # torch.save(model.state_dict(), 'home/TextING-pytorch/ids2012-CNNparams.pkl')
    print("Test Precision, Recall and F1-Score...")
    print(metrics.classification_report(labels, preds, digits=4))
    print("Macro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(labels, preds, average='macro'))
    print("Micro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(labels, preds, average='micro'))
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )