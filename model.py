import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import nn as gnn


class GRUUint(nn.Module):

    def __init__(self, hid_dim, act):
        super(GRUUint, self).__init__()
        self.act = act
        self.lin_z0 = nn.Linear(hid_dim, hid_dim)
        self.lin_z1 = nn.Linear(hid_dim, hid_dim)
        self.lin_r0 = nn.Linear(hid_dim, hid_dim)
        self.lin_r1 = nn.Linear(hid_dim, hid_dim)
        self.lin_h0 = nn.Linear(hid_dim, hid_dim)
        self.lin_h1 = nn.Linear(hid_dim, hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, a):
        z = (self.lin_z0(a) + self.lin_z1(x)).sigmoid()
        r = (self.lin_r0(a) + self.lin_r1(x)).sigmoid()
        h = self.act((self.lin_h0(a) + self.lin_h1(x * r)))
        return h * z + x * (1 - z)


class GraphLayer(gnn.MessagePassing):

    def __init__(self, in_dim, out_dim, dropout=0.5,
                 act=torch.relu, bias=False, step=2):
        super(GraphLayer, self).__init__(aggr='add')
        self.step = step
        self.act = act
        self.encode = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        self.gru = GRUUint(out_dim, act=act)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, g):
        #import pudb;pu.db
        #x.size:7551.300
        x = self.encode(x)
        x = self.act(x)
        b=g.length
        #import pudb;pu.db
        for _ in range(self.step):
            c=g.edge_index
            d=self.dropout(g.edge_attr)
            #import pudb;pu.db
            a = self.propagate(edge_index=g.edge_index, x=x, edge_attr=self.dropout(g.edge_attr))
            
            x = self.gru(x, a)
        #import pudb;pu.db
        x = self.graph2batch(x, g.length)
        return x

    def message(self, x_j, edge_attr):
        return x_j * edge_attr.unsqueeze(-1)

    def update(self, inputs):
        return inputs

    def graph2batch(self, x, length):
        x_list = []
        c=length
        #import pudb;pu.db
        for l in length:
            
            x_list.append(x[:l])
            
            x = x[l:]
        x = pad_sequence(x_list, batch_first=True)
        return x


class ReadoutLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5,
                 act=torch.relu, bias=False):
        super(ReadoutLayer, self).__init__()
        self.act = act
        self.bias = bias
        self.att = nn.Linear(in_dim, 1, bias=True)
        self.emb = nn.Linear(in_dim, in_dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        #import pudb;pu.db
        #x.size:[64, 251, 96];[64, 213, 96]
        att = self.att(x).sigmoid()
        emb = self.act(self.emb(x))
        x = att * emb
        x = self.__max(x, mask) + self.__mean(x, mask)
        x = self.mlp(x)


        # x1= x.tolist()
        # with open(f"home/TextING-pytorch/score", "w") as f:
        #     f.write("\n".join(str(x) for x in x1))
        # import pudb;pu.db
        return x

    def __max(self, x, mask):
        return (x + (mask - 1) * 1e9).max(1)[0]

    def __mean(self, x, mask):
        return (x * mask).sum(1) / mask.sum(1)


class Model(nn.Module):
    def __init__(self, num_words, num_classes, in_dim=300, hid_dim=96,
                 step=2, dropout=0.5, word2vec=None, freeze=True):
        super(Model, self).__init__()
        if word2vec is None:
            self.embed = nn.Embedding(num_words + 1, in_dim, num_words)
        else:
            self.embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(word2vec).float(), freeze, num_words)
        
        self.gcn = GraphLayer(in_dim, hid_dim, act=torch.tanh, dropout=dropout, step=step)
        self.read = ReadoutLayer(hid_dim, num_classes, act=torch.tanh, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, g):
        mask = self.get_mask(g)
        x = self.embed(g.x)
        #import pudb;pu.db
        x = self.gcn(x, g)
        x = self.read(x, mask)
        return x

    def get_mask(self, g):
        mask = pad_sequence([torch.ones(l) for l in g.length], batch_first=True).unsqueeze(-1)
        if g.x.is_cuda: mask = mask.cuda()
        return mask


