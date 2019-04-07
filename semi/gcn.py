import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            adj_uns = torch.unsqueeze(adj, dim=0)
            seq_fts_uns = torch.unsqueeze(seq_fts, dim=0)
            #print("adj uns:", adj_uns.shape)
            #print("seq fts uns:", seq_fts_uns.shape)
            out = torch.bmm(adj_uns, seq_fts_uns)
            out = torch.squeeze(out, dim=0)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class GCNet(nn.Module):
    def __init__(self, nb_features, nb_classes):
        super(GCNet, self).__init__()
        self.gcn1 = GCN(nb_features, 64, nn.ReLU())
        self.gcn2 = GCN(64, nb_classes, lambda x: x)

    def forward(self, fts, adj):
        h_1 = self.gcn1(fts, adj)
        logits = self.gcn2(h_1, adj)
        return logits

