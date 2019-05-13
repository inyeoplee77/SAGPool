import os.path as osp
from math import ceil
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv as GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from GCNConv_sagpool_2hop import SAGPool
from earlystopping import Earlystop
import random
import argparse
import pickle
import os
import csv
import time

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='Fixed seed for reproducibility')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--ratio', type=float, default=0.8,
                    help='Keeping node ratio')
parser.add_argument('--dataname', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/Letter-high(med,low)/TWITTER-Real-Graph-Partial/COIL-DEL/COIL-RAG/FRANKENSTEIN')
parser.add_argument('--mode', type=str, default='GCNConv_search',
                    help='GCNConv_search/GCNConv_seed')

args = parser.parse_args()
#args.batch_size=128
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.ratio != 0.8:
    raise ValueError

if args.weight_decay == 0.01 or args.lr == 0.00005:
    raise ValueError

random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

def my_transform(data):
    data.x = F.normalize(data.x,p=2,dim=-1)
    #data.two_hop = make_two_hop(data.edge_index)
    return data

def transform(data):
    #data.x = F.normalize(data.x,p=2,dim=-1)
    data.two_hop = make_two_hop(data.edge_index)
    return data

def make_two_hop(edge_index):
    sp_A = torch.sparse.FloatTensor(edge_index.to('cpu'), torch.ones(edge_index.shape[1]),torch.Size([edge_index.shape[1],edge_index.shape[1]])) 
    ds_A = sp_A.to_dense() 
    return (ds_A + sp_A.matmul(ds_A)).to_sparse()._indices().to(device)


if not os.path.exists(os.path.join('./data',args.dataname)):
    os.mkdir(os.path.join('./data',args.dataname))

data_seed_dir = os.path.join('./data',args.dataname,str(args.seed)+'.pkl')

if not os.path.exists(data_seed_dir):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataname)
    dataset = TUDataset(path, name= args.dataname,pre_transform=my_transform)
    max_nodes = max([x.num_nodes for x in dataset])
    dataset.max_nodes = max_nodes
    dataset = dataset.shuffle()
    dataset = dataset.shuffle()
    with open(data_seed_dir, 'wb') as f:
        pickle.dump(dataset, f)
    print('Seed Data Saved : ',data_seed_dir)
else:
    with open(data_seed_dir, 'rb') as f:
        dataset = pickle.load(f)
    print('Seed Data Loaded : ',data_seed_dir)
dataset.transform = transform

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        nhid = args.nhid

        self.conv1 = GraphConv(dataset.num_features, nhid)
        self.pool1 = SAGPool(nhid, ratio=args.ratio)
        self.conv2 = GraphConv(nhid, nhid)
        self.pool2 = SAGPool(nhid, ratio=args.ratio)
        self.conv3 = GraphConv(nhid, nhid)
        self.pool3 = SAGPool(nhid, ratio=args.ratio)

        self.lin1 = torch.nn.Linear(nhid*2, nhid)
        self.lin2 = torch.nn.Linear(nhid, int(nhid/2))
        self.lin3 = torch.nn.Linear(int(nhid/2), dataset.num_classes)

    def forward(self, data):
        x, edge_index, two_hop, batch = data.x, data.edge_index, data.two_hop, data.batch
        #two_hop = make_two_hop(edge_index)

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, two_hop,_, batch, _ = self.pool1(x, edge_index, two_hop, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, two_hop, _, batch, _ = self.pool2(x, edge_index, two_hop, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, _, batch, _ = self.pool3(x, edge_index, two_hop, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(loader):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)



def test(loader):
    model.eval()

    correct = 0
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss = F.nll_loss(output, data.y)

        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.sum(param**2)
        loss += args.weight_decay * torch.sqrt(l2_reg)
        
        loss_all += data.num_graphs * loss.item()
    return correct / len(loader.dataset), loss_all / len(loader.dataset) 

# split data to 10-fold
fold_size = ceil(len(dataset)/10)
folded_data = []
for i in range(10):
    if i != 9:
        folded_data.append(dataset[i*fold_size:(i+1)*fold_size])
    elif i == 9:
        folded_data.append(dataset[i*fold_size:])

final_acc = 0.
best_acc = 0.

for fold in range(10):

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    earlystop = Earlystop(window_size = 50)
    train_dataset = []
    test_dataset = []
    for j in range(10):
        if j == fold:
            test_dataset = folded_data[j]
        else:
            train_dataset.extend(folded_data[j])
    valid_idx = int(len(train_dataset)*0.1)
    valid_dataset = train_dataset[:valid_idx]
    train_dataset = train_dataset[valid_idx:]

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    
    fold_best_acc = 0.
    fold_val_loss = 100.
    fold_val_acc = 0.
    patience = 0
    
    for epoch in range(1, 4001):
        t = time.time()
        loss = train(train_loader)
        val_acc,val_loss = test(valid_loader)
        train_acc,_ = test(train_loader)
        
        
        if fold_val_loss >= val_loss:
            fold_val_loss = val_loss

            test_acc,test_loss = test(test_loader)
            fold_best_acc = test_acc
        
            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Valid Acc: {:.5f}, Test Acc: {:.5f}, Time: {:.2f}s'.
                format(epoch, loss, train_acc, val_acc, test_acc,time.time()-t))
            patience=0
        elif patience >= 50:
            print("{:01d} Fold Early Stop Epoch: {:03d}".format(fold,epoch))
            break
        else:
            patience += 1
        '''
        if val_acc >= fold_val_acc:
            fold_val_acc = val_acc

            test_acc,test_loss = test(test_loader)
            fold_best_acc = test_acc
        
            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Valid Acc: {:.5f}, Test Acc: {:.5f}'.
                format(epoch, loss, train_acc, val_acc, test_acc))
        '''
        '''
        stop_TF_loss = earlystop.loss_stop(val_loss)
        stop_TF_acc = earlystop.acc_stop(val_acc)
        if stop_TF_loss:
            print("{:01d} Fold Early Stop Epoch: {:03d}".format(fold,epoch))
            break
        '''
    best_acc += fold_best_acc

    del model
    del optimizer
print('Final Accuracy : ', best_acc/10)

f = open(os.path.join('./data',args.dataname,args.mode+'_2hop_sagpool_result.csv'), 'a', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow([best_acc/10,args.seed,args.lr,args.nhid,args.weight_decay,args.batch_size,args.ratio])
f.close()