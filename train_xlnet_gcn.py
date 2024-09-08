import torch as th
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import logging
from torch.optim import lr_scheduler
from model import XLG_Net
import pickle
import random

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for xlnet')
parser.add_argument('--batch_size', type=int, default=64, help='the default is 64')
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--model_init', type=str, default='xlnet_base_cased')
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--checkpoint_dir', default=None,
                    help='checkpoint directory, XLGNet_[dataset] if not specified')
parser.add_argument('--n_hidden', type=int, default=200,
                    help='the dimension of gcnii hidden layer')

parser.add_argument('-m', '--m', type=float, default=0.6, help='the factor balancing XLNetMix and GCNII prediction')
parser.add_argument('--dataset', default='mr', choices=['R52', 'ohsumed', 'mr'])
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--bert_lr', type=float, default=1e-6)
parser.add_argument('--gcn_lr', type=float, default=2e-3)
parser.add_argument('--mix_layer_set', nargs='+', default=[9, 10, 12], type=int, help='define a set of mix layers')
parser.add_argument('--gcn_layers', type=int, default=8)

parser.add_argument('--alpha', default=0.75, type=float, help='alpha for beta distribution')
parser.add_argument('--beta', default=-1, type=float, help='another param for beta distribution')
parser.add_argument('--tau', default=1, type=float, help='tau for dirichlet distribution')
parser.add_argument('--seed', type=int, default=77, help="random seed for initialization")
parser.add_argument('--n_sample', type=int, default=2, help='num of aug samples')
parser.add_argument('--gcnii_alpha', type=float, default=0.1, help='gcnii alpha')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')

args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
model_init = args.model_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr
gcnii_alpha = args.gcnii_alpha

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/XLGNet_{}'.format(dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
gpu = th.device('cuda:0')

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

# Data Preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
adj: n*n sparse adjacency matrix
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
'''

# compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# Model
model = XLG_Net(nb_class=nb_class, pretrained_model=f'./pre_model/{model_init}', m=m, n_layer=gcn_layers,
                n_hidden=n_hidden, dropout=dropout, alpha=gcnii_alpha)

# load documents and compute input encodings
corpse_file = './data/corpus/' + dataset + '_shuffle.txt'
with open(corpse_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')


def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask


input_ids, attention_mask = encode_input(text, model.tokenizer)
# train + valid + vocab(零张量) + test
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat(
    [attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

with open(f"./mix_data/{dataset}/{dataset}_de.pkl", 'rb') as f1, open(
        f"./mix_data/{dataset}/{dataset}_ru.pkl", 'rb') as f2:
    de_data = list(pickle.load(f1).values())
    ru_data = list(pickle.load(f2).values())
input_ids1 = model.tokenizer(de_data, max_length=128, truncation=True, padding='max_length',
                             return_tensors='pt').input_ids
input_ids2 = model.tokenizer(ru_data, max_length=128, truncation=True, padding='max_length',
                             return_tensors='pt').input_ids
attention_mask1 = model.tokenizer(de_data, max_length=128, truncation=True, padding='max_length',
                                  return_tensors='pt').attention_mask
attention_mask2 = model.tokenizer(de_data, max_length=128, truncation=True, padding='max_length',
                                  return_tensors='pt').attention_mask

input_ids1 = th.cat([input_ids1[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids1[-nb_test:]])
attention_mask1 = th.cat(
    [attention_mask1[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask1[-nb_test:]])
input_ids2 = th.cat([input_ids2[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids2[-nb_test:]])
attention_mask2 = th.cat(
    [attention_mask2[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask2[-nb_test:]])

# transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
# y存储的就是标签id
y = y.argmax(axis=1)

# document mask used for update feature
doc_mask = train_mask + val_mask + test_mask

# build DGL Graph
# 图归一化, sp.eye(adj.shape[0]) 创建了一个单位矩阵，然后将它与原始邻接矩阵相加，以确保每个节点都有一个自连接。
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
# 将稀疏矩阵转为tensor
adj = sys_normalized_adjacency(adj)
adj = sparse_mx_to_torch_sparse_tensor(adj)
adj = adj.to(gpu)
# 将编码后的数据存储在图节点中
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['input_ids1'], g.ndata['attention_mask1'] = input_ids1, attention_mask1
g.ndata['input_ids2'], g.ndata['attention_mask2'] = input_ids2, attention_mask2
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

logger.info('graph information:')
logger.info(str(g))

# create index loader
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node - nb_test, nb_node, dtype=th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)


# Training
def update_feature():
    global model, g, doc_mask
    # mask掉vocab
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.xlnet_model(hidden_states=input_ids, attention_mask=attention_mask)[:, -1]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    # 更新图，不包含vocab
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)


optimizer = th.optim.Adam([
    {'params': model.xlnet_model.parameters(), 'lr': bert_lr},
    {'params': model.classifier.parameters(), 'lr': bert_lr},
    {'params': model.gcnii.params1, 'weight_decay': args.wd1, 'lr': gcn_lr},
    {'params': model.gcnii.params2, 'weight_decay': args.wd2, 'lr': bert_lr},
], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.15)

set_seed(args)
# mix_layer_set这个列表中随机选择一个混合层进行数据混合，减去1，因为通常混合层的索引是从0开始的。
mix_layer = np.random.choice(args.mix_layer_set, 1)[0] - 1
# 确定lam的值，lam为混合时原始数据的混合权值。
# 从Beta分布中随机抽样的一个值
if args.beta == -1:
    lam = np.random.beta(args.alpha, args.alpha)
else:
    lam = np.random.beta(args.alpha, args.beta)
# 确保混合后的数据更接近原始数据，lam要更大
lam = max(lam, 1 - lam)
# dirichlet分布，获取混合增强数据时各个增强数据的隐藏层的权重
ws = np.random.dirichlet([args.tau] * args.n_sample)


def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx,) = [x.to(gpu) for x in batch]
    # 将0/1转化为bool型
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    # bertgcn的预测矩阵 batch * n_class
    # 混合预测结果
    mix_outputs = model(g, adj, idx, lam, ws, mix_layer, 1)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(mix_outputs, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = mix_outputs.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx,) = [x.to(gpu) for x in batch]
        y_pred = model(g, adj, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true


evaluator = Engine(test_step)
metrics = {
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'xlnet_model': model.xlnet_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcnii': model.gcnii.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc


log_training_results.best_val_acc = 0
g = update_feature()
trainer.run(idx_loader, max_epochs=nb_epochs)
