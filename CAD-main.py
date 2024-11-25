import numpy as np
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.utils import column_or_1d
from sklearn.metrics import roc_auc_score, average_precision_score
import math
import json
import random
import matplotlib.pyplot as plt
import argparse
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def setup_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class ODDataset(Dataset):
    def __init__(self, y):
        self.y = torch.Tensor(y)

    def __getitem__(self, idx):
        return idx, self.y[idx]
    
    def __len__(self):
        return len(self.y)

class distance(nn.Module): 
    def __init__(self, hidden_dim, n_channels=4):
        super(distance, self).__init__()
        self.W = nn.Parameter(torch.randn(n_channels, hidden_dim, hidden_dim))

    def forward(self, x0, X): 
        x0 = torch.matmul(x0.unsqueeze(1).unsqueeze(1), self.W).squeeze() # N1, C, H
        out = torch.tanh(torch.matmul(x0, X.T)).squeeze() # N1, C, N
        return out.mean(dim=1)

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_channels=4):
        super(Model, self).__init__()
        middle_dim = int((input_dim + hidden_dim) / 2)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, middle_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(middle_dim, hidden_dim),
        )
        
        self.distance = distance(hidden_dim, n_channels)
    
    def forward(self, x0_idx, X):
        X = self.fc(X)
        x0 = X[x0_idx]
        out = self.distance(x0, X)
        return out



class NCELoss2(nn.Module):
    def __init__(self):
        super(NCELoss2, self).__init__()
    
    def forward(self, out, y_batch): 
        # out = torch.clamp(out, max=30)
        out.diagonal().fill_(0)
        d0 = out[y_batch == 0]
        d00 = d0[:, y_batch == 0].mean(dim=1)
        d00 = torch.exp(d00)
        d00_sum = d00.sum()
        loss = torch.tensor([]).to(device)
        if torch.any(y_batch == 1):
            d1 = out[y_batch == 1]
            d10 = d1[:, y_batch == 0].mean(dim=1)
            d10 = torch.exp(d10)
            loss1 = -torch.log(d10 / (d10 + d00_sum) + 1e-7)
            loss = loss1
        if torch.any(y_batch == 3):
            d3 = out[y_batch == 3]
            d30 = d3[:, y_batch == 0].mean(dim=1)
            d30 = torch.exp(d30)
            loss2 = 0.3 * -torch.log(d30 / (d30 + d00_sum) + 1e-7)
            loss = torch.cat((loss, loss2), dim=0)

        return loss.mean()
    

def normal_sample_denoising(X, y):
    try:
        clf = CBLOF()
        clf.fit(X)
    except:
        clf = IForest()
        clf.fit(X)
    scores = clf.decision_scores_
    m = np.mean(scores)
    d = np.std(scores)
    threshold = m + 3 * d  
    indices = np.where(scores < threshold)[0]
    n_pseudos = len(y) - len(indices)
    print(sum(y[indices]), len(y[indices]))
    return indices, n_pseudos


def train(config, anomaly_train_loader, normal_train_loader, test_loader, train_size, X_train, y_train, X_inference, y_true, train_normal_indice, train_anomaly_indice):
    d, hidden_layer, n_channels, n_epochs, best_loss, step, early_stop_count = config.dimention, config.hidden_layer, config.n_channels, config.n_epochs, math.inf, 0, 0
    best_loss = 100000
    best_roc = 0
    best_prn = 0

    criterion2 = NCELoss2() 

    model = Model(d, hidden_layer, n_channels=n_channels).to(device)
    X_train = torch.FloatTensor(X_train).to(device)
    X_inference = torch.FloatTensor(X_inference).to(device)
    ps_anomaly1 = None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0.01*config.learning_rate)
    train_loss = []
    roc = []
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_bar = tqdm(normal_train_loader, position=0, leave=True)
        for x0_idx, y_batch in train_bar:
            n_anomaly = random.randint(config.min_anomaly, config.max_anomaly)
            anomaly_idx, anomaly_y = next(iter(anomaly_train_loader))
            indices = torch.randperm(anomaly_idx.shape[0])[:n_anomaly]
            anomaly_idx, anomaly_y = anomaly_idx[indices], anomaly_y[indices]

            x0_idx = torch.cat((anomaly_idx, x0_idx), dim=0)
            y_batch = torch.cat((anomaly_y, y_batch), dim=0)
            optimizer.zero_grad()
                
            H = model.fc(X_train[x0_idx])
            D = model.distance(H, H)
            loss1 = criterion2(D, y_batch)
        
            loss1.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss1.detach().item())
            train_bar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_bar.set_postfix({'loss': loss1.detach().item()})
        scheduler.step()
        mean_loss = np.mean(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}], loss: {mean_loss}]')
        train_loss.append(mean_loss)


        model.eval()  
        num_label = config.n_pseudos
        H = model.fc(X_inference)
        H_mean = H.mean(dim=0)
        Y = torch.tensor([]) 
        Out = torch.tensor([])
        for x0_idx, y in test_loader:
            h = H[x0_idx]
            out = model.distance(h, H_mean.unsqueeze(0)).detach().cpu()
            Out = torch.cat((Out, out), dim=-1)
            Y = torch.cat((Y, y), dim=-1)
        

        if epoch > int(config.n_epochs/2):   # Stage 2 - Anomalous sample expansion
            _, expanded_anomalies = torch.topk(Out, num_label)
            correct1 = y_true[expanded_anomalies].sum()
            print(correct1, correct1/num_label)
            expanded_anomalies = expanded_anomalies[expanded_anomalies < train_size]
            y_train_new = y_train.copy()
            y_train_new[expanded_anomalies] = 3
            train_set = ODDataset(y_train_new)
            normal_train_set = Subset(train_set, train_normal_indice)
            anomaly_train_set = Subset(train_set, train_anomaly_indice)
            normal_train_loader = DataLoader(normal_train_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)
            anomaly_train_loader = DataLoader(anomaly_train_set, batch_size=len(train_anomaly_indice), shuffle=True, pin_memory=True) 


        out, y_test = Out[train_size:], Y[train_size:]
        score = out.detach().cpu().numpy()
        score = column_or_1d(score)
        y_test = column_or_1d(y_test)
        test_roc = np.round(roc_auc_score(y_test, score), decimals=4)
        test_prn = np.round(average_precision_score(y_true=y_test, y_score=score, pos_label=1), decimals=4)
        print('test roc:{}, test pr:{}'.format(test_roc, test_prn))
        roc.append(test_roc)
        if epoch > int(config.n_epochs/2):
            if mean_loss < best_loss:
                print('best model saved...')
                best_loss = mean_loss
                best_roc = test_roc
                best_prn = test_prn
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count == config.early_stop:
                print(score, y)
                print('early stop...')
                break

    print(f'best roc:{best_roc}, best prn:{best_prn}')
    return best_roc, best_prn

def run(config, rocs, prns, seed):
    setup_seed(seed)
    print(f'seed: {seed}')

    
    data = json.load(open('./processed_data6/'+config.dataset, 'r'))
    X = np.array(data[0])
    y = np.array(data[1])
    
    print(X.shape, y.shape) 
    config.dimention = X.shape[1]
    config.hidden_layer = next((x for x in [4, 8, 16, 32] if x >= X.shape[1] / 4), 32)
    config.n_channels = 3 * config.hidden_layer
    num_total_anomaly = sum(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    y = np.concatenate((y_train, y_test), axis=0)

    print(X_train.shape, X.shape, sum(y_train), sum(y))


    num_total_anomaly = sum(y)
    num_train_anomaly = int(config.labeled_ratio * num_total_anomaly * 0.8)
    train_anomaly_indice = np.where(y_train == 1)[0]
    train_anomaly_indice = train_anomaly_indice[:num_train_anomaly]
    train_normal_indice, config.n_pseudos = normal_sample_denoising(X_train, y_train)  # Stage 1 - Normal sample denoising
    
    num_train_normals = len(train_normal_indice)
    num_train_anomalies = len(train_anomaly_indice)
    num_train_samples = len(y_train)

    y_unlabel = np.full((len(y_train)), 2)
    y_unlabel[train_normal_indice] = 0
    y_unlabel[train_anomaly_indice] = 1
    X_inference = np.concatenate((X_train, X_test), axis=0)
    y_inference = np.concatenate((y_unlabel, np.full(len(y_test), 2)), axis=0)
    # train_unlabeled_indice = np.where(y_unlabel == 2)[0]

    train_set = ODDataset(y_unlabel)
    # unlabeled_train_set = Subset(train_set, train_unlabeled_indice)
    normal_train_set = Subset(train_set, train_normal_indice)
    anomaly_train_set = Subset(train_set, train_anomaly_indice)
    test_set = ODDataset(y)

    # unlabel_train_loader = DataLoader(unlabeled_train_set, batch_size=config.batch_size, shuffle=True, pin_memory=True) 
    normal_train_loader = DataLoader(normal_train_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    anomaly_train_loader = DataLoader(anomaly_train_set, batch_size=num_train_anomalies, shuffle=True, pin_memory=True) 
    test_loader = DataLoader(test_set, batch_size=config.batch_size*10, shuffle=False, pin_memory=True)
    

    print(f'inlier: {sum(y_unlabel==0)}, anomaly: {sum(y_unlabel==1)}, unlabeled: {sum(y_unlabel==2)}')
    print(f'inlier: {sum(y_inference==0)}, anomaly: {sum(y_inference==1)}, unlabeled: {sum(y_inference==2)}')


    config.max_anomaly = num_train_anomalies
    config.min_anomaly = int(num_train_anomalies* 0.5) 
    if X.shape[0] < 50000:   # smaller number of learning epochs is enough for large datasets
        config.n_epochs = 60
        config.early_stop = 40
    else:
        config.n_epochs = 30
        config.early_stop = 20

    print(config)
    # exit()
    roc, prn = train(config, anomaly_train_loader, normal_train_loader, test_loader, num_train_samples, X_train, y_unlabel, X_inference, y, train_normal_indice, train_anomaly_indice)
    rocs.append(roc)
    prns.append(prn)

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument('--dataset', default='Cardiotocography.json', type=str, help='Dataset') 
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size, you may increase it when dealing with large datasets')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--labeled_ratio', default=0.05, type=float, help='Ratio of labeled anomalies')
    parser.add_argument('-o', default=None, type=str, help='Output file')
    return parser.parse_args()


config = parse_args()
seed_list = range(10)
rocs = []
prns = []
for seed in seed_list:
    run(config, rocs, prns, seed)
    print(rocs)
    print(prns)
    print(np.mean(rocs), np.mean(prns))

if config.o is not None:
    with open(config.o, 'a') as f:
        f.write(str(config)+'\n')
        f.write(str(rocs)+'\n')
        f.write(str(prns)+'\n')
        f.write(str(np.mean(rocs))+'  ' +str(np.mean(prns))+'\n\n')
