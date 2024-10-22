import scipy.io as scio
import os
import h5py
import json
import numpy as np
from scipy.stats import zscore
import os

from scipy.io import loadmat

from sklearn.utils import check_array


def read_from_ODDS():
    datasets = []
    files = os.listdir('./ODDSdataset/')
    for file in files:
        # print(file)
        if not file.endswith('.mat'):
            continue
        path = './dataset/' + file
        try:
            data = scio.loadmat(path)   
            temp = {'name': file.split('.')[0], 'x': data['X'], 'y': data['y']}
            datasets.append(temp)
        except NotImplementedError:
            data = h5py.File(path, 'r')  
            temp = {'name': file.split('.')[0], 'x': np.array(data['X']), 'y': np.array(data['y'])}
            if temp['name'] in ['http', 'smtp']:
                temp['x'] = temp['x'].T
                temp['y'] = temp['y'].reshape((-1,))
            datasets.append(temp)
            data.close()  
    return datasets

def read_from_adbench():
    from adbench.datasets.data_generator import DataGenerator
    D = DataGenerator(generate_duplicates=False, test_size=0.95)
    D.dataset = '11_donors'
    d = D.generator(la=1.0) 
    # print(d['X_train'].shape, d['y_train'].shape)
    # print(sum(d['y_train']), sum(d['y_test']))
    X = np.concatenate((d['X_train'], d['X_test']), axis=0)
    y = np.concatenate((d['y_train'], d['y_test']), axis=0)
    dataset = {}
    dataset['X'] = X
    dataset['y'] = y
    dataset['name'] = 'donors'
    datasets = [dataset]
    return datasets

def fix_nan(X):
    col_mean = np.nanmean(X, axis = 0) 
    col_mean[np.isnan(col_mean)] = 0
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1]) 
    return X

def preprocess(datasets):
    not_include = []
    for i in range(len(datasets)):
        X = datasets[i]['X']
        y = datasets[i]['y']
        y = np.squeeze(y)
        name = datasets[i]['name']
        print(name)
        print(X.shape)
        std = np.std(X, axis=0)
        mean = np.mean(X, axis=0)
        indices = np.where(std > 0)[0]
        X = X[:, indices]
        X = zscore(X, axis=0)
        print(X.shape)
        # break
        with open('./processed_data/' + name + '.json', 'w') as f_obj:
            json.dump((X.tolist(), y.tolist()), f_obj)
    print(not_include)
    return  



# datasets = read_from_ODDS()
datasets = read_from_adbench()
meta_mat_transformed = preprocess(datasets)



