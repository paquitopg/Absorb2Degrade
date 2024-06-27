#!/usr/bin/env python
# coding: utf-8

# In[ ]:

'This notebook determines the applicability domain of a model.'
'It uses the k-NN approach developed by sahigara&Al'

'further development :'
'allow for the use of different distance metrics'
'Optimize the choice of the parameter k'


import numpy as np
import pandas as pd
#from sklearn.neighbors import NearestNeighbors
import sys
from tqdm import tqdm


# In[1]:
def get_trainset(data_path: str):
    return pd.read_parquet(data_path)

def get_testset(data_path: str):
    return pd.read_parquet(data_path)

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def get_thresholds(trainset, k):
    thresholds = np.zeros(len(trainset))
    K_i = np.zeros(len(trainset))
    mat = np.zeros((len(trainset), len(trainset))) 
    mean = np.zeros(len(trainset))

    for i in range(len(trainset)):
        for j in range(len(trainset)):
            mat[i,j] = euclidean_distance(trainset.iloc[i], trainset.iloc[j])
        mat[i] = np.sort(mat[i])
        mean[i] = np.mean(mat[i][1:k+1])

    q1mean = np.percentile(mean, 25)
    q3mean = np.percentile(mean, 75)
    interquartile = q3mean - q1mean
    refvalue = q3mean + 1.5*interquartile

    for i in range(len(trainset)):
        for j in range(1, len(trainset)):
            if mat[i][j] <= refvalue:
                K_i[i] = K_i[i] + 1
            else:
               mat[i][j] = 0
        if K_i[i] != 0:
            thresholds[i] = np.sum(mat[i])/K_i[i]
    non_zero_thresholds = thresholds[thresholds != 0]
    min_threshold = np.min(non_zero_thresholds)
    for i in range(len(trainset)):
        if K_i[i] == 0:
            thresholds[i] = min_threshold
        
    return mat, mean, thresholds, refvalue


# In[2]:


def determine_AD(trainset, testset, k):
    thresholds = get_thresholds(trainset, k)
    AD = np.zeros(len(testset))
    for i in tqdm(range(len(testset))):
        for j in range(len(trainset)):
            if euclidean_distance(testset.iloc[i], trainset.iloc[j]) <= thresholds[j]:
                AD[i] = 1
                break
    return AD

if __name__ == '__main__':
    if len(sys.argv) == 4: 
        trainset = get_trainset(sys.argv[1]) 
        testset = get_testset(sys.argv[2])
        k = int(sys.argv[3])
        result = determine_AD(trainset, testset, k)
        df = pd.DataFrame(result, columns=['AD'])
        df.to_csv(f'AD_k={k}.csv', index=False)
    else:
        print("Usage: AD_determination.py trainset testset k")
