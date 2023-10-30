#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 23:44:53 2022

@author: jgjang
"""

from sklearn.decomposition import IncrementalPCA
import scipy.io as sio
import glob
import os
import numpy as np
from pickle import dump
from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Model update

class HSI_PCA():
    def __init__(self, load=False):
        self.path = 'data/'

        self.train_files = glob.glob('data/train/*mat')
        self.val_files = glob.glob('data/val/*mat')
        self.test_files = glob.glob('data/test/*mat')
        self.test_clean_files = glob.glob('data/test_clean/*mat')
        self.test_effect_files = glob.glob('data/test_effect/*mat')

        mean_std = sio.loadmat('./data/mean_std.mat')

        self.mean = mean_std['mean'] # (1,127)
        self.std = mean_std['std'] # (1,127)
        self.n_batches = 200
        self.n_components= 30
        self.in_channel = 127
        self.inc_pca = IncrementalPCA(n_components=self.n_components)
        if load:
            self.load_data()
            if os.path.isfile('./data/pca_mean_std.mat'):
                self.mean_pca = sio.loadmat('./data/pca_mean_std.mat')['mean']
                self.std_pca = sio.loadmat('./data/pca_mean_std.mat')['std']

    def normalize(self, img):
        img = img.reshape([-1,self.in_channel])
        #  mean = np.mean(img, axis=0, keepdims=True)
        #  std = np.std(img, axis=0, keepdims=True)
        img_org = img +0.0
        _max = np.max(img, axis=0, keepdims=True)
        tmp = img / _max
        mean = np.zeros([1,self.in_channel])
        for i in range(self.in_channel):
            tmp2 =  np.bincount(np.uint8(tmp[:,i]*255))
            tmp2[0] = 0
            mean[0,i] = np.argmax(tmp2) / 255.0 * _max[0,i]

        img = (img - self.mean) / self.std
        #  img = img / mean
        img[img_org.sum(axis=1) == 0, :] = 0
        return img


    def update_pca(self):
        self.train_files = np.random.permutation(self.train_files)
        for X_batch in tqdm(np.array_split(self.train_files,self.n_batches)):
            # Load batch_data
            data = list()
            for ind in X_batch:
                org_ = sio.loadmat(ind)['data']
                normalized_ = self.normalize(org_)
                data.append(normalized_)
            data = np.concatenate(data,axis=0).reshape([-1, self.in_channel ])
            self.inc_pca.partial_fit(data)
            print('Explained variance ratio 1st component: ', self.inc_pca.explained_variance_ratio_[0])
            print('Explained variance ratio 2nd component: ', self.inc_pca.explained_variance_ratio_[0:2].sum())
            print('Explained variance ratio 3rd component: ', self.inc_pca.explained_variance_ratio_[0:3].sum())
            print('Explained variance ratio 4th component: ', self.inc_pca.explained_variance_ratio_[0:4].sum())
            print('Explained variance ratio 5th component: ', self.inc_pca.explained_variance_ratio_[0:5].sum())
            print('Explained variance ratio total: ', self.inc_pca.explained_variance_ratio_.sum())
        print('PCA update done')
    def plot(self):
        # plt.plot(inc_pca.singular_values_)
        plt.xlim((0,20))
        plt.xlabel('Component index')
        plt.ylabel('Cumulative covariance')
        # plt.ylim((0.99,1))
        plt.plot(np.cumsum(self.inc_pca.explained_variance_ratio_[1::])/self.inc_pca.explained_variance_ratio_[1::].sum())
        
        plt.show()
    def save_data(self):
        dump(self.inc_pca, open('./data/model.pkl', 'wb'))
        
    def load_data(self):
        self.inc_pca = load(open('./data/model.pkl', 'rb'))
    
   
    def compute_mean_std(self):
        mean = np.zeros([1,self.n_components])
        var = np.zeros([1,self.n_components])
        self.train_files = np.random.permutation(self.train_files)
        for ind in tqdm(self.train_files):
            org_ = sio.loadmat(ind)['data']
            org_ = self.normalize(org_)
            pca_result = self.inc_pca.transform(org_.reshape([-1,self.in_channel])).astype(np.float32)
            mean += np.mean(pca_result,axis=0,keepdims=True)
        mean /= len(self.train_files)

        for X_batch in tqdm(self.train_files):
            org_ = sio.loadmat(ind)['data']
            org_ = self.normalize(org_)
            pca_result = self.inc_pca.transform(org_.reshape([-1,self.in_channel])).astype(np.float32)
            var += np.sum((pca_result - mean)**2,axis=0,keepdims=True)
        var /= len(self.train_files)
        std = np.sqrt(var)
        sio.savemat('./data/pca_mean_std.mat',{'mean':mean,'std':std})
        print('mean: ',mean)
        print('std: ',std)
        print('done')

    def transform(self,img):
        shp = img.shape[0:2]+ (self.n_components,)
        img = img.reshape([-1,self.in_channel])
        img = self.normalize(img)
        #  img = (img - self.mean) / self.std
        img = self.inc_pca.transform(img)
        #  img = (img - self.mean_pca)/self.std_pca
        return img.reshape(shp).astype(np.float32)

        #  return self.inc_pca.transform(img.reshape([-1,self.in_channel])).reshape(shp).astype(np.float32)
        
    def inverse_transform(self,img):
        shp = img.shape[0:2] + (self.in_channel,)
        img = img.reshape([-1,self.n_components])
        img = img * self.std_pca + self.mean_pca
        img = self.inc_pca.inverse_transform(img)
        img = img * self.std + self.mean
        return img.reshape(shp)
        #  img = self.inc_pca.inverse_transform(img.reshape([-1,self.n_components])).reshape(shp)
        #  return img

    def save_imgs(self):
        if not os.path.isdir('./data/train_pca'):
            os.mkdir('./data/train_pca')
        for ind in tqdm(self.train_files):
            org_ = sio.loadmat(ind)['data']
            pca_result = self.transform(org_).astype(np.float32)
            sio.savemat('./data/train_pca/'+ind.split('/')[-1],{'data':pca_result})
        print('done')
        if not os.path.isdir('./data/test_pca'):
            os.mkdir('./data/test_pca')
        for ind in tqdm(self.test_files):
            org_ = sio.loadmat(ind)['data']
            pca_result = self.transform(org_).astype(np.float32)
            sio.savemat('./data/test_pca/'+ind.split('/')[-1],{'data':pca_result})
        print('done')
        if not os.path.isdir('./data/val_pca'):
            os.mkdir('./data/val_pca')
        for ind in tqdm(self.val_files):
            org_ = sio.loadmat(ind)['data']
            pca_result = self.transform(org_).astype(np.float32)
            sio.savemat('./data/val_pca/'+ind.split('/')[-1],{'data':pca_result})
        print('done')

        if not os.path.isdir('./data/test_clean_pca'):
            os.mkdir('./data/test_clean_pca')
        for ind in tqdm(self.test_clean_files):
            org_ = sio.loadmat(ind)['data']
            pca_result = self.transform(org_).astype(np.float32)
            sio.savemat('./data/test_clean_pca/'+ind.split('/')[-1],{'data':pca_result})
        print('done')

        if not os.path.isdir('./data/test_effect_pca'):
            os.mkdir('./data/test_effect_pca')
        for ind in tqdm(self.test_effect_files):
            org_ = sio.loadmat(ind)['data']
            pca_result = self.transform(org_).astype(np.float32)
            sio.savemat('./data/test_effect_pca/'+ind.split('/')[-1],{'data':pca_result})

        
if __name__ == '__main__':
    print('start')
    pca = HSI_PCA()
    pca.update_pca()
    pca.save_data()
    #  pca = HSI_PCA(load=True)
    pca.save_imgs()
    #  pca.compute_mean_std()
    #  pca = HSI_PCA(load=True)
    #  A = sio.loadmat('data/train/1006.mat')['data']
    #  A = HSI_PCA.transform(A)
    #  print(A[122,122,:])
    #  print(A[122,123,:])
    #  print(A[122,126,:])
    #  print(A[138,186,:])
    #  print(A[18,85,:])
    #  A = np.ones([1,10,127])
    #  print(HSI_PCA.transform(A)[0,0,:])
    #
    #  A = np.ones([1, 10,127])
    #  B = np.random.normal(size=[10,10,127])
    #  A = np.concatenate([A,B],axis=0)
    #  print(HSI_PCA.transform(A)[0,0,:])
