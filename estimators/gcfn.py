#GCFN this code comes from the paper "General Control Function for Instrumental Variables"
#It can be found on https://github.com/rajesh-lab/gcfn-code
#I slightly modified the code to be able to use it as a class GCFN as shown in the example below.



from __future__ import print_function

import math
import copy
import argparse
import sys
import datetime
import time
import pdb
import os


import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.distributions as dists

import pytorch_lightning as pl


from utils_models import VDE, OutcomeModel, permutation, evaluate
from utils_data import generate_data_additive_treatment_process, generate_data_multiplicative_treatment_process, XZETY_Dataset

import matplotlib.pyplot as plt


class GCFN():
    def __init__(self):
        self.out_model = None
        self.args=None
        
    
    def fit(self,X,Y,Z): 
        dirpath = os.path.dirname(os.path.realpath(__file__)) # no end-slash in this name
        print(dirpath)
        torch.set_printoptions(precision=5, linewidth=140)
        
        parser = argparse.ArgumentParser(description='The General Control Function Method')
        parser.add_argument('--save_dir', type=str, default=os.path.realpath(__file__))
        parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for training (default: 500)')
        #number of epochs first stage can be changed here
        parser.add_argument('--max_epochs', type=int, default=20, metavar='N', #20
                            help='number of epochs to train (default: 2)')
        #number of epochs second stage can be changed here
        parser.add_argument('--max_out_epochs', type=int, default=20, metavar='N',#20
                            help='number of outcome epochs to train (default: 2)')
        parser.add_argument('--no-fig', action='store_true',
                            default=False, help='saves figures')
        parser.add_argument('--seed', type=int, default=1000, metavar='S',
                            help='random seed (default: 1000)')
        parser.add_argument('--lambda_', type=float, default=0.3)
        parser.add_argument('--verbose', type=int, default=0)
        parser.add_argument('--sample_size', type=int, default=len(X))
        parser.add_argument('--prefix', type=str, default="DUMMY_RUN")
        parser.add_argument('--pdb', type=bool, default=False)
        parser.add_argument('--load_vde', type=str, default=None,
                            help='path to vde model you want to load')
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--zhat_dim', type=int, default=1)#50
        parser.add_argument('--t_dim', type=int, default=20)
        parser.add_argument('--vde_d', type=int, default=0)
        parser.add_argument('--vde_K', type=int, default=2)
        parser.add_argument('--out_d', type=int, default=50)
        parser.add_argument('--out_K', type=int, default=2)
        parser.add_argument('--experiment', type=str, default='add')
        parser.add_argument('--recon_likelihood', type=str, default='cat')
        parser.add_argument('--gpus', type=int, default=1,
                            help='set None to disable gpu')
        
        
        self.args = parser.parse_args()
        
        self.args.gpus=None
        
        self.args.cuda = self.args.gpus is not None and torch.cuda.is_available()
        if self.args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        torch.manual_seed(self.args.seed)
        
  
        # VDE learning rates
        self.args.lr_enc = 1e-2
        self.args.lr_dec = 1e-2
        self.args.lr_qzhat = 1e-2
        self.args.lr_lb = 1e-5
        
        # OUT learning rates
        self.args.lr_out = 1e-3
        self.args.lr_beta = 1e-2
        self.args.out_lr_lb = 1e-5
        
    
        
        """ ========================= data variables and data loaders ========================= """
        batch_size = self.args.batch_size

        
        def get_shape(tensor):
            if tensor is None:
                return 0
            else:
                if len(tensor.shape) > 1:
                    return tensor.shape[1]
                elif len(tensor.shape) == 1:
                    return 1
                else:
                    assert False, 'input has no dimensions'
        
        
        # updating args with
        self.args.d_x = get_shape(None)
        self.args.d_e = get_shape(torch.tensor(Z, dtype=torch.float))
        if self.args.vde_d == 0:
            self.args.vde_d = 100
        
        filename_prefix = "_{}_lambda_{}_zhatdim{}_bs{}_e{}_ss{}_seed{}_".format(self.args.prefix,
                                                                                 str(self.args.lambda_).replace(
                                                                                     '.', ''),
                                                                                 self.args.zhat_dim,
                                                                                 self.args.batch_size,
                                                                                 self.args.max_epochs,
                                                                                 self.args.sample_size,
                                                                                 self.args.seed)
        self.args.filename = filename_prefix
        
    
        
        """ ========================= data variables and data loaders ========================= """
       

        
        #the first torch.tensor(Z, dtype=torch.float) just needs to have the same dimension as the confounder H, normally we should put H. 
        #It is however not used in the regression.
        full_dataset =XZETY_Dataset( torch.tensor(Z, dtype=torch.float), torch.tensor(Z, dtype=torch.float), torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float),None)
        
        m_train = int(0.8*self.args.sample_size)
        m_val = int(0.1*self.args.sample_size)
        m_test = self.args.sample_size - m_train - m_val
        
        
        train_data, val_data, test_data = random_split(
            full_dataset, [m_train, m_val, m_test])
        train_loader, val_loader, test_loader = DataLoader(train_data, batch_size=self.args.batch_size), DataLoader(
            val_data, batch_size=m_val), DataLoader(test_data, batch_size=m_test)
        
        train_x = train_data.dataset[train_data.indices][0]
        train_t = train_data.dataset[train_data.indices][3]
        train_eps = train_data.dataset[train_data.indices][2]
        
        
        vde_model = VDE(self.args)
        vde_trainer = pl.Trainer( gpus=self.args.gpus,max_epochs=self.args.max_epochs,logger=None)
        vde_trainer.fit(vde_model, train_loader, val_loader)
        
        self.out_model = OutcomeModel(self.args, vde_model=vde_model, ce_fn=None)
        # assert False, "figure out what to give here."
        
        
        
        zhat_samples = vde_model.sample_zhat_from_data((None, None, train_eps, train_t, None)).view(-1)
        zhat_samples = permutation(zhat_samples.detach())
        self.out_model.zhat = zhat_samples
        
        # ------------
        # outcome training
        # ------------
        outcome_trainer = pl.Trainer(gpus=self.args.gpus, max_epochs=self.args.max_out_epochs,
                                     logger=None)
        outcome_trainer.fit(self.out_model, train_loader, val_loader)
        outcome_trainer.test(self.out_model, test_dataloaders=test_loader)
        
    def predict(self,X):
        
        effect_pred = self.out_model.predict_y_do_t(torch.tensor(X, dtype=torch.float)).detach().cpu()
        return effect_pred.numpy()
       
"""
#Example

n = 2000

Z = np.random.normal(0, 4, n)
H = np.random.normal(0, 2, n)
X = Z + H + np.random.normal(0,1,n)
Y = abs(X)  +  H + np.random.normal(0, 1, n)



estimator = GCFN()
estimator.fit(X, Y,Z)
predict = estimator.predict(X)

# Plotting
plt.scatter(X, Y, marker='o', s=0.25, label="True")
orderX = np.argsort(X)

#estimator
plt.plot(X[orderX], predict[orderX], color="red", linewidth=3, label="Predicted")

#true causal effect
plt.plot(X[orderX], np.abs(X)[orderX], color="green", linewidth=3, label="True")

plt.legend(loc="upper right")
plt.show()
"""